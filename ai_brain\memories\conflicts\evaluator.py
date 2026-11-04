"""
Evidence evaluation for MemoryConflict instances.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ai_brain.memories.conflicts.models import MemoryConflict
from ai_brain.memories.conflicts.types import ConflictStatus, ConflictType
from ai_brain.memories.models import Memory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConflictAssessment:
    """
    Evidence-based evaluation outcome for a MemoryConflict.
    """
    conflict_id: str
    is_resolvable: bool
    recommended_status: ConflictStatus
    winning_memory_id: Optional[str] = None
    losing_memory_ids: Tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.5
    reasoning: str = ""
    evidence_scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


class ConflictEvaluator:
    """
    Evaluates evidence surrounding a MemoryConflict.
    
    Examines:
    - Independent agent consensus (corroboration by distinct agents).
    - Event frequency and supporting event volume.
    - Explicit superseding or migration statements.
    - Stored confidence and importance metrics.
    - Temporal context without relying solely on recency.
    """

    def evaluate(
        self,
        conflict: MemoryConflict,
        memories: Sequence[Memory],
    ) -> ConflictAssessment:
        """
        Evaluate evidence for a specific conflict against the memory pool.
        
        Args:
            conflict: The MemoryConflict under review.
            memories: Context memories containing the memories involved in conflict.
            
        Returns:
            ConflictAssessment detailing resolution feasibility, winner, and evidence scoring.
        """
        involved_map = {m.memory_id: m for m in memories if m.memory_id in conflict.memory_ids}

        # If not all memories are available, conflict cannot be evaluated reliably
        if len(involved_map) < len(conflict.memory_ids):
            return ConflictAssessment(
                conflict_id=conflict.conflict_id,
                is_resolvable=False,
                recommended_status=ConflictStatus.UNRESOLVED,
                confidence=0.0,
                reasoning="Missing involved memories from evaluation context",
            )

        # Check for temporal migration evidence
        if conflict.conflict_type == ConflictType.TEMPORAL:
            return self._evaluate_temporal(conflict, list(involved_map.values()))

        # Check for task status conflict
        if conflict.conflict_type == ConflictType.STATUS:
            return self._evaluate_status(conflict, list(involved_map.values()))

        # General evidence scoring for factual / technical / decision / source conflicts
        scores: Dict[str, float] = {}
        rationale: Dict[str, List[str]] = defaultdict(list)

        for m_id, m in involved_map.items():
            score = 0.0

            # 1. Distinct event grounding (deduplicated, diminishing returns, hard capped at 0.8)
            unique_events = list(dict.fromkeys(m.source_event_ids))
            num_events = len(unique_events)
            if num_events == 0:
                event_score = 0.0
            elif num_events == 1:
                event_score = 0.5
            elif num_events == 2:
                event_score = 0.7
            else:
                event_score = 0.8
            score += event_score
            rationale[m_id].append(f"{num_events} unique events (+{event_score:.2f})")

            # 2. Independent agent corroboration (deduplicated, diminishing returns, hard capped at 1.3)
            unique_agents = list(dict.fromkeys(m.source_agent_ids))
            num_agents = len(unique_agents)
            if num_agents <= 1:
                agent_score = 0.2 if num_agents == 1 else 0.0
            elif num_agents == 2:
                agent_score = 1.0  # Clear independent corroboration boost
            else:
                agent_score = 1.3
            score += agent_score
            rationale[m_id].append(f"{num_agents} unique agents (+{agent_score:.2f})")

            # 3. Substantive verification, confirmation, or correction (capped at 2.0)
            verif_score = 0.0
            if m.metadata.get("verified") is True or m.metadata.get("confirmed") is True:
                verif_score += 1.5
            if m.metadata.get("correction") is True:
                verif_score += 1.5
            if m.metadata.get("tool_result") is not None:
                verif_score += 1.0
            verif_score = min(2.0, verif_score)
            if verif_score > 0:
                score += verif_score
                rationale[m_id].append(f"verification (+{verif_score:.2f})")

            # 4. Self-reported confidence (restricted weighting: max impact +/-0.20 across full [0, 1] range)
            # Cannot force resolution by itself (resolution threshold is 0.80)
            conf_weight = round((m.confidence - 0.5) * 0.4, 2)
            score += conf_weight
            rationale[m_id].append(f"confidence_weight ({conf_weight:+.2f})")

            scores[m_id] = round(score, 2)

        # Sort by score
        sorted_candidates = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top_score = sorted_candidates[0]
        second_id, second_score = sorted_candidates[1]

        # Resolution threshold: margin must be >= 0.80
        margin = round(top_score - second_score, 2)
        if margin >= 0.80:
            losing_ids = tuple(m_id for m_id in involved_map.keys() if m_id != top_id)
            win_mem = involved_map[top_id]
            reason = (
                f"Memory '{top_id}' established by strong evidence ({', '.join(rationale[top_id])}) "
                f"exceeding alternative '{second_id}' (margin: {margin:.2f})."
            )
            return ConflictAssessment(
                conflict_id=conflict.conflict_id,
                is_resolvable=True,
                recommended_status=ConflictStatus.RESOLVED,
                winning_memory_id=top_id,
                losing_memory_ids=losing_ids,
                confidence=min(0.95, round(0.75 + min(0.20, margin * 0.05), 2)),
                reasoning=reason,
                evidence_scores=scores,
                details={"winner_content": win_mem.content, "margin": margin},
            )
        else:
            return ConflictAssessment(
                conflict_id=conflict.conflict_id,
                is_resolvable=False,
                recommended_status=ConflictStatus.UNRESOLVED,
                confidence=0.5,
                reasoning=(
                    f"Evidence is balanced or inconclusive between '{top_id}' (score: {top_score}) "
                    f"and '{second_id}' (score: {second_score}, margin: {margin:.2f}). Claims remain disputed."
                ),
                evidence_scores=scores,
                details={"margin": margin},
            )

    def _evaluate_temporal(
        self,
        conflict: MemoryConflict,
        memories: List[Memory],
    ) -> ConflictAssessment:
        """Evaluate temporal conflicts where one claim describes migration or subsequent update."""
        sorted_by_time = sorted(memories, key=lambda m: m.created_at)
        earlier = sorted_by_time[0]
        later = sorted_by_time[-1]

        later_lower = later.content.lower()
        migration_indicators = ["migrated", "switched to", "replaced with", "now uses"]
        
        if any(ind in later_lower for ind in migration_indicators):
            reason = (
                f"Later memory '{later.memory_id}' explicitly documents a chronological transition: "
                f"'{later.content}' updates earlier state '{earlier.content}'."
            )
            return ConflictAssessment(
                conflict_id=conflict.conflict_id,
                is_resolvable=True,
                recommended_status=ConflictStatus.RESOLVED,
                winning_memory_id=later.memory_id,
                losing_memory_ids=(earlier.memory_id,),
                confidence=0.92,
                reasoning=reason,
                details={"transition_type": "chronological_migration"},
            )

        # If later claim does not explicitly document transition, evidence is insufficient
        return ConflictAssessment(
            conflict_id=conflict.conflict_id,
            is_resolvable=False,
            recommended_status=ConflictStatus.UNRESOLVED,
            confidence=0.5,
            reasoning=f"Claims differ temporally between {earlier.created_at} and {later.created_at} without documented migration.",
        )

    def _evaluate_status(
        self,
        conflict: MemoryConflict,
        memories: List[Memory],
    ) -> ConflictAssessment:
        """Evaluate task status conflicts (e.g. active vs completed)."""
        active_mems = [m for m in memories if m.status == "ACTIVE"]
        archived_mems = [m for m in memories if m.status in ("ARCHIVED", "SUPERSEDED")]

        if active_mems and archived_mems:
            archived = archived_mems[0]
            # If archived task has later timestamp or explicitly documents completion
            if archived.created_at >= active_mems[0].created_at or archived.metadata.get("completed"):
                return ConflictAssessment(
                    conflict_id=conflict.conflict_id,
                    is_resolvable=True,
                    recommended_status=ConflictStatus.RESOLVED,
                    winning_memory_id=archived.memory_id,
                    losing_memory_ids=(active_mems[0].memory_id,),
                    confidence=0.95,
                    reasoning=f"Task completed in '{archived.memory_id}' supersedes active task state in '{active_mems[0].memory_id}'.",
                )

        return ConflictAssessment(
            conflict_id=conflict.conflict_id,
            is_resolvable=False,
            recommended_status=ConflictStatus.UNRESOLVED,
            confidence=0.5,
            reasoning="Uncertain task completion status; insufficient evidence to resolve.",
        )
