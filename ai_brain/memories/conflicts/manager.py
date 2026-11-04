"""
Conflict Manager for AI Brain.
Manages conflict resolution, lifecycle status transitions, and integration with MemoryUpdater.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Sequence, Tuple

from ai_brain.memories.conflicts.evaluator import ConflictAssessment, ConflictEvaluator
from ai_brain.memories.conflicts.models import MemoryConflict
from ai_brain.memories.conflicts.types import ConflictStatus
from ai_brain.memories.exceptions import ConflictResolutionError
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus
from ai_brain.memories.updating.updater import MemoryUpdater

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConflictResolutionResult:
    """
    Structured outcome of a conflict resolution operation.
    """
    conflict: MemoryConflict
    status: ConflictStatus
    action_taken: str
    affected_memories: Tuple[Memory, ...]
    updated_memory: Optional[Memory]
    superseded_memories: Tuple[Memory, ...]
    message: str


class ConflictManager:
    """
    Coordinates conflict lifecycle transitions and evidence-supported resolutions.
    
    Responsibilities:
    - Oversees state transitions: DETECTED -> UNDER_REVIEW -> RESOLVED / UNRESOLVED / DISMISSED.
    - Integrates cleanly with MemoryUpdater for memory supersession upon resolution.
    - Preserves all conflicting claims when evidence is insufficient without silently picking a winner.
    - Records dismissals and reasons without mutating underlying valid memories.
    """

    def __init__(
        self,
        updater: Optional[MemoryUpdater] = None,
        evaluator: Optional[ConflictEvaluator] = None,
    ) -> None:
        self.updater = updater or MemoryUpdater()
        self.evaluator = evaluator or ConflictEvaluator()
        self._tracked_conflicts: Dict[str, MemoryConflict] = {}

    def track(self, conflict: MemoryConflict) -> None:
        """Register a conflict in the manager."""
        self._tracked_conflicts[conflict.conflict_id] = conflict

    def get_conflict(self, conflict_id: str) -> Optional[MemoryConflict]:
        """Retrieve a tracked conflict by ID."""
        return self._tracked_conflicts.get(conflict_id)

    def all_conflicts(self) -> List[MemoryConflict]:
        """Return all tracked conflicts."""
        return list(self._tracked_conflicts.values())

    def resolve(
        self,
        conflict: MemoryConflict,
        evidence: ConflictAssessment,
        memories: Sequence[Memory],
    ) -> ConflictResolutionResult:
        """
        Resolve a conflict using evidence assessment.
        
        Args:
            conflict: The MemoryConflict to resolve.
            evidence: ConflictAssessment provided by ConflictEvaluator.
            memories: Collection containing the involved Memory objects.
            
        Returns:
            ConflictResolutionResult detailing state changes and affected memories.
        """
        mem_map = {m.memory_id: m for m in memories if m.memory_id in conflict.memory_ids}

        # Case 1: Resolvable with sufficient evidence
        if evidence.is_resolvable and evidence.winning_memory_id:
            winner = mem_map.get(evidence.winning_memory_id)
            if not winner:
                raise ConflictResolutionError(
                    f"Winning memory '{evidence.winning_memory_id}' not found in provided memories"
                )

            superseded_list: List[Memory] = []
            affected_list: List[Memory] = [winner]

            # Supersede losing memories via MemoryUpdater
            current_winner = winner
            for loser_id in evidence.losing_memory_ids:
                loser = mem_map.get(loser_id)
                if loser:
                    update_res = self.updater.update(
                        existing_memory=loser,
                        incoming_memory=current_winner,
                        strategy="supersede",
                    )
                    if update_res.previous_memory:
                        superseded_list.append(update_res.previous_memory)
                    if update_res.current_memory:
                        current_winner = update_res.current_memory
                    affected_list.extend(update_res.affected_memories)

            resolution_data = {
                "winner_id": evidence.winning_memory_id,
                "losing_ids": list(evidence.losing_memory_ids),
                "reasoning": evidence.reasoning,
                "confidence": evidence.confidence,
                "resolved_at": evidence.recommended_status.value,
            }

            resolved_conflict = conflict.with_status(
                ConflictStatus.RESOLVED,
                resolution=resolution_data,
                metadata={"action": "superseded_losing_memories"},
            )
            self._tracked_conflicts[resolved_conflict.conflict_id] = resolved_conflict

            seen_ids = set()
            deduped_affected = []
            for mem in affected_list:
                if mem.memory_id not in seen_ids:
                    seen_ids.add(mem.memory_id)
                    deduped_affected.append(mem)

            return ConflictResolutionResult(
                conflict=resolved_conflict,
                status=ConflictStatus.RESOLVED,
                action_taken="RESOLVED_VIA_EVIDENCE",
                affected_memories=tuple(deduped_affected),
                updated_memory=current_winner,
                superseded_memories=tuple(superseded_list),
                message=f"Conflict '{conflict.conflict_id}' resolved. Winning memory: '{evidence.winning_memory_id}'.",
            )

        # Case 2: Insufficient evidence -> remains UNRESOLVED
        unresolved_conflict = conflict.with_status(
            ConflictStatus.UNRESOLVED,
            resolution={"status": "unresolved", "reason": evidence.reasoning},
            metadata={"evaluation_confidence": evidence.confidence},
        )
        self._tracked_conflicts[unresolved_conflict.conflict_id] = unresolved_conflict

        # Ensure all involved memories remain marked DISPUTED
        disputed_mems = []
        for m in mem_map.values():
            if m.status != MemoryStatus.DISPUTED.value:
                m_disp = m.with_updates(
                    status=MemoryStatus.DISPUTED.value,
                    relationships={"conflict_id": conflict.conflict_id},
                    memory_id=m.memory_id,
                )
                disputed_mems.append(m_disp)
            else:
                disputed_mems.append(m)

        return ConflictResolutionResult(
            conflict=unresolved_conflict,
            status=ConflictStatus.UNRESOLVED,
            action_taken="PRESERVED_AS_DISPUTED",
            affected_memories=tuple(disputed_mems),
            updated_memory=None,
            superseded_memories=(),
            message=f"Conflict '{conflict.conflict_id}' remains unresolved. Claims preserved as DISPUTED.",
        )

    def dismiss(self, conflict: MemoryConflict, reason: str) -> ConflictResolutionResult:
        """
        Dismiss a conflict that was determined not to be a genuine disagreement.
        
        Args:
            conflict: The MemoryConflict to dismiss.
            reason: Detailed rationale for dismissal.
            
        Returns:
            ConflictResolutionResult documenting dismissal.
        """
        if not reason or not isinstance(reason, str) or not reason.strip():
            raise ConflictResolutionError("Dismissal reason must be a non-empty string")

        dismissed = conflict.with_status(
            ConflictStatus.DISMISSED,
            resolution={"dismissal_reason": reason.strip()},
            metadata={"dismissed": True},
        )
        self._tracked_conflicts[dismissed.conflict_id] = dismissed

        return ConflictResolutionResult(
            conflict=dismissed,
            status=ConflictStatus.DISMISSED,
            action_taken="DISMISSED",
            affected_memories=(),
            updated_memory=None,
            superseded_memories=(),
            message=f"Conflict '{conflict.conflict_id}' dismissed: {reason.strip()}",
        )
