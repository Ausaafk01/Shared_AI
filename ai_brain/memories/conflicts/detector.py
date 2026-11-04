"""
Conflict detection between Memory objects.
"""

from collections import defaultdict
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ai_brain.memories.conflicts.models import MemoryConflict
from ai_brain.memories.conflicts.types import ConflictStatus, ConflictType
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus, MemoryType
from ai_brain.memories.updating.comparison import extract_memory_subject

logger = logging.getLogger(__name__)


TECH_ENTITIES: Dict[str, List[str]] = {
    "tech:database": ["postgresql", "postgres", "mysql", "sqlite", "mongodb", "mariadb"],
    "tech:authentication": ["jwt", "oauth", "session", "cookie"],
    "tech:caching": ["redis", "memcached"],
    "tech:language": ["python", "typescript", "javascript", "go", "rust"],
    "tech:framework": ["fastapi", "django", "flask", "express", "next.js", "nextjs"],
    "tech:orm": ["sqlalchemy", "prisma", "alembic", "django orm"],
    "tech:testing": ["pytest", "unittest", "jest"],
    "tech:os": ["linux", "ubuntu", "windows", "macos"],
}

ENTITY_CANONICAL: Dict[str, str] = {
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "sessions": "session",
    "cookies": "cookie",
    "next.js": "nextjs",
    "nextjs": "nextjs",
}


def _is_temporal_transition(m1: Memory, m2: Memory) -> bool:
    """Check if two memories reflect a chronological/temporal progression rather than direct simultaneous contradiction."""
    earlier, later = (m1, m2) if m1.created_at <= m2.created_at else (m2, m1)
    later_text = later.content.lower()
    
    progression_words = [
        "migrated", "switched", "upgraded", "replaced", "deprecated",
        "yesterday", "subsequently", "now uses", "removed",
    ]
    return any(w in later_text for w in progression_words)


def _is_source_or_provenance_disagreement(m1: Memory, m2: Memory) -> bool:
    """
    Check if the disagreement specifically concerns the credibility, provenance,
    reliability, or attribution of the evidence itself.
    """
    if m1.metadata.get("provenance_dispute") or m2.metadata.get("provenance_dispute"):
        return True
    if m1.metadata.get("attribution_dispute") or m2.metadata.get("attribution_dispute"):
        return True

    prov_patterns = [
        r"\b(?:author(?:ed|ship)?|created by|author of|attributed to|attribution|misattributed|misattribution)\b",
        r"\b(?:credibility|reliable source|unreliable source|untrustworthy|trustworthy source)\b",
        r"\b(?:provenance|authentic|authenticity|unauthentic|falsified|fabricated|counterfeit)\b",
        r"\b(?:source event|event validity|invalid event|fake event|phantom event)\b",
    ]
    t1, t2 = m1.content.lower(), m2.content.lower()
    return any(re.search(p, t1) for p in prov_patterns) or any(re.search(p, t2) for p in prov_patterns)


def _are_claims_contradictory(m1: Memory, m2: Memory, subject: str) -> bool:
    """
    Determine whether two memories on the same subject make mutually contradictory claims.
    Returns False when claims corroborate each other or discuss non-conflicting aspects.
    """
    c1, c2 = m1.content.strip().lower(), m2.content.strip().lower()
    if c1 == c2:
        return False

    # Check technical configuration / ports
    ports1 = set(re.findall(r"\bport\b[^\d]*(\d+)", c1))
    ports2 = set(re.findall(r"\bport\b[^\d]*(\d+)", c2))
    if ports1 and ports2:
        return ports1 != ports2

    # Check provenance / attribution claims
    if subject == "meta:provenance" or _is_source_or_provenance_disagreement(m1, m2):
        auth1 = set(re.findall(r"\b(?:authored by|created by|author of|attributed to)\s+([a-zA-Z0-9_\-]+)\b", c1))
        auth2 = set(re.findall(r"\b(?:authored by|created by|author of|attributed to)\s+([a-zA-Z0-9_\-]+)\b", c2))
        if auth1 and auth2:
            return auth1 != auth2
        verif_negation = ["falsified", "fabricated", "unreliable", "fake", "invalid", "unauthentic", "misattributed"]
        pos1 = not any(w in c1 for w in verif_negation)
        pos2 = not any(w in c2 for w in verif_negation)
        if pos1 != pos2:
            return True
        return True

    # Check known domain entities
    if subject in TECH_ENTITIES:
        known = TECH_ENTITIES[subject]
        ent1 = {ENTITY_CANONICAL.get(e, e) for e in known if re.search(r"\b" + re.escape(e) + r"\b", c1)}
        ent2 = {ENTITY_CANONICAL.get(e, e) for e in known if re.search(r"\b" + re.escape(e) + r"\b", c2)}
        if ent1 and ent2:
            if ent1 == ent2:
                negations = ["not", "removed", "deprecated", "rejected", "no longer"]
                neg1 = any(n in c1 for n in negations)
                neg2 = any(n in c2 for n in negations)
                return neg1 != neg2
            if ent1.isdisjoint(ent2):
                return True

    # Check Decision memories
    if m1.memory_type == MemoryType.DECISION or m2.memory_type == MemoryType.DECISION:
        return True

    # Check opposing negation keywords
    negations = ["not", "removed", "deprecated", "rejected", "no longer", "disabled", "stopped"]
    neg1 = any(n in c1 for n in negations)
    neg2 = any(n in c2 for n in negations)
    if neg1 != neg2:
        return True

    return False


class ConflictDetector:
    """
    Identifies, classifies, and documents conflicts between Memories.
    
    Guarantees:
    - Never flags conflicts across different projects.
    - Accurately classifies into FACTUAL, TEMPORAL, DECISION, STATUS, TECHNICAL, or SOURCE.
    - Treats disjoint agent sets as provenance information, never as independent SOURCE conflicts.
    - Treats multiple agents independently supporting the same claim as corroboration, not conflict.
    - Uses SOURCE only for actual disputes regarding evidence credibility, provenance, reliability, or attribution.
    - Prevents duplicate conflict records for the same disagreement.
    """

    def __init__(self, ml_conflict_model: Optional[Any] = None) -> None:
        self.ml_conflict_model = ml_conflict_model
        self._tracked_conflict_signatures: Set[Tuple[str, Tuple[str, ...]]] = set()

    def detect_pair(self, m1: Memory, m2: Memory) -> Optional[MemoryConflict]:
        """
        Evaluate a single pair of Memories for conflict.
        
        Returns:
            MemoryConflict instance if a conflict exists, otherwise None.
        """
        # Rule 1: Never compare memories from different projects
        if m1.project_id != m2.project_id:
            return None

        # Rule 2: Cannot conflict with itself or identical memory_id
        if m1.memory_id == m2.memory_id:
            return None

        # Rule 3: Exact same content is not a conflict (corroboration if distinct agents)
        if m1.content.strip().lower() == m2.content.strip().lower():
            return None

        # Rule 4: Status conflict (e.g. task is unresolved vs completed)
        if m1.memory_type == MemoryType.TASK and m2.memory_type == MemoryType.TASK:
            subj1 = extract_memory_subject(m1.content) or m1.content.lower()
            subj2 = extract_memory_subject(m2.content) or m2.content.lower()
            if subj1 == subj2 or any(w in m1.content.lower() for w in m2.content.lower().split()):
                if (m1.status == MemoryStatus.ACTIVE.value and m2.status in (MemoryStatus.ARCHIVED.value, MemoryStatus.SUPERSEDED.value)) or (
                    m2.status == MemoryStatus.ACTIVE.value and m1.status in (MemoryStatus.ARCHIVED.value, MemoryStatus.SUPERSEDED.value)
                ):
                    return self._create_conflict(
                        m1, m2, ConflictType.STATUS,
                        f"Task status conflict on '{m1.content}': one asserts active work while another asserts completion."
                    )

        # Rule 5: Same subject check
        subj1 = extract_memory_subject(m1.content)
        subj2 = extract_memory_subject(m2.content)

        if not subj1 or subj1 != subj2:
            return None

        # Rule 5b: Optional ML conflict classification when model is trained
        if self.ml_conflict_model is not None and getattr(self.ml_conflict_model, "is_trained", False):
            try:
                ml_pred = self.ml_conflict_model.predict((m1, m2))
                if not ml_pred.fallback_used and ml_pred.confidence >= 0.70:
                    val = str(ml_pred.target_value).upper()
                    if val != "NONE" and val in [c.value for c in ConflictType]:
                        return self._create_conflict(
                            m1, m2, ConflictType(val),
                            f"ML classified conflict: {ml_pred.explanation}",
                            confidence=ml_pred.confidence,
                        )
                    elif val == "NONE":
                        # ML indicates non-conflicting compatible elaboration
                        return None
            except Exception:
                pass  # Fall back to deterministic rules below

        # Rule 6: Check if claims are actually contradictory
        # Multiple agents independently supporting the same claim are corroboration, not a conflict
        if not _are_claims_contradictory(m1, m2, subj1):
            return None

        # Rule 7: Actual provenance/credibility/attribution disagreement -> SOURCE conflict
        if _is_source_or_provenance_disagreement(m1, m2):
            return self._create_conflict(
                m1, m2, ConflictType.SOURCE,
                f"Disagreement regarding evidence credibility, provenance, or attribution on '{subj1}': '{m1.content}' vs '{m2.content}'.",
            )

        # Rule 8: Temporal progression
        if _is_temporal_transition(m1, m2):
            return self._create_conflict(
                m1, m2, ConflictType.TEMPORAL,
                f"Temporal disagreement on '{subj1}': claims describe different points in time or migration.",
                confidence=0.85,
            )

        # Rule 9: Technical configuration conflict
        if ("port" in m1.content.lower() and "port" in m2.content.lower()) or "config" in subj1:
            return self._create_conflict(
                m1, m2, ConflictType.TECHNICAL,
                f"Conflicting technical configuration for '{subj1}': '{m1.content}' vs '{m2.content}'.",
            )

        # Rule 10: Decision conflict
        if m1.memory_type == MemoryType.DECISION or m2.memory_type == MemoryType.DECISION:
            return self._create_conflict(
                m1, m2, ConflictType.DECISION,
                f"Conflicting decisions recorded for '{subj1}': '{m1.content}' vs '{m2.content}'.",
            )

        # Rule 11: General factual conflict (disjoint agent sets are preserved in provenance, not classified as SOURCE)
        return self._create_conflict(
            m1, m2, ConflictType.FACTUAL,
            f"Factual disagreement on '{subj1}': '{m1.content}' vs '{m2.content}'.",
        )

    def _create_conflict(
        self,
        m1: Memory,
        m2: Memory,
        conflict_type: ConflictType,
        description: str,
        confidence: float = 1.0,
    ) -> MemoryConflict:
        """Helper to create a MemoryConflict with preserved provenance and evidence."""
        sorted_ids = tuple(sorted([m1.memory_id, m2.memory_id]))

        all_events = list(dict.fromkeys(list(m1.source_event_ids) + list(m2.source_event_ids)))
        all_agents = list(dict.fromkeys(list(m1.source_agent_ids) + list(m2.source_agent_ids)))

        evidence = {
            "source_event_ids": all_events,
            "source_agent_ids": all_agents,
            "memory_timestamps": {
                m1.memory_id: m1.created_at.isoformat(),
                m2.memory_id: m2.created_at.isoformat(),
            },
            "claims": {
                m1.memory_id: m1.content,
                m2.memory_id: m2.content,
            },
        }

        return MemoryConflict.create(
            project_id=m1.project_id,
            memory_ids=sorted_ids,
            conflict_type=conflict_type,
            description=description,
            confidence=confidence,
            evidence=evidence,
            status=ConflictStatus.DETECTED,
        )

    def detect_many(self, memories: Sequence[Memory]) -> List[MemoryConflict]:
        """
        Evaluate a collection of memories for conflicts.
        
        Groups by project, pairs candidate memories, and avoids duplicate conflict records.
        """
        if not memories or len(memories) < 2:
            return []

        # Group by project to strictly enforce project boundary
        by_project: Dict[str, List[Memory]] = defaultdict(list)
        for m in memories:
            by_project[m.project_id].append(m)

        detected_conflicts: List[MemoryConflict] = []

        for project_id, proj_mems in by_project.items():
            seen_pairs: Set[Tuple[str, str]] = set()

            for i in range(len(proj_mems)):
                for j in range(i + 1, len(proj_mems)):
                    m1, m2 = proj_mems[i], proj_mems[j]
                    pair_key = tuple(sorted([m1.memory_id, m2.memory_id]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    conflict = self.detect_pair(m1, m2)
                    if conflict:
                        # Avoid duplicate conflict signatures
                        sig = (project_id, conflict.memory_ids)
                        if sig not in self._tracked_conflict_signatures:
                            self._tracked_conflict_signatures.add(sig)
                            detected_conflicts.append(conflict)

        return detected_conflicts

    def clear_cache(self) -> None:
        """Clear the duplicate conflict signature cache."""
        self._tracked_conflict_signatures.clear()

