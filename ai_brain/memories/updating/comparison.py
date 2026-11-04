"""
Comparison and relationship detection between Memory instances.
"""

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any, Optional

from ai_brain.memories.models import Memory


class MemoryRelation(StrEnum):
    """Relationship classification between an existing memory and incoming information."""
    IDENTICAL = "IDENTICAL"          # Byte-for-byte or exact content match
    DUPLICATE = "DUPLICATE"          # Semantically equivalent / unchanged information
    UPDATE = "UPDATE"                # Incoming replaces or updates the existing memory
    CONTRADICTORY = "CONTRADICTORY"  # Conflicting assertions on the same subject without clear resolution
    DISTINCT = "DISTINCT"            # Unrelated knowledge (separate subject)


@dataclass(frozen=True)
class ComparisonResult:
    """Result of comparing an incoming Memory against an existing Memory."""
    relation: MemoryRelation
    subject: Optional[str] = None
    similarity: float = 0.0
    reason: str = ""


# Common technical subjects to match updates on specific configurations
KNOWN_SUBJECT_PATTERNS = [
    (r"\b(?:database|postgres(?:ql)?|mysql|sqlite|mongodb|mariadb)\b", "tech:database"),
    (r"\b(?:auth(?:entication)?|jwt|oauth|session(?:s)?|cookie(?:s)?)\b", "tech:authentication"),
    (r"\b(?:cache|caching|redis|memcached)\b", "tech:caching"),
    (r"\b(?:language|python|typescript|javascript|go|rust)\b", "tech:language"),
    (r"\b(?:framework|fastapi|django|flask|express|next\.?js)\b", "tech:framework"),
    (r"\b(?:orm|sqlalchemy|prisma|django orm|alembic)\b", "tech:orm"),
    (r"\b(?:test(?:ing)?|pytest|unittest|jest)\b", "tech:testing"),
    (r"\b(?:os|operating system|linux|ubuntu|windows|macos)\b", "tech:os"),
    (r"\b(?:author(?:ed|ship)?|attributed|attribution|provenance|credibility|authenticity)\b", "meta:provenance"),
]


def extract_memory_subject(content: str) -> Optional[str]:
    """Identify the core subject of a memory content string."""
    lower = content.lower()
    for pattern, subject in KNOWN_SUBJECT_PATTERNS:
        if re.search(pattern, lower):
            return subject
    return None


class MemoryComparator:
    """Deterministic comparator with optional ML-assisted duplicate and contradiction classification."""

    def __init__(self, ml_duplicate_model: Optional[Any] = None) -> None:
        self.ml_duplicate_model = ml_duplicate_model

    def compare(self, existing: Memory, incoming: Memory) -> ComparisonResult:
        """
        Compare an existing Memory against an incoming Memory.
        
        Args:
            existing: Currently known Memory.
            incoming: Candidate or new Memory.
            
        Returns:
            ComparisonResult indicating relation, subject, and reasoning.
        """
        # 1. Project check
        if existing.project_id != incoming.project_id:
            return ComparisonResult(
                relation=MemoryRelation.DISTINCT,
                reason="Memories belong to different projects",
            )

        # 2. Explicit predecessor relationship
        if (
            incoming.parent_memory_id == existing.memory_id
            or incoming.root_memory_id == existing.root_memory_id
            and incoming.version > existing.version
        ):
            return ComparisonResult(
                relation=MemoryRelation.UPDATE,
                subject=extract_memory_subject(incoming.content),
                reason="Incoming memory explicitly references existing memory as parent or version update",
            )

        # 3. Exact content check
        existing_norm = existing.content.strip().lower()
        incoming_norm = incoming.content.strip().lower()

        if existing_norm == incoming_norm:
            if existing.memory_type == incoming.memory_type:
                return ComparisonResult(
                    relation=MemoryRelation.IDENTICAL,
                    similarity=1.0,
                    reason="Contents and memory types are identical",
                )
            return ComparisonResult(
                relation=MemoryRelation.DUPLICATE,
                similarity=0.9,
                reason="Contents are identical with different memory types",
            )

        # 4. Check if types are different
        if existing.memory_type != incoming.memory_type:
            # Different memory types represent distinct knowledge unless an explicit link is present
            return ComparisonResult(
                relation=MemoryRelation.DISTINCT,
                reason=f"Different memory types ({existing.memory_type} vs {incoming.memory_type}) represent distinct knowledge",
            )

        # 4b. Optional ML duplicate classification when trained model is configured
        if self.ml_duplicate_model is not None and getattr(self.ml_duplicate_model, "is_trained", False):
            try:
                ml_pred = self.ml_duplicate_model.predict((existing, incoming))
                if not ml_pred.fallback_used and ml_pred.confidence >= 0.70:
                    target = str(ml_pred.target_value)
                    if target == "EXACT_DUPLICATE":
                        return ComparisonResult(
                            relation=MemoryRelation.IDENTICAL,
                            similarity=1.0,
                            reason=f"ML detected exact duplicate: {ml_pred.explanation}",
                        )
                    elif target == "SEMANTIC_DUPLICATE":
                        return ComparisonResult(
                            relation=MemoryRelation.DUPLICATE,
                            similarity=ml_pred.confidence,
                            reason=f"ML detected semantic duplicate: {ml_pred.explanation}",
                        )
                    elif target == "CONTRADICTORY":
                        return ComparisonResult(
                            relation=MemoryRelation.CONTRADICTORY,
                            subject=extract_memory_subject(incoming.content),
                            similarity=ml_pred.confidence,
                            reason=f"ML detected contradiction: {ml_pred.explanation}",
                        )
            except Exception:
                pass  # Gracefully fall back to deterministic logic below

        # 5. Subject-based comparison
        subj_existing = extract_memory_subject(existing.content)
        subj_incoming = extract_memory_subject(incoming.content)

        if subj_existing and subj_existing == subj_incoming:
            # Check for explicit replacement language in incoming
            replacement_keywords = [
                "removed", "replaced", "switched to", "migrated to", "deprecated",
                "decided to use", "now uses",
            ]
            if any(k in incoming_norm for k in replacement_keywords):
                return ComparisonResult(
                    relation=MemoryRelation.UPDATE,
                    subject=subj_incoming,
                    reason=f"Incoming explicitly documents update/replacement on subject '{subj_incoming}'",
                )

            # Check if incoming has higher confidence or explicit update metadata
            if incoming.metadata.get("updates") == existing.memory_id or incoming.metadata.get("supersedes") == existing.memory_id:
                return ComparisonResult(
                    relation=MemoryRelation.UPDATE,
                    subject=subj_incoming,
                    reason="Incoming metadata explicitly references existing memory",
                )

            # Check for conflicting claims on the same subject
            # e.g., "The project uses MySQL" vs "The project uses PostgreSQL"
            if incoming.confidence >= existing.confidence and incoming.created_at >= existing.created_at:
                # Newer observation with equal or higher confidence represents an update/supersession
                return ComparisonResult(
                    relation=MemoryRelation.UPDATE,
                    subject=subj_incoming,
                    reason=f"Newer observation on subject '{subj_incoming}' supersedes previous knowledge",
                )
            else:
                # Ambiguous or contradictory assertions without clear temporal superseding
                return ComparisonResult(
                    relation=MemoryRelation.CONTRADICTORY,
                    subject=subj_incoming,
                    reason=f"Contradictory assertions on subject '{subj_incoming}' without clear authority",
                )

        # 6. Default to distinct knowledge
        return ComparisonResult(
            relation=MemoryRelation.DISTINCT,
            reason="Distinct knowledge on unrelated subjects",
        )
