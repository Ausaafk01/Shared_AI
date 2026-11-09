"""
Context item prioritization and ranking for Context Builder.
"""

from datetime import datetime, timezone
import math
from typing import Optional, Sequence

from ai_brain.context.models import PriorityBreakdown
from ai_brain.retrieval.models import RetrievalItem


class MemoryPrioritizer:
    """
    Ranks retrieved candidate items using a deterministic multi-factor formula.
    
    Guarantees:
    - Direct relevance is the primary gate (relevance == 0 yields priority == 0).
    - High importance or confidence cannot elevate irrelevant memories.
    - Decisions and active tasks receive category boosts.
    - Disputed memories are discounted to avoid presenting disputed claims as facts.
    - Provides inspectable PriorityBreakdown for transparency.
    """

    def __init__(
        self,
        *,
        disputed_multiplier: float = 0.65,
        superseded_multiplier: float = 0.20,
        archived_multiplier: float = 0.40,
        decision_bonus: float = 1.15,
        task_bonus: float = 1.10,
        fact_bonus: float = 1.05,
    ) -> None:
        self.disputed_multiplier = disputed_multiplier
        self.superseded_multiplier = superseded_multiplier
        self.archived_multiplier = archived_multiplier
        self.decision_bonus = decision_bonus
        self.task_bonus = task_bonus
        self.fact_bonus = fact_bonus

    def get_type_bonus(self, memory_type: str) -> float:
        """Calculate bonus multiplier based on memory type."""
        m_type = memory_type.strip().upper()
        if m_type == "DECISION":
            return self.decision_bonus
        elif m_type == "TASK":
            return self.task_bonus
        elif m_type in ("FACT", "CODE_KNOWLEDGE"):
            return self.fact_bonus
        return 1.0

    def get_status_multiplier(self, status: str, is_disputed: bool) -> float:
        """Resolve status discount multiplier."""
        s = status.strip().upper()
        if is_disputed or s == "DISPUTED":
            return self.disputed_multiplier
        elif s == "ACTIVE":
            return 1.0
        elif s == "ARCHIVED":
            return self.archived_multiplier
        elif s == "SUPERSEDED":
            return self.superseded_multiplier
        return 0.8

    def extract_importance(self, item: RetrievalItem) -> float:
        """Extract or infer importance score in [0.0, 1.0]."""
        if "importance" in item.metadata:
            try:
                return max(0.0, min(1.0, float(item.metadata["importance"])))
            except (ValueError, TypeError):
                pass
        if hasattr(item, "score_breakdown") and item.score_breakdown:
            f = item.score_breakdown.importance_factor
            # factor = 0.8 + 0.2 * imp => imp = (factor - 0.8) / 0.2
            return max(0.0, min(1.0, (f - 0.8) / 0.2))
        return 0.5

    def extract_confidence(self, item: RetrievalItem) -> float:
        """Extract or infer confidence score in [0.0, 1.0]."""
        if "confidence" in item.metadata:
            try:
                return max(0.0, min(1.0, float(item.metadata["confidence"])))
            except (ValueError, TypeError):
                pass
        if hasattr(item, "score_breakdown") and item.score_breakdown:
            f = item.score_breakdown.confidence_factor
            return max(0.0, min(1.0, (f - 0.8) / 0.2))
        return 1.0

    def calculate_recency_factor(
        self,
        created_at: datetime,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """Calculate recency factor using exponential decay."""
        ref = reference_time or datetime.now(timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        cat = created_at if created_at.tzinfo is not None else created_at.replace(tzinfo=timezone.utc)
        age_seconds = max(0.0, (ref - cat).total_seconds())
        age_days = age_seconds / 86400.0
        return 0.9 + 0.1 * math.exp(-age_days / 30.0)

    def compute_priority(
        self,
        item: RetrievalItem,
        reference_time: Optional[datetime] = None,
    ) -> tuple[float, PriorityBreakdown]:
        """
        Compute the priority score and inspectable breakdown for a retrieval item.
        """
        relevance = max(0.0, min(1.0, float(item.score)))
        importance = self.extract_importance(item)
        confidence = self.extract_confidence(item)

        type_bonus = self.get_type_bonus(item.memory_type)
        status_mult = self.get_status_multiplier(item.status, item.is_disputed)

        if hasattr(item, "score_breakdown") and item.score_breakdown and item.score_breakdown.recency_factor:
            recency = item.score_breakdown.recency_factor
        else:
            recency = self.calculate_recency_factor(item.created_at, reference_time)

        # Relevance is the primary gate
        if relevance <= 0.0:
            final_p = 0.0
        else:
            imp_weight = 0.7 + 0.3 * importance
            conf_weight = 0.8 + 0.2 * confidence
            raw = relevance * imp_weight * conf_weight * type_bonus * recency * status_mult
            final_p = max(0.0, min(1.0, raw))

        breakdown = PriorityBreakdown(
            relevance_score=round(relevance, 4),
            importance_score=round(importance, 4),
            confidence_score=round(confidence, 4),
            type_bonus=round(type_bonus, 4),
            recency_factor=round(recency, 4),
            status_multiplier=round(status_mult, 4),
            final_priority=round(final_p, 4),
        )

        return round(final_p, 4), breakdown

    def prioritize(
        self,
        items: Sequence[RetrievalItem],
        reference_time: Optional[datetime] = None,
    ) -> list[tuple[RetrievalItem, float, PriorityBreakdown]]:
        """
        Score and rank items in descending order of priority score.
        Tie-breaking is deterministic (by score, created_at, memory_id).
        """
        scored = []
        for it in items:
            p_score, breakdown = self.compute_priority(it, reference_time=reference_time)
            scored.append((it, p_score, breakdown))

        # Sort descending by priority_score, then score, then created_at, then memory_id
        scored.sort(
            key=lambda tup: (
                -tup[1],
                -tup[0].score,
                -tup[0].created_at.timestamp(),
                tup[0].memory_id,
            )
        )
        return scored

