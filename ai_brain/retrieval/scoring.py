"""
Transparent, multi-factor memory relevance scoring component.
"""

from datetime import datetime, timezone
import math
from typing import Optional, Sequence

from ai_brain.retrieval.exceptions import ScoringError
from ai_brain.retrieval.fusion import FusedCandidate
from ai_brain.retrieval.models import RetrievalItem, ScoreBreakdown


class RelevanceScorer:
    """
    Computes a transparent relevance score from vector, graph, confidence, importance,
    status, recency, and provenance signals.
    
    Formula:
        base_relevance = fused_score (or vector/graph score)
        importance_factor = 0.8 + 0.2 * importance
        confidence_factor = 0.8 + 0.2 * confidence
        recency_factor = 0.9 + 0.1 * exp(-age_in_days / 30.0)
        status_multiplier = ACTIVE: 1.0, DISPUTED: 0.7, ARCHIVED: 0.4, SUPERSEDED: 0.2
        provenance_bonus = extra_agent_bonus + extra_event_bonus (up to 0.10)
        
        final_score = clamp_0_1((base_relevance * importance_factor * confidence_factor *
                                 recency_factor * status_multiplier) + provenance_bonus)
    
    Guarantees:
    - Normalized final score in [0.0, 1.0].
    - Fully deterministic for identical inputs.
    - Explicitly separates confidence from importance.
    - Applies a defined discount to DISPUTED memories so they do not masquerade as facts.
    - Returns a granular ScoreBreakdown for explainability.
    """

    def __init__(
        self,
        *,
        half_life_days: float = 30.0,
        disputed_multiplier: float = 0.7,
        superseded_multiplier: float = 0.2,
        archived_multiplier: float = 0.4,
    ) -> None:
        if half_life_days <= 0:
            raise ScoringError(f"half_life_days must be positive, got {half_life_days}")
        self.half_life_days = half_life_days
        self.disputed_multiplier = disputed_multiplier
        self.superseded_multiplier = superseded_multiplier
        self.archived_multiplier = archived_multiplier

    def get_status_multiplier(self, status: str, is_disputed: bool) -> float:
        """Resolve status multiplier."""
        s = status.upper()
        if is_disputed or s == "DISPUTED":
            return self.disputed_multiplier
        elif s == "ACTIVE":
            return 1.0
        elif s == "ARCHIVED":
            return self.archived_multiplier
        elif s == "SUPERSEDED":
            return self.superseded_multiplier
        return 0.5

    def calculate_recency_factor(self, created_at: datetime, reference_time: Optional[datetime] = None) -> float:
        """Compute recency factor decaying with age."""
        ref = reference_time or datetime.now(timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        c_at = created_at if created_at.tzinfo is not None else created_at.replace(tzinfo=timezone.utc)

        age_seconds = max(0.0, (ref - c_at).total_seconds())
        age_days = age_seconds / 86400.0

        decay = math.exp(-age_days / self.half_life_days)
        factor = 0.9 + 0.1 * decay
        return max(0.9, min(1.0, factor))

    def calculate_provenance_bonus(
        self,
        source_agent_ids: Sequence[str],
        source_event_ids: Sequence[str],
    ) -> float:
        """Compute corroboration bonus for multiple agents and supporting events."""
        # Extra distinct agents beyond the first: +0.02 each, max +0.06
        unique_agents = set(source_agent_ids)
        agent_bonus = min(0.06, max(0, len(unique_agents) - 1) * 0.02)

        # Extra distinct events beyond the first: +0.01 each, max +0.04
        unique_events = set(source_event_ids)
        event_bonus = min(0.04, max(0, len(unique_events) - 1) * 0.01)

        return round(agent_bonus + event_bonus, 4)

    def score_candidate(
        self,
        candidate: FusedCandidate,
        *,
        reference_time: Optional[datetime] = None,
    ) -> RetrievalItem:
        """
        Evaluate and score a FusedCandidate, returning a fully enriched RetrievalItem.
        """
        base_score = float(candidate.fused_score)

        # Importance & Confidence factors (strictly separated)
        imp = max(0.0, min(1.0, float(candidate.importance)))
        conf = max(0.0, min(1.0, float(candidate.confidence)))
        f_imp = 0.8 + 0.2 * imp
        f_conf = 0.8 + 0.2 * conf

        # Recency factor
        f_rec = self.calculate_recency_factor(candidate.created_at, reference_time)

        # Status multiplier
        m_stat = self.get_status_multiplier(candidate.status, candidate.is_disputed)

        # Provenance bonus
        b_prov = self.calculate_provenance_bonus(candidate.source_agent_ids, candidate.source_event_ids)

        # Final score calculation
        raw_final = (base_score * f_imp * f_conf * f_rec * m_stat) + b_prov
        final_score = max(0.0, min(1.0, raw_final))

        breakdown = ScoreBreakdown(
            base_score=round(base_score, 4),
            vector_score=round(candidate.vector_score, 4) if candidate.vector_score is not None else None,
            graph_score=round(candidate.graph_score, 4) if candidate.graph_score is not None else None,
            importance_factor=round(f_imp, 4),
            confidence_factor=round(f_conf, 4),
            recency_factor=round(f_rec, 4),
            status_multiplier=round(m_stat, 4),
            provenance_bonus=round(b_prov, 4),
            final_score=round(final_score, 4),
        )

        return RetrievalItem(
            memory_id=candidate.memory_id,
            root_memory_id=candidate.root_memory_id,
            version=candidate.version,
            project_id=candidate.project_id,
            content=candidate.content,
            memory_type=candidate.memory_type,
            status=candidate.status,
            score=round(final_score, 4),
            score_breakdown=breakdown,
            vector_score=candidate.vector_score,
            graph_score=candidate.graph_score,
            graph_metadata=candidate.graph_metadata,
            sources=candidate.sources,
            source_event_ids=candidate.source_event_ids,
            source_agent_ids=candidate.source_agent_ids,
            is_disputed=candidate.is_disputed,
            created_at=candidate.created_at,
            metadata=candidate.metadata,
        )

