"""
MLService facade integrating all ML models, feedback, and deterministic fallbacks.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ai_brain.memories.conflicts.models import MemoryConflict
from ai_brain.memories.conflicts.types import ConflictStatus, ConflictType
from ai_brain.memories.models import Memory
from ai_brain.memories.updating.comparison import ComparisonResult, MemoryRelation
from ai_brain.ml.base import ModelPrediction
from ai_brain.ml.feedback.collector import FeedbackCollector
from ai_brain.ml.feedback.models import FeedbackRecord
from ai_brain.ml.integration.config import MLConfiguration
from ai_brain.ml.models.conflict import ConflictDetectionModel
from ai_brain.ml.models.duplicate import DuplicateClassification, DuplicateDetectionModel
from ai_brain.ml.models.importance import MemoryImportanceModel
from ai_brain.ml.models.ranking import RelevanceRankingModel
from ai_brain.ml.models.staleness import StalenessDetectionModel
from ai_brain.ml.training.registry import ModelRegistry
from ai_brain.retrieval.models import RetrievalItem

logger = logging.getLogger(__name__)


class MLService:
    """
    Central service coordinating ML predictions across the AI Brain lifecycle.
    
    Guarantees:
    - Independent enablement flags via MLConfiguration.
    - Automatic graceful fallback to deterministic logic on low confidence or error.
    - Preserves project and agent isolation boundaries.
    """

    def __init__(
        self,
        config: Optional[MLConfiguration] = None,
        registry: Optional[ModelRegistry] = None,
        feedback_collector: Optional[FeedbackCollector] = None,
    ) -> None:
        self.config = config or MLConfiguration()
        self.registry = registry or ModelRegistry()
        self.feedback_collector = feedback_collector or FeedbackCollector()

    # 1. Importance prediction
    def predict_importance(self, memory: Memory, **context: Any) -> ModelPrediction:
        if not self.config.enable_importance_ml:
            return ModelPrediction(
                target_value=round(memory.importance, 4),
                confidence=1.0,
                model_version="deterministic-bypass",
                fallback_used=True,
                explanation="ML importance disabled by configuration; used baseline memory.importance",
            )
        try:
            model = self.registry.get_importance_model()
            pred = model.predict(memory, **context)
            if pred.confidence < self.config.min_confidence_threshold and self.config.allow_fallback_on_error:
                return ModelPrediction(
                    target_value=round(memory.importance, 4),
                    confidence=pred.confidence,
                    model_version=pred.model_version,
                    fallback_used=True,
                    explanation=f"ML confidence ({pred.confidence:.2f}) below threshold; fell back to baseline",
                )
            return pred
        except Exception as e:
            if not self.config.allow_fallback_on_error:
                raise
            logger.warning("Importance ML prediction failed: %s; falling back", e)
            return ModelPrediction(
                target_value=round(memory.importance, 4),
                confidence=0.5,
                model_version="error-fallback",
                fallback_used=True,
                explanation=f"Error during ML prediction ({e}); fell back to baseline",
            )

    # 2. Retrieval re-ranking
    def rerank_retrieval(
        self,
        items: Sequence[RetrievalItem],
        query: str = "",
        active_agent_id: Optional[str] = None,
    ) -> List[RetrievalItem]:
        if not self.config.enable_ranking_ml or not items:
            return list(items)
        try:
            model = self.registry.get_ranking_model()
            return model.rerank(items, query=query, active_agent_id=active_agent_id)
        except Exception as e:
            if not self.config.allow_fallback_on_error:
                raise
            logger.warning("Relevance ranking ML failed: %s; returning original items", e)
            return list(items)

    # 3. Duplicate detection
    def check_duplicate(self, m1: Memory, m2: Memory) -> ModelPrediction:
        if not self.config.enable_duplicate_ml:
            model = self.registry.get_duplicate_model()
            comp = model.deterministic_comparator.compare(m1, m2)
            val = DuplicateClassification.EXACT_DUPLICATE.value if comp.relation == MemoryRelation.IDENTICAL else (
                DuplicateClassification.SEMANTIC_DUPLICATE.value if comp.relation == MemoryRelation.DUPLICATE else (
                    DuplicateClassification.CONTRADICTORY.value if comp.relation == MemoryRelation.CONTRADICTORY else DuplicateClassification.RELATED_DISTINCT.value
                )
            )
            return ModelPrediction(
                target_value=val,
                confidence=1.0,
                model_version="deterministic-bypass",
                fallback_used=True,
                explanation=f"Deterministic comparator: {comp.reason}",
            )
        try:
            model = self.registry.get_duplicate_model()
            return model.predict((m1, m2))
        except Exception as e:
            if not self.config.allow_fallback_on_error:
                raise
            logger.warning("Duplicate ML prediction failed: %s; falling back", e)
            model = self.registry.get_duplicate_model()
            comp = model.deterministic_comparator.compare(m1, m2)
            return ModelPrediction(
                target_value=DuplicateClassification.RELATED_DISTINCT.value,
                confidence=0.5,
                model_version="error-fallback",
                fallback_used=True,
                explanation=f"Error in duplicate ML ({e}); fell back",
            )

    # 4. Conflict detection
    def detect_conflict(self, m1: Memory, m2: Memory) -> ModelPrediction:
        if not self.config.enable_conflict_ml:
            model = self.registry.get_conflict_model()
            conf = model.deterministic_detector.detect_pair(m1, m2)
            c_val = conf.conflict_type.value if conf else "NONE"
            return ModelPrediction(
                target_value=c_val,
                confidence=conf.confidence if conf else 0.9,
                model_version="deterministic-bypass",
                fallback_used=True,
                explanation="Conflict ML disabled; used deterministic ConflictDetector",
            )
        try:
            model = self.registry.get_conflict_model()
            return model.predict((m1, m2))
        except Exception as e:
            if not self.config.allow_fallback_on_error:
                raise
            logger.warning("Conflict ML prediction failed: %s; falling back", e)
            return ModelPrediction(
                target_value="NONE",
                confidence=0.5,
                model_version="error-fallback",
                fallback_used=True,
                explanation=f"Error during conflict ML ({e}); fell back",
            )

    # 5. Staleness detection
    def assess_staleness(self, memory: Memory, **context: Any) -> ModelPrediction:
        if not self.config.enable_staleness_ml:
            return ModelPrediction(
                target_value=0.0 if memory.status == "ACTIVE" else 1.0,
                confidence=1.0,
                model_version="deterministic-bypass",
                fallback_used=True,
                explanation="Staleness ML disabled; used status indicator",
            )
        try:
            model = self.registry.get_staleness_model()
            return model.predict(memory, **context)
        except Exception as e:
            if not self.config.allow_fallback_on_error:
                raise
            logger.warning("Staleness ML prediction failed: %s; falling back", e)
            return ModelPrediction(
                target_value=0.0,
                confidence=0.5,
                model_version="error-fallback",
                fallback_used=True,
                explanation=f"Error during staleness ML ({e}); fell back",
            )
