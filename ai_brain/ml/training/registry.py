"""
Model registry for managing active models, versions, and fallback policies.
"""

from typing import Dict

from ai_brain.ml.base import BaseMLModel, ModelType
from ai_brain.ml.exceptions import ModelNotFoundError
from ai_brain.ml.models.conflict import ConflictDetectionModel
from ai_brain.ml.models.duplicate import DuplicateDetectionModel
from ai_brain.ml.models.importance import MemoryImportanceModel
from ai_brain.ml.models.ranking import RelevanceRankingModel
from ai_brain.ml.models.staleness import StalenessDetectionModel


class ModelRegistry:
    """
    Registry for managing and serving active ML models across AI Brain.
    
    Guarantees:
    - Provides default models with deterministic fallback if untrained.
    - Model versions can be dynamically loaded and registered.
    """

    def __init__(self) -> None:
        self._models: Dict[ModelType, BaseMLModel] = {
            ModelType.IMPORTANCE: MemoryImportanceModel(),
            ModelType.RANKING: RelevanceRankingModel(),
            ModelType.DUPLICATE: DuplicateDetectionModel(),
            ModelType.CONFLICT: ConflictDetectionModel(),
            ModelType.STALENESS: StalenessDetectionModel(),
        }

    def register_model(self, model: BaseMLModel) -> None:
        self._models[model.model_type] = model

    def get_model(self, model_type: ModelType) -> BaseMLModel:
        if model_type not in self._models:
            raise ModelNotFoundError(f"No model registered for type: {model_type}")
        return self._models[model_type]

    def get_importance_model(self) -> MemoryImportanceModel:
        return self._models[ModelType.IMPORTANCE]  # type: ignore

    def get_ranking_model(self) -> RelevanceRankingModel:
        return self._models[ModelType.RANKING]  # type: ignore

    def get_duplicate_model(self) -> DuplicateDetectionModel:
        return self._models[ModelType.DUPLICATE]  # type: ignore

    def get_conflict_model(self) -> ConflictDetectionModel:
        return self._models[ModelType.CONFLICT]  # type: ignore

    def get_staleness_model(self) -> StalenessDetectionModel:
        return self._models[ModelType.STALENESS]  # type: ignore
