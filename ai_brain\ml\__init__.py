"""
AI Brain Machine Learning Layer.

Provides modular, offline, calibrated ML models for memory importance,
relevance ranking, duplicate detection, conflict detection, and staleness detection,
with full provenance tracking, explicit dataset validation, feedback collection,
and seamless fallback to deterministic logic.
"""

from .base import (
    BaseFeatureExtractor,
    BaseMLModel,
    ModelMetadata,
    ModelPrediction,
    ModelType,
)
from .exceptions import (
    DatasetValidationError,
    FallbackTriggeredWarning,
    FeatureExtractionError,
    FeedbackValidationError,
    MLError,
    ModelNotFoundError,
    ModelNotTrainedError,
    ModelSerializationError,
)
from .feedback import (
    FeedbackCollector,
    FeedbackRecord,
    FeedbackSignal,
    FeedbackStore,
    FeedbackType,
    SQLiteFeedbackStore,
)
from .integration import (
    MLConfiguration,
    MLService,
)
from .models import (
    ConflictDetectionModel,
    DuplicateClassification,
    DuplicateDetectionModel,
    MemoryImportanceModel,
    RelevanceRankingModel,
    StalenessDetectionModel,
)
from .training import (
    ClassificationMetrics,
    ModelEvaluationReport,
    ModelEvaluator,
    ModelRegistry,
    RankingMetrics,
    RegressionMetrics,
    TrainingPipeline,
)

__all__ = [
    # Base
    "BaseMLModel",
    "ModelMetadata",
    "ModelPrediction",
    "ModelType",
    "BaseFeatureExtractor",
    # Exceptions
    "MLError",
    "ModelNotTrainedError",
    "ModelNotFoundError",
    "FeatureExtractionError",
    "DatasetValidationError",
    "FeedbackValidationError",
    "ModelSerializationError",
    "FallbackTriggeredWarning",
    # Models
    "MemoryImportanceModel",
    "RelevanceRankingModel",
    "DuplicateClassification",
    "DuplicateDetectionModel",
    "ConflictDetectionModel",
    "StalenessDetectionModel",
    # Training & Evaluation
    "TrainingPipeline",
    "ModelRegistry",
    "ModelEvaluator",
    "ModelEvaluationReport",
    "ClassificationMetrics",
    "RankingMetrics",
    "RegressionMetrics",
    # Feedback
    "FeedbackType",
    "FeedbackSignal",
    "FeedbackRecord",
    "FeedbackStore",
    "SQLiteFeedbackStore",
    "FeedbackCollector",
    # Integration
    "MLConfiguration",
    "MLService",
]
