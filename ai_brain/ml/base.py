"""
Base abstractions and schemas for AI Brain ML models and features.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple, Union


class ModelType(StrEnum):
    IMPORTANCE = "IMPORTANCE"
    RANKING = "RANKING"
    DUPLICATE = "DUPLICATE"
    CONFLICT = "CONFLICT"
    STALENESS = "STALENESS"


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata describing a specific trained model version."""
    model_name: str
    model_type: ModelType
    version: str
    created_at: datetime
    feature_names: Tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    metrics: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    training_data_provenance: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "feature_names": list(self.feature_names),
            "parameters": dict(self.parameters),
            "metrics": dict(self.metrics),
            "training_data_provenance": self.training_data_provenance,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelMetadata":
        return cls(
            model_name=str(data["model_name"]),
            model_type=ModelType(data["model_type"]),
            version=str(data["version"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            feature_names=tuple(data.get("feature_names", ())),
            parameters=MappingProxyType(dict(data.get("parameters", {}))),
            metrics=MappingProxyType(dict(data.get("metrics", {}))),
            training_data_provenance=data.get("training_data_provenance"),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class ModelPrediction:
    """Standardized output representation for ML model predictions."""
    target_value: Union[float, str, bool]
    confidence: float
    probabilities: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    feature_contributions: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    model_version: str = "unknown"
    fallback_used: bool = False
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_value": self.target_value,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "feature_contributions": {k: round(v, 4) for k, v in self.feature_contributions.items()},
            "model_version": self.model_version,
            "fallback_used": self.fallback_used,
            "explanation": self.explanation,
        }


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extraction from domain models."""

    @property
    @abstractmethod
    def feature_names(self) -> Tuple[str, ...]:
        """Names of features produced by this extractor."""
        pass

    @abstractmethod
    def extract(self, item: Any, **context: Any) -> Dict[str, float]:
        """Extract a numeric feature map from the given entity and context."""
        pass


class BaseMLModel(ABC):
    """Abstract base class for all AI Brain ML models."""

    def __init__(self, metadata: Optional[ModelMetadata] = None) -> None:
        self.metadata = metadata
        self._is_trained = metadata is not None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Type of ML model."""
        pass

    @abstractmethod
    def predict(self, item: Any, **context: Any) -> ModelPrediction:
        """Run inference on a single item with fallback to deterministic logic."""
        pass

    @abstractmethod
    def train(self, dataset: Any, **kwargs: Any) -> Dict[str, float]:
        """Train model parameters on a validated dataset and return evaluation metrics."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Serialize model to a portable file format."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "BaseMLModel":
        """Deserialize model from a portable file format."""
        pass
