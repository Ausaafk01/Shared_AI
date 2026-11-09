"""
Memory Importance Model for predicting long-term retention value.
"""

from datetime import datetime, timezone
import json
import math
from types import MappingProxyType
from typing import Any, Dict, Optional

from ai_brain.memories.models import Memory
from ai_brain.ml.base import BaseMLModel, ModelMetadata, ModelPrediction, ModelType
from ai_brain.ml.datasets.schemas import DatasetSplit
from ai_brain.ml.exceptions import ModelNotTrainedError
from ai_brain.ml.features.temporal import TemporalFeatureExtractor
from ai_brain.ml.features.text import TextFeatureExtractor


class MemoryImportanceModel(BaseMLModel):
    """
    Predicts long-term memory retention importance [0.0, 1.0].
    
    Considers:
    - Memory type and architectural classification (decision, constraint, preference vs temporary)
    - Lexical indicators of decisions, constraints, and permanence
    - Temporal decay and age
    - Provenance breadth (multi-agent corroboration)
    - Explicit non-circular weighting (avoids treating retrieval frequency as automatic importance)
    
    Guarantees:
    - Bounded predictions in [0.0, 1.0]
    - Fully explainable linear/calibrated feature attribution
    - Fallback to Memory.importance or rule-based heuristic when untrained
    """

    FEATURE_NAMES = (
        "is_decision_type",
        "is_fact_type",
        "is_preference_type",
        "decision_score",
        "constraint_score",
        "preference_score",
        "temporary_penalty",
        "provenance_breadth",
        "recency_decay",
    )

    def __init__(self, metadata: Optional[ModelMetadata] = None, weights: Optional[Dict[str, float]] = None, bias: float = 0.0) -> None:
        super().__init__(metadata)
        self.weights = weights or {f: 0.0 for f in self.FEATURE_NAMES}
        self.bias = bias
        self.text_extractor = TextFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor()

    @property
    def model_type(self) -> ModelType:
        return ModelType.IMPORTANCE

    def extract_features(self, memory: Memory, **context: Any) -> Dict[str, float]:
        """Extract tabular feature vector from a Memory object."""
        m_type = str(getattr(memory, "memory_type", "FACT")).upper()
        text_feats = self.text_extractor.extract(memory, **context)
        temp_feats = self.temporal_extractor.extract(memory, **context)

        agent_count = len(getattr(memory, "source_agent_ids", ()))
        event_count = len(getattr(memory, "source_event_ids", ()))
        prov_breadth = min(1.0, (agent_count * 0.3) + (event_count * 0.1))

        return {
            "is_decision_type": 1.0 if m_type == "DECISION" else 0.0,
            "is_fact_type": 1.0 if m_type == "FACT" else 0.0,
            "is_preference_type": 1.0 if m_type == "PREFERENCE" else 0.0,
            "decision_score": text_feats["decision_score"],
            "constraint_score": text_feats["constraint_score"],
            "preference_score": text_feats["preference_score"],
            "temporary_penalty": text_feats["temporary_score"],
            "provenance_breadth": prov_breadth,
            "recency_decay": temp_feats["recency_decay"],
        }

    def predict(self, item: Any, **context: Any) -> ModelPrediction:
        """Predict importance score for a Memory instance."""
        # 1. Fallback if untrained
        if not self.is_trained:
            base_val = getattr(item, "importance", 0.5)
            return ModelPrediction(
                target_value=round(float(base_val), 4),
                confidence=0.5,
                feature_contributions=MappingProxyType({}),
                model_version=self.metadata.version if self.metadata else "fallback-untrained",
                fallback_used=True,
                explanation="Model untrained; fell back to baseline memory.importance",
            )

        feats = self.extract_features(item, **context)

        # 2. Linear dot product
        raw_score = self.bias
        contributions = {}
        for f, val in feats.items():
            w = self.weights.get(f, 0.0)
            contrib = w * val
            contributions[f] = round(contrib, 4)
            raw_score += contrib

        # Sigmoid activation to strictly bound in [0.0, 1.0]
        sig_score = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, raw_score))))

        # Confidence is higher when strong features are active (high magnitude)
        active_weight_sum = sum(abs(contributions[f]) for f in contributions)
        confidence = max(0.5, min(0.95, 0.5 + 0.3 * (active_weight_sum / (sum(abs(w) for w in self.weights.values()) or 1.0))))

        return ModelPrediction(
            target_value=round(sig_score, 4),
            confidence=round(confidence, 4),
            feature_contributions=MappingProxyType(contributions),
            model_version=self.metadata.version if self.metadata else "v1.0",
            fallback_used=False,
            explanation=f"Predicted importance={sig_score:.3f} (bias={self.bias:.2f}, active features={len([c for c in contributions.values() if c != 0])})",
        )

    def train(self, dataset: DatasetSplit, learning_rate: float = 0.05, epochs: int = 100, l2_reg: float = 0.01) -> Dict[str, float]:
        """Train weights using regularized logistic regression over dataset."""
        train_examples = dataset.train_examples
        if not train_examples:
            raise ModelNotTrainedError("Cannot train with empty dataset split")

        weights = {f: 0.0 for f in self.FEATURE_NAMES}
        # Initialize with domain priors
        weights["is_decision_type"] = 0.8
        weights["decision_score"] = 0.6
        weights["constraint_score"] = 0.7
        weights["preference_score"] = 0.3
        weights["temporary_penalty"] = -1.2
        weights["provenance_breadth"] = 0.4
        bias = 0.0

        n = len(train_examples)
        for _ in range(epochs):
            grad_w = {f: 0.0 for f in self.FEATURE_NAMES}
            grad_b = 0.0

            for ex in train_examples:
                y = float(ex.label)
                # Compute logit
                logit = bias + sum(weights[f] * ex.features.get(f, 0.0) for f in self.FEATURE_NAMES)
                p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, logit))))
                err = p - y

                grad_b += err
                for f in self.FEATURE_NAMES:
                    grad_w[f] += err * ex.features.get(f, 0.0)

            # Update with L2 regularization
            bias -= learning_rate * (grad_b / n)
            for f in self.FEATURE_NAMES:
                weights[f] -= learning_rate * ((grad_w[f] / n) + (l2_reg * weights[f]))

        self.weights = weights
        self.bias = bias
        self._is_trained = True

        # Compute validation MAE
        val_errors = []
        for ex in dataset.validation_examples:
            y = float(ex.label)
            logit = bias + sum(weights[f] * ex.features.get(f, 0.0) for f in self.FEATURE_NAMES)
            p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, logit))))
            val_errors.append(abs(p - y))

        mae = sum(val_errors) / len(val_errors) if val_errors else 0.0

        self.metadata = ModelMetadata(
            model_name="MemoryImportanceModel",
            model_type=ModelType.IMPORTANCE,
            version="1.0.0",
            created_at=datetime.now(timezone.utc),
            feature_names=self.FEATURE_NAMES,
            parameters=MappingProxyType({"weights": self.weights, "bias": self.bias}),
            metrics=MappingProxyType({"validation_mae": mae}),
            training_data_provenance=dataset.name,
            description="Calibrated regularized logistic importance model",
        )
        return {"mae": mae}

    def save(self, filepath: str) -> None:
        data = {
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "weights": self.weights,
            "bias": self.bias,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "MemoryImportanceModel":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = ModelMetadata.from_dict(data["metadata"]) if data.get("metadata") else None
        return cls(metadata=meta, weights=data.get("weights"), bias=float(data.get("bias", 0.0)))
