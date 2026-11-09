"""
Staleness Detection Model for predicting whether a memory is outdated without deleting it.
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


class StalenessDetectionModel(BaseMLModel):
    """
    Predicts probability of memory staleness / obsolescence [0.0, 1.0].
    
    Guarantees:
    - Never automatically deletes or purges potentially stale memories.
    - Preserves full memory history, provenance, and versions.
    - Emits explainable risk factors for retrieval discount or maintenance review.
    - Falls back to age-based heuristic when untrained.
    """

    FEATURE_NAMES = (
        "age_days",
        "days_since_update",
        "is_temporary_type",
        "is_decision_type",
        "has_temporal_keywords",
        "subsequent_project_event_count",
        "has_superseding_child",
        "confirmation_gap_factor",
    )

    def __init__(self, metadata: Optional[ModelMetadata] = None, weights: Optional[Dict[str, float]] = None, bias: float = 0.0) -> None:
        super().__init__(metadata)
        self.weights = weights or {f: 0.0 for f in self.FEATURE_NAMES}
        self.bias = bias
        self.temporal_extractor = TemporalFeatureExtractor()

    @property
    def model_type(self) -> ModelType:
        return ModelType.STALENESS

    def extract_staleness_features(self, memory: Memory, **context: Any) -> Dict[str, float]:
        """Extract staleness indicators."""
        ref_time = context.get("reference_time") or datetime.now(timezone.utc)
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        temp_feats = self.temporal_extractor.extract(memory, reference_time=ref_time)
        age_days = temp_feats["age_days"]
        days_since_upd = temp_feats["days_since_update"]

        m_type = str(getattr(memory, "memory_type", "FACT")).upper()
        is_temp = 1.0 if ("TEMP" in m_type or temp_feats["has_temporal_keywords"] > 0) else 0.0
        is_dec = 1.0 if m_type == "DECISION" else 0.0

        subseq_events = float(context.get("subsequent_project_events", 0))
        has_child = 1.0 if getattr(memory, "status", "") == "SUPERSEDED" or context.get("has_child_version") else 0.0

        # Confirmation gap: long periods without corroborating events
        agent_count = len(getattr(memory, "source_agent_ids", ()))
        conf_gap = max(0.0, min(1.0, (age_days / 60.0) * (1.0 / max(1, agent_count))))

        return {
            "age_days": round(min(365.0, age_days) / 365.0, 4),
            "days_since_update": round(min(365.0, days_since_upd) / 365.0, 4),
            "is_temporary_type": is_temp,
            "is_decision_type": is_dec,
            "has_temporal_keywords": temp_feats["has_temporal_keywords"],
            "subsequent_project_event_count": round(min(100.0, subseq_events) / 100.0, 4),
            "has_superseding_child": has_child,
            "confirmation_gap_factor": round(conf_gap, 4),
        }

    def predict(self, item: Any, **context: Any) -> ModelPrediction:
        """Predict staleness probability."""
        if not self.is_trained:
            # Deterministic fallback based on status and age
            status = getattr(item, "status", "ACTIVE")
            if status == "SUPERSEDED":
                return ModelPrediction(
                    target_value=1.0,
                    confidence=1.0,
                    feature_contributions=MappingProxyType({}),
                    model_version="deterministic",
                    fallback_used=True,
                    explanation="Status is SUPERSEDED; definitely stale",
                )
            # Simple heuristic
            feats = self.extract_staleness_features(item, **context)
            heuristic_prob = min(0.9, feats["age_days"] * 0.4 + feats["is_temporary_type"] * 0.5)
            return ModelPrediction(
                target_value=round(heuristic_prob, 4),
                confidence=0.6,
                feature_contributions=MappingProxyType({}),
                model_version=self.metadata.version if self.metadata else "fallback-untrained",
                fallback_used=True,
                explanation=f"Deterministic heuristic staleness={heuristic_prob:.2f}",
            )

        feats = self.extract_staleness_features(item, **context)

        raw_score = self.bias
        contributions = {}
        for f, val in feats.items():
            w = self.weights.get(f, 0.0)
            contrib = w * val
            contributions[f] = round(contrib, 4)
            raw_score += contrib

        sig_prob = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, raw_score))))

        is_stale = sig_prob >= 0.65
        conf = max(0.6, min(0.95, 0.5 + abs(sig_prob - 0.5) * 0.9))

        return ModelPrediction(
            target_value=round(sig_prob, 4),
            confidence=round(conf, 4),
            probabilities=MappingProxyType({"is_stale": round(sig_prob, 4), "is_fresh": round(1.0 - sig_prob, 4)}),
            feature_contributions=MappingProxyType(contributions),
            model_version=self.metadata.version if self.metadata else "v1.0",
            fallback_used=False,
            explanation=f"{'Likely stale' if is_stale else 'Fresh'} (p={sig_prob:.3f}, age_contrib={contributions.get('age_days', 0):.2f})",
        )

    def train(self, dataset: DatasetSplit, learning_rate: float = 0.05, epochs: int = 100, l2_reg: float = 0.01) -> Dict[str, float]:
        train_examples = dataset.train_examples
        if not train_examples:
            raise ModelNotTrainedError("Cannot train with empty dataset split")

        weights = {f: 0.0 for f in self.FEATURE_NAMES}
        weights["age_days"] = 1.5
        weights["days_since_update"] = 1.0
        weights["is_temporary_type"] = 2.0
        weights["is_decision_type"] = -1.2  # Decisions are durable
        weights["has_superseding_child"] = 3.5
        bias = -1.0

        n = len(train_examples)
        for _ in range(epochs):
            grad_w = {f: 0.0 for f in self.FEATURE_NAMES}
            grad_b = 0.0

            for ex in train_examples:
                y = float(ex.label)
                logit = bias + sum(weights[f] * ex.features.get(f, 0.0) for f in self.FEATURE_NAMES)
                p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, logit))))
                err = p - y

                grad_b += err
                for f in self.FEATURE_NAMES:
                    grad_w[f] += err * ex.features.get(f, 0.0)

            bias -= learning_rate * (grad_b / n)
            for f in self.FEATURE_NAMES:
                weights[f] -= learning_rate * ((grad_w[f] / n) + (l2_reg * weights[f]))

        self.weights = weights
        self.bias = bias
        self._is_trained = True

        from ai_brain.ml.training.evaluator import ModelEvaluator

        val_true, val_pred, val_prob = [], [], []
        val_errors = []
        for ex in dataset.validation_examples:
            y = float(ex.label)
            logit = bias + sum(weights[f] * ex.features.get(f, 0.0) for f in self.FEATURE_NAMES)
            p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, logit))))
            val_errors.append(abs(p - y))
            val_true.append(1 if y >= 0.5 else 0)
            val_prob.append(p)
            val_pred.append(1 if p >= 0.5 else 0)

        mae = sum(val_errors) / len(val_errors) if val_errors else 0.0
        clf_metrics = ModelEvaluator.evaluate_binary_classification(val_true, val_pred, val_prob)
        metrics = clf_metrics.to_dict()
        metrics["mae"] = round(mae, 4)

        self.metadata = ModelMetadata(
            model_name="StalenessDetectionModel",
            model_type=ModelType.STALENESS,
            version="1.0.0",
            created_at=datetime.now(timezone.utc),
            feature_names=self.FEATURE_NAMES,
            parameters=MappingProxyType({"weights": self.weights, "bias": self.bias}),
            metrics=MappingProxyType(metrics),
            training_data_provenance=dataset.name,
            description="Calibrated regularized logistic obsolescence risk model",
        )
        return metrics

    def save(self, filepath: str) -> None:
        data = {
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "weights": self.weights,
            "bias": self.bias,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "StalenessDetectionModel":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = ModelMetadata.from_dict(data["metadata"]) if data.get("metadata") else None
        return cls(metadata=meta, weights=data.get("weights"), bias=float(data.get("bias", 0.0)))
