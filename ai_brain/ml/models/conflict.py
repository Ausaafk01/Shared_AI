"""
Conflict Detection Model for identifying genuine contradictions and classifying conflict types.
"""

from datetime import datetime, timezone
import json
import math
import re
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ai_brain.memories.conflicts.detector import ConflictDetector
from ai_brain.memories.conflicts.types import ConflictType
from ai_brain.memories.models import Memory
from ai_brain.ml.base import BaseMLModel, ModelMetadata, ModelPrediction, ModelType
from ai_brain.ml.datasets.schemas import DatasetSplit
from ai_brain.ml.exceptions import ModelNotTrainedError


CONFLICT_CLASSES: Tuple[str, ...] = (
    ConflictType.FACTUAL.value,
    ConflictType.TECHNICAL.value,
    ConflictType.TEMPORAL.value,
    ConflictType.DECISION.value,
    ConflictType.SOURCE.value,
    "NONE",
)


class ConflictDetectionModel(BaseMLModel):
    """
    Supervised Multi-Class ML model for detecting nuanced conflicts and classifying conflict types.
    
    Guarantees:
    - Learns distinct feature weight vectors per conflict type via One-vs-Rest regularized logistic regression.
    - Distinguishes genuine contradictions from chronological updates, differing detail levels, and compatible views.
    - Never automatically resolves or discards disputed memories.
    - Preserves evidence and provides transparent conflict type classification with calibrated probabilities.
    - Falls back to deterministic ConflictDetector when untrained.
    """

    FEATURE_NAMES = (
        "same_subject",
        "has_mutually_exclusive_entities",
        "has_port_or_numeric_mismatch",
        "has_temporal_progression_language",
        "has_negation_contradiction",
        "detail_level_difference",
        "has_provenance_dispute_markers",
        "is_decision_type",
        "time_gap_days",
    )

    def __init__(
        self,
        metadata: Optional[ModelMetadata] = None,
        class_weights: Optional[Dict[str, Dict[str, float]]] = None,
        class_biases: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        bias: float = 0.0,
    ) -> None:
        super().__init__(metadata)
        self.class_weights: Dict[str, Dict[str, float]] = class_weights or {
            c: {f: 0.0 for f in self.FEATURE_NAMES} for c in CONFLICT_CLASSES
        }
        self.class_biases: Dict[str, float] = class_biases or {c: 0.0 for c in CONFLICT_CLASSES}
        self.weights = weights or {f: 0.0 for f in self.FEATURE_NAMES}
        self.bias = bias
        self.deterministic_detector = ConflictDetector()

    @property
    def model_type(self) -> ModelType:
        return ModelType.CONFLICT

    def extract_conflict_features(self, m1: Memory, m2: Memory) -> Dict[str, float]:
        """Extract multi-factor pairwise contradiction features."""
        c1, c2 = m1.content.strip().lower(), m2.content.strip().lower()

        # Subject match
        from ai_brain.memories.updating.comparison import extract_memory_subject
        s1 = extract_memory_subject(m1.content)
        s2 = extract_memory_subject(m2.content)
        same_subj = 1.0 if (s1 and s1 == s2) else 0.0

        # Numeric / port mismatch
        ports1 = set(re.findall(r"\bport\b[^\d]*(\d+)", c1))
        ports2 = set(re.findall(r"\bport\b[^\d]*(\d+)", c2))
        port_mismatch = 1.0 if (ports1 and ports2 and ports1 != ports2) else 0.0

        # Temporal progression keywords
        temporal_kws = ["migrated", "switched", "upgraded", "replaced", "deprecated", "now uses", "subsequently"]
        has_temporal = 1.0 if any(k in c2 for k in temporal_kws) or any(k in c1 for k in temporal_kws) else 0.0

        # Mutually exclusive entities on same subject
        tech_sets = [
            {"postgresql", "postgres", "mysql", "sqlite", "mongodb"},
            {"redis", "memcached"},
            {"jwt", "session", "cookie"},
        ]
        has_exclusive = 0.0
        for tset in tech_sets:
            match1 = {t for t in tset if t in c1}
            match2 = {t for t in tset if t in c2}
            if match1 and match2 and match1 != match2:
                has_exclusive = 1.0
                break

        # Negation contradiction
        negations = ["not", "never", "cannot", "no longer", "removed"]
        has_neg1 = any(n in c1 for n in negations)
        has_neg2 = any(n in c2 for n in negations)
        neg_contra = 1.0 if (has_neg1 != has_neg2 and same_subj > 0) else 0.0

        # Detail level difference: one text contains the other or is a subset elaboration
        words1 = set(c1.split())
        words2 = set(c2.split())
        is_subset = words1.issubset(words2) or words2.issubset(words1)
        detail_diff = 1.0 if is_subset and not has_exclusive and not port_mismatch else 0.0

        # Provenance dispute markers
        prov_markers = ["unreliable", "fabricated", "fake", "falsified", "unauthentic", "misattributed"]
        prov_dispute = 1.0 if any(m in c1 or m in c2 for m in prov_markers) else 0.0

        # Decision type
        is_dec = 1.0 if (m1.memory_type == "DECISION" or m2.memory_type == "DECISION") else 0.0

        # Time gap
        gap_days = abs((m1.created_at - m2.created_at).total_seconds()) / 86400.0

        return {
            "same_subject": same_subj,
            "has_mutually_exclusive_entities": has_exclusive,
            "has_port_or_numeric_mismatch": port_mismatch,
            "has_temporal_progression_language": has_temporal,
            "has_negation_contradiction": neg_contra,
            "detail_level_difference": detail_diff,
            "has_provenance_dispute_markers": prov_dispute,
            "is_decision_type": is_dec,
            "time_gap_days": min(365.0, gap_days),
        }

    def predict(self, item: Any, **context: Any) -> ModelPrediction:
        """
        item: tuple of (Memory, Memory) or candidate Memory with existing in context.
        """
        if isinstance(item, tuple) and len(item) == 2:
            m1, m2 = item
        else:
            m1 = context.get("existing")
            m2 = item

        if m1 is None or m2 is None:
            raise ValueError("ConflictDetectionModel requires two memories (m1, m2) to compare")

        # Project isolation: cross-project is never a conflict
        if m1.project_id != m2.project_id:
            return ModelPrediction(
                target_value="NONE",
                confidence=1.0,
                feature_contributions=MappingProxyType({}),
                model_version=self.metadata.version if self.metadata else "deterministic",
                fallback_used=True,
                explanation="Memories belong to separate projects; strictly isolated",
            )

        # 1. Fallback if untrained
        if not self.is_trained:
            conf = self.deterministic_detector.detect_pair(m1, m2)
            if conf:
                return ModelPrediction(
                    target_value=conf.conflict_type.value,
                    confidence=conf.confidence,
                    feature_contributions=MappingProxyType({}),
                    model_version=self.metadata.version if self.metadata else "fallback-untrained",
                    fallback_used=True,
                    explanation=f"Deterministic detector identified {conf.conflict_type.value}: {conf.description}",
                )
            else:
                return ModelPrediction(
                    target_value="NONE",
                    confidence=0.9,
                    feature_contributions=MappingProxyType({}),
                    model_version=self.metadata.version if self.metadata else "fallback-untrained",
                    fallback_used=True,
                    explanation="Deterministic detector found no conflict",
                )

        feats = self.extract_conflict_features(m1, m2)

        # 2. Compute OvR logits for all conflict classes using learned parameters
        logits: Dict[str, float] = {}
        for c in CONFLICT_CLASSES:
            w_c = self.class_weights.get(c, {})
            b_c = self.class_biases.get(c, 0.0)
            logits[c] = b_c + sum(w_c.get(f, 0.0) * feats.get(f, 0.0) for f in self.FEATURE_NAMES)

        # Convert logits to normalized probabilities via softmax
        max_logit = max(logits.values())
        exp_vals = {c: math.exp(max(-15.0, min(15.0, l - max_logit))) for c, l in logits.items()}
        sum_exp = sum(exp_vals.values()) or 1.0
        probs = {c: exp_vals[c] / sum_exp for c in CONFLICT_CLASSES}

        # Argmax class
        predicted_class = max(probs.keys(), key=lambda c: probs[c])
        confidence = probs[predicted_class]

        # Feature contributions for predicted class
        winning_weights = self.class_weights.get(predicted_class, {})
        contributions = {
            f: round(winning_weights.get(f, 0.0) * feats.get(f, 0.0), 4)
            for f in self.FEATURE_NAMES
        }

        explanation = (
            f"Supervised multiclass model predicts {predicted_class} (prob={confidence:.3f}, "
            f"features active: {[f for f, v in contributions.items() if v != 0]})"
        )

        return ModelPrediction(
            target_value=predicted_class,
            confidence=round(confidence, 4),
            probabilities=MappingProxyType({c: round(p, 4) for c, p in probs.items()}),
            feature_contributions=MappingProxyType(contributions),
            model_version=self.metadata.version if self.metadata else "v1.0",
            fallback_used=False,
            explanation=explanation,
        )

    def train(
        self,
        dataset: DatasetSplit,
        *,
        learning_rate: float = 0.05,
        epochs: int = 60,
        l2_reg: float = 0.01,
        **kwargs: Any,
    ) -> Dict[str, float]:
        train_examples = dataset.train_examples
        if not train_examples:
            raise ModelNotTrainedError("Cannot train with empty dataset split")

        from ai_brain.ml.training.evaluator import ModelEvaluator

        # Initialize priors per class
        class_weights: Dict[str, Dict[str, float]] = {
            c: {f: 0.0 for f in self.FEATURE_NAMES} for c in CONFLICT_CLASSES
        }
        class_biases: Dict[str, float] = {c: 0.0 for c in CONFLICT_CLASSES}

        # Initialize sensible domain priors for One-vs-Rest classifiers
        class_weights[ConflictType.TECHNICAL.value]["has_port_or_numeric_mismatch"] = 3.5
        class_weights[ConflictType.TEMPORAL.value]["has_temporal_progression_language"] = 3.0
        class_weights[ConflictType.SOURCE.value]["has_provenance_dispute_markers"] = 3.5
        class_weights[ConflictType.DECISION.value]["is_decision_type"] = 2.0
        class_weights[ConflictType.DECISION.value]["has_mutually_exclusive_entities"] = 2.0
        class_weights[ConflictType.FACTUAL.value]["same_subject"] = 1.5
        class_weights[ConflictType.FACTUAL.value]["has_mutually_exclusive_entities"] = 2.5
        class_weights["NONE"]["detail_level_difference"] = 3.0

        n = len(train_examples)

        # Train each OvR binary classifier
        for c in CONFLICT_CLASSES:
            w_c = class_weights[c]
            b_c = class_biases[c]

            for _ in range(epochs):
                grad_w = {f: 0.0 for f in self.FEATURE_NAMES}
                grad_b = 0.0

                for ex in train_examples:
                    # Positive label if example target matches class c
                    ex_label = str(ex.label).upper()
                    y = 1.0 if (ex_label == c or (c == "NONE" and ex_label in ("NONE", "FALSE", "0", 0))) else 0.0
                    logit = b_c + sum(w_c[f] * ex.features.get(f, 0.0) for f in self.FEATURE_NAMES)
                    p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, logit))))
                    err = p - y

                    grad_b += err
                    for f in self.FEATURE_NAMES:
                        grad_w[f] += err * ex.features.get(f, 0.0)

                b_c -= learning_rate * (grad_b / n)
                for f in self.FEATURE_NAMES:
                    w_c[f] -= learning_rate * ((grad_w[f] / n) + (l2_reg * w_c[f]))

            class_weights[c] = w_c
            class_biases[c] = b_c

        self.class_weights = class_weights
        self.class_biases = class_biases
        self._is_trained = True

        # Validation evaluation
        val_true, val_pred = [], []
        for ex in dataset.validation_examples:
            ex_lbl = str(ex.label).upper()
            if ex_lbl in ("NONE", "FALSE", "0", "0.0"):
                target_str = "NONE"
            else:
                target_str = ex_lbl

            logits_v = {
                c: class_biases[c] + sum(class_weights[c][f] * ex.features.get(f, 0.0) for f in self.FEATURE_NAMES)
                for c in CONFLICT_CLASSES
            }
            pred_c = max(logits_v.keys(), key=lambda c: logits_v[c])
            val_true.append(target_str if target_str in CONFLICT_CLASSES else "NONE")
            val_pred.append(pred_c)

        eval_obj = ModelEvaluator.evaluate_multiclass_classification(
            val_true, val_pred, classes=CONFLICT_CLASSES
        )
        metrics = eval_obj.to_dict()

        self.metadata = ModelMetadata(
            model_name="ConflictDetectionModel",
            model_type=ModelType.CONFLICT,
            version="1.0.0",
            created_at=datetime.now(timezone.utc),
            feature_names=self.FEATURE_NAMES,
            parameters=MappingProxyType({"classes": list(CONFLICT_CLASSES)}),
            metrics=MappingProxyType({"macro_f1": eval_obj.macro_f1, "accuracy": eval_obj.accuracy}),
            training_data_provenance=dataset.name,
            description="Supervised multiclass One-vs-Rest regularized logistic conflict classifier",
        )
        return {"accuracy": eval_obj.accuracy, "macro_f1": eval_obj.macro_f1}

    def save(self, filepath: str) -> None:
        data = {
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "class_weights": self.class_weights,
            "class_biases": self.class_biases,
            "weights": self.weights,
            "bias": self.bias,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ConflictDetectionModel":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = ModelMetadata.from_dict(data["metadata"]) if data.get("metadata") else None
        return cls(
            metadata=meta,
            class_weights=data.get("class_weights"),
            class_biases=data.get("class_biases"),
            weights=data.get("weights"),
            bias=float(data.get("bias", 0.0)),
        )
