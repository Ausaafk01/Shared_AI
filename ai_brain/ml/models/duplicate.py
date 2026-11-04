"""
Duplicate Detection Model for distinguishing exact duplicates, semantic duplicates,
related distinct memories, and contradictory claims.
"""

from datetime import datetime, timezone
from enum import StrEnum
import json
import math
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ai_brain.memories.models import Memory
from ai_brain.memories.updating.comparison import MemoryComparator, MemoryRelation
from ai_brain.ml.base import BaseMLModel, ModelMetadata, ModelPrediction, ModelType
from ai_brain.ml.datasets.schemas import DatasetSplit
from ai_brain.ml.exceptions import ModelNotTrainedError


class DuplicateClassification(StrEnum):
    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    SEMANTIC_DUPLICATE = "SEMANTIC_DUPLICATE"
    RELATED_DISTINCT = "RELATED_DISTINCT"
    CONTRADICTORY = "CONTRADICTORY"


class DuplicateDetectionModel(BaseMLModel):
    """
    ML model for fine-grained classification of memory pairs.
    
    Guarantees:
    - Never merges memories solely on embedding similarity.
    - Explicitly flags contradictory claims as CONTRADICTORY rather than duplicates.
    - Preserves provenance and history.
    - Falls back to deterministic MemoryComparator when untrained.
    """

    FEATURE_NAMES = (
        "exact_content_match",
        "jaccard_word_similarity",
        "char_length_ratio",
        "same_memory_type",
        "same_project_id",
        "common_tech_entity_count",
        "has_negation_contrast",
        "subject_match",
        "provenance_overlap",
    )

    def __init__(self, metadata: Optional[ModelMetadata] = None, weights: Optional[Dict[str, float]] = None, bias: float = 0.0) -> None:
        super().__init__(metadata)
        self.weights = weights or {f: 0.0 for f in self.FEATURE_NAMES}
        self.bias = bias
        self.deterministic_comparator = MemoryComparator()

    @property
    def model_type(self) -> ModelType:
        return ModelType.DUPLICATE

    def extract_pair_features(self, m1: Memory, m2: Memory) -> Dict[str, float]:
        """Extract pairwise features comparing existing m1 and candidate m2."""
        c1, c2 = m1.content.strip().lower(), m2.content.strip().lower()
        exact = 1.0 if c1 == c2 else 0.0

        w1 = set(c1.split())
        w2 = set(c2.split())
        inter = len(w1.intersection(w2))
        union = len(w1.union(w2))
        jaccard = inter / union if union > 0 else 0.0

        len1, len2 = len(c1), len(c2)
        len_ratio = min(len1, len2) / max(1, max(len1, len2))

        same_type = 1.0 if m1.memory_type == m2.memory_type else 0.0
        same_proj = 1.0 if m1.project_id == m2.project_id else 0.0

        # Tech terms
        tech_terms = ["postgres", "postgresql", "mysql", "redis", "mongodb", "sqlite", "jwt", "oauth"]
        t1 = {t for t in tech_terms if t in c1}
        t2 = {t for t in tech_terms if t in c2}
        common_tech = float(len(t1.intersection(t2)))

        # Negation contrast
        negations = ["not", "never", "cannot", "deprecated", "removed", "no longer"]
        has_neg1 = any(n in c1 for n in negations)
        has_neg2 = any(n in c2 for n in negations)
        neg_contrast = 1.0 if (has_neg1 != has_neg2 and common_tech > 0) else 0.0

        # Provenance overlap
        a1 = set(m1.source_agent_ids)
        a2 = set(m2.source_agent_ids)
        prov_overlap = len(a1.intersection(a2)) / max(1, len(a1.union(a2))) if (a1 or a2) else 0.0

        # Subject match
        from ai_brain.memories.updating.comparison import extract_memory_subject
        s1 = extract_memory_subject(m1.content)
        s2 = extract_memory_subject(m2.content)
        subj_match = 1.0 if (s1 and s1 == s2) else 0.0

        return {
            "exact_content_match": exact,
            "jaccard_word_similarity": round(jaccard, 4),
            "char_length_ratio": round(len_ratio, 4),
            "same_memory_type": same_type,
            "same_project_id": same_proj,
            "common_tech_entity_count": common_tech,
            "has_negation_contrast": neg_contrast,
            "subject_match": subj_match,
            "provenance_overlap": round(prov_overlap, 4),
        }

    def predict(self, item: Any, **context: Any) -> ModelPrediction:
        """
        item: tuple of (Memory, Memory) or incoming Memory with existing passed in context.
        """
        if isinstance(item, tuple) and len(item) == 2:
            m1, m2 = item
        else:
            m1 = context.get("existing")
            m2 = item

        if m1 is None or m2 is None:
            raise ValueError("DuplicateDetectionModel requires two memories (m1, m2) to compare")

        # 1. Fallback if untrained or cross-project
        if not self.is_trained:
            comp = self.deterministic_comparator.compare(m1, m2)
            # Map deterministic relation to classification
            if comp.relation == MemoryRelation.IDENTICAL:
                cls_val = DuplicateClassification.EXACT_DUPLICATE.value
                conf = 1.0
            elif comp.relation == MemoryRelation.DUPLICATE:
                cls_val = DuplicateClassification.SEMANTIC_DUPLICATE.value
                conf = 0.9
            elif comp.relation == MemoryRelation.CONTRADICTORY:
                cls_val = DuplicateClassification.CONTRADICTORY.value
                conf = 0.85
            else:
                cls_val = DuplicateClassification.RELATED_DISTINCT.value
                conf = 0.8

            return ModelPrediction(
                target_value=cls_val,
                confidence=conf,
                feature_contributions=MappingProxyType({}),
                model_version=self.metadata.version if self.metadata else "fallback-untrained",
                fallback_used=True,
                explanation=f"Deterministic fallback: {comp.reason}",
            )

        feats = self.extract_pair_features(m1, m2)

        # 2. Exact match check
        if feats["exact_content_match"] == 1.0:
            return ModelPrediction(
                target_value=DuplicateClassification.EXACT_DUPLICATE.value,
                confidence=1.0,
                feature_contributions=MappingProxyType(feats),
                model_version=self.metadata.version if self.metadata else "v1.0",
                fallback_used=False,
                explanation="Byte-for-byte exact content match",
            )

        # 3. Contradiction gate: negation contrast or conflicting assertions on same subject
        if feats["has_negation_contrast"] > 0.0 or (feats["subject_match"] > 0.0 and feats["jaccard_word_similarity"] > 0.35 and feats["exact_content_match"] == 0.0):
            from ai_brain.memories.conflicts.detector import ConflictDetector
            det = ConflictDetector()
            conf = det.detect_pair(m1, m2)
            if conf:
                return ModelPrediction(
                    target_value=DuplicateClassification.CONTRADICTORY.value,
                    confidence=0.92,
                    feature_contributions=MappingProxyType(feats),
                    model_version=self.metadata.version if self.metadata else "v1.0",
                    fallback_used=False,
                    explanation=f"Contradiction gate: detected genuine dispute '{conf.description}'",
                )

        # 4. Learned logistic prediction
        raw_logit = self.bias
        contributions = {}
        for f in self.FEATURE_NAMES:
            val = feats.get(f, 0.0)
            w = self.weights.get(f, 0.0)
            contrib = w * val
            contributions[f] = round(contrib, 4)
            raw_logit += contrib

        prob = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, raw_logit))))

        if prob >= 0.5:
            pred_class = DuplicateClassification.SEMANTIC_DUPLICATE.value
            conf = min(0.95, max(0.65, prob))
            expl = f"Supervised logistic model predicts semantic duplicate (p={prob:.3f}, logit={raw_logit:.2f})"
        else:
            pred_class = DuplicateClassification.RELATED_DISTINCT.value
            conf = min(0.95, max(0.65, 1.0 - prob))
            expl = f"Supervised logistic model predicts distinct knowledge (p={prob:.3f} below duplicate threshold)"

        return ModelPrediction(
            target_value=pred_class,
            confidence=round(conf, 4),
            probabilities=MappingProxyType({"duplicate_prob": round(prob, 4), "distinct_prob": round(1.0 - prob, 4)}),
            feature_contributions=MappingProxyType(contributions),
            model_version=self.metadata.version if self.metadata else "v1.0",
            fallback_used=False,
            explanation=expl,
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

        weights = {f: 0.0 for f in self.FEATURE_NAMES}
        weights["exact_content_match"] = 3.0
        weights["jaccard_word_similarity"] = 2.0
        weights["char_length_ratio"] = 1.0
        weights["same_memory_type"] = 1.0
        weights["subject_match"] = 1.5
        weights["has_negation_contrast"] = -3.0
        bias = -1.0

        n = len(train_examples)
        for _ in range(epochs):
            grad_w = {f: 0.0 for f in self.FEATURE_NAMES}
            grad_b = 0.0

            for ex in train_examples:
                y = 1.0 if (ex.label in (1, 1.0, True, "SEMANTIC_DUPLICATE", "EXACT_DUPLICATE")) else 0.0
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

        val_true, val_pred, val_prob = [], [], []
        for ex in dataset.validation_examples:
            y = 1 if (ex.label in (1, 1.0, True, "SEMANTIC_DUPLICATE", "EXACT_DUPLICATE")) else 0
            logit = bias + sum(weights[f] * ex.features.get(f, 0.0) for f in self.FEATURE_NAMES)
            p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, logit))))
            val_true.append(y)
            val_prob.append(p)
            val_pred.append(1 if p >= 0.5 else 0)

        metrics_obj = ModelEvaluator.evaluate_binary_classification(val_true, val_pred, val_prob)
        metrics = metrics_obj.to_dict()

        self.metadata = ModelMetadata(
            model_name="DuplicateDetectionModel",
            model_type=ModelType.DUPLICATE,
            version="1.0.0",
            created_at=datetime.now(timezone.utc),
            feature_names=self.FEATURE_NAMES,
            parameters=MappingProxyType({"weights": self.weights, "bias": self.bias, "threshold": 0.5}),
            metrics=MappingProxyType(metrics),
            training_data_provenance=dataset.name,
            description="Supervised regularized logistic regression for pair duplicate detection",
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
    def load(cls, filepath: str) -> "DuplicateDetectionModel":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = ModelMetadata.from_dict(data["metadata"]) if data.get("metadata") else None
        return cls(metadata=meta, weights=data.get("weights"), bias=float(data.get("bias", 0.0)))
