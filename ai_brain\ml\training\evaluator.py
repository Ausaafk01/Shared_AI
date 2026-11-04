"""
Evaluation metrics and comparative reporting for AI Brain ML models.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    brier_score: float  # Calibration metric (lower is better calibrated)
    total_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "brier_score": round(self.brier_score, 4),
            "total_samples": self.total_samples,
        }


@dataclass(frozen=True)
class RankingMetrics:
    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float
    total_queries: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "mrr": round(self.mrr, 4),
            "total_queries": self.total_queries,
        }


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2_score: float
    total_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "r2_score": round(self.r2_score, 4),
            "total_samples": self.total_samples,
        }


@dataclass(frozen=True)
class ModelEvaluationReport:
    """Comprehensive evaluation report comparing ML performance against deterministic baseline."""
    model_name: str
    model_version: str
    dataset_name: str
    dataset_size: int
    synthetic_ratio: float
    metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    improvement_delta: Dict[str, float]
    is_suitable_for_optional_integration: bool
    evaluation_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dataset_name": self.dataset_name,
            "dataset_size": self.dataset_size,
            "synthetic_ratio": round(self.synthetic_ratio, 4),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "baseline_metrics": {k: round(v, 4) for k, v in self.baseline_metrics.items()},
            "improvement_delta": {k: round(v, 4) for k, v in self.improvement_delta.items()},
            "is_suitable_for_optional_integration": self.is_suitable_for_optional_integration,
            "evaluation_notes": self.evaluation_notes,
        }


class ModelEvaluator:
    """Computes exact metrics for classification, ranking, and regression tasks."""

    @staticmethod
    def evaluate_binary_classification(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_prob: Optional[Sequence[float]] = None,
    ) -> ClassificationMetrics:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have identical length")
        n = len(y_true)
        if n == 0:
            return ClassificationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

        accuracy = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        if y_prob is not None and len(y_prob) == n:
            brier = sum((p - y) ** 2 for p, y in zip(y_prob, y_true)) / n
        else:
            brier = sum((p - y) ** 2 for p, y in zip(y_pred, y_true)) / n

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            brier_score=brier,
            total_samples=n,
        )

    @staticmethod
    def evaluate_regression(
        y_true: Sequence[float],
        y_pred: Sequence[float],
    ) -> RegressionMetrics:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have identical length")
        n = len(y_true)
        if n == 0:
            return RegressionMetrics(0.0, 0.0, 0.0, 0)

        mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n
        mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n
        rmse = math.sqrt(mse)

        mean_y = sum(y_true) / n
        ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
        ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return RegressionMetrics(mae=mae, rmse=rmse, r2_score=r2, total_samples=n)

    @staticmethod
    def evaluate_ranking(
        ranked_relevance_lists: Sequence[Sequence[float]],
    ) -> RankingMetrics:
        """
        Evaluate NDCG and MRR across queries.
        ranked_relevance_lists: list of lists where each element is the binary or graded relevance of ranked items.
        """
        if not ranked_relevance_lists:
            return RankingMetrics(0.0, 0.0, 0.0, 0)

        def dcg_at_k(rels: Sequence[float], k: int) -> float:
            score = 0.0
            for i, rel in enumerate(rels[:k]):
                score += (2 ** rel - 1) / math.log2(i + 2)
            return score

        def ndcg_at_k(rels: Sequence[float], k: int) -> float:
            actual = dcg_at_k(rels, k)
            ideal = dcg_at_k(sorted(rels, reverse=True), k)
            return actual / ideal if ideal > 0 else (1.0 if actual == 0 else 0.0)

        ndcg_5_scores = [ndcg_at_k(r, 5) for r in ranked_relevance_lists if len(r) > 0]
        ndcg_10_scores = [ndcg_at_k(r, 10) for r in ranked_relevance_lists if len(r) > 0]

        # MRR
        rr_scores = []
        for r in ranked_relevance_lists:
            rr = 0.0
            for idx, rel in enumerate(r):
                if rel > 0:
                    rr = 1.0 / (idx + 1)
                    break
            rr_scores.append(rr)

        avg_5 = sum(ndcg_5_scores) / len(ndcg_5_scores) if ndcg_5_scores else 0.0
        avg_10 = sum(ndcg_10_scores) / len(ndcg_10_scores) if ndcg_10_scores else 0.0
        avg_mrr = sum(rr_scores) / len(rr_scores) if rr_scores else 0.0

        return RankingMetrics(
            ndcg_at_5=avg_5,
            ndcg_at_10=avg_10,
            mrr=avg_mrr,
            total_queries=len(ranked_relevance_lists),
        )
