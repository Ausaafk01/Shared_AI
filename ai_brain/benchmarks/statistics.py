"""
Statistical analysis tools for AI Brain benchmark comparisons.
Computes paired differences, effect sizes (Cohen's d), confidence intervals, and statistical tests.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ai_brain.benchmarks.models import PairedComparison


@dataclass(frozen=True)
class MetricStatisticalSummary:
    metric_name: str
    n_samples: int
    mean_without_brain: float
    std_without_brain: float
    mean_with_brain: float
    std_with_brain: float
    mean_difference: float  # With Brain - Without Brain
    std_difference: float
    standard_error: float
    confidence_interval_95: Tuple[float, float]
    cohens_d: float  # Paired effect size
    t_statistic: float
    degrees_of_freedom: int
    is_statistically_significant: bool
    limitation_notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "n_samples": self.n_samples,
            "mean_without_brain": round(self.mean_without_brain, 3),
            "std_without_brain": round(self.std_without_brain, 3),
            "mean_with_brain": round(self.mean_with_brain, 3),
            "std_with_brain": round(self.std_with_brain, 3),
            "mean_difference": round(self.mean_difference, 3),
            "std_difference": round(self.std_difference, 3),
            "standard_error": round(self.standard_error, 3),
            "confidence_interval_95": (
                round(self.confidence_interval_95[0], 3),
                round(self.confidence_interval_95[1], 3),
            ),
            "cohens_d": round(self.cohens_d, 3),
            "t_statistic": round(self.t_statistic, 3),
            "degrees_of_freedom": self.degrees_of_freedom,
            "is_statistically_significant": self.is_statistically_significant,
            "limitation_notes": self.limitation_notes,
        }


class BenchmarkStatistics:
    """Computes paired statistical tests and effect sizes across benchmark conditions."""

    @staticmethod
    def _t_critical(df: int) -> float:
        # Two-tailed 95% critical t values
        table = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            15: 2.131, 20: 2.086, 30: 2.042,
        }
        return table.get(df, 1.96)

    @classmethod
    def analyze_paired_metric(
        cls,
        name: str,
        values_without: Sequence[float],
        values_with: Sequence[float],
        higher_is_better: bool = True,
    ) -> MetricStatisticalSummary:
        n = len(values_without)
        if n != len(values_with):
            raise ValueError("Arrays must have identical length")
        if n < 2:
            raise ValueError("Need at least 2 samples for statistical analysis")

        df = n - 1
        t_crit = cls._t_critical(df)

        m_wo = sum(values_without) / n
        s_wo = math.sqrt(sum((x - m_wo) ** 2 for x in values_without) / df) if df > 0 else 0.0

        m_w = sum(values_with) / n
        s_w = math.sqrt(sum((x - m_w) ** 2 for x in values_with) / df) if df > 0 else 0.0

        diffs = [w - wo for w, wo in zip(values_with, values_without)]
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / df if df > 0 else 0.0
        std_diff = math.sqrt(var_diff)
        se_diff = std_diff / math.sqrt(n)

        ci = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)

        cohens_d = (mean_diff / std_diff) if std_diff > 0 else (0.0 if mean_diff == 0 else float("inf"))
        t_stat = (mean_diff / se_diff) if se_diff > 0 else 0.0

        # Significance: 95% CI does not overlap 0
        is_sig = (ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)

        notes = (
            f"Evaluated on {n} scenarios. With sample size N={n}, effect sizes should be interpreted with caution. "
            f"While statistical tests indicate {'a significant' if is_sig else 'no significant'} difference at p<0.05, "
            f"broader real-world multi-agent trials are recommended."
        )

        return MetricStatisticalSummary(
            metric_name=name,
            n_samples=n,
            mean_without_brain=m_wo,
            std_without_brain=s_wo,
            mean_with_brain=m_w,
            std_with_brain=s_w,
            mean_difference=mean_diff,
            std_difference=std_diff,
            standard_error=se_diff,
            confidence_interval_95=ci,
            cohens_d=cohens_d,
            t_statistic=t_stat,
            degrees_of_freedom=df,
            is_statistically_significant=is_sig,
            limitation_notes=notes,
        )

    @classmethod
    def compute_all_comparisons(
        cls,
        comparisons: Sequence[PairedComparison],
    ) -> Dict[str, MetricStatisticalSummary]:
        # 1. Task Success
        succ_wo = [1.0 if c.result_without_brain.task_success else 0.0 for c in comparisons]
        succ_w = [1.0 if c.result_with_brain.task_success else 0.0 for c in comparisons]
        task_summary = cls.analyze_paired_metric("Task Continuation Success Rate", succ_wo, succ_w, higher_is_better=True)

        # 2. Repeated Explanations Needed
        re_wo = [float(c.result_without_brain.repeated_explanations_needed) for c in comparisons]
        re_w = [float(c.result_with_brain.repeated_explanations_needed) for c in comparisons]
        re_summary = cls.analyze_paired_metric("Repeated Explanations Needed (Lower is Better)", re_wo, re_w, higher_is_better=False)

        # 3. Total Tokens Consumed
        tok_wo = [float(c.result_without_brain.total_tokens) for c in comparisons]
        tok_w = [float(c.result_with_brain.total_tokens) for c in comparisons]
        tok_summary = cls.analyze_paired_metric("Total Tokens Consumed per Query", tok_wo, tok_w, higher_is_better=False)

        # 4. Decisions Maintained
        dec_wo = [float(c.result_without_brain.decisions_maintained_count) for c in comparisons]
        dec_w = [float(c.result_with_brain.decisions_maintained_count) for c in comparisons]
        dec_summary = cls.analyze_paired_metric("Historical Decisions Maintained", dec_wo, dec_w, higher_is_better=True)

        return {
            "task_success": task_summary,
            "repeated_explanations": re_summary,
            "total_tokens": tok_summary,
            "decisions_maintained": dec_summary,
        }
