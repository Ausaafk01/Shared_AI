"""
Statistical analysis tools for AI Brain benchmark comparisons.
Computes paired differences, effect sizes (Cohen's d), confidence intervals,
exact McNemar's test for binary paired outcomes, and statistical significance.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ai_brain.benchmarks.models import PairedComparison


@dataclass(frozen=True)
class McNemarTestResult:
    """Exact McNemar's test for paired binary outcomes (e.g. task success/failure)."""
    contingency_table: Dict[str, int]  # {"both_pass": a, "only_with_brain_pass": b, "only_without_brain_pass": c, "both_fail": d}
    n_total: int
    b_discordant_with_brain: int       # Condition A failed, Condition B passed
    c_discordant_without_brain: int    # Condition A passed, Condition B failed
    exact_p_value: float               # Two-tailed exact binomial test on discordant pairs
    chi2_statistic: Optional[float]    # Asymptotic chi2 with Edwards continuity correction
    odds_ratio: Optional[float]        # Discordant odds ratio (b / c)
    is_statistically_significant: bool # True if exact_p_value < 0.05
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contingency_table": self.contingency_table,
            "n_total": self.n_total,
            "b_discordant_with_brain": self.b_discordant_with_brain,
            "c_discordant_without_brain": self.c_discordant_without_brain,
            "exact_p_value": self.exact_p_value,
            "chi2_statistic": round(self.chi2_statistic, 3) if self.chi2_statistic is not None else None,
            "odds_ratio": round(self.odds_ratio, 3) if (self.odds_ratio is not None and not math.isinf(self.odds_ratio)) else "inf",
            "is_statistically_significant": self.is_statistically_significant,
            "interpretation": self.interpretation,
        }


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
            15: 2.131, 20: 2.086, 25: 2.060, 29: 2.045, 30: 2.042,
        }
        return table.get(df, 1.96)

    @classmethod
    def compute_mcnemar_test(
        cls,
        pairs: Sequence[Tuple[bool, bool]],  # (success_without_brain, success_with_brain)
    ) -> McNemarTestResult:
        """
        Computes exact McNemar's test for paired binary observations.
        Null hypothesis H0: p_b = p_c (marginal probabilities of success are identical).
        """
        n = len(pairs)
        if n == 0:
            raise ValueError("Pairs list cannot be empty")

        a = sum(1 for wo, w in pairs if wo and w)       # Both pass
        b = sum(1 for wo, w in pairs if not wo and w)   # Only With Brain passes
        c = sum(1 for wo, w in pairs if wo and not w)   # Only Without Brain passes
        d = sum(1 for wo, w in pairs if not wo and not w) # Both fail

        n_disc = b + c
        if n_disc == 0:
            p_val = 1.0
            chi2 = 0.0
            odds_ratio = 1.0
        else:
            # Exact two-sided binomial test under H0: p=0.5
            k = min(b, c)
            # Probability of observing k or fewer successes in n_disc trials with p=0.5
            p_cum = sum(math.comb(n_disc, i) * (0.5 ** n_disc) for i in range(k + 1))
            p_val = min(1.0, 2.0 * p_cum)

            # Asymptotic chi-square with Edwards continuity correction
            chi2 = ((abs(b - c) - 1.0) ** 2) / float(n_disc)
            odds_ratio = (b / c) if c > 0 else float("inf")

        is_sig = p_val < 0.05
        interp = (
            f"N={n} paired scenarios. Contingency table: Both Pass={a}, Brain-Only Pass={b}, "
            f"Baseline-Only Pass={c}, Both Fail={d}. Discordant pairs={n_disc}. "
            f"Exact binomial p-value = {p_val:.2e}. "
            f"The difference is {'statistically significant (p < 0.05)' if is_sig else 'not statistically significant'}."
        )

        return McNemarTestResult(
            contingency_table={
                "both_pass": a,
                "only_with_brain_pass": b,
                "only_without_brain_pass": c,
                "both_fail": d,
            },
            n_total=n,
            b_discordant_with_brain=b,
            c_discordant_without_brain=c,
            exact_p_value=p_val,
            chi2_statistic=chi2,
            odds_ratio=odds_ratio,
            is_statistically_significant=is_sig,
            interpretation=interp,
        )

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
            f"Evaluated on N={n} paired scenarios. "
            f"Effect size Cohen's d={cohens_d:.2f}. "
            f"Paired t-test indicates {'a statistically significant' if is_sig else 'no statistically significant'} "
            f"difference at two-tailed alpha=0.05 (t={t_stat:.2f}, df={df})."
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
    ) -> Dict[str, Any]:
        # 1. Exact McNemar's Test for Binary Task Continuation
        succ_pairs = [(c.result_without_brain.task_success, c.result_with_brain.task_success) for c in comparisons]
        mcnemar_summary = cls.compute_mcnemar_test(succ_pairs)

        # 2. Continuous Metric Paired Analyses
        # Task Success as continuous (0/1) for Cohen's d and CI
        succ_wo = [1.0 if c.result_without_brain.task_success else 0.0 for c in comparisons]
        succ_w = [1.0 if c.result_with_brain.task_success else 0.0 for c in comparisons]
        task_summary = cls.analyze_paired_metric("Task Continuation Success Rate", succ_wo, succ_w, higher_is_better=True)

        # Partial Success Score
        partial_wo = [c.result_without_brain.partial_success_score for c in comparisons]
        partial_w = [c.result_with_brain.partial_success_score for c in comparisons]
        partial_summary = cls.analyze_paired_metric("Partial Credit Success Score", partial_wo, partial_w, higher_is_better=True)

        # Repeated Explanations Needed
        re_wo = [float(c.result_without_brain.repeated_explanations_needed) for c in comparisons]
        re_w = [float(c.result_with_brain.repeated_explanations_needed) for c in comparisons]
        re_summary = cls.analyze_paired_metric("Repeated Explanations Needed (Lower is Better)", re_wo, re_w, higher_is_better=False)

        # Total Tokens Consumed
        tok_wo = [float(c.result_without_brain.total_tokens) for c in comparisons]
        tok_w = [float(c.result_with_brain.total_tokens) for c in comparisons]
        tok_summary = cls.analyze_paired_metric("Total Tokens Consumed per Query", tok_wo, tok_w, higher_is_better=False)

        # Decisions Maintained
        dec_wo = [float(c.result_without_brain.decisions_maintained_count) for c in comparisons]
        dec_w = [float(c.result_with_brain.decisions_maintained_count) for c in comparisons]
        dec_summary = cls.analyze_paired_metric("Historical Decisions Maintained", dec_wo, dec_w, higher_is_better=True)

        return {
            "mcnemar_test": mcnemar_summary,
            "task_success": task_summary,
            "partial_success": partial_summary,
            "repeated_explanations": re_summary,
            "total_tokens": tok_summary,
            "decisions_maintained": dec_summary,
        }
