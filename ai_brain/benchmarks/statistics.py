"""
Statistical analysis tools for AI Brain benchmark comparisons.
Computes paired differences, effect sizes (Cohen's d), confidence intervals,
exact McNemar's test for binary paired outcomes, and cluster-aware category-stratified analysis.
"""

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ai_brain.benchmarks.models import PairedComparison, ScenarioCategory


@dataclass(frozen=True)
class McNemarTestResult:
    """Exact McNemar's test for paired binary outcomes (e.g. task success/failure)."""
    comparison_name: str
    contingency_table: Dict[str, int]  # {"both_pass": a, "only_second_pass": b, "only_first_pass": c, "both_fail": d}
    n_total: int
    b_discordant_second: int           # First condition failed, Second condition passed
    c_discordant_first: int            # First condition passed, Second condition failed
    exact_p_value: float               # Two-tailed exact binomial test on discordant pairs
    chi2_statistic: Optional[float]    # Asymptotic chi2 with Edwards continuity correction
    odds_ratio: Optional[float]        # Discordant odds ratio (b / c)
    is_statistically_significant: bool # True if exact_p_value < 0.05
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_name": self.comparison_name,
            "contingency_table": self.contingency_table,
            "n_total": self.n_total,
            "b_discordant_second": self.b_discordant_second,
            "c_discordant_first": self.c_discordant_first,
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


@dataclass(frozen=True)
class CategoryStatisticalSummary:
    """Summary of performance within a specific scenario category."""
    category_name: str
    n_scenarios: int
    success_rate_cold_start: float
    success_rate_raw_history: float
    success_rate_ai_brain: float
    partial_score_cold_start: float
    partial_score_raw_history: float
    partial_score_ai_brain: float
    avg_tokens_cold_start: float
    avg_tokens_raw_history: float
    avg_tokens_ai_brain: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category_name": self.category_name,
            "n_scenarios": self.n_scenarios,
            "success_rate_cold_start": round(self.success_rate_cold_start, 3),
            "success_rate_raw_history": round(self.success_rate_raw_history, 3),
            "success_rate_ai_brain": round(self.success_rate_ai_brain, 3),
            "partial_score_cold_start": round(self.partial_score_cold_start, 3),
            "partial_score_raw_history": round(self.partial_score_raw_history, 3),
            "partial_score_ai_brain": round(self.partial_score_ai_brain, 3),
            "avg_tokens_cold_start": round(self.avg_tokens_cold_start, 1),
            "avg_tokens_raw_history": round(self.avg_tokens_raw_history, 1),
            "avg_tokens_ai_brain": round(self.avg_tokens_ai_brain, 1),
        }


@dataclass(frozen=True)
class ClusterAwareStatisticalSummary:
    """Cluster-stratified category-level analysis accounting for within-category dependencies."""
    metric_name: str
    n_clusters: int                  # 10 scenario categories
    cluster_size: int                # 3 scenarios per category
    cluster_mean_diff: float         # Mean difference across cluster means (C - A)
    cluster_std_error: float         # Between-cluster standard error
    confidence_interval_95: Tuple[float, float]
    t_statistic: float
    degrees_of_freedom: int          # df = K - 1 = 9
    is_statistically_significant: bool
    notes: str
    category_differences: Tuple[float, ...] = ()  # Paired differences per category (C - A)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "n_clusters": self.n_clusters,
            "cluster_size": self.cluster_size,
            "cluster_mean_diff": round(self.cluster_mean_diff, 4),
            "cluster_std_error": round(self.cluster_std_error, 4),
            "confidence_interval_95": (
                round(self.confidence_interval_95[0], 4),
                round(self.confidence_interval_95[1], 4),
            ),
            "t_statistic": round(self.t_statistic, 3),
            "degrees_of_freedom": self.degrees_of_freedom,
            "is_statistically_significant": self.is_statistically_significant,
            "notes": self.notes,
            "category_differences": [round(d, 4) for d in self.category_differences],
        }


class BenchmarkStatistics:
    """Computes paired statistical tests, cluster-aware category tests, and effect sizes."""

    @staticmethod
    def _t_critical(df: int) -> float:
        table = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            15: 2.131, 20: 2.086, 25: 2.060, 29: 2.045, 30: 2.042,
        }
        return table.get(df, 1.96)

    @classmethod
    def compute_mcnemar_test(
        cls,
        pairs: Sequence[Tuple[bool, bool]],
        comparison_name: str = "Condition A vs Condition C",
    ) -> McNemarTestResult:
        """
        Computes exact McNemar's test for paired binary observations.
        Null hypothesis H0: Marginal probabilities of success are identical.
        """
        n = len(pairs)
        if n == 0:
            raise ValueError("Pairs list cannot be empty")

        a = sum(1 for c1, c2 in pairs if c1 and c2)       # Both pass
        b = sum(1 for c1, c2 in pairs if not c1 and c2)   # Only second condition passes
        c = sum(1 for c1, c2 in pairs if c1 and not c2)   # Only first condition passes
        d = sum(1 for c1, c2 in pairs if not c1 and not c2) # Both fail

        n_disc = b + c
        if n_disc == 0:
            p_val = 1.0
            chi2 = 0.0
            odds_ratio = 1.0
        else:
            k = min(b, c)
            p_cum = sum(math.comb(n_disc, i) * (0.5 ** n_disc) for i in range(k + 1))
            p_val = min(1.0, 2.0 * p_cum)
            chi2 = ((abs(b - c) - 1.0) ** 2) / float(n_disc)
            odds_ratio = (b / c) if c > 0 else float("inf")

        is_sig = p_val < 0.05
        interp = (
            f"{comparison_name} (N={n}). Contingency table: Both Pass={a}, Second Only Pass={b}, "
            f"First Only Pass={c}, Both Fail={d}. Discordant pairs={n_disc}. "
            f"Exact binomial p-value = {p_val:.2e}. "
            f"The difference is {'statistically significant (p < 0.05)' if is_sig else 'not statistically significant'}."
        )

        return McNemarTestResult(
            comparison_name=comparison_name,
            contingency_table={
                "both_pass": a,
                "only_second_pass": b,
                "only_first_pass": c,
                "both_fail": d,
            },
            n_total=n,
            b_discordant_second=b,
            c_discordant_first=c,
            exact_p_value=p_val,
            chi2_statistic=chi2,
            odds_ratio=odds_ratio,
            is_statistically_significant=is_sig,
            interpretation=interp,
        )

    @classmethod
    def compute_cluster_aware_analysis(
        cls,
        comparisons: Sequence[PairedComparison],
        metric_getter: Callable[[PairedComparison], Tuple[float, float]],  # Returns (val_c, val_a)
        metric_name: str,
    ) -> ClusterAwareStatisticalSummary:
        """
        Computes cluster-stratified analysis treating the 10 categories as primary clusters (df=9).
        Averages within each category cluster before computing standard error.
        """
        by_cat: Dict[ScenarioCategory, List[Tuple[float, float]]] = {}
        for comp in comparisons:
            by_cat.setdefault(comp.category, []).append(metric_getter(comp))

        k_clusters = len(by_cat)
        cat_diffs: List[float] = []
        for cat, pairs in by_cat.items():
            mean_c = sum(p[0] for p in pairs) / len(pairs)
            mean_a = sum(p[1] for p in pairs) / len(pairs)
            cat_diffs.append(mean_c - mean_a)

        if k_clusters < 2:
            single_mean = cat_diffs[0] if cat_diffs else 0.0
            return ClusterAwareStatisticalSummary(
                metric_name=metric_name,
                n_clusters=k_clusters,
                degrees_of_freedom=0,
                cluster_mean_difference=single_mean,
                cluster_standard_error=0.0,
                t_statistic=0.0,
                confidence_interval_95=(single_mean, single_mean),
                is_statistically_significant=False,
                notes="Single category evaluated; cluster variance undefined (requires >= 2 categories)",
            )

        cluster_mean_diff = sum(cat_diffs) / k_clusters
        df = k_clusters - 1
        var_cluster = sum((cd - cluster_mean_diff) ** 2 for cd in cat_diffs) / df if df > 0 else 0.0
        se_cluster = math.sqrt(var_cluster / k_clusters)

        t_crit = cls._t_critical(df)
        ci = (cluster_mean_diff - t_crit * se_cluster, cluster_mean_diff + t_crit * se_cluster)
        t_stat = (cluster_mean_diff / se_cluster) if se_cluster > 0 else 0.0
        is_sig = (ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)

        notes = (
            f"Cluster-stratified across {k_clusters} categories (cluster size=3, df={df}). "
            f"Direction defined as Condition C minus Condition A (C - A). "
            f"Exploratory and conditional on treating the {k_clusters} categories as independent clusters; "
            f"these results do not establish general performance across real-world agent tasks, "
            f"and statistical significance does not constitute proof of general superiority. "
            f"Between-category SE={se_cluster:.4f}, t={t_stat:.3f}. Difference is "
            f"{'statistically significant (p < 0.05)' if is_sig else 'not statistically significant'}."
        )

        return ClusterAwareStatisticalSummary(
            metric_name=metric_name,
            n_clusters=k_clusters,
            cluster_size=3,
            cluster_mean_diff=cluster_mean_diff,
            cluster_std_error=se_cluster,
            confidence_interval_95=ci,
            t_statistic=t_stat,
            degrees_of_freedom=df,
            is_statistically_significant=is_sig,
            notes=notes,
            category_differences=tuple(cat_diffs),
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

        is_sig = (ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)

        notes = (
            f"Unclustered paired evaluation across N={n} scenarios (df={df}). "
            f"Effect size Cohen's d={cohens_d:.2f}. "
            f"Paired t-test indicates {'a statistically significant' if is_sig else 'no statistically significant'} "
            f"difference at two-tailed alpha=0.05 (t={t_stat:.2f})."
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
        # 1. Exact McNemar's Tests for Binary Task Success
        # Cold Start (A) vs Raw History (B)
        pairs_ab = [
            (c.result_without_brain.task_success, c.result_raw_history.task_success if c.result_raw_history else False)
            for c in comparisons
        ]
        mcnemar_ab = cls.compute_mcnemar_test(pairs_ab, comparison_name="Condition A (Cold Start) vs Condition B (Raw History)")

        # Cold Start (A) vs AI Brain (C)
        pairs_ac = [
            (c.result_without_brain.task_success, c.result_with_brain.task_success)
            for c in comparisons
        ]
        mcnemar_ac = cls.compute_mcnemar_test(pairs_ac, comparison_name="Condition A (Cold Start) vs Condition C (AI Brain)")

        # Raw History (B) vs AI Brain (C)
        pairs_bc = [
            (c.result_raw_history.task_success if c.result_raw_history else False, c.result_with_brain.task_success)
            for c in comparisons
        ]
        mcnemar_bc = cls.compute_mcnemar_test(pairs_bc, comparison_name="Condition B (Raw History) vs Condition C (AI Brain)")

        # 2. Cluster-Aware Category-Stratified Analyses (K=10 clusters, df=9)
        cluster_task_success = cls.compute_cluster_aware_analysis(
            comparisons,
            lambda c: (1.0 if c.result_with_brain.task_success else 0.0, 1.0 if c.result_without_brain.task_success else 0.0),
            "Task Continuation Success (Cluster-Stratified, df=9)",
        )
        cluster_partial = cls.compute_cluster_aware_analysis(
            comparisons,
            lambda c: (c.result_with_brain.partial_success_score, c.result_without_brain.partial_success_score),
            "Partial Credit Score (Cluster-Stratified, df=9)",
        )
        cluster_tokens = cls.compute_cluster_aware_analysis(
            comparisons,
            lambda c: (float(c.result_with_brain.total_tokens), float(c.result_without_brain.total_tokens)),
            "Total Tokens Consumed Overhead (Cluster-Stratified, df=9)",
        )
        cluster_explanations = cls.compute_cluster_aware_analysis(
            comparisons,
            lambda c: (float(c.result_with_brain.repeated_explanations_needed), float(c.result_without_brain.repeated_explanations_needed)),
            "Repeated Explanations Needed (Cluster-Stratified, df=9)",
        )
        cluster_decisions = cls.compute_cluster_aware_analysis(
            comparisons,
            lambda c: (float(c.result_with_brain.decisions_maintained_count), float(c.result_without_brain.decisions_maintained_count)),
            "Decisions Maintained (Cluster-Stratified, df=9)",
        )

        # 3. Category Summaries Breakdown (All 10 Categories)
        cat_summaries: Dict[str, CategoryStatisticalSummary] = {}
        for cat in ScenarioCategory:
            cat_comps = [c for c in comparisons if c.category == cat]
            if cat_comps:
                n_c = len(cat_comps)
                cat_summaries[cat.value] = CategoryStatisticalSummary(
                    category_name=cat.value,
                    n_scenarios=n_c,
                    success_rate_cold_start=sum(1 for c in cat_comps if c.result_without_brain.task_success) / n_c,
                    success_rate_raw_history=sum(1 for c in cat_comps if c.result_raw_history and c.result_raw_history.task_success) / n_c,
                    success_rate_ai_brain=sum(1 for c in cat_comps if c.result_with_brain.task_success) / n_c,
                    partial_score_cold_start=sum(c.result_without_brain.partial_success_score for c in cat_comps) / n_c,
                    partial_score_raw_history=sum(c.result_raw_history.partial_success_score for c in cat_comps if c.result_raw_history) / n_c,
                    partial_score_ai_brain=sum(c.result_with_brain.partial_success_score for c in cat_comps) / n_c,
                    avg_tokens_cold_start=sum(c.result_without_brain.total_tokens for c in cat_comps) / n_c,
                    avg_tokens_raw_history=sum(c.result_raw_history.total_tokens for c in cat_comps if c.result_raw_history) / n_c,
                    avg_tokens_ai_brain=sum(c.result_with_brain.total_tokens for c in cat_comps) / n_c,
                )

        # 4. Continuous Metric Paired Analyses (Condition A vs C)
        succ_wo = [1.0 if c.result_without_brain.task_success else 0.0 for c in comparisons]
        succ_w = [1.0 if c.result_with_brain.task_success else 0.0 for c in comparisons]
        task_summary = cls.analyze_paired_metric("Task Continuation Success Rate", succ_wo, succ_w, higher_is_better=True)

        partial_wo = [c.result_without_brain.partial_success_score for c in comparisons]
        partial_w = [c.result_with_brain.partial_success_score for c in comparisons]
        partial_summary = cls.analyze_paired_metric("Partial Credit Success Score", partial_wo, partial_w, higher_is_better=True)

        re_wo = [float(c.result_without_brain.repeated_explanations_needed) for c in comparisons]
        re_w = [float(c.result_with_brain.repeated_explanations_needed) for c in comparisons]
        re_summary = cls.analyze_paired_metric("Repeated Explanations Needed (Lower is Better)", re_wo, re_w, higher_is_better=False)

        tok_wo = [float(c.result_without_brain.total_tokens) for c in comparisons]
        tok_w = [float(c.result_with_brain.total_tokens) for c in comparisons]
        tok_summary = cls.analyze_paired_metric("Total Tokens Consumed per Query", tok_wo, tok_w, higher_is_better=False)

        dec_wo = [float(c.result_without_brain.decisions_maintained_count) for c in comparisons]
        dec_w = [float(c.result_with_brain.decisions_maintained_count) for c in comparisons]
        dec_summary = cls.analyze_paired_metric("Historical Decisions Maintained", dec_wo, dec_w, higher_is_better=True)

        # 5. Continuous Paired Analyses (Condition B vs C)
        succ_raw = [1.0 if (c.result_raw_history and c.result_raw_history.task_success) else 0.0 for c in comparisons]
        partial_raw = [c.result_raw_history.partial_success_score if c.result_raw_history else 0.0 for c in comparisons]
        tok_raw = [float(c.result_raw_history.total_tokens) if c.result_raw_history else 0.0 for c in comparisons]

        raw_task_summary = cls.analyze_paired_metric("Raw History vs Brain: Task Success", succ_raw, succ_w, higher_is_better=True)
        raw_partial_summary = cls.analyze_paired_metric("Raw History vs Brain: Partial Score", partial_raw, partial_w, higher_is_better=True)
        raw_tok_summary = cls.analyze_paired_metric("Raw History vs Brain: Tokens Consumed", tok_raw, tok_w, higher_is_better=False)

        return {
            "mcnemar_cold_vs_raw": mcnemar_ab,
            "mcnemar_test": mcnemar_ac,
            "mcnemar_raw_vs_brain": mcnemar_bc,
            "cluster_task_success": cluster_task_success,
            "cluster_partial_success": cluster_partial,
            "cluster_tokens": cluster_tokens,
            "cluster_repeated_explanations": cluster_explanations,
            "cluster_decisions_maintained": cluster_decisions,
            "category_summaries": {k: v.to_dict() for k, v in cat_summaries.items()},
            "task_success": task_summary,
            "partial_success": partial_summary,
            "repeated_explanations": re_summary,
            "total_tokens": tok_summary,
            "decisions_maintained": dec_summary,
            "raw_task_success": raw_task_summary,
            "raw_partial_success": raw_partial_summary,
            "raw_total_tokens": raw_tok_summary,
        }
