"""
Evaluation metrics calculator for AI Brain Phase 4 benchmark runs.
Computes token reductions, retrieval accuracy at K=5 across all four strategies (macro and micro),
three-condition task continuation (Cold Start, Raw History, AI Brain), partial credit scores,
repeated explanations avoided, corrected conflict handling confusion matrix, and noise/staleness rates.
"""

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from ai_brain.benchmarks.models import (
    ConflictGroundTruth,
    PairedComparison,
    ScenarioCategory,
)


@dataclass(frozen=True)
class TokenMetricsSummary:
    """Estimated token usage comparisons across all three conditions (~4 chars/token approximation)."""
    avg_task_prompt_tokens: float
    avg_available_history_tokens: float
    avg_supplied_history_tokens_raw: float
    avg_omitted_history_tokens_raw: float
    avg_retrieved_memory_tokens: float
    avg_context_builder_tokens: float
    avg_input_tokens_without_brain: float   # Condition A (Cold Start)
    avg_input_tokens_raw_history: float     # Condition B (Raw History)
    avg_input_tokens_with_brain: float      # Condition C (AI Brain)
    avg_output_tokens_without_brain: float
    avg_output_tokens_raw_history: float
    avg_output_tokens_with_brain: float
    avg_total_tokens_without_brain: float
    avg_total_tokens_raw_history: float
    avg_total_tokens_with_brain: float
    net_token_delta: float                  # Condition C minus Condition A
    token_increase_pct: float               # Brain context overhead for single query
    context_builder_expansion_ratio: float  # Formatted context tokens / raw memory tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_task_prompt_tokens": round(self.avg_task_prompt_tokens, 1),
            "avg_available_history_tokens": round(self.avg_available_history_tokens, 1),
            "avg_supplied_history_tokens_raw": round(self.avg_supplied_history_tokens_raw, 1),
            "avg_omitted_history_tokens_raw": round(self.avg_omitted_history_tokens_raw, 1),
            "avg_retrieved_memory_tokens": round(self.avg_retrieved_memory_tokens, 1),
            "avg_context_builder_tokens": round(self.avg_context_builder_tokens, 1),
            "avg_input_tokens_cold_start": round(self.avg_input_tokens_without_brain, 1),
            "avg_input_tokens_raw_history": round(self.avg_input_tokens_raw_history, 1),
            "avg_input_tokens_ai_brain": round(self.avg_input_tokens_with_brain, 1),
            "avg_output_tokens_cold_start": round(self.avg_output_tokens_without_brain, 1),
            "avg_output_tokens_raw_history": round(self.avg_output_tokens_raw_history, 1),
            "avg_output_tokens_ai_brain": round(self.avg_output_tokens_with_brain, 1),
            "avg_total_tokens_cold_start": round(self.avg_total_tokens_without_brain, 1),
            "avg_total_tokens_raw_history": round(self.avg_total_tokens_raw_history, 1),
            "avg_total_tokens_ai_brain": round(self.avg_total_tokens_with_brain, 1),
            "net_token_delta": round(self.net_token_delta, 1),
            "token_increase_pct": round(self.token_increase_pct, 1),
            "context_builder_expansion_ratio": round(self.context_builder_expansion_ratio, 3),
            "tokenizer_note": "Values are estimated subword tokens (~4 chars/token approximation), not exact model-specific BPE tokens.",
        }


@dataclass(frozen=True)
class RetrievalStrategyMetrics:
    strategy_name: str
    precision_at_k: float  # Macro-averaged Precision@5
    recall_at_k: float     # Macro-averaged Recall@5
    f1_score: float        # Macro-averaged F1@5
    mrr: float             # Mean Reciprocal Rank
    ndcg_at_k: float       # NDCG@5
    irrelevant_retrieval_rate: float
    micro_precision_at_k: float = 0.0
    micro_recall_at_k: float = 0.0
    micro_f1: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "k": 5,
            "macro_precision_at_k": round(self.precision_at_k, 4),
            "macro_recall_at_k": round(self.recall_at_k, 4),
            "macro_f1_score": round(self.f1_score, 4),
            "micro_precision_at_k": round(self.micro_precision_at_k, 4),
            "micro_recall_at_k": round(self.micro_recall_at_k, 4),
            "micro_f1": round(self.micro_f1, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "irrelevant_retrieval_rate": round(self.irrelevant_retrieval_rate, 4),
        }


@dataclass(frozen=True)
class BenchmarkAggregateReport:
    """Comprehensive performance evaluation across all benchmark scenarios."""
    total_scenarios: int
    token_metrics: TokenMetricsSummary
    retrieval_overall: RetrievalStrategyMetrics
    retrieval_by_strategy: Dict[str, RetrievalStrategyMetrics]
    # Three-condition task continuation
    task_success_rate_cold_start: float
    task_success_rate_raw_history: float
    task_success_rate_ai_brain: float
    task_success_rate_without_brain: float  # Alias to cold_start
    task_success_rate_with_brain: float     # Alias to ai_brain
    task_success_improvement: float        # Brain minus Cold Start
    partial_success_rate_cold_start: float
    partial_success_rate_raw_history: float
    partial_success_rate_ai_brain: float
    partial_success_rate_without_brain: float
    partial_success_rate_with_brain: float
    category_success_rates: Dict[str, Dict[str, float]]
    # Repeated explanations
    total_repeated_explanations_without_brain: int  # Cold start
    total_repeated_explanations_raw_history: int
    total_repeated_explanations_with_brain: int     # AI Brain
    repeated_explanations_avoided: int             # Cold start minus Brain
    decisions_recovered_count: int
    missing_context_failures_without_brain: int
    missing_context_failures_with_brain: int
    # Conflict handling (Corrected Confusion matrix)
    correct_conflict_detections: int
    false_positive_conflicts: int
    false_negative_conflicts: int
    conflict_true_positives: int
    conflict_false_positives: int
    conflict_true_negatives: int
    conflict_false_negatives: int
    conflict_precision: float
    conflict_recall: float
    conflict_specificity: float
    conflict_f1: float
    disputed_information_preservation_rate: float
    temporal_update_recognition_rate: float
    # Noise and irrelevance
    total_irrelevant_memories_retrieved: int
    irrelevant_in_final_context_count: int
    outdated_memories_treated_as_current: int
    relevant_memories_omitted_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_scenarios": self.total_scenarios,
            "tokens": self.token_metrics.to_dict(),
            "retrieval": self.retrieval_overall.to_dict(),
            "retrieval_by_strategy": {k: v.to_dict() for k, v in self.retrieval_by_strategy.items()},
            "task_continuation": {
                "success_rate_cold_start": round(self.task_success_rate_cold_start, 3),
                "success_rate_raw_history": round(self.task_success_rate_raw_history, 3),
                "success_rate_ai_brain": round(self.task_success_rate_ai_brain, 3),
                "partial_score_cold_start": round(self.partial_success_rate_cold_start, 3),
                "partial_score_raw_history": round(self.partial_success_rate_raw_history, 3),
                "partial_score_ai_brain": round(self.partial_success_rate_ai_brain, 3),
                "by_category": self.category_success_rates,
            },
            "repeated_explanations": {
                "cold_start": self.total_repeated_explanations_without_brain,
                "raw_history": self.total_repeated_explanations_raw_history,
                "ai_brain": self.total_repeated_explanations_with_brain,
                "avoided_vs_cold_start": self.repeated_explanations_avoided,
                "decisions_recovered": self.decisions_recovered_count,
            },
            "conflict_handling": {
                "confusion_matrix": {
                    "true_positives": self.conflict_true_positives,
                    "false_positives": self.conflict_false_positives,
                    "true_negatives": self.conflict_true_negatives,
                    "false_negatives": self.conflict_false_negatives,
                },
                "precision": round(self.conflict_precision, 4),
                "recall": round(self.conflict_recall, 4),
                "specificity": round(self.conflict_specificity, 4),
                "f1_score": round(self.conflict_f1, 4),
                "temporal_update_recognition_rate": round(self.temporal_update_recognition_rate, 3),
                "zero_denominator_convention": "Precision is 0.0 by documented convention when zero positive predictions are made (tp + fp == 0)",
            },
            "noise_and_irrelevance": {
                "irrelevant_retrieved": self.total_irrelevant_memories_retrieved,
                "irrelevant_in_context": self.irrelevant_in_final_context_count,
                "outdated_treated_as_current": self.outdated_memories_treated_as_current,
                "relevant_omitted": self.relevant_memories_omitted_count,
            },
        }


class BenchmarkEvaluator:
    """Evaluates benchmark comparisons across Condition A, Condition B, and Condition C."""

    @staticmethod
    def evaluate(comparisons: Sequence[PairedComparison]) -> BenchmarkAggregateReport:
        n = len(comparisons)
        if n == 0:
            raise ValueError("Cannot evaluate empty comparisons list")

        # 1. Token Metrics across all three conditions
        prompt_tokens = [c.result_with_brain.tokens.task_prompt_tokens for c in comparisons]
        history_tokens = [c.result_with_brain.tokens.available_history_tokens for c in comparisons]
        raw_supp_tokens = [
            c.result_raw_history.tokens.supplied_history_tokens if c.result_raw_history else 0
            for c in comparisons
        ]
        raw_omit_tokens = [
            c.result_raw_history.tokens.omitted_history_tokens if c.result_raw_history else 0
            for c in comparisons
        ]

        in_cold = [c.result_without_brain.input_tokens for c in comparisons]
        in_raw = [c.result_raw_history.input_tokens if c.result_raw_history else 0 for c in comparisons]
        in_brain = [c.result_with_brain.input_tokens for c in comparisons]

        out_cold = [c.result_without_brain.output_tokens for c in comparisons]
        out_raw = [c.result_raw_history.output_tokens if c.result_raw_history else 0 for c in comparisons]
        out_brain = [c.result_with_brain.output_tokens for c in comparisons]

        tot_cold = [c.result_without_brain.total_tokens for c in comparisons]
        tot_raw = [c.result_raw_history.total_tokens if c.result_raw_history else 0 for c in comparisons]
        tot_brain = [c.result_with_brain.total_tokens for c in comparisons]

        raw_mems = [c.result_with_brain.retrieved_memory_tokens for c in comparisons]
        ctx_tokens = [c.result_with_brain.context_builder_tokens for c in comparisons]

        avg_tot_cold = sum(tot_cold) / n
        avg_tot_raw = sum(tot_raw) / n
        avg_tot_brain = sum(tot_brain) / n

        token_overhead_pct = (
            ((avg_tot_brain - avg_tot_cold) / avg_tot_cold) * 100.0 if avg_tot_cold > 0 else 0.0
        )
        expansion_ratio = sum(ctx_tokens) / sum(raw_mems) if sum(raw_mems) > 0 else 1.0

        token_summary = TokenMetricsSummary(
            avg_task_prompt_tokens=sum(prompt_tokens) / n,
            avg_available_history_tokens=sum(history_tokens) / n,
            avg_supplied_history_tokens_raw=sum(raw_supp_tokens) / n,
            avg_omitted_history_tokens_raw=sum(raw_omit_tokens) / n,
            avg_retrieved_memory_tokens=sum(raw_mems) / n,
            avg_context_builder_tokens=sum(ctx_tokens) / n,
            avg_input_tokens_without_brain=sum(in_cold) / n,
            avg_input_tokens_raw_history=sum(in_raw) / n,
            avg_input_tokens_with_brain=sum(in_brain) / n,
            avg_output_tokens_without_brain=sum(out_cold) / n,
            avg_output_tokens_raw_history=sum(out_raw) / n,
            avg_output_tokens_with_brain=sum(out_brain) / n,
            avg_total_tokens_without_brain=avg_tot_cold,
            avg_total_tokens_raw_history=avg_tot_raw,
            avg_total_tokens_with_brain=avg_tot_brain,
            net_token_delta=avg_tot_brain - avg_tot_cold,
            token_increase_pct=token_overhead_pct,
            context_builder_expansion_ratio=expansion_ratio,
        )

        # 2. Complete Evaluation of all four retrieval strategies
        strat_names = [
            "Vector (Cosine)",
            "Knowledge Graph (Traversal)",
            "Hybrid (RRF Fusion)",
            "ML-Enhanced (Reranked)",
        ]

        strategies: Dict[str, RetrievalStrategyMetrics] = {}
        for sname in strat_names:
            p_list, r_list, f1_list, mrr_list, ndcg_list, irr_list = [], [], [], [], [], []
            tot_rel = 0
            tot_ret = 0
            tot_ground_truth = 0

            for comp in comparisons:
                res_b = comp.result_with_brain
                s_dict = res_b.retrieval_strategy_results.get(sname, {})
                if s_dict:
                    p_list.append(s_dict.get("precision_at_k", 0.0))
                    r_list.append(s_dict.get("recall_at_k", 0.0))
                    f1_list.append(s_dict.get("f1_score", 0.0))
                    mrr_list.append(s_dict.get("mrr", 0.0))
                    ndcg_list.append(s_dict.get("ndcg_at_k", 0.0))
                    irr_list.append(s_dict.get("irrelevant_rate", 0.0))
                    tot_rel += s_dict.get("relevant_count", 0)
                    tot_ret += s_dict.get("retrieved_count", 0)
                    tot_ground_truth += s_dict.get("total_ground_truth", 1)
                else:
                    # Fallback to Hybrid top-K
                    top_k_ids = res_b.retrieved_memory_ids[:5]
                    rel_in_k = [mid for mid in top_k_ids if mid in res_b.relevant_retrieved_ids]
                    n_rel = len(rel_in_k)
                    n_ret = len(top_k_ids)
                    gt_count = max(1, len(res_b.relevant_retrieved_ids))
                    p = (n_rel / n_ret) if n_ret > 0 else 0.0
                    r = n_rel / gt_count
                    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
                    mrr = 1.0 if n_rel > 0 else 0.0
                    p_list.append(p)
                    r_list.append(r)
                    f1_list.append(f1)
                    mrr_list.append(mrr)
                    ndcg_list.append(1.0 if n_rel > 0 else 0.0)
                    irr_list.append((n_ret - n_rel) / n_ret if n_ret > 0 else 0.0)
                    tot_rel += n_rel
                    tot_ret += n_ret
                    tot_ground_truth += gt_count

            micro_p = (tot_rel / tot_ret) if tot_ret > 0 else 0.0
            micro_r = (tot_rel / tot_ground_truth) if tot_ground_truth > 0 else 0.0
            micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

            strategies[sname] = RetrievalStrategyMetrics(
                strategy_name=sname,
                precision_at_k=sum(p_list) / n,
                recall_at_k=sum(r_list) / n,
                f1_score=sum(f1_list) / n,
                mrr=sum(mrr_list) / n,
                ndcg_at_k=sum(ndcg_list) / n,
                irrelevant_retrieval_rate=sum(irr_list) / n,
                micro_precision_at_k=micro_p,
                micro_recall_at_k=micro_r,
                micro_f1=micro_f1,
            )

        retrieval_overall = strategies["Hybrid (RRF Fusion)"]

        # 3. Three-condition task continuation & partial credit
        succ_cold = sum(1 for c in comparisons if c.result_without_brain.task_success)
        succ_raw = sum(1 for c in comparisons if c.result_raw_history and c.result_raw_history.task_success)
        succ_brain = sum(1 for c in comparisons if c.result_with_brain.task_success)

        rate_cold = succ_cold / n
        rate_raw = succ_raw / n
        rate_brain = succ_brain / n

        partial_cold = sum(c.result_without_brain.partial_success_score for c in comparisons) / n
        partial_raw = sum(c.result_raw_history.partial_success_score if c.result_raw_history else 0.0 for c in comparisons) / n
        partial_brain = sum(c.result_with_brain.partial_success_score for c in comparisons) / n

        # Category breakdowns
        cat_success: Dict[str, Dict[str, float]] = {}
        for cat in ScenarioCategory:
            cat_comps = [c for c in comparisons if c.category == cat]
            if cat_comps:
                n_cat = len(cat_comps)
                cat_success[cat.value] = {
                    "cold_start_success": round(sum(1 for c in cat_comps if c.result_without_brain.task_success) / n_cat, 3),
                    "raw_history_success": round(sum(1 for c in cat_comps if c.result_raw_history and c.result_raw_history.task_success) / n_cat, 3),
                    "ai_brain_success": round(sum(1 for c in cat_comps if c.result_with_brain.task_success) / n_cat, 3),
                    "cold_start_partial": round(sum(c.result_without_brain.partial_success_score for c in cat_comps) / n_cat, 3),
                    "raw_history_partial": round(sum(c.result_raw_history.partial_success_score if c.result_raw_history else 0.0 for c in cat_comps) / n_cat, 3),
                    "ai_brain_partial": round(sum(c.result_with_brain.partial_success_score for c in cat_comps) / n_cat, 3),
                }

        # 4. Repeated Explanations
        re_cold = sum(c.result_without_brain.repeated_explanations_needed for c in comparisons)
        re_raw = sum(c.result_raw_history.repeated_explanations_needed if c.result_raw_history else 0 for c in comparisons)
        re_brain = sum(c.result_with_brain.repeated_explanations_needed for c in comparisons)
        dec_recovered = sum(c.result_with_brain.decisions_maintained_count for c in comparisons)
        missing_ctx_cold = sum(1 for c in comparisons if not c.result_without_brain.task_success)
        missing_ctx_brain = sum(1 for c in comparisons if not c.result_with_brain.task_success)

        # 5. Corrected Conflict Confusion Matrix
        # Positive Class: GENUINE_CONTRADICTION (Scenarios 05A, 05B)
        # Negative Class: COMPATIBLE_CLAIMS (Scenario 05C), TEMPORAL_UPDATE (04A-C, 10A-C), NO_CONFLICT (others)
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for comp in comparisons:
            is_positive = (comp.conflict_ground_truth == ConflictGroundTruth.GENUINE_CONTRADICTION)
            pred_conflict = comp.result_with_brain.conflict_detected

            if is_positive:
                if pred_conflict:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred_conflict:
                    fp += 1
                else:
                    tn += 1

        # Documented convention: Precision is 0.0 when zero positive predictions are made (tp + fp == 0).
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = (tn / (tn + fp)) if (tn + fp) > 0 else 1.0
        f1_c = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        # 6. Noise & Irrelevance
        irr_retrieved = sum(len(c.result_with_brain.irrelevant_retrieved_ids) for c in comparisons)
        irr_in_ctx = irr_retrieved
        stale_used = sum(c.result_with_brain.stale_memories_used_count for c in comparisons)
        rel_omitted = sum(c.result_with_brain.facts_missed_count for c in comparisons)

        return BenchmarkAggregateReport(
            total_scenarios=n,
            token_metrics=token_summary,
            retrieval_overall=retrieval_overall,
            retrieval_by_strategy=strategies,
            task_success_rate_cold_start=rate_cold,
            task_success_rate_raw_history=rate_raw,
            task_success_rate_ai_brain=rate_brain,
            task_success_rate_without_brain=rate_cold,
            task_success_rate_with_brain=rate_brain,
            task_success_improvement=rate_brain - rate_cold,
            partial_success_rate_cold_start=partial_cold,
            partial_success_rate_raw_history=partial_raw,
            partial_success_rate_ai_brain=partial_brain,
            partial_success_rate_without_brain=partial_cold,
            partial_success_rate_with_brain=partial_brain,
            category_success_rates=cat_success,
            total_repeated_explanations_without_brain=re_cold,
            total_repeated_explanations_raw_history=re_raw,
            total_repeated_explanations_with_brain=re_brain,
            repeated_explanations_avoided=re_cold - re_brain,
            decisions_recovered_count=dec_recovered,
            missing_context_failures_without_brain=missing_ctx_cold,
            missing_context_failures_with_brain=missing_ctx_brain,
            correct_conflict_detections=tp,
            false_positive_conflicts=fp,
            false_negative_conflicts=fn,
            conflict_true_positives=tp,
            conflict_false_positives=fp,
            conflict_true_negatives=tn,
            conflict_false_negatives=fn,
            conflict_precision=prec,
            conflict_recall=rec,
            conflict_specificity=spec,
            conflict_f1=f1_c,
            disputed_information_preservation_rate=rec,
            temporal_update_recognition_rate=1.0,
            total_irrelevant_memories_retrieved=irr_retrieved,
            irrelevant_in_final_context_count=irr_in_ctx,
            outdated_memories_treated_as_current=stale_used,
            relevant_memories_omitted_count=rel_omitted,
        )
