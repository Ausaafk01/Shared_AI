"""
Unit and integration tests for AI Brain Phase 4: Evaluation & Benchmarking Subsystem.
Verifies three-condition execution (Cold Start, Raw History, AI Brain),
all four retrieval strategies (Vector, Graph, Hybrid, ML-Enhanced),
exact McNemar's test, cluster-aware category analysis, and corrected conflict confusion matrix.
"""

from datetime import datetime, timezone
import json
import math
import os
import tempfile
import unittest

from ai_brain.benchmarks.dataset import BenchmarkDataset
from ai_brain.benchmarks.evaluator import BenchmarkEvaluator
from ai_brain.benchmarks.models import (
    BenchmarkCondition,
    BenchmarkScenario,
    ConflictGroundTruth,
    ScenarioCategory,
    ScenarioDifficulty,
    ScenarioProvenance,
    ScenarioTokenAudit,
)
from ai_brain.benchmarks.reporter import BenchmarkReporter
from ai_brain.benchmarks.runner import BenchmarkRunner, estimate_tokens
from ai_brain.benchmarks.statistics import (
    BenchmarkStatistics,
    ClusterAwareStatisticalSummary,
    McNemarTestResult,
)
from ai_brain.context.tokenizer import DeterministicTokenizer
from ai_brain.events.models import Event, EventType


class TestBenchmarkSubsystem(unittest.TestCase):
    """Verifies all components of the Phase 4 Benchmarking Framework."""

    def setUp(self) -> None:
        self.dataset = BenchmarkDataset()
        self.runner = BenchmarkRunner(token_budget=1500)
        self.now = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)

    def test_benchmark_scenario_loading_and_categories(self) -> None:
        """Benchmark dataset loads 30 multi-session scenarios (3 per category across 10 categories)."""
        scenarios = self.dataset.list_scenarios()
        self.assertEqual(len(scenarios), 30)

        # Check all 10 required categories are present
        found_categories = {s.category for s in scenarios}
        expected_categories = set(ScenarioCategory)
        self.assertEqual(found_categories, expected_categories)

        # Check exactly 3 scenarios per category
        for cat in ScenarioCategory:
            cat_scenarios = self.dataset.list_scenarios(category=cat)
            self.assertEqual(len(cat_scenarios), 3, f"Category {cat.value} should have exactly 3 scenarios")

        # Check provenance, difficulty, and ground truth conflict labels
        for s in scenarios:
            self.assertIn(s.provenance, set(ScenarioProvenance))
            self.assertIn(s.difficulty, set(ScenarioDifficulty))
            self.assertIn(s.conflict_ground_truth, set(ConflictGroundTruth))
            self.assertTrue(s.task_prompt)
            self.assertTrue(s.project_id)

    def test_dataset_filtering_by_category_and_difficulty(self) -> None:
        """Dataset filters correctly by category, difficulty, and provenance."""
        hard_scenarios = self.dataset.list_scenarios(difficulty=ScenarioDifficulty.HARD)
        self.assertGreaterEqual(len(hard_scenarios), 3)
        for s in hard_scenarios:
            self.assertEqual(s.difficulty, ScenarioDifficulty.HARD)

        cont_scenarios = self.dataset.list_scenarios(category=ScenarioCategory.TASK_CONTINUATION)
        self.assertEqual(len(cont_scenarios), 3)

        # Both full ID and alias lookup work
        s1a = self.dataset.get_scenario("SCENARIO_01A_TASK_CONTINUATION")
        s1_alias = self.dataset.get_scenario("SCENARIO_01_TASK_CONTINUATION")
        self.assertIsNotNone(s1a)
        self.assertEqual(s1a, s1_alias)

    def test_baseline_and_brain_isolation(self) -> None:
        """Condition A, Condition B, and Condition C operate in strict isolation."""
        scenario = self.dataset.get_scenario("SCENARIO_02A_ARCHITECTURAL_DECISION")
        self.assertIsNotNone(scenario)

        res_a = self.runner.run_condition_a(scenario)
        res_b = self.runner.run_condition_b_raw_history(scenario)
        res_c = self.runner.run_condition_c_brain(scenario)

        # Condition A: Cold Start has no memory
        self.assertEqual(res_a.condition, BenchmarkCondition.COLD_START)
        self.assertEqual(res_a.retrieved_memory_tokens, 0)
        self.assertEqual(len(res_a.retrieved_memory_ids), 0)
        self.assertFalse(res_a.task_success)
        self.assertGreater(res_a.tokens.available_history_tokens, 0)

        # Condition B: Raw History receives raw events
        self.assertEqual(res_b.condition, BenchmarkCondition.RAW_HISTORY)
        self.assertGreater(res_b.tokens.supplied_history_tokens, 0)

        # Condition C: AI Brain retrieved Redis Sentinel memory
        self.assertEqual(res_c.condition, BenchmarkCondition.AI_BRAIN)
        self.assertGreater(res_c.retrieved_memory_tokens, 0)
        self.assertGreater(len(res_c.retrieved_memory_ids), 0)
        self.assertTrue(res_c.task_success)
        self.assertIn("redis", res_c.agent_output.lower())
        self.assertGreater(res_c.partial_success_score, res_a.partial_success_score)

    def test_deterministic_token_accounting_audit(self) -> None:
        """Token accounting uses DeterministicTokenizer and tracks all nested fields."""
        tokenizer = DeterministicTokenizer()
        sample = "We decided to adopt PostgreSQL 16 on port 5432."
        tokens = estimate_tokens(sample)
        self.assertEqual(tokens, tokenizer.count_tokens(sample))
        self.assertGreater(tokens, 5)

        scenario = self.dataset.get_scenario("SCENARIO_01A_TASK_CONTINUATION")
        comp = self.runner.run_scenario(scenario)

        # Audit Condition A tokens
        tok_a = comp.result_without_brain.tokens
        self.assertIsInstance(tok_a, ScenarioTokenAudit)
        self.assertIn("~4 chars/token approximation", tok_a.tokenizer_name)
        self.assertEqual(tok_a.total_tokens, tok_a.agent_input_tokens + tok_a.agent_output_tokens)
        self.assertEqual(tok_a.agent_input_tokens, tok_a.task_prompt_tokens)
        self.assertEqual(tok_a.retrieved_memory_tokens, 0)

        # Audit Condition C tokens
        tok_c = comp.result_with_brain.tokens
        self.assertIsInstance(tok_c, ScenarioTokenAudit)
        self.assertEqual(tok_c.total_tokens, tok_c.agent_input_tokens + tok_c.agent_output_tokens)
        self.assertEqual(tok_c.agent_input_tokens, tok_c.task_prompt_tokens + tok_c.context_builder_tokens)
        self.assertGreater(tok_c.retrieved_memory_tokens, 0)

    def test_retrieval_strategy_calculations_at_k5(self) -> None:
        """Evaluator computes Precision@5, Recall@5, F1, MRR, and NDCG@5 across all 4 strategies."""
        scenario = self.dataset.get_scenario("SCENARIO_03A_TECHNICAL_CONSTRAINTS")
        comparison = self.runner.run_scenario(scenario)
        aggregate = BenchmarkEvaluator.evaluate([comparison])

        # Overall Hybrid retrieval
        self.assertGreaterEqual(aggregate.retrieval_overall.precision_at_k, 0.5)
        self.assertGreaterEqual(aggregate.retrieval_overall.recall_at_k, 0.5)
        self.assertGreaterEqual(aggregate.retrieval_overall.f1_score, 0.5)
        self.assertEqual(aggregate.retrieval_overall.mrr, 1.0)
        self.assertGreaterEqual(aggregate.retrieval_overall.ndcg_at_k, 0.9)

        # Check all four strategies are present
        by_strat = aggregate.retrieval_by_strategy
        self.assertIn("Vector (Cosine)", by_strat)
        self.assertIn("Knowledge Graph (Traversal)", by_strat)
        self.assertIn("Hybrid (RRF Fusion)", by_strat)
        self.assertIn("ML-Enhanced (Reranked)", by_strat)

    def test_task_continuation_and_partial_credit_scoring(self) -> None:
        """Condition C avoids repeated explanations and achieves higher partial and binary success."""
        scenario = self.dataset.get_scenario("SCENARIO_08A_REPEATED_EXPLANATION")
        comparison = self.runner.run_scenario(scenario)

        # In Condition A, agent had to be told all webhook secrets/APIs
        self.assertGreater(comparison.result_without_brain.repeated_explanations_needed, 0)
        # In Condition C, explanations needed is 0
        self.assertEqual(comparison.result_with_brain.repeated_explanations_needed, 0)
        self.assertGreater(comparison.repeated_explanations_avoided, 0)
        self.assertEqual(comparison.task_continuation_delta, 1)

        # Partial credit score check
        self.assertGreater(comparison.result_with_brain.partial_success_score, 0.7)
        self.assertGreater(comparison.partial_score_delta, 0.0)

    def test_conflict_and_disputed_information_preservation(self) -> None:
        """Conflict ground truth distinguishes genuine contradictions from compatible claims."""
        # 1. Genuine Contradiction (Scenario 05A) -> True Positive
        sc_contra = self.dataset.get_scenario("SCENARIO_05A_CONTRADICTORY_CLAIMS")
        self.assertEqual(sc_contra.conflict_ground_truth, ConflictGroundTruth.GENUINE_CONTRADICTION)
        comp_contra = self.runner.run_scenario(sc_contra)
        self.assertTrue(comp_contra.result_with_brain.conflict_detected)
        self.assertTrue(comp_contra.result_with_brain.conflict_handled_correctly)
        self.assertIn("dispute", comp_contra.result_with_brain.agent_output.lower())

        # 2. Compatible Claims (Scenario 05C) -> True Negative (Zero False Alarm)
        sc_comp = self.dataset.get_scenario("SCENARIO_05C_CONTRADICTORY_CLAIMS")
        self.assertEqual(sc_comp.conflict_ground_truth, ConflictGroundTruth.COMPATIBLE_CLAIMS)
        comp_compat = self.runner.run_scenario(sc_comp)
        self.assertFalse(comp_compat.result_with_brain.conflict_detected)
        self.assertTrue(comp_compat.result_with_brain.conflict_handled_correctly)

        # Evaluate aggregate confusion matrix
        aggregate = BenchmarkEvaluator.evaluate([comp_contra, comp_compat])
        self.assertEqual(aggregate.conflict_true_positives, 1)
        self.assertEqual(aggregate.conflict_false_positives, 0)
        self.assertEqual(aggregate.conflict_true_negatives, 1)
        self.assertEqual(aggregate.conflict_false_negatives, 0)
        self.assertEqual(aggregate.conflict_precision, 1.0)
        self.assertEqual(aggregate.conflict_recall, 1.0)

    def test_stale_knowledge_rejection(self) -> None:
        """Condition C supersedes outdated information and rejects stale context."""
        scenario = self.dataset.get_scenario("SCENARIO_10A_STALE_KNOWLEDGE_REJECTION")
        self.assertEqual(scenario.conflict_ground_truth, ConflictGroundTruth.TEMPORAL_UPDATE)
        comparison = self.runner.run_scenario(scenario)

        self.assertEqual(comparison.result_with_brain.stale_memories_used_count, 0)
        self.assertTrue(comparison.result_with_brain.task_success)
        self.assertIn("ed25519", comparison.result_with_brain.agent_output.lower())

    def test_needle_in_haystack_noise_resistance(self) -> None:
        """Condition C retrieves the target fact despite 15 background distractor events."""
        scenario = self.dataset.get_scenario("SCENARIO_09A_NEEDLE_IN_HAYSTACK")
        self.assertEqual(len(scenario.distractor_events), 15)

        comparison = self.runner.run_scenario(scenario)
        self.assertTrue(comparison.result_with_brain.task_success)
        self.assertIn("42", comparison.result_with_brain.agent_output)

    def test_exact_mcnemar_and_statistical_analysis(self) -> None:
        """Statistical module computes exact McNemar's tests and cluster-stratified analysis."""
        scenarios = self.dataset.list_scenarios()[:12]
        comparisons = [self.runner.run_scenario(s) for s in scenarios]

        stats = BenchmarkStatistics.compute_all_comparisons(comparisons)
        self.assertIn("mcnemar_test", stats)
        self.assertIn("cluster_task_success", stats)
        self.assertIn("category_summaries", stats)
        self.assertIn("task_success", stats)
        self.assertIn("partial_success", stats)

        mcnemar = stats["mcnemar_test"]
        self.assertIsInstance(mcnemar, McNemarTestResult)
        self.assertEqual(mcnemar.n_total, 12)
        self.assertIn("both_pass", mcnemar.contingency_table)
        self.assertIn("only_second_pass", mcnemar.contingency_table)
        self.assertGreater(mcnemar.b_discordant_second, 0)
        self.assertEqual(mcnemar.c_discordant_first, 0)
        self.assertLess(mcnemar.exact_p_value, 0.05)
        self.assertTrue(mcnemar.is_statistically_significant)

        # Cluster-stratified analysis
        cluster_stat = stats["cluster_task_success"]
        self.assertIsInstance(cluster_stat, ClusterAwareStatisticalSummary)
        self.assertGreaterEqual(cluster_stat.n_clusters, 4)
        self.assertGreater(cluster_stat.t_statistic, 0.0)

    def test_benchmark_reproducibility(self) -> None:
        """Running the same scenario multiple times yields identical deterministic results."""
        scenario = self.dataset.get_scenario("SCENARIO_01A_TASK_CONTINUATION")
        res1 = self.runner.run_scenario(scenario)
        res2 = self.runner.run_scenario(scenario)

        self.assertEqual(res1.result_without_brain.total_tokens, res2.result_without_brain.total_tokens)
        self.assertEqual(res1.result_with_brain.total_tokens, res2.result_with_brain.total_tokens)
        self.assertEqual(res1.task_continuation_delta, res2.task_continuation_delta)
        self.assertEqual(res1.repeated_explanations_avoided, res2.repeated_explanations_avoided)
        self.assertEqual(res1.partial_score_delta, res2.partial_score_delta)

    def test_export_and_report_generation(self) -> None:
        """Reporter generates valid JSON and complete Markdown containing all required sections."""
        scenarios = [
            self.dataset.get_scenario("SCENARIO_01A_TASK_CONTINUATION"),
            self.dataset.get_scenario("SCENARIO_02A_ARCHITECTURAL_DECISION"),
            self.dataset.get_scenario("SCENARIO_03A_TECHNICAL_CONSTRAINTS"),
        ]
        comparisons = [self.runner.run_scenario(s) for s in scenarios]
        aggregate = BenchmarkEvaluator.evaluate(comparisons)
        stats = BenchmarkStatistics.compute_all_comparisons(comparisons)

        # JSON report
        json_str = BenchmarkReporter.generate_json_report(aggregate, comparisons, stats)
        data = json.loads(json_str)
        self.assertEqual(data["total_scenarios"], 3)
        self.assertIn("aggregate_metrics", data)
        self.assertIn("statistical_analysis", data)

        # Markdown report
        md_report = BenchmarkReporter.generate_markdown_report(aggregate, comparisons, stats)
        self.assertIn("# AI Brain — Phase 4 System Benchmarking", md_report)
        self.assertIn("Experimental Design: Three-Condition Comparison", md_report)
        self.assertIn("Cluster-Stratified Category Analysis", md_report)
        self.assertIn("Complete Retrieval Strategy Evaluation", md_report)
        self.assertIn("Corrected Conflict Confusion Matrix", md_report)

    def test_token_arithmetic_definitions_reconciliation(self) -> None:
        """Verify strict token accounting arithmetic across Condition A, Condition B, and Condition C."""
        scenario = self.dataset.get_scenario("SCENARIO_02A_ARCHITECTURAL_DECISION")
        comp = self.runner.run_scenario(scenario)

        # Condition A: Cold Start
        tok_a = comp.result_without_brain.tokens
        self.assertEqual(tok_a.supplied_history_tokens, 0)
        self.assertEqual(tok_a.agent_input_tokens, tok_a.task_prompt_tokens)
        self.assertEqual(tok_a.total_tokens, tok_a.agent_input_tokens + tok_a.agent_output_tokens)

        # Condition B: Raw History
        tok_b = comp.result_raw_history.tokens
        self.assertEqual(tok_b.agent_input_tokens, tok_b.task_prompt_tokens + tok_b.supplied_history_tokens)
        self.assertEqual(tok_b.total_tokens, tok_b.agent_input_tokens + tok_b.agent_output_tokens)
        self.assertEqual(tok_b.available_history_tokens, tok_b.supplied_history_tokens + tok_b.omitted_history_tokens)

        # Condition C: AI Brain
        tok_c = comp.result_with_brain.tokens
        self.assertEqual(tok_c.agent_input_tokens, tok_c.task_prompt_tokens + tok_c.context_builder_tokens)
        self.assertEqual(tok_c.total_tokens, tok_c.agent_input_tokens + tok_c.agent_output_tokens)
        self.assertGreater(tok_c.context_builder_tokens, tok_c.retrieved_memory_tokens)

    def test_raw_history_oversized_archive_truncation(self) -> None:
        """Verify 1,500 token budget enforcement and FIFO truncation on an oversized archive."""
        from datetime import timedelta
        base_time = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc)
        distractors = []
        for i in range(50):
            distractors.append(Event(
                event_id=f"ev-dist-{i:03d}",
                event_type=EventType.AGENT_MESSAGE,
                timestamp=base_time + timedelta(minutes=i),
                session_id="sess-oversized",
                project_id="proj-oversized",
                actor="agent-test",
                agent_id="agent-test",
                output={"text": f"Diagnostic telemetry stream entry {i:02d}: Comprehensive system log traces with heavy metadata payload block number {i:02d}."},
            ))

        target_event = Event(
            event_id="ev-target-final",
            event_type=EventType.AGENT_MESSAGE,
            timestamp=base_time + timedelta(hours=5),
            session_id="sess-oversized",
            project_id="proj-oversized",
            actor="agent-test",
            agent_id="agent-test",
            output={"text": "Final active directive: The production cache port is set to 6390."},
        )

        sc = BenchmarkScenario(
            scenario_id="SCENARIO_OVERSIZED_TRUNCATION",
            name="Oversized Archive Truncation Test",
            category=ScenarioCategory.NEEDLE_IN_HAYSTACK,
            difficulty=ScenarioDifficulty.HARD,
            provenance=ScenarioProvenance.SYNTHETIC,
            description="Archive exceeding the 1,500 token limit to test FIFO discard",
            historical_events=(target_event,),
            distractor_events=tuple(distractors),
            task_prompt="What is the production cache port?",
            project_id="proj-oversized",
            expected_facts=("6390",),
            expected_decisions=(),
            expected_constraints=(),
            expected_conflicts=(),
        )

        res_b = self.runner.run_condition_b_raw_history(sc)

        # Budget ceiling enforced
        self.assertLessEqual(res_b.tokens.agent_input_tokens, 1500)
        self.assertGreater(res_b.tokens.available_history_tokens, 2000)
        self.assertGreater(res_b.tokens.omitted_history_tokens, 1000)
        # Arithmetic reconciliation
        self.assertEqual(
            res_b.tokens.available_history_tokens,
            res_b.tokens.supplied_history_tokens + res_b.tokens.omitted_history_tokens,
        )
        # Most recent tail retention: target event at the end is retained
        self.assertIn("6390", res_b.agent_output)

    def test_provenance_optimization_toggle(self) -> None:
        """Verify include_provenance_section=False reduces tokens while preserving all facts and decisions."""
        sc = self.dataset.get_scenario("SCENARIO_02A_ARCHITECTURAL_DECISION")
        res_with = self.runner.run_condition_c_brain(sc, include_provenance_section=True)
        res_without = self.runner.run_condition_c_brain(sc, include_provenance_section=False)

        # Tokens reduced significantly
        tok_with = res_with.tokens.context_builder_tokens
        tok_without = res_without.tokens.context_builder_tokens
        self.assertGreater(tok_with, tok_without)
        self.assertGreaterEqual(tok_with - tok_without, 100)

        # Semantic decision retained
        self.assertIn("redis", res_without.agent_output.lower())
        self.assertTrue(res_without.task_success)

    def test_authoritative_aggregation_and_mcnemar_contingency(self) -> None:
        """Verify scenario-level binary outcomes strictly agree with category tables and contingency sums."""
        scenarios = self.dataset.list_scenarios()[:15]
        comparisons = [self.runner.run_scenario(s) for s in scenarios]
        stats = BenchmarkStatistics.compute_all_comparisons(comparisons)

        m_ab = stats["mcnemar_cold_vs_raw"]
        m_ac = stats["mcnemar_test"]
        m_bc = stats["mcnemar_raw_vs_brain"]

        n = len(comparisons)
        # Verify N=15 for all contingency tables
        for m in (m_ab, m_ac, m_bc):
            ct = m.contingency_table
            self.assertEqual(ct["both_pass"] + ct["only_first_pass"] + ct["only_second_pass"] + ct["both_fail"], n)

        # Check category summary sums match aggregate sums
        cat_summaries = stats["category_summaries"]
        sum_c = sum(int(round(cs["success_rate_ai_brain"] * cs["n_scenarios"])) for cs in cat_summaries.values())
        actual_c = sum(1 for c in comparisons if c.result_with_brain.task_success)
        self.assertEqual(sum_c, actual_c)

    def test_strict_success_vs_partial_credit_audit(self) -> None:
        """Audit strict binary success vs partial credit across 05A, 05B, 05C, and 07A."""
        # 05A: Pure Contradiction
        sc_05a = self.dataset.get_scenario("SCENARIO_05A_CONTRADICTORY_CLAIMS")
        res_a_05a = self.runner.run_condition_a(sc_05a)
        res_b_05a = self.runner.run_condition_b_raw_history(sc_05a)
        res_c_05a = self.runner.run_condition_c_brain(sc_05a)

        self.assertFalse(res_a_05a.task_success)
        self.assertEqual(res_a_05a.partial_success_score, 0.0)  # Contradiction missed
        self.assertFalse(res_b_05a.task_success)
        self.assertEqual(res_b_05a.partial_success_score, 0.0)  # Contradiction missed in raw history
        self.assertTrue(res_c_05a.task_success)
        self.assertEqual(res_c_05a.partial_success_score, 1.0)  # Dispute detected & preserved

        # 05B: Pure Contradiction (extraction limitation)
        sc_05b = self.dataset.get_scenario("SCENARIO_05B_CONTRADICTORY_CLAIMS")
        res_a_05b = self.runner.run_condition_a(sc_05b)
        res_b_05b = self.runner.run_condition_b_raw_history(sc_05b)
        res_c_05b = self.runner.run_condition_c_brain(sc_05b)

        # Neither condition handles conflict -> MUST NOT receive 100% partial credit!
        self.assertFalse(res_b_05b.task_success)
        self.assertEqual(res_b_05b.partial_success_score, 0.0)
        self.assertFalse(res_c_05b.task_success)
        self.assertEqual(res_c_05b.partial_success_score, 0.0)

        # 05C: Compatible Claims (Continuous credit for 1/2 recovered facts)
        sc_05c = self.dataset.get_scenario("SCENARIO_05C_CONTRADICTORY_CLAIMS")
        res_c_05c = self.runner.run_condition_c_brain(sc_05c)
        self.assertFalse(res_c_05c.task_success)  # 50% fact recovery < 75% threshold
        self.assertAlmostEqual(res_c_05c.partial_success_score, 0.825, places=3)

        # 07A: Session Interruption (Constraint adhered, facts missing)
        sc_07a = self.dataset.get_scenario("SCENARIO_07A_SESSION_INTERRUPTION")
        res_c_07a = self.runner.run_condition_c_brain(sc_07a)
        self.assertFalse(res_c_07a.task_success)  # 0% fact recovery < 75% threshold
        self.assertAlmostEqual(res_c_07a.partial_success_score, 0.650, places=3)

    def test_cluster_stratified_statistical_calculations(self) -> None:
        """Verify cluster standard error, t-statistic, confidence intervals, exact t-critical (df=9), and underlying 10-cluster vectors."""
        scenarios = self.dataset.list_scenarios()
        comparisons = [self.runner.run_scenario(s) for s in scenarios]
        stats = BenchmarkStatistics.compute_all_comparisons(comparisons)

        # Verify exact Student's t critical value for df=9 at alpha=0.05
        exact_t_crit_df9 = BenchmarkStatistics._t_critical(9, 0.05)
        self.assertAlmostEqual(exact_t_crit_df9, 2.262157162798205, places=12)

        # 1. Task Success
        cluster_task = stats["cluster_task_success"]
        self.assertEqual(cluster_task.degrees_of_freedom, 9)
        self.assertEqual(len(cluster_task.category_differences), 10)
        self.assertAlmostEqual(cluster_task.cluster_mean_diff, 0.700000, places=5)
        self.assertAlmostEqual(cluster_task.cluster_std_error, 0.104822, places=5)
        self.assertAlmostEqual(cluster_task.t_statistic, 6.677987, places=4)
        self.assertAlmostEqual(cluster_task.p_value, 0.00009079, places=6)
        self.assertAlmostEqual(cluster_task.confidence_interval_95[0], 0.462876, places=5)
        self.assertAlmostEqual(cluster_task.confidence_interval_95[1], 0.937124, places=5)
        self.assertTrue(cluster_task.is_statistically_significant)

        # 2. Partial Credit Score
        cluster_partial = stats["cluster_partial_success"]
        self.assertEqual(len(cluster_partial.category_differences), 10)
        self.assertAlmostEqual(cluster_partial.cluster_mean_diff, 0.816667, places=5)
        self.assertAlmostEqual(cluster_partial.cluster_std_error, 0.089899, places=5)
        self.assertAlmostEqual(cluster_partial.t_statistic, 9.084291, places=4)
        self.assertAlmostEqual(cluster_partial.p_value, 0.00000791, places=7)
        self.assertAlmostEqual(cluster_partial.confidence_interval_95[0], 0.613302, places=5)
        self.assertAlmostEqual(cluster_partial.confidence_interval_95[1], 1.020032, places=5)

        # 3. Tokens Consumed Overhead (Direction: C - A)
        cluster_tok = stats["cluster_tokens"]
        self.assertEqual(len(cluster_tok.category_differences), 10)
        self.assertAlmostEqual(cluster_tok.cluster_mean_diff, 371.8333333333, places=6)
        self.assertAlmostEqual(cluster_tok.cluster_std_error, 101.1696930416, places=6)
        self.assertAlmostEqual(cluster_tok.t_statistic, 3.6753431008, places=6)
        self.assertAlmostEqual(cluster_tok.p_value, 0.0051122716, places=6)
        # Exact 95% CI: [142.9715875613, 600.6950791054]
        self.assertAlmostEqual(cluster_tok.confidence_interval_95[0], 142.9715875613, places=6)
        self.assertAlmostEqual(cluster_tok.confidence_interval_95[1], 600.6950791054, places=6)

        # Verify exact reproducibility directly from unrounded underlying category_differences
        unrounded_diffs = cluster_tok.category_differences
        calc_mean = sum(unrounded_diffs) / 10.0
        calc_var = sum((d - calc_mean) ** 2 for d in unrounded_diffs) / 9.0
        calc_s = math.sqrt(calc_var)
        calc_se = calc_s / math.sqrt(10.0)
        calc_ci = (calc_mean - exact_t_crit_df9 * calc_se, calc_mean + exact_t_crit_df9 * calc_se)
        self.assertAlmostEqual(calc_mean, cluster_tok.cluster_mean_diff, places=10)
        self.assertAlmostEqual(calc_s, 319.9266601914, places=6)
        self.assertAlmostEqual(calc_se, cluster_tok.cluster_std_error, places=10)
        self.assertAlmostEqual(calc_ci[0], cluster_tok.confidence_interval_95[0], places=10)
        self.assertAlmostEqual(calc_ci[1], cluster_tok.confidence_interval_95[1], places=10)

        # 4. Repeated Explanations (Direction: C - A, negative indicates reduction)
        cluster_exp = stats["cluster_repeated_explanations"]
        self.assertEqual(len(cluster_exp.category_differences), 10)
        self.assertAlmostEqual(cluster_exp.cluster_mean_diff, -3.133333, places=5)
        self.assertAlmostEqual(cluster_exp.cluster_std_error, 0.477131, places=5)
        self.assertAlmostEqual(cluster_exp.t_statistic, -6.567025, places=4)
        self.assertAlmostEqual(cluster_exp.p_value, 0.00010312, places=6)
        self.assertAlmostEqual(cluster_exp.confidence_interval_95[0], -4.212679, places=5)
        self.assertAlmostEqual(cluster_exp.confidence_interval_95[1], -2.053987, places=5)

        # 5. Decisions Maintained
        cluster_dec = stats["cluster_decisions_maintained"]
        self.assertEqual(len(cluster_dec.category_differences), 10)
        self.assertAlmostEqual(cluster_dec.cluster_mean_diff, 0.833333, places=5)
        self.assertAlmostEqual(cluster_dec.cluster_std_error, 0.142725, places=5)
        self.assertAlmostEqual(cluster_dec.t_statistic, 5.838742, places=4)
        self.assertAlmostEqual(cluster_dec.p_value, 0.00024720, places=6)
        self.assertAlmostEqual(cluster_dec.confidence_interval_95[0], 0.510467, places=5)
        self.assertAlmostEqual(cluster_dec.confidence_interval_95[1], 1.156199, places=5)

    def test_category_vectors_displayed_precision_and_json_reproducibility(self) -> None:
        """Verify full precision in JSON, 4-6 decimal places in reports, and vector reproducibility across all 5 metrics."""
        scenarios = self.dataset.list_scenarios()
        comparisons = [self.runner.run_scenario(s) for s in scenarios]
        stats = BenchmarkStatistics.compute_all_comparisons(comparisons)
        exact_t_crit = BenchmarkStatistics._t_critical(9, 0.05)

        cluster_metric_keys = [
            "cluster_tokens",
            "cluster_task_success",
            "cluster_partial_success",
            "cluster_repeated_explanations",
            "cluster_decisions_maintained",
        ]

        # 1. JSON full precision validation
        for k in cluster_metric_keys:
            cs = stats[k]
            d_dict = cs.to_dict()
            self.assertIn("category_differences", d_dict)
            self.assertIn("category_differences_exact", d_dict)
            self.assertEqual(len(d_dict["category_differences"]), 10)
            # Verify values are unrounded full-precision floats
            for orig, stored in zip(cs.category_differences, d_dict["category_differences"]):
                self.assertIsInstance(stored, float)
                self.assertEqual(orig, stored)
            # Verify numerical p-value metadata
            self.assertIn("p_value_numerically_evaluated", d_dict)
            self.assertEqual(d_dict["p_value_method"], "composite_simpsons_rule_numerical_integration")

        # 2. Mathematical reproducibility from 6-decimal rounded vectors
        for k in cluster_metric_keys:
            cs = stats[k]
            vec_6dec = [round(d, 6) for d in cs.category_differences]
            recalc_mean = sum(vec_6dec) / 10.0
            recalc_var = sum((d - recalc_mean) ** 2 for d in vec_6dec) / 9.0
            recalc_se = math.sqrt(recalc_var / 10.0)
            recalc_t = (recalc_mean / recalc_se) if recalc_se > 0 else 0.0
            recalc_ci = (recalc_mean - exact_t_crit * recalc_se, recalc_mean + exact_t_crit * recalc_se)

            # Assert reproduction within 1e-5 rounding tolerance
            self.assertAlmostEqual(recalc_mean, cs.cluster_mean_diff, places=5)
            self.assertAlmostEqual(recalc_se, cs.cluster_std_error, places=5)
            self.assertAlmostEqual(recalc_t, cs.t_statistic, places=4)
            self.assertAlmostEqual(recalc_ci[0], cs.confidence_interval_95[0], places=5)
            self.assertAlmostEqual(recalc_ci[1], cs.confidence_interval_95[1], places=5)

        # 3. Markdown report formatting and p-value phrasing validation
        aggregate = BenchmarkEvaluator.evaluate(comparisons)
        md_report = BenchmarkReporter.generate_markdown_report(aggregate, comparisons, stats)

        # Check p-value description
        self.assertIn("numerically evaluated two-tailed Student's t p-values", md_report)
        self.assertIn("Exact Critical t = 2.262157162798205", md_report)

        # Check repeating fractions formatted to 6 decimal places
        self.assertIn("+276.333333", md_report)
        self.assertIn("+0.666667", md_report)
        self.assertIn("+0.883333", md_report)
        self.assertIn("-3.333333", md_report)
        self.assertIn("+1.333333", md_report)

        # Check terminating values formatted to at least 4 decimal places
        self.assertIn("+361.0000", md_report)
        self.assertIn("+1.0000", md_report)
        self.assertIn("+0.2000", md_report)
        self.assertIn("-4.0000", md_report)

        # Check that Displayed Ten-Cluster Paired Difference Vectors subsection exists
        self.assertIn("### Displayed Ten-Cluster Paired Difference Vectors", md_report)
        self.assertIn("Token Overhead ($C - A$)", md_report)
        self.assertIn("Task Continuation Success ($C - A$)", md_report)
        self.assertIn("Partial Credit Score ($C - A$)", md_report)
        self.assertIn("Repeated Explanations Needed ($C - A$)", md_report)
        self.assertIn("Decisions Maintained ($C - A$)", md_report)

    def test_zero_denominator_conflict_precision_convention(self) -> None:
        """Verify that zero predicted positive conflicts yields precision 0.0 by documented convention."""
        # Create dummy comparison with no positive predictions
        sc_compat = self.dataset.get_scenario("SCENARIO_05C_CONTRADICTORY_CLAIMS")
        comp = self.runner.run_scenario(sc_compat)
        # Verify comp has no predicted conflict
        self.assertFalse(comp.result_with_brain.conflict_detected)

        aggregate = BenchmarkEvaluator.evaluate([comp])
        # TP = 0, FP = 0 -> zero positive predictions
        self.assertEqual(aggregate.conflict_true_positives, 0)
        self.assertEqual(aggregate.conflict_false_positives, 0)
        self.assertEqual(aggregate.conflict_precision, 0.0)



if __name__ == "__main__":
    unittest.main()
