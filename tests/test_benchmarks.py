"""
Unit and integration tests for AI Brain Phase 4: Evaluation & Benchmarking Subsystem.
Verifies scenario loading (30 scenarios across 10 categories), strict condition isolation,
exact subword token audit, McNemar's test, partial-credit scoring, and reporting.
"""

from datetime import datetime, timezone
import json
import os
import tempfile
import unittest

from ai_brain.benchmarks.dataset import BenchmarkDataset
from ai_brain.benchmarks.evaluator import BenchmarkEvaluator
from ai_brain.benchmarks.models import (
    BenchmarkCondition,
    BenchmarkScenario,
    ScenarioCategory,
    ScenarioDifficulty,
    ScenarioProvenance,
    ScenarioTokenAudit,
)
from ai_brain.benchmarks.reporter import BenchmarkReporter
from ai_brain.benchmarks.runner import BenchmarkRunner, estimate_tokens
from ai_brain.benchmarks.statistics import BenchmarkStatistics, McNemarTestResult
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

        # Check provenance and required fields
        for s in scenarios:
            self.assertIn(s.provenance, set(ScenarioProvenance))
            self.assertIn(s.difficulty, set(ScenarioDifficulty))
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
        """Condition A and Condition B operate in strict isolation without sharing state."""
        scenario = self.dataset.get_scenario("SCENARIO_02A_ARCHITECTURAL_DECISION")
        self.assertIsNotNone(scenario)

        res_a = self.runner.run_condition_a(scenario)
        res_b = self.runner.run_condition_b(scenario)

        # Condition A has no memory of Redis Sentinel
        self.assertEqual(res_a.condition, BenchmarkCondition.WITHOUT_BRAIN)
        self.assertEqual(res_a.retrieved_memory_tokens, 0)
        self.assertEqual(len(res_a.retrieved_memory_ids), 0)
        self.assertFalse(res_a.task_success)
        self.assertGreater(res_a.tokens.available_history_tokens, 0)

        # Condition B retrieved Redis Sentinel memory
        self.assertEqual(res_b.condition, BenchmarkCondition.WITH_BRAIN)
        self.assertGreater(res_b.retrieved_memory_tokens, 0)
        self.assertGreater(len(res_b.retrieved_memory_ids), 0)
        self.assertTrue(res_b.task_success)
        self.assertIn("redis", res_b.agent_output.lower())
        self.assertGreater(res_b.partial_success_score, res_a.partial_success_score)

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
        self.assertEqual(tok_a.tokenizer_name, "DeterministicTokenizer")
        self.assertEqual(tok_a.total_tokens, tok_a.agent_input_tokens + tok_a.agent_output_tokens)
        self.assertEqual(tok_a.agent_input_tokens, tok_a.task_prompt_tokens)
        self.assertEqual(tok_a.retrieved_memory_tokens, 0)

        # Audit Condition B tokens
        tok_b = comp.result_with_brain.tokens
        self.assertIsInstance(tok_b, ScenarioTokenAudit)
        self.assertEqual(tok_b.total_tokens, tok_b.agent_input_tokens + tok_b.agent_output_tokens)
        self.assertEqual(tok_b.agent_input_tokens, tok_b.task_prompt_tokens + tok_b.context_builder_tokens)
        self.assertGreater(tok_b.retrieved_memory_tokens, 0)

    def test_retrieval_metric_calculations_at_k5(self) -> None:
        """Evaluator correctly computes Precision@5, Recall@5, F1, MRR, and NDCG@5."""
        scenario = self.dataset.get_scenario("SCENARIO_03A_TECHNICAL_CONSTRAINTS")
        comparison = self.runner.run_scenario(scenario)
        aggregate = BenchmarkEvaluator.evaluate([comparison])

        self.assertGreaterEqual(aggregate.retrieval_overall.precision_at_k, 0.5)
        self.assertGreaterEqual(aggregate.retrieval_overall.recall_at_k, 0.5)
        self.assertGreaterEqual(aggregate.retrieval_overall.f1_score, 0.5)
        self.assertEqual(aggregate.retrieval_overall.mrr, 1.0)
        self.assertGreaterEqual(aggregate.retrieval_overall.ndcg_at_k, 0.9)

        # Check strategies breakdown
        by_strat = aggregate.retrieval_by_strategy
        self.assertIn("Vector (Cosine Alone)", by_strat)
        self.assertIn("Knowledge Graph (Traversal Alone)", by_strat)
        self.assertIn("Hybrid (RRF Fusion, Top-5)", by_strat)

    def test_task_continuation_and_partial_credit_scoring(self) -> None:
        """Condition B avoids repeated explanations and achieves higher partial and binary success."""
        scenario = self.dataset.get_scenario("SCENARIO_08A_REPEATED_EXPLANATION")
        comparison = self.runner.run_scenario(scenario)

        # In Condition A, agent had to be told all webhook secrets/APIs
        self.assertGreater(comparison.result_without_brain.repeated_explanations_needed, 0)
        # In Condition B, explanations needed is 0
        self.assertEqual(comparison.result_with_brain.repeated_explanations_needed, 0)
        self.assertGreater(comparison.repeated_explanations_avoided, 0)
        self.assertEqual(comparison.task_continuation_delta, 1)

        # Partial credit score check
        self.assertGreater(comparison.result_with_brain.partial_success_score, 0.7)
        self.assertGreater(comparison.partial_score_delta, 0.0)

    def test_conflict_and_disputed_information_preservation(self) -> None:
        """Conflict scenario preserves dispute status and tracks confusion matrix."""
        scenario = self.dataset.get_scenario("SCENARIO_05A_CONTRADICTORY_CLAIMS")
        comparison = self.runner.run_scenario(scenario)

        # Condition A doesn't know about the conflict
        self.assertFalse(comparison.result_without_brain.conflict_detected)

        # Condition B identifies the contradiction and flags dispute
        self.assertTrue(comparison.result_with_brain.conflict_detected)
        self.assertTrue(comparison.result_with_brain.conflict_handled_correctly)
        self.assertIn("dispute", comparison.result_with_brain.agent_output.lower())

        # Evaluate aggregate confusion matrix
        aggregate = BenchmarkEvaluator.evaluate([comparison])
        self.assertEqual(aggregate.conflict_true_positives, 1)
        self.assertEqual(aggregate.conflict_false_positives, 0)

    def test_stale_knowledge_rejection(self) -> None:
        """Condition B supersedes outdated information and rejects stale context."""
        scenario = self.dataset.get_scenario("SCENARIO_10A_STALE_KNOWLEDGE_REJECTION")
        comparison = self.runner.run_scenario(scenario)

        self.assertEqual(comparison.result_with_brain.stale_memories_used_count, 0)
        self.assertTrue(comparison.result_with_brain.task_success)
        self.assertIn("ed25519", comparison.result_with_brain.agent_output.lower())

    def test_needle_in_haystack_noise_resistance(self) -> None:
        """Condition B retrieves the target fact despite 15 background distractor events."""
        scenario = self.dataset.get_scenario("SCENARIO_09A_NEEDLE_IN_HAYSTACK")
        self.assertEqual(len(scenario.distractor_events), 15)

        comparison = self.runner.run_scenario(scenario)
        self.assertTrue(comparison.result_with_brain.task_success)
        self.assertIn("42", comparison.result_with_brain.agent_output)

    def test_exact_mcnemar_and_statistical_analysis(self) -> None:
        """Statistical module computes exact McNemar's test, paired t-tests, and Cohen's d."""
        scenarios = self.dataset.list_scenarios()[:8]
        comparisons = [self.runner.run_scenario(s) for s in scenarios]

        stats = BenchmarkStatistics.compute_all_comparisons(comparisons)
        self.assertIn("mcnemar_test", stats)
        self.assertIn("task_success", stats)
        self.assertIn("partial_success", stats)
        self.assertIn("repeated_explanations", stats)
        self.assertIn("total_tokens", stats)

        mcnemar = stats["mcnemar_test"]
        self.assertIsInstance(mcnemar, McNemarTestResult)
        self.assertEqual(mcnemar.n_total, 8)
        self.assertIn("both_pass", mcnemar.contingency_table)
        self.assertIn("only_with_brain_pass", mcnemar.contingency_table)
        self.assertGreater(mcnemar.b_discordant_with_brain, 0)
        self.assertEqual(mcnemar.c_discordant_without_brain, 0)
        self.assertLess(mcnemar.exact_p_value, 0.05)
        self.assertTrue(mcnemar.is_statistically_significant)

        # Cohen's d and t-statistic
        partial_stat = stats["partial_success"]
        self.assertEqual(partial_stat.n_samples, 8)
        self.assertGreater(partial_stat.cohens_d, 0.0)
        self.assertIsInstance(partial_stat.confidence_interval_95, tuple)

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
        scenarios = self.dataset.list_scenarios()[:3]
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
        self.assertIn("Experimental Design & Condition Reassessment", md_report)
        self.assertIn("Rigorous Paired Statistical Analysis", md_report)
        self.assertIn("Exact McNemar's Test", md_report)
        self.assertIn("Fine-Grained Token Accounting Audit", md_report)
        self.assertIn("Conflict Handling & Explanation Avoidance", md_report)
        self.assertIn("Scenario-Level Partial Credit Scoring Breakdown", md_report)
        self.assertIn("Answers to the Eight Core Evaluation Questions", md_report)


if __name__ == "__main__":
    unittest.main()
