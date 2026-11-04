"""
Unit and integration tests for AI Brain Phase 4: Evaluation & Benchmarking Subsystem.
Verifies scenario loading, isolation, metric calculations, token tracking, statistical analysis, and report generation.
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
)
from ai_brain.benchmarks.reporter import BenchmarkReporter
from ai_brain.benchmarks.runner import BenchmarkRunner, estimate_tokens
from ai_brain.benchmarks.statistics import BenchmarkStatistics
from ai_brain.events.models import Event, EventType


class TestBenchmarkSubsystem(unittest.TestCase):
    """Verifies all components of the Phase 4 Benchmarking Framework."""

    def setUp(self) -> None:
        self.dataset = BenchmarkDataset()
        self.runner = BenchmarkRunner(token_budget=1500)
        self.now = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)

    def test_benchmark_scenario_loading_and_categories(self) -> None:
        """Benchmark dataset loads all 10 standard multi-session scenario categories."""
        scenarios = self.dataset.list_scenarios()
        self.assertEqual(len(scenarios), 10)

        # Check all required categories are present
        found_categories = {s.category for s in scenarios}
        expected_categories = set(ScenarioCategory)
        self.assertEqual(found_categories, expected_categories)

        # Check provenance is explicitly documented
        for s in scenarios:
            self.assertIn(s.provenance, set(ScenarioProvenance))
            self.assertIn(s.difficulty, set(ScenarioDifficulty))
            self.assertTrue(s.task_prompt)
            self.assertTrue(s.project_id)

    def test_dataset_filtering_by_category_and_difficulty(self) -> None:
        """Dataset filters correctly by category, difficulty, and provenance."""
        hard_scenarios = self.dataset.list_scenarios(difficulty=ScenarioDifficulty.HARD)
        self.assertGreaterEqual(len(hard_scenarios), 2)
        for s in hard_scenarios:
            self.assertEqual(s.difficulty, ScenarioDifficulty.HARD)

        cont_scenarios = self.dataset.list_scenarios(category=ScenarioCategory.TASK_CONTINUATION)
        self.assertEqual(len(cont_scenarios), 1)
        self.assertEqual(cont_scenarios[0].scenario_id, "SCENARIO_01_TASK_CONTINUATION")

    def test_baseline_and_brain_isolation(self) -> None:
        """Condition A and Condition B operate in strict isolation without sharing state."""
        scenario = self.dataset.get_scenario("SCENARIO_02_ARCHITECTURAL_DECISION")
        self.assertIsNotNone(scenario)

        res_a = self.runner.run_condition_a(scenario)
        res_b = self.runner.run_condition_b(scenario)

        # Condition A has no memory of Redis Sentinel
        self.assertEqual(res_a.condition, BenchmarkCondition.WITHOUT_BRAIN)
        self.assertEqual(res_a.retrieved_memory_tokens, 0)
        self.assertEqual(len(res_a.retrieved_memory_ids), 0)
        self.assertFalse(res_a.task_success)

        # Condition B retrieved Redis Sentinel memory
        self.assertEqual(res_b.condition, BenchmarkCondition.WITH_BRAIN)
        self.assertGreater(res_b.retrieved_memory_tokens, 0)
        self.assertGreater(len(res_b.retrieved_memory_ids), 0)
        self.assertTrue(res_b.task_success)
        self.assertIn("redis", res_b.agent_output.lower())

    def test_token_measurement_consistency(self) -> None:
        """Token estimation is consistent, deterministic, and positive for non-empty text."""
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("   "), 0)
        
        sample = "We decided to adopt PostgreSQL 16 on port 5432."
        tokens = estimate_tokens(sample)
        self.assertGreater(tokens, 5)
        self.assertLess(tokens, 30)

        # Same input yields identical token count
        self.assertEqual(estimate_tokens(sample), tokens)

    def test_retrieval_metric_calculations(self) -> None:
        """Evaluator correctly computes Precision, Recall, F1, MRR, and NDCG."""
        scenario = self.dataset.get_scenario("SCENARIO_03_TECHNICAL_CONSTRAINTS")
        comparison = self.runner.run_scenario(scenario)
        aggregate = BenchmarkEvaluator.evaluate([comparison])

        self.assertGreaterEqual(aggregate.retrieval_overall.precision_at_k, 0.5)
        self.assertGreaterEqual(aggregate.retrieval_overall.recall_at_k, 0.5)
        self.assertGreaterEqual(aggregate.retrieval_overall.f1_score, 0.5)
        self.assertEqual(aggregate.retrieval_overall.mrr, 1.0)
        self.assertEqual(aggregate.retrieval_overall.ndcg_at_k, 1.0)

        # Check strategies breakdown
        by_strat = aggregate.retrieval_by_strategy
        self.assertIn("Vector (Cosine)", by_strat)
        self.assertIn("Knowledge Graph (Traversal)", by_strat)
        self.assertIn("Hybrid (RRF Fusion)", by_strat)
        self.assertIn("ML-Enhanced (Reranked)", by_strat)

    def test_task_continuation_and_explanation_avoidance(self) -> None:
        """Condition B avoids repeated explanations and achieves higher continuation success."""
        scenario = self.dataset.get_scenario("SCENARIO_08_REPEATED_EXPLANATION")
        comparison = self.runner.run_scenario(scenario)

        # In Condition A, agent had to be told all webhook secrets/APIs
        self.assertGreater(comparison.result_without_brain.repeated_explanations_needed, 0)
        # In Condition B, explanations needed is 0
        self.assertEqual(comparison.result_with_brain.repeated_explanations_needed, 0)
        # Repeated explanations avoided is positive
        self.assertGreater(comparison.repeated_explanations_avoided, 0)
        self.assertEqual(comparison.task_continuation_delta, 1)

    def test_conflict_and_disputed_information_preservation(self) -> None:
        """Conflict scenario preserves dispute status and emits warnings in Condition B."""
        scenario = self.dataset.get_scenario("SCENARIO_05_CONTRADICTORY_CLAIMS")
        comparison = self.runner.run_scenario(scenario)

        # Condition A doesn't know about the conflict
        self.assertFalse(comparison.result_without_brain.conflict_detected)

        # Condition B identifies the contradiction and flags dispute
        self.assertTrue(comparison.result_with_brain.conflict_detected)
        self.assertTrue(comparison.result_with_brain.conflict_handled_correctly)
        self.assertIn("dispute", comparison.result_with_brain.agent_output.lower())

    def test_stale_knowledge_rejection(self) -> None:
        """Condition B supersedes outdated information and rejects stale context."""
        scenario = self.dataset.get_scenario("SCENARIO_10_STALE_KNOWLEDGE_REJECTION")
        comparison = self.runner.run_scenario(scenario)

        self.assertEqual(comparison.result_with_brain.stale_memories_used_count, 0)
        self.assertTrue(comparison.result_with_brain.task_success)
        self.assertIn("ed25519", comparison.result_with_brain.agent_output.lower())

    def test_needle_in_haystack_noise_resistance(self) -> None:
        """Condition B retrieves the target fact despite 15 background distractor events."""
        scenario = self.dataset.get_scenario("SCENARIO_09_NEEDLE_IN_HAYSTACK")
        self.assertEqual(len(scenario.distractor_events), 15)

        comparison = self.runner.run_scenario(scenario)
        self.assertTrue(comparison.result_with_brain.task_success)
        self.assertIn("42", comparison.result_with_brain.agent_output)

    def test_statistical_calculations_and_cohens_d(self) -> None:
        """Statistical module computes paired t-statistics, Cohen's d effect size, and confidence intervals."""
        scenarios = self.dataset.list_scenarios()
        comparisons = [self.runner.run_scenario(s) for s in scenarios[:4]]

        stats = BenchmarkStatistics.compute_all_comparisons(comparisons)
        self.assertIn("task_success", stats)
        self.assertIn("repeated_explanations", stats)
        self.assertIn("total_tokens", stats)

        task_stat = stats["task_success"]
        self.assertEqual(task_stat.n_samples, 4)
        self.assertGreater(task_stat.cohens_d, 0.0)
        self.assertIsInstance(task_stat.confidence_interval_95, tuple)
        self.assertEqual(len(task_stat.confidence_interval_95), 2)

    def test_benchmark_reproducibility(self) -> None:
        """Running the same scenario multiple times yields identical deterministic results."""
        scenario = self.dataset.get_scenario("SCENARIO_01_TASK_CONTINUATION")
        res1 = self.runner.run_scenario(scenario)
        res2 = self.runner.run_scenario(scenario)

        self.assertEqual(res1.result_without_brain.total_tokens, res2.result_without_brain.total_tokens)
        self.assertEqual(res1.result_with_brain.total_tokens, res2.result_with_brain.total_tokens)
        self.assertEqual(res1.task_continuation_delta, res2.task_continuation_delta)
        self.assertEqual(res1.repeated_explanations_avoided, res2.repeated_explanations_avoided)

    def test_export_and_report_generation(self) -> None:
        """Reporter generates valid JSON and complete Markdown containing all required questions."""
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
        self.assertIn("Does AI Brain reduce total context consumption?", md_report)
        self.assertIn("Does it improve retrieval quality?", md_report)
        self.assertIn("Does it improve task continuation?", md_report)
        self.assertIn("Which scenarios benefit most?", md_report)
        self.assertIn("Which scenarios perform worse with Brain?", md_report)


if __name__ == "__main__":
    unittest.main()
