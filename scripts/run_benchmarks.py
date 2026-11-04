"""
Benchmark CLI runner for AI Brain Phase 4 evaluation.
Runs all 10 multi-session scenarios under Condition A (Without Brain) and Condition B (With Brain).
Generates human-readable Markdown and machine-readable JSON reports.
"""

from datetime import datetime, timezone
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_brain.benchmarks.dataset import BenchmarkDataset
from ai_brain.benchmarks.evaluator import BenchmarkEvaluator
from ai_brain.benchmarks.reporter import BenchmarkReporter
from ai_brain.benchmarks.runner import BenchmarkRunner
from ai_brain.benchmarks.statistics import BenchmarkStatistics


def main() -> None:
    print("=" * 80)
    print("AI BRAIN — PHASE 4 BENCHMARK EXECUTION FRAMEWORK")
    print("=" * 80)
    print("Initializing benchmark dataset and isolated execution runner...")

    dataset = BenchmarkDataset()
    scenarios = dataset.list_scenarios()
    print(f"Loaded {len(scenarios)} multi-session benchmark scenarios across all categories.\n")

    runner = BenchmarkRunner(token_budget=1500)
    comparisons = []

    for idx, scenario in enumerate(scenarios, 1):
        print(f"[{idx:02d}/{len(scenarios):02d}] Running Scenario: {scenario.scenario_id}")
        print(f"     Name: {scenario.name}")
        print(f"     Category: {scenario.category.value} | Difficulty: {scenario.difficulty.value}")
        comp = runner.run_scenario(scenario)
        comparisons.append(comp)

        status_wo = "PASS" if comp.result_without_brain.task_success else "FAIL"
        status_w = "PASS" if comp.result_with_brain.task_success else "FAIL"
        print(f"     Condition A (Without Brain): {status_wo} ({comp.result_without_brain.total_tokens} tokens)")
        print(f"     Condition B (With Brain)   : {status_w} ({comp.result_with_brain.total_tokens} tokens)")
        print(f"     Explanations Saved: {comp.repeated_explanations_avoided} facts\n")

    print("-" * 80)
    print("Aggregating performance metrics and statistical analyses...")
    aggregate = BenchmarkEvaluator.evaluate(comparisons)
    stats = BenchmarkStatistics.compute_all_comparisons(comparisons)

    # Generate Markdown and JSON reports
    md_report = BenchmarkReporter.generate_markdown_report(aggregate, comparisons, stats)
    json_report = BenchmarkReporter.generate_json_report(aggregate, comparisons, stats)

    # Save to disk
    report_md_path = os.path.join(os.path.dirname(__file__), "..", "benchmarks_report.md")
    report_json_path = os.path.join(os.path.dirname(__file__), "..", "benchmarks_report.json")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    with open(report_json_path, "w", encoding="utf-8") as f:
        f.write(json_report)

    print(f"Saved Markdown report to: {os.path.abspath(report_md_path)}")
    print(f"Saved JSON report to    : {os.path.abspath(report_json_path)}")
    print("=" * 80)
    print("\n" + md_report)


if __name__ == "__main__":
    main()
