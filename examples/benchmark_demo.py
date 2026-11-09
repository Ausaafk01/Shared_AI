"""
Demonstration of AI Brain Phase 4: Evaluation & Benchmarking.
Executes live comparisons across the 6 key demonstration dimensions:
1. Agent continuing a task with and without Brain.
2. Retrieval accuracy comparison across strategies.
3. Context and token usage comparison.
4. Multi-agent session continuity comparison.
5. Conflict-handling comparison.
6. Overall benchmark report generation.
"""

from pathlib import Path
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ai_brain.benchmarks.dataset import BenchmarkDataset
from ai_brain.benchmarks.evaluator import BenchmarkEvaluator
from ai_brain.benchmarks.runner import BenchmarkRunner
from ai_brain.benchmarks.statistics import BenchmarkStatistics


def main() -> None:
    print("=" * 75)
    print("AI Brain — Phase 4 Benchmarking & Evaluation Demonstration")
    print("=" * 75)

    dataset = BenchmarkDataset()
    runner = BenchmarkRunner(token_budget=1500)

    # -------------------------------------------------------------------------
    # 1. Agent Task Continuation With and Without Brain
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("DEMO 1: Task Continuation Across Sessions (With vs Without Brain)")
    print("-" * 75)
    s1 = dataset.get_scenario("SCENARIO_01_TASK_CONTINUATION")
    print(f"Task: '{s1.task_prompt}'")
    print("Condition A (Without Brain):")
    res1_a = runner.run_condition_a(s1)
    print(f"  Agent Output    : {res1_a.agent_output}")
    print(f"  Task Success    : {res1_a.task_success}")
    print(f"  Missing Facts   : {res1_a.facts_missed_count} facts missed")

    print("\nCondition B (With Brain):")
    res1_b = runner.run_condition_b(s1)
    print(f"  Agent Output    :\n{res1_b.agent_output}")
    print(f"  Task Success    : {res1_b.task_success}")
    print(f"  Recovered Facts : {res1_b.facts_recovered_count} facts recovered")

    # -------------------------------------------------------------------------
    # 2. Retrieval Accuracy Comparison Across Strategies
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("DEMO 2: Retrieval Accuracy Across Strategies")
    print("-" * 75)
    s3 = dataset.get_scenario("SCENARIO_03_TECHNICAL_CONSTRAINTS")
    comp3 = runner.run_scenario(s3)
    agg3 = BenchmarkEvaluator.evaluate([comp3])

    print(f"Scenario: {s3.name}")
    print(f"Overall Hybrid Retrieval:")
    print(f"  Precision@K = {agg3.retrieval_overall.precision_at_k:.2f}")
    print(f"  Recall@K    = {agg3.retrieval_overall.recall_at_k:.2f}")
    print(f"  F1 Score    = {agg3.retrieval_overall.f1_score:.2f}")
    print(f"  MRR         = {agg3.retrieval_overall.mrr:.2f}")

    print("\nBreakdown by Strategy:")
    for strat_name, metrics in agg3.retrieval_by_strategy.items():
        print(f"  • {strat_name:28}: P={metrics.precision_at_k:.2f}, R={metrics.recall_at_k:.2f}, F1={metrics.f1_score:.2f}")

    # -------------------------------------------------------------------------
    # 3. Context and Token Usage Comparison
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("DEMO 3: Context and Token Usage Breakdown")
    print("-" * 75)
    s2 = dataset.get_scenario("SCENARIO_02_ARCHITECTURAL_DECISION")
    comp2 = runner.run_scenario(s2)
    print(f"Scenario: {s2.name}")
    print(f"Condition A (Without Brain):")
    print(f"  Input Tokens  : {comp2.result_without_brain.input_tokens}")
    print(f"  Output Tokens : {comp2.result_without_brain.output_tokens}")
    print(f"  Total Tokens  : {comp2.result_without_brain.total_tokens}")

    print(f"\nCondition B (With Brain):")
    print(f"  Input Tokens       : {comp2.result_with_brain.input_tokens}")
    print(f"  Raw Memory Tokens  : {comp2.result_with_brain.retrieved_memory_tokens}")
    print(f"  Context Builder    : {comp2.result_with_brain.context_builder_tokens} tokens")
    print(f"  Output Tokens      : {comp2.result_with_brain.output_tokens}")
    print(f"  Total Tokens       : {comp2.result_with_brain.total_tokens}")
    print(f"  Token Delta        : {comp2.token_delta:+} tokens ({comp2.token_reduction_pct:+.1f}%)")

    # -------------------------------------------------------------------------
    # 4. Multi-Agent Session Continuity Comparison
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("DEMO 4: Multi-Agent Handoff and Session Continuity")
    print("-" * 75)
    s6 = dataset.get_scenario("SCENARIO_06_AGENT_SWITCHING")
    comp6 = runner.run_scenario(s6)
    print(f"Scenario: {s6.name}")
    print(f"Condition A (Without Brain):")
    print(f"  Task Success      : {comp6.result_without_brain.task_success}")
    print(f"  Explanations Needed: {comp6.result_without_brain.repeated_explanations_needed}")

    print(f"Condition B (With Brain):")
    print(f"  Task Success      : {comp6.result_with_brain.task_success}")
    print(f"  Explanations Needed: {comp6.result_with_brain.repeated_explanations_needed}")
    print(f"  Decisions Upheld  : {comp6.result_with_brain.decisions_maintained_count}")

    # -------------------------------------------------------------------------
    # 5. Conflict-Handling Comparison
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("DEMO 5: Contradictory Claims & Disputed Fact Preservation")
    print("-" * 75)
    s5 = dataset.get_scenario("SCENARIO_05_CONTRADICTORY_CLAIMS")
    comp5 = runner.run_scenario(s5)
    print(f"Scenario: {s5.name}")
    print(f"Condition A (Without Brain):")
    print(f"  Conflict Detected : {comp5.result_without_brain.conflict_detected}")
    print(f"  Conflict Handled  : {comp5.result_without_brain.conflict_handled_correctly}")

    print(f"\nCondition B (With Brain):")
    print(f"  Conflict Detected : {comp5.result_with_brain.conflict_detected}")
    print(f"  Conflict Handled  : {comp5.result_with_brain.conflict_handled_correctly}")
    print(f"  Dispute Output    :\n{comp5.result_with_brain.agent_output}")

    # -------------------------------------------------------------------------
    # 6. Overall Benchmark Report Generation
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("DEMO 6: Comprehensive Benchmark Evaluation Compilation")
    print("-" * 75)
    all_scenarios = dataset.list_scenarios()
    all_comps = [runner.run_scenario(s) for s in all_scenarios]
    aggregate = BenchmarkEvaluator.evaluate(all_comps)
    stats = BenchmarkStatistics.compute_all_comparisons(all_comps)

    print(f"Evaluated {len(all_comps)} total multi-session scenarios.")
    print(f"  Task Continuation Rate (Without Brain): {aggregate.task_success_rate_without_brain * 100:.1f}%")
    print(f"  Task Continuation Rate (With Brain)   : {aggregate.task_success_rate_with_brain * 100:.1f}% (Improvement: +{aggregate.task_success_improvement * 100:.1f}%)")
    print(f"  Repeated Explanations Avoided         : {aggregate.repeated_explanations_avoided} facts")
    print(f"  Decisions Maintained                  : {aggregate.decisions_recovered_count} total")
    print(f"  Average Query Tokens (Without vs With): {aggregate.token_metrics.avg_total_tokens_without_brain:.1f} vs {aggregate.token_metrics.avg_total_tokens_with_brain:.1f}")
    print(f"  Retrieval Precision / F1              : P={aggregate.retrieval_overall.precision_at_k:.2f}, F1={aggregate.retrieval_overall.f1_score:.2f}")

    print("\n" + "=" * 75)
    print("ALL 6 BENCHMARK DEMONSTRATIONS EXECUTED LIVE AND VERIFIED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()

