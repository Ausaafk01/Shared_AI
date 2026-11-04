"""
Data models for AI Brain Phase 4 benchmarking and evaluation framework.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ai_brain.events.models import Event


class ScenarioCategory(Enum):
    TASK_CONTINUATION = "TASK_CONTINUATION"
    ARCHITECTURAL_DECISION = "ARCHITECTURAL_DECISION"
    TECHNICAL_CONSTRAINTS = "TECHNICAL_CONSTRAINTS"
    UPDATED_REQUIREMENTS = "UPDATED_REQUIREMENTS"
    CONTRADICTORY_CLAIMS = "CONTRADICTORY_CLAIMS"
    AGENT_SWITCHING = "AGENT_SWITCHING"
    SESSION_INTERRUPTION = "SESSION_INTERRUPTION"
    REPEATED_EXPLANATION = "REPEATED_EXPLANATION"
    NEEDLE_IN_HAYSTACK = "NEEDLE_IN_HAYSTACK"
    STALE_KNOWLEDGE_REJECTION = "STALE_KNOWLEDGE_REJECTION"


class ScenarioDifficulty(Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class ScenarioProvenance(Enum):
    MANUALLY_AUTHORED = "MANUALLY_AUTHORED"
    SYNTHETIC = "SYNTHETIC"
    REAL_WORLD_DERIVED = "REAL_WORLD_DERIVED"


class BenchmarkCondition(Enum):
    WITHOUT_BRAIN = "WITHOUT_BRAIN"  # Condition A: No persistent memory, current turn only
    WITH_BRAIN = "WITH_BRAIN"        # Condition B: Complete persistent Brain functionality


@dataclass(frozen=True)
class BenchmarkScenario:
    """A realistic multi-session scenario evaluated across Condition A and Condition B."""
    scenario_id: str
    name: str
    category: ScenarioCategory
    difficulty: ScenarioDifficulty
    provenance: ScenarioProvenance
    description: str
    historical_events: Tuple[Event, ...]  # Seed events establishing prior knowledge
    task_prompt: str                      # Current prompt delivered to the agent
    project_id: str
    expected_facts: Tuple[str, ...]       # Facts that must be recalled / respected
    expected_decisions: Tuple[str, ...]   # Historical decisions that must be maintained
    expected_constraints: Tuple[str, ...] # Technical constraints that must be observed
    expected_conflicts: Tuple[str, ...]   # Contradictions that should be flagged or handled
    distractor_events: Tuple[Event, ...] = () # Irrelevant noise events to test filter precision
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "provenance": self.provenance.value,
            "description": self.description,
            "historical_events_count": len(self.historical_events),
            "distractor_events_count": len(self.distractor_events),
            "task_prompt": self.task_prompt,
            "project_id": self.project_id,
            "expected_facts": list(self.expected_facts),
            "expected_decisions": list(self.expected_decisions),
            "expected_constraints": list(self.expected_constraints),
            "expected_conflicts": list(self.expected_conflicts),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ConditionRunResult:
    """Execution output and telemetry for a single scenario under one experimental condition."""
    scenario_id: str
    condition: BenchmarkCondition
    agent_id: str
    agent_output: str
    # Token metrics
    input_tokens: int
    output_tokens: int
    retrieved_memory_tokens: int
    context_builder_tokens: int
    total_tokens: int
    # Retrieval metrics (With Brain only)
    retrieved_memory_ids: Tuple[str, ...]
    relevant_retrieved_ids: Tuple[str, ...]
    irrelevant_retrieved_ids: Tuple[str, ...]
    # Evaluation outcomes
    task_success: bool
    facts_recovered_count: int
    facts_missed_count: int
    decisions_maintained_count: int
    constraints_violated_count: int
    repeated_explanations_needed: int
    conflict_detected: bool
    conflict_handled_correctly: bool
    stale_memories_used_count: int
    execution_time_ms: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "condition": self.condition.value,
            "agent_id": self.agent_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retrieved_memory_tokens": self.retrieved_memory_tokens,
            "context_builder_tokens": self.context_builder_tokens,
            "total_tokens": self.total_tokens,
            "retrieved_count": len(self.retrieved_memory_ids),
            "relevant_retrieved_count": len(self.relevant_retrieved_ids),
            "irrelevant_retrieved_count": len(self.irrelevant_retrieved_ids),
            "task_success": self.task_success,
            "facts_recovered_count": self.facts_recovered_count,
            "facts_missed_count": self.facts_missed_count,
            "decisions_maintained_count": self.decisions_maintained_count,
            "constraints_violated_count": self.constraints_violated_count,
            "repeated_explanations_needed": self.repeated_explanations_needed,
            "conflict_detected": self.conflict_detected,
            "conflict_handled_correctly": self.conflict_handled_correctly,
            "stale_memories_used_count": self.stale_memories_used_count,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class PairedComparison:
    """Direct comparison between Condition A (Without Brain) and Condition B (With Brain) for a scenario."""
    scenario_id: str
    scenario_name: str
    category: ScenarioCategory
    difficulty: ScenarioDifficulty
    result_without_brain: ConditionRunResult
    result_with_brain: ConditionRunResult
    # Deltas (B - A): negative token delta means Brain saved tokens
    token_delta: int
    token_reduction_pct: float
    task_continuation_delta: int  # 1 if B succeeded and A failed, 0 if equal, -1 if A succeeded and B failed
    repeated_explanations_avoided: int  # explanations needed in A minus explanations needed in B
    retrieval_f1: float
    conflict_handling_delta: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "without_brain": self.result_without_brain.to_dict(),
            "with_brain": self.result_with_brain.to_dict(),
            "token_delta": self.token_delta,
            "token_reduction_pct": round(self.token_reduction_pct, 2),
            "task_continuation_delta": self.task_continuation_delta,
            "repeated_explanations_avoided": self.repeated_explanations_avoided,
            "retrieval_f1": round(self.retrieval_f1, 4),
            "conflict_handling_delta": self.conflict_handling_delta,
        }
