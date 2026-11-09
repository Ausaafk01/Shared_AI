"""
Data models for AI Brain Phase 4 benchmarking and evaluation framework.
Provides fine-grained token accounting, partial-credit evaluation,
three-condition comparison (Cold Start, Raw History, AI Brain), and conflict ground truth.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

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


class ConflictGroundTruth(Enum):
    """Ground truth conflict classification for rigorous confusion matrix evaluation."""
    GENUINE_CONTRADICTION = "GENUINE_CONTRADICTION"  # Positive class: Direct mutual factual contradiction
    COMPATIBLE_CLAIMS = "COMPATIBLE_CLAIMS"          # Negative class: Different aspects of same system
    TEMPORAL_UPDATE = "TEMPORAL_UPDATE"              # Negative class: Valid chronological supersession
    NO_CONFLICT = "NO_CONFLICT"                      # Negative class: Standard non-conflicting knowledge


class BenchmarkCondition(Enum):
    COLD_START = "COLD_START"        # Condition A: Stateless cold start (prompt only, zero history)
    RAW_HISTORY = "RAW_HISTORY"      # Condition B: Raw uncurated event log supplied up to budget
    AI_BRAIN = "AI_BRAIN"            # Condition C: Persistent Brain (selective budgeted retrieval)

    # Backwards-compatible aliases
    WITHOUT_BRAIN = "COLD_START"
    WITH_BRAIN = "AI_BRAIN"


@dataclass(frozen=True)
class BenchmarkScenario:
    """A realistic multi-session scenario evaluated across Condition A, Condition B, and Condition C."""
    scenario_id: str
    name: str
    category: ScenarioCategory
    difficulty: ScenarioDifficulty
    provenance: ScenarioProvenance
    description: str
    historical_events: Tuple[Event, ...]  # Seed events establishing prior knowledge
    task_prompt: str                      # Current prompt delivered to the agent
    project_id: str
    expected_facts: Tuple[str, ...]       # Primary facts that must be recalled / respected
    expected_decisions: Tuple[str, ...]   # Historical decisions that must be maintained
    expected_constraints: Tuple[str, ...] # Technical constraints that must be observed
    expected_conflicts: Tuple[str, ...]   # Contradictions that should be flagged or handled
    distractor_events: Tuple[Event, ...] = () # Irrelevant noise events to test filter precision
    conflict_ground_truth: ConflictGroundTruth = ConflictGroundTruth.NO_CONFLICT
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "provenance": self.provenance.value,
            "conflict_ground_truth": self.conflict_ground_truth.value,
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
class ScenarioTokenAudit:
    """Estimated token accounting for a single scenario run without double-counting."""
    task_prompt_tokens: int              # Estimated tokens in user prompt delivered to the agent
    available_history_tokens: int        # Estimated tokens in historical seed events + distractors (archive size)
    supplied_history_tokens: int         # Estimated tokens of historical context actually supplied to agent
    omitted_history_tokens: int          # Estimated tokens of historical context omitted due to budget
    retrieved_memory_tokens: int         # Estimated tokens in raw memories retrieved from index
    context_builder_tokens: int          # Estimated tokens formatted by Context Builder into prompt
    agent_input_tokens: int              # Estimated input tokens delivered to agent model
    agent_output_tokens: int             # Estimated tokens generated by the agent model
    total_tokens: int                    # Net total: agent_input_tokens + agent_output_tokens
    tokenizer_name: str = "DeterministicTokenizer (~4 chars/token approximation)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_prompt_tokens": self.task_prompt_tokens,
            "available_history_tokens": self.available_history_tokens,
            "supplied_history_tokens": self.supplied_history_tokens,
            "omitted_history_tokens": self.omitted_history_tokens,
            "retrieved_memory_tokens": self.retrieved_memory_tokens,
            "context_builder_tokens": self.context_builder_tokens,
            "agent_input_tokens": self.agent_input_tokens,
            "agent_output_tokens": self.agent_output_tokens,
            "total_tokens": self.total_tokens,
            "tokenizer_name": self.tokenizer_name,
        }


@dataclass(frozen=True)
class ConditionRunResult:
    """Execution output, token audit, and telemetry for a single scenario under one condition."""
    scenario_id: str
    condition: BenchmarkCondition
    agent_id: str
    agent_output: str
    # Token audit
    tokens: ScenarioTokenAudit
    # Retrieval metrics
    retrieved_memory_ids: Tuple[str, ...]
    relevant_retrieved_ids: Tuple[str, ...]
    irrelevant_retrieved_ids: Tuple[str, ...]
    # Partial-credit evaluation scores (0.0 to 1.0)
    facts_recovery_rate: float
    decisions_maintenance_rate: float
    constraints_adherence_rate: float
    partial_success_score: float         # Weighted composite score
    task_success: bool                   # Strict binary pass/fail
    # Discrete counts
    facts_recovered_count: int
    facts_missed_count: int
    decisions_maintained_count: int
    constraints_violated_count: int
    repeated_explanations_needed: int
    # Conflict & staleness
    conflict_detected: bool
    conflict_handled_correctly: bool
    stale_memories_used_count: int
    execution_time_ms: float
    missing_details_notes: Tuple[str, ...] = ()
    error_message: Optional[str] = None
    retrieval_strategy_results: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.tokens.total_tokens

    @property
    def input_tokens(self) -> int:
        return self.tokens.agent_input_tokens

    @property
    def output_tokens(self) -> int:
        return self.tokens.agent_output_tokens

    @property
    def retrieved_memory_tokens(self) -> int:
        return self.tokens.retrieved_memory_tokens

    @property
    def context_builder_tokens(self) -> int:
        return self.tokens.context_builder_tokens

    @property
    def supplied_history_tokens(self) -> int:
        return self.tokens.supplied_history_tokens

    @property
    def omitted_history_tokens(self) -> int:
        return self.tokens.omitted_history_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "condition": self.condition.value,
            "agent_id": self.agent_id,
            "tokens": self.tokens.to_dict(),
            "retrieved_count": len(self.retrieved_memory_ids),
            "relevant_retrieved_count": len(self.relevant_retrieved_ids),
            "irrelevant_retrieved_count": len(self.irrelevant_retrieved_ids),
            "facts_recovery_rate": round(self.facts_recovery_rate, 3),
            "decisions_maintenance_rate": round(self.decisions_maintenance_rate, 3),
            "constraints_adherence_rate": round(self.constraints_adherence_rate, 3),
            "partial_success_score": round(self.partial_success_score, 3),
            "task_success": self.task_success,
            "facts_recovered_count": self.facts_recovered_count,
            "facts_missed_count": self.facts_missed_count,
            "decisions_maintained_count": self.decisions_maintained_count,
            "constraints_violated_count": self.constraints_violated_count,
            "repeated_explanations_needed": self.repeated_explanations_needed,
            "conflict_detected": self.conflict_detected,
            "conflict_handled_correctly": self.conflict_handled_correctly,
            "stale_memories_used_count": self.stale_memories_used_count,
            "missing_details_notes": list(self.missing_details_notes),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class PairedComparison:
    """Comparison across Condition A (Cold Start), Condition B (Raw History), and Condition C (AI Brain)."""
    scenario_id: str
    scenario_name: str
    category: ScenarioCategory
    difficulty: ScenarioDifficulty
    result_without_brain: ConditionRunResult  # Condition A: Cold Start (Stateless)
    result_with_brain: ConditionRunResult     # Condition C: AI Brain (Selective Memory)
    result_raw_history: Optional[ConditionRunResult] = None  # Condition B: Raw History
    conflict_ground_truth: ConflictGroundTruth = ConflictGroundTruth.NO_CONFLICT
    # Deltas (C - A)
    token_delta: int = 0
    token_reduction_pct: float = 0.0
    task_continuation_delta: int = 0         # +1 if C passed and A failed; 0 if equal; -1 if A passed and C failed
    partial_score_delta: float = 0.0
    repeated_explanations_avoided: int = 0  # Explanations needed in A minus explanations needed in C
    retrieval_precision_at_k: float = 0.0
    retrieval_recall_at_k: float = 0.0
    retrieval_f1: float = 0.0
    conflict_handling_delta: int = 0

    @property
    def result_cold_start(self) -> ConditionRunResult:
        return self.result_without_brain

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "conflict_ground_truth": self.conflict_ground_truth.value,
            "condition_a_cold_start": self.result_without_brain.to_dict(),
            "condition_c_ai_brain": self.result_with_brain.to_dict(),
            "without_brain": self.result_without_brain.to_dict(),
            "with_brain": self.result_with_brain.to_dict(),
            "token_delta": self.token_delta,
            "token_reduction_pct": round(self.token_reduction_pct, 2),
            "task_continuation_delta": self.task_continuation_delta,
            "partial_score_delta": round(self.partial_score_delta, 3),
            "repeated_explanations_avoided": self.repeated_explanations_avoided,
            "retrieval_precision_at_k": round(self.retrieval_precision_at_k, 4),
            "retrieval_recall_at_k": round(self.retrieval_recall_at_k, 4),
            "retrieval_f1": round(self.retrieval_f1, 4),
            "conflict_handling_delta": self.conflict_handling_delta,
        }
        if self.result_raw_history:
            d["condition_b_raw_history"] = self.result_raw_history.to_dict()
        return d
