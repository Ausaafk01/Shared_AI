"""Pydantic request and response schemas for AI Brain REST API (M5.5).

All models provide strict field validation, detailed descriptions, and OpenAPI 3.1 compatibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Common & System Schemas
# =============================================================================
class ErrorResponse(BaseModel):
    """Standardized error response payload."""
    model_config = ConfigDict(extra="forbid")

    error: str = Field(..., description="Machine-readable error classification code")
    message: str = Field(..., description="Human-readable sanitized error description")
    request_id: Optional[str] = Field(default=None, description="Correlation identifier for tracing")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Safe contextual metadata")


class HealthResponse(BaseModel):
    """Liveness probe response."""
    status: str = Field(default="ok", description="Service operational status")
    version: str = Field(default="1.0.0", description="API version")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ReadyResponse(BaseModel):
    """Readiness probe response."""
    status: str = Field(default="ready", description="Readiness status")
    backend: str = Field(..., description="Active storage backend (sqlite or postgres)")
    database: str = Field(default="connected", description="Database connection status")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# =============================================================================
# Interaction Lifecycle Schemas
# =============================================================================
class InteractionPrepareRequest(BaseModel):
    """Request payload for preparing an agent interaction context."""
    model_config = ConfigDict(extra="forbid")

    input_text: str = Field(..., min_length=1, max_length=100000, description="Task prompt or user query")
    session_id: Optional[str] = Field(default=None, description="Optional active session identifier")
    agent_id: Optional[str] = Field(default=None, description="Identity of executing agent")
    idempotency_key: Optional[str] = Field(default=None, max_length=128, description="Unique client idempotency key")
    max_context_tokens: Optional[int] = Field(default=None, ge=50, le=32000, description="Token budget for context")
    search_query: Optional[str] = Field(default=None, description="Optional custom retrieval search query")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary interaction metadata")


class ContextItemDTO(BaseModel):
    """Knowledge memory item selected for interaction context."""
    memory_id: str
    content: str
    memory_type: str
    status: str
    relevance_score: Optional[float] = None
    importance_score: Optional[float] = None
    source_agents: List[str] = []


class ContextPackageDTO(BaseModel):
    """Structured context delivery package."""
    project_id: str
    query: str
    token_budget: int
    tokens_used: int
    items: List[ContextItemDTO] = []
    disputed_omitted: bool = False


class InteractionPrepareResponse(BaseModel):
    """Response returned upon preparing an interaction context."""
    interaction_id: str
    project_id: str
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    status: str = "PREPARED"
    tokens_used: int
    prompt_context_markdown: str
    context_package: Optional[ContextPackageDTO] = None
    is_cached: bool = False


class InteractionCompleteRequest(BaseModel):
    """Request payload for completing an interaction with agent output."""
    model_config = ConfigDict(extra="forbid")

    agent_response: str = Field(..., min_length=1, description="Raw completion text produced by agent")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    agent_id: Optional[str] = Field(default=None, description="Agent identifier")
    decision_records: Optional[List[Dict[str, Any]]] = Field(default=None, description="Explicit decisions made")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Execution metrics (e.g. latency, tokens)")


class InteractionCompleteResponse(BaseModel):
    """Response returned upon completing an interaction and extracting memories."""
    interaction_id: str
    status: str = "COMPLETED"
    memories_created: List[Dict[str, Any]] = []
    memories_updated: List[Dict[str, Any]] = []
    conflicts_detected: List[Dict[str, Any]] = []
    graph_nodes_created: int = 0
    vectors_indexed: int = 0
    execution_time_ms: Optional[float] = None


class InteractionRecordDTO(BaseModel):
    """Stored persistent interaction record."""
    interaction_id: str
    project_id: str
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    status: str
    input_text: str
    output_text: Optional[str] = None
    decision_records: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    created_at: str


# =============================================================================
# Memory Management Schemas
# =============================================================================
class MemoryCreateRequest(BaseModel):
    """Request to create a new persistent memory item."""
    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1, max_length=100000, description="Knowledge text content")
    memory_type: str = Field(default="FACT", description="Classification: FACT, DECISION, PREFERENCE, CONSTRAINT, TASK")
    importance: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Base importance score")
    source_agent_id: Optional[str] = Field(default=None, description="Originating agent identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Custom metadata tags")


class MemoryUpdateRequest(BaseModel):
    """Request to update an existing memory subject to versioning rules."""
    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1, max_length=100000, description="Updated knowledge content")
    source_event_ids: Optional[List[str]] = Field(default=None, description="Attributed event identifiers")
    source_agent_ids: Optional[List[str]] = Field(default=None, description="Attributed agent identifiers")
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Updated importance score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata tags")


class MemoryDTO(BaseModel):
    """Representation of an immutable memory version."""
    memory_id: str
    project_id: str
    content: str
    memory_type: str
    status: str
    version: int
    root_memory_id: Optional[str] = None
    parent_memory_id: Optional[str] = None
    importance: float = 0.5
    importance_score: float = 0.5
    confidence: float = 1.0
    source_event_ids: List[str] = []
    source_agent_ids: List[str] = []
    metadata: Dict[str, Any] = {}
    created_at: str
    updated_at: str


class MemoryUpdateResponse(BaseModel):
    """Result of memory update processing."""
    result_type: str = Field(..., description="Outcome: UPDATED, NO_CHANGE, CONFLICT")
    memory: MemoryDTO
    version: int
    superseded_memory_id: Optional[str] = None


# =============================================================================
# Retrieval Schemas
# =============================================================================
class RetrievalQueryRequest(BaseModel):
    """Request payload for multi-strategy memory retrieval."""
    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(..., min_length=1, description="Search query string")
    strategy: str = Field(default="HYBRID", description="Retrieval strategy: HYBRID, VECTOR, GRAPH")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of items to return")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum cosine threshold")
    include_superseded: bool = Field(default=False, description="Whether to include historical superseded memories")
    include_archived: bool = Field(default=False, description="Whether to include archived memories")
    metadata_filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata key-value filter requirements")


class RetrievalResultItemDTO(BaseModel):
    """Attributed retrieval match item."""
    memory_id: str
    content: str
    score: float
    sources: List[str]
    memory_type: str
    status: str
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    graph_traversal_info: Optional[Dict[str, Any]] = None


class RetrievalQueryResponse(BaseModel):
    """Attributed retrieval query results."""
    query: str
    strategy: str
    items: List[RetrievalResultItemDTO]
    warnings: List[str] = []
    count: int


# =============================================================================
# Session Continuity Schemas
# =============================================================================
class SessionCreateRequest(BaseModel):
    """Request payload to create a new agent session."""
    model_config = ConfigDict(extra="forbid")

    primary_agent_id: Optional[str] = Field(default=None, description="Initial primary agent identifier")
    summary: Optional[str] = Field(default=None, description="Initial session objective or summary")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Custom session metadata")


class SwitchAgentRequest(BaseModel):
    """Request to handover a session to a different agent."""
    model_config = ConfigDict(extra="forbid")

    new_agent_id: str = Field(..., min_length=1, description="Identifier of agent taking over session")
    reason: Optional[str] = Field(default=None, description="Handoff rationale")


class SessionRecordDTO(BaseModel):
    """Persistent session state record."""
    session_id: str
    project_id: str
    primary_agent_id: Optional[str] = None
    status: str
    summary: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: str
    updated_at: str


# =============================================================================
# Workflow Schemas
# =============================================================================
class CollaborationStepDTO(BaseModel):
    """Configuration for a single step in a multi-agent collaboration workflow."""
    step_id: Optional[str] = None
    step_index: int = Field(default=1, ge=1)
    name: str = Field(..., min_length=1)
    agent_id: str = Field(..., min_length=1)
    instruction: str = Field(..., min_length=1)
    query: Optional[str] = None
    required_role: str = "assistant"


class WorkflowCreateRequest(BaseModel):
    """Request to instantiate a multi-agent collaboration workflow."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="Multi-Agent Workflow", min_length=1)
    session_id: Optional[str] = Field(default=None)
    workflow_type: str = Field(default="RESEARCH_CODING_REVIEW", description="RESEARCH_CODING_REVIEW or CUSTOM")
    task_description: str = Field(..., min_length=1, description="Overall task prompt")
    steps: Optional[List[CollaborationStepDTO]] = None
    metadata: Optional[Dict[str, Any]] = None


class WorkflowExecuteRequest(BaseModel):
    """Optional execution parameters for a workflow."""
    model_config = ConfigDict(extra="forbid")

    additional_context: Optional[str] = Field(default=None, description="Optional runtime context for execution")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional runtime execution metadata")


class CollaborationStepResultDTO(BaseModel):
    """Result of an individual workflow step execution."""
    step_index: int
    name: str
    agent_id: str
    status: str
    output: Optional[str] = None
    error_message: Optional[str] = None


class WorkflowExecutionResponse(BaseModel):
    """Result of workflow execution."""
    workflow_id: str
    project_id: str
    session_id: str
    status: str
    task_description: str
    completed_steps: List[CollaborationStepResultDTO] = []
    final_output: Optional[str] = None


class WorkflowRecordDTO(BaseModel):
    """Stored workflow metadata."""
    workflow_id: str
    project_id: str
    session_id: str
    task_description: str
    status: str
    step_count: int
    completed_step_count: int
    created_at: str
    updated_at: str


# =============================================================================
# Background Task Schemas (M5.7)
# =============================================================================
class TaskSubmitRequest(BaseModel):
    """Request payload for submitting an asynchronous background task."""
    model_config = ConfigDict(extra="forbid")

    task_type: str = Field(..., min_length=1, max_length=100, description="Registered task type (e.g. interaction_execution, memory_maintenance)")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Task payload dictionary")
    idempotency_key: Optional[str] = Field(default=None, min_length=1, max_length=128, description="Client idempotency key")


class TaskSubmitResponse(BaseModel):
    """Response returned upon submitting a background task."""
    task_id: str
    project_id: str
    task_type: str
    status: str
    idempotency_key: Optional[str] = None
    created_at: str


class TaskStatusResponse(BaseModel):
    """Response payload representing the state and result of a background task."""
    task_id: str
    project_id: str
    task_type: str
    status: str
    retry_count: int
    idempotency_key: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str

