"""
Immutable data models for the Brain Loop subsystem.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Union
import uuid

from ai_brain.context.models import ContextPackage
from ai_brain.loop.exceptions import InputValidationError, ResponseValidationError
from ai_brain.memories.conflicts.models import MemoryConflict
from ai_brain.memories.models import Memory
from ai_brain.memories.updating.updater import UpdateAction, UpdateResult
from ai_brain.retrieval.models import QueryFilters, RetrievalResponse
from ai_brain.retrieval.types import RetrievalMethod, normalize_retrieval_method


def _freeze_container(data: Any) -> Any:
    """Recursively freeze dicts and sequences for immutability."""
    if isinstance(data, (dict, MappingProxyType)):
        return MappingProxyType({k: _freeze_container(v) for k, v in data.items()})
    elif isinstance(data, (list, tuple)):
        return tuple(_freeze_container(v) for v in data)
    elif isinstance(data, (set, frozenset)):
        return frozenset(_freeze_container(v) for v in data)
    return data


def _unfreeze_container(data: Any) -> Any:
    """Recursively unfreeze containers for serialization."""
    if isinstance(data, (MappingProxyType, dict)):
        return {k: _unfreeze_container(v) for k, v in data.items()}
    elif isinstance(data, (tuple, list, frozenset, set)):
        return [_unfreeze_container(v) for v in data]
    return data


def _json_serializer(obj: Any) -> Any:
    """JSON serializer helper for datetime, enums, UUIDs, and proxies."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, StrEnum):
        return obj.value
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, (MappingProxyType, Mapping)):
        return dict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class InteractionStatus(StrEnum):
    """Lifecycle statuses for a Brain Loop interaction."""
    INITIALIZED = "INITIALIZED"
    INPUT_RECEIVED = "INPUT_RECEIVED"
    CONTEXT_PREPARED = "CONTEXT_PREPARED"
    RESPONSE_RECEIVED = "RESPONSE_RECEIVED"
    COMPLETED = "COMPLETED"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"


class LoopStage(StrEnum):
    """Execution stages of an interaction in Brain Loop."""
    INITIALIZED = "INITIALIZED"
    INPUT_PERSISTED = "INPUT_PERSISTED"
    CONTEXT_RETRIEVED = "CONTEXT_RETRIEVED"
    CONTEXT_DELIVERED = "CONTEXT_DELIVERED"
    RESPONSE_PERSISTED = "RESPONSE_PERSISTED"
    MEMORY_EXTRACTED = "MEMORY_EXTRACTED"
    MEMORY_UPDATED = "MEMORY_UPDATED"
    CONFLICTS_CHECKED = "CONFLICTS_CHECKED"
    GRAPH_SYNCED = "GRAPH_SYNCED"
    VECTOR_SYNCED = "VECTOR_SYNCED"
    COMPLETED = "COMPLETED"


def update_result_to_dict(res: UpdateResult) -> dict[str, Any]:
    return {
        "action": res.action.value,
        "created": res.created,
        "superseded": res.superseded,
        "duplicate": res.duplicate,
        "conflict": res.conflict,
        "current_memory": res.current_memory.to_dict() if res.current_memory else None,
        "previous_memory": res.previous_memory.to_dict() if res.previous_memory else None,
        "affected_memories": [m.to_dict() for m in res.affected_memories],
        "message": res.message,
    }


def update_result_from_dict(d: dict[str, Any]) -> UpdateResult:
    return UpdateResult(
        action=UpdateAction(d["action"]),
        created=d["created"],
        superseded=d["superseded"],
        duplicate=d["duplicate"],
        conflict=d["conflict"],
        current_memory=Memory.from_dict(d["current_memory"]) if d.get("current_memory") else None,
        previous_memory=Memory.from_dict(d["previous_memory"]) if d.get("previous_memory") else None,
        affected_memories=tuple(Memory.from_dict(m) for m in d.get("affected_memories", ())),
        message=d.get("message", ""),
    )


@dataclass(frozen=True, kw_only=True)
class AgentInteractionInput:
    """
    Immutable representation of an incoming agent interaction request.
    """
    interaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    idempotency_key: Optional[str] = None
    project_id: str
    agent_id: str
    session_id: str
    content: str
    query: Optional[str] = None
    token_budget: Optional[int] = None
    retrieval_method: str = RetrievalMethod.HYBRID.value
    retrieval_filters: Optional[QueryFilters] = None
    top_k: int = 10
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.interaction_id or not isinstance(self.interaction_id, str):
            object.__setattr__(self, "interaction_id", str(uuid.uuid4()))
        else:
            object.__setattr__(self, "interaction_id", self.interaction_id.strip())

        if self.idempotency_key is not None:
            if not isinstance(self.idempotency_key, str) or not self.idempotency_key.strip():
                raise InputValidationError("idempotency_key, if provided, must be a non-empty string")
            object.__setattr__(self, "idempotency_key", self.idempotency_key.strip())

        if not self.project_id or not isinstance(self.project_id, str) or not self.project_id.strip():
            raise InputValidationError("project_id must be a non-empty string")
        object.__setattr__(self, "project_id", self.project_id.strip())

        if not self.agent_id or not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise InputValidationError("agent_id must be a non-empty string")
        object.__setattr__(self, "agent_id", self.agent_id.strip())

        if not self.session_id or not isinstance(self.session_id, str) or not self.session_id.strip():
            raise InputValidationError("session_id must be a non-empty string")
        object.__setattr__(self, "session_id", self.session_id.strip())

        if not self.content or not isinstance(self.content, str) or not self.content.strip():
            raise InputValidationError("content must be a non-empty string")
        object.__setattr__(self, "content", self.content.strip())

        resolved_query = self.query.strip() if self.query and isinstance(self.query, str) else self.content.strip()
        object.__setattr__(self, "query", resolved_query)

        norm_method = normalize_retrieval_method(self.retrieval_method)
        object.__setattr__(self, "retrieval_method", norm_method.value)

        if self.token_budget is not None and self.token_budget <= 0:
            raise InputValidationError(f"token_budget must be positive, got {self.token_budget}")

        if self.top_k <= 0:
            raise InputValidationError(f"top_k must be positive, got {self.top_k}")

        object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))

        if not isinstance(self.created_at, datetime):
            raise InputValidationError("created_at must be a datetime object")
        c_utc = (
            self.created_at.replace(tzinfo=timezone.utc)
            if self.created_at.tzinfo is None
            else self.created_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "created_at", c_utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "idempotency_key": self.idempotency_key,
            "project_id": self.project_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "content": self.content,
            "query": self.query,
            "token_budget": self.token_budget,
            "retrieval_method": self.retrieval_method,
            "retrieval_filters": self.retrieval_filters.to_dict() if self.retrieval_filters else None,
            "top_k": self.top_k,
            "metadata": _unfreeze_container(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentInteractionInput":
        filters = (
            QueryFilters.from_dict(data["retrieval_filters"])
            if data.get("retrieval_filters")
            else None
        )
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc)
        )
        return cls(
            interaction_id=data.get("interaction_id") or str(uuid.uuid4()),
            idempotency_key=data.get("idempotency_key"),
            project_id=data["project_id"],
            agent_id=data["agent_id"],
            session_id=data["session_id"],
            content=data["content"],
            query=data.get("query"),
            token_budget=data.get("token_budget"),
            retrieval_method=data.get("retrieval_method", RetrievalMethod.HYBRID.value),
            retrieval_filters=filters,
            top_k=data.get("top_k", 10),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AgentInteractionInput":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True, kw_only=True)
class InteractionContextResult:
    """
    Immutable representation of context assembled and delivered for an interaction.
    """
    interaction_id: str
    project_id: str
    agent_id: str
    session_id: str
    input_event_id: str
    context_event_id: Optional[str] = None
    context_package: ContextPackage
    retrieval_response: Optional[RetrievalResponse] = None
    status: InteractionStatus = InteractionStatus.CONTEXT_PREPARED
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))
        if not isinstance(self.created_at, datetime):
            raise InputValidationError("created_at must be a datetime object")
        c_utc = (
            self.created_at.replace(tzinfo=timezone.utc)
            if self.created_at.tzinfo is None
            else self.created_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "created_at", c_utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "project_id": self.project_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "input_event_id": self.input_event_id,
            "context_event_id": self.context_event_id,
            "context_package": self.context_package.to_dict(),
            "retrieval_response": self.retrieval_response.to_dict() if self.retrieval_response else None,
            "status": self.status.value,
            "metadata": _unfreeze_container(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionContextResult":
        return cls(
            interaction_id=data["interaction_id"],
            project_id=data["project_id"],
            agent_id=data["agent_id"],
            session_id=data["session_id"],
            input_event_id=data["input_event_id"],
            context_event_id=data.get("context_event_id"),
            context_package=ContextPackage.from_dict(data["context_package"]),
            retrieval_response=(
                RetrievalResponse.from_dict(data["retrieval_response"])
                if data.get("retrieval_response")
                else None
            ),
            status=InteractionStatus(data.get("status", InteractionStatus.CONTEXT_PREPARED.value)),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "InteractionContextResult":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True, kw_only=True)
class AgentResponseInput:
    """
    Immutable representation of an agent's response to an interaction.
    """
    interaction_id: str
    project_id: str
    agent_id: str
    session_id: str
    content: str
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.interaction_id or not isinstance(self.interaction_id, str) or not self.interaction_id.strip():
            raise ResponseValidationError("interaction_id must be a non-empty string referencing the interaction")
        object.__setattr__(self, "interaction_id", self.interaction_id.strip())

        if not self.project_id or not isinstance(self.project_id, str) or not self.project_id.strip():
            raise ResponseValidationError("project_id must be a non-empty string")
        object.__setattr__(self, "project_id", self.project_id.strip())

        if not self.agent_id or not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ResponseValidationError("agent_id must be a non-empty string")
        object.__setattr__(self, "agent_id", self.agent_id.strip())

        if not self.session_id or not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ResponseValidationError("session_id must be a non-empty string")
        object.__setattr__(self, "session_id", self.session_id.strip())

        if not self.content or not isinstance(self.content, str) or not self.content.strip():
            raise ResponseValidationError("content must be a non-empty string")
        object.__setattr__(self, "content", self.content.strip())

        object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))

        if not isinstance(self.created_at, datetime):
            raise ResponseValidationError("created_at must be a datetime object")
        c_utc = (
            self.created_at.replace(tzinfo=timezone.utc)
            if self.created_at.tzinfo is None
            else self.created_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "created_at", c_utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "project_id": self.project_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "content": self.content,
            "metadata": _unfreeze_container(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentResponseInput":
        return cls(
            interaction_id=data["interaction_id"],
            project_id=data["project_id"],
            agent_id=data["agent_id"],
            session_id=data["session_id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AgentResponseInput":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True, kw_only=True)
class InteractionCompletionResult:
    """
    Immutable representation of the outcome of completing an interaction lifecycle.
    """
    interaction_id: str
    project_id: str
    agent_id: str
    session_id: str
    status: InteractionStatus
    response_event_id: str
    completion_event_id: Optional[str] = None
    extracted_memories: tuple[Memory, ...] = field(default_factory=tuple)
    update_results: tuple[UpdateResult, ...] = field(default_factory=tuple)
    conflicts_detected: tuple[MemoryConflict, ...] = field(default_factory=tuple)
    graph_nodes_integrated: int = 0
    vectors_indexed: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)
    is_partial_success: bool = False
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "extracted_memories", tuple(self.extracted_memories))
        object.__setattr__(self, "update_results", tuple(self.update_results))
        object.__setattr__(self, "conflicts_detected", tuple(self.conflicts_detected))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))

        if not isinstance(self.created_at, datetime):
            raise InputValidationError("created_at must be a datetime object")
        c_utc = (
            self.created_at.replace(tzinfo=timezone.utc)
            if self.created_at.tzinfo is None
            else self.created_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "created_at", c_utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "project_id": self.project_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "response_event_id": self.response_event_id,
            "completion_event_id": self.completion_event_id,
            "extracted_memories": [m.to_dict() for m in self.extracted_memories],
            "update_results": [update_result_to_dict(r) for r in self.update_results],
            "conflicts_detected": [c.to_dict() for c in self.conflicts_detected],
            "graph_nodes_integrated": self.graph_nodes_integrated,
            "vectors_indexed": self.vectors_indexed,
            "warnings": list(self.warnings),
            "is_partial_success": self.is_partial_success,
            "metadata": _unfreeze_container(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionCompletionResult":
        return cls(
            interaction_id=data["interaction_id"],
            project_id=data["project_id"],
            agent_id=data["agent_id"],
            session_id=data["session_id"],
            status=InteractionStatus(data["status"]),
            response_event_id=data["response_event_id"],
            completion_event_id=data.get("completion_event_id"),
            extracted_memories=tuple(Memory.from_dict(m) for m in data.get("extracted_memories", ())),
            update_results=tuple(update_result_from_dict(r) for r in data.get("update_results", ())),
            conflicts_detected=tuple(MemoryConflict.from_dict(c) for c in data.get("conflicts_detected", ())),
            graph_nodes_integrated=data.get("graph_nodes_integrated", 0),
            vectors_indexed=data.get("vectors_indexed", 0),
            warnings=tuple(data.get("warnings", ())),
            is_partial_success=data.get("is_partial_success", False),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "InteractionCompletionResult":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True, kw_only=True)
class InteractionRecord:
    """
    Internal tracking record preserving interaction state across lifecycle stages.
    """
    interaction_id: str
    project_id: str
    agent_id: str
    session_id: str
    input: AgentInteractionInput
    status: InteractionStatus
    idempotency_key: Optional[str] = None
    current_stage: str = LoopStage.INITIALIZED.value
    stages_completed: tuple[str, ...] = field(default_factory=tuple)
    input_event_id: Optional[str] = None
    context_event_id: Optional[str] = None
    context_result: Optional[InteractionContextResult] = None
    response: Optional[AgentResponseInput] = None
    response_event_id: Optional[str] = None
    completion_event_id: Optional[str] = None
    completion_result: Optional[InteractionCompletionResult] = None
    error_message: Optional[str] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages_completed", tuple(self.stages_completed))
        object.__setattr__(self, "warnings", tuple(self.warnings))

        if not isinstance(self.created_at, datetime):
            raise InputValidationError("created_at must be a datetime object")
        c_utc = (
            self.created_at.replace(tzinfo=timezone.utc)
            if self.created_at.tzinfo is None
            else self.created_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "created_at", c_utc)

        if not isinstance(self.updated_at, datetime):
            raise InputValidationError("updated_at must be a datetime object")
        u_utc = (
            self.updated_at.replace(tzinfo=timezone.utc)
            if self.updated_at.tzinfo is None
            else self.updated_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "updated_at", u_utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "idempotency_key": self.idempotency_key,
            "project_id": self.project_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "current_stage": self.current_stage,
            "stages_completed": list(self.stages_completed),
            "input": self.input.to_dict(),
            "input_event_id": self.input_event_id,
            "context_event_id": self.context_event_id,
            "context_result": self.context_result.to_dict() if self.context_result else None,
            "response": self.response.to_dict() if self.response else None,
            "response_event_id": self.response_event_id,
            "completion_event_id": self.completion_event_id,
            "completion_result": self.completion_result.to_dict() if self.completion_result else None,
            "error_message": self.error_message,
            "warnings": list(self.warnings),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionRecord":
        return cls(
            interaction_id=data["interaction_id"],
            idempotency_key=data.get("idempotency_key"),
            project_id=data["project_id"],
            agent_id=data["agent_id"],
            session_id=data["session_id"],
            status=InteractionStatus(data["status"]),
            current_stage=data.get("current_stage", LoopStage.INITIALIZED.value),
            stages_completed=tuple(data.get("stages_completed", ())),
            input=AgentInteractionInput.from_dict(data["input"]),
            input_event_id=data.get("input_event_id"),
            context_event_id=data.get("context_event_id"),
            context_result=InteractionContextResult.from_dict(data["context_result"]) if data.get("context_result") else None,
            response=AgentResponseInput.from_dict(data["response"]) if data.get("response") else None,
            response_event_id=data.get("response_event_id"),
            completion_event_id=data.get("completion_event_id"),
            completion_result=InteractionCompletionResult.from_dict(data["completion_result"]) if data.get("completion_result") else None,
            error_message=data.get("error_message"),
            warnings=tuple(data.get("warnings", ())),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "InteractionRecord":
        return cls.from_dict(json.loads(json_str))
