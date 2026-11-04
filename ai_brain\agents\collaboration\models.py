"""
Models for multi-agent collaboration workflows in AI Brain.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Union
import uuid

from ai_brain.agents.exceptions import CollaborationWorkflowError
from ai_brain.agents.models import AgentExecutionResponse
from ai_brain.loop.models import _freeze_container, _json_serializer, _unfreeze_container


class WorkflowStatus(StrEnum):
    """Execution status of a multi-agent collaboration workflow."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"


@dataclass(frozen=True, kw_only=True)
class CollaborationStep:
    """A single ordered step in a multi-agent collaboration workflow."""
    step_index: int
    name: str
    agent_id: str
    instruction: str
    query: Optional[str] = None
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "name": self.name,
            "agent_id": self.agent_id,
            "instruction": self.instruction,
            "query": self.query,
            "required": self.required,
            "metadata": _unfreeze_container(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CollaborationStep":
        return cls(
            step_index=data["step_index"],
            name=data["name"],
            agent_id=data["agent_id"],
            instruction=data["instruction"],
            query=data.get("query"),
            required=data.get("required", True),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, kw_only=True)
class CollaborationStepResult:
    """Execution outcome of an individual collaboration step."""
    step_index: int
    name: str
    agent_id: str
    status: str  # "COMPLETED", "FAILED", "SKIPPED"
    output_content: str = ""
    interaction_id: Optional[str] = None
    memories_created_count: int = 0
    conflicts_detected_count: int = 0
    error_message: Optional[str] = None
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "name": self.name,
            "agent_id": self.agent_id,
            "status": self.status,
            "output_content": self.output_content,
            "interaction_id": self.interaction_id,
            "memories_created_count": self.memories_created_count,
            "conflicts_detected_count": self.conflicts_detected_count,
            "error_message": self.error_message,
            "executed_at": self.executed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CollaborationStepResult":
        return cls(
            step_index=data["step_index"],
            name=data["name"],
            agent_id=data["agent_id"],
            status=data["status"],
            output_content=data.get("output_content", ""),
            interaction_id=data.get("interaction_id"),
            memories_created_count=data.get("memories_created_count", 0),
            conflicts_detected_count=data.get("conflicts_detected_count", 0),
            error_message=data.get("error_message"),
            executed_at=datetime.fromisoformat(data["executed_at"]) if "executed_at" in data else datetime.now(timezone.utc),
        )


@dataclass(frozen=True, kw_only=True)
class CollaborationWorkflowRecord:
    """Full lifecycle tracking record for a multi-agent collaboration."""
    workflow_id: str
    project_id: str
    session_id: str
    task_description: str
    steps: tuple[CollaborationStep, ...]
    completed_steps: tuple[CollaborationStepResult, ...] = field(default_factory=tuple)
    status: WorkflowStatus = WorkflowStatus.PENDING
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "completed_steps", tuple(self.completed_steps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "task_description": self.task_description,
            "steps": [s.to_dict() for s in self.steps],
            "completed_steps": [s.to_dict() for s in self.completed_steps],
            "status": self.status.value,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CollaborationWorkflowRecord":
        return cls(
            workflow_id=data["workflow_id"],
            project_id=data["project_id"],
            session_id=data["session_id"],
            task_description=data["task_description"],
            steps=tuple(CollaborationStep.from_dict(s) for s in data["steps"]),
            completed_steps=tuple(CollaborationStepResult.from_dict(s) for s in data.get("completed_steps", ())),
            status=WorkflowStatus(data.get("status", WorkflowStatus.PENDING.value)),
            error_message=data.get("error_message"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CollaborationWorkflowRecord":
        return cls.from_dict(json.loads(json_str))

