"""
Data models for session continuity and management in AI Brain.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Union
import uuid

from ai_brain.agents.exceptions import SessionError
from ai_brain.loop.models import _freeze_container, _json_serializer, _unfreeze_container


class SessionStatus(StrEnum):
    """Lifecycle statuses for an active or historical agent session."""
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ARCHIVED = "ARCHIVED"


@dataclass(frozen=True, kw_only=True)
class SessionRecord:
    """
    Immutable representation of an agent session within a project.
    """
    session_id: str
    project_id: str
    title: str = ""
    current_agent_id: Optional[str] = None
    participating_agent_ids: tuple[str, ...] = field(default_factory=tuple)
    status: SessionStatus = SessionStatus.ACTIVE
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.session_id or not isinstance(self.session_id, str) or not self.session_id.strip():
            raise SessionError("session_id must be a non-empty string")
        object.__setattr__(self, "session_id", self.session_id.strip())

        if not self.project_id or not isinstance(self.project_id, str) or not self.project_id.strip():
            raise SessionError("project_id must be a non-empty string")
        object.__setattr__(self, "project_id", self.project_id.strip())

        object.__setattr__(self, "title", self.title.strip() if self.title else f"Session {self.session_id[:8]}")

        if self.current_agent_id is not None:
            object.__setattr__(self, "current_agent_id", str(self.current_agent_id).strip())

        part_agents = list(self.participating_agent_ids)
        if self.current_agent_id and self.current_agent_id not in part_agents:
            part_agents.append(self.current_agent_id)
        object.__setattr__(self, "participating_agent_ids", tuple(part_agents))

        if not isinstance(self.status, SessionStatus):
            try:
                object.__setattr__(self, "status", SessionStatus(str(self.status).upper()))
            except ValueError:
                object.__setattr__(self, "status", SessionStatus.ACTIVE)

        object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))

        if not isinstance(self.created_at, datetime):
            raise SessionError("created_at must be a datetime object")
        c_utc = (
            self.created_at.replace(tzinfo=timezone.utc)
            if self.created_at.tzinfo is None
            else self.created_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "created_at", c_utc)

        if not isinstance(self.updated_at, datetime):
            raise SessionError("updated_at must be a datetime object")
        u_utc = (
            self.updated_at.replace(tzinfo=timezone.utc)
            if self.updated_at.tzinfo is None
            else self.updated_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "updated_at", u_utc)

    def with_agent(self, new_agent_id: str) -> "SessionRecord":
        """Return updated session with new current agent and updated participants."""
        clean_new = new_agent_id.strip()
        part = list(self.participating_agent_ids)
        if clean_new not in part:
            part.append(clean_new)
        return SessionRecord(
            session_id=self.session_id,
            project_id=self.project_id,
            title=self.title,
            current_agent_id=clean_new,
            participating_agent_ids=tuple(part),
            status=SessionStatus.ACTIVE,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def with_status(self, status: SessionStatus) -> "SessionRecord":
        """Return updated session with new lifecycle status."""
        return SessionRecord(
            session_id=self.session_id,
            project_id=self.project_id,
            title=self.title,
            current_agent_id=self.current_agent_id,
            participating_agent_ids=self.participating_agent_ids,
            status=status,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "title": self.title,
            "current_agent_id": self.current_agent_id,
            "participating_agent_ids": list(self.participating_agent_ids),
            "status": self.status.value,
            "metadata": _unfreeze_container(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRecord":
        return cls(
            session_id=data["session_id"],
            project_id=data["project_id"],
            title=data.get("title", ""),
            current_agent_id=data.get("current_agent_id"),
            participating_agent_ids=tuple(data.get("participating_agent_ids", ())),
            status=SessionStatus(data.get("status", SessionStatus.ACTIVE.value)),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SessionRecord":
        return cls.from_dict(json.loads(json_str))

