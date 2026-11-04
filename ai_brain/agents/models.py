"""
Immutable data models for the Multi-Agent System in AI Brain.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Union
import uuid

from ai_brain.agents.exceptions import AgentRegistrationError
from ai_brain.context.models import ContextPackage
from ai_brain.loop.models import InteractionCompletionResult, _freeze_container, _json_serializer, _unfreeze_container


class AgentType(StrEnum):
    """Categorization of agent specializations."""
    RESEARCH = "RESEARCH"
    CODING = "CODING"
    REVIEW = "REVIEW"
    GENERAL = "GENERAL"
    CUSTOM = "CUSTOM"


class AgentStatus(StrEnum):
    """Lifecycle statuses for an agent."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    BUSY = "BUSY"
    ERROR = "ERROR"
    DECOMMISSIONED = "DECOMMISSIONED"


class AgentCapability(StrEnum):
    """Core capabilities an agent may declare."""
    INFORMATION_GATHERING = "INFORMATION_GATHERING"
    CODE_DEVELOPMENT = "CODE_DEVELOPMENT"
    CODE_REVIEW = "CODE_REVIEW"
    ARCHITECTURE_DESIGN = "ARCHITECTURE_DESIGN"
    TASK_MANAGEMENT = "TASK_MANAGEMENT"
    REASONING = "REASONING"
    CONTRADICTION_DETECTION = "CONTRADICTION_DETECTION"


# Keys prohibited from being stored in ordinary agent configuration records
_PROHIBITED_SECRET_KEYS = frozenset({
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "password",
    "token",
    "access_token",
    "auth_token",
    "private_key",
})


def _validate_no_secrets(data: Any, path: str = "") -> None:
    """Recursively ensure no API keys or passwords are in ordinary configuration records."""
    if isinstance(data, (dict, Mapping, MappingProxyType)):
        for k, v in data.items():
            lower_k = str(k).lower().replace("-", "_").replace(" ", "_")
            current_path = f"{path}.{k}" if path else str(k)
            if any(secret_term == lower_k or lower_k.endswith(f"_{secret_term}") for secret_term in _PROHIBITED_SECRET_KEYS):
                raise AgentRegistrationError(
                    f"Prohibited sensitive credential '{current_path}' found in agent configuration. "
                    "API keys and secrets must remain outside Brain storage."
                )
            _validate_no_secrets(v, current_path)
    elif isinstance(data, (list, tuple, set, frozenset)):
        for item in data:
            _validate_no_secrets(item, path)


@dataclass(frozen=True, kw_only=True)
class AgentRecord:
    """
    Immutable representation of an agent's registered identity and metadata.
    """
    agent_id: str
    name: str
    provider: str
    agent_type: AgentType = AgentType.GENERAL
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    configuration: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    status: AgentStatus = AgentStatus.ACTIVE
    supported_interaction_methods: tuple[str, ...] = field(default_factory=lambda: ("sync", "prepare_and_complete"))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.agent_id or not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise AgentRegistrationError("agent_id must be a non-empty string")
        object.__setattr__(self, "agent_id", self.agent_id.strip())

        if not self.name or not isinstance(self.name, str) or not self.name.strip():
            raise AgentRegistrationError("name must be a non-empty string")
        object.__setattr__(self, "name", self.name.strip())

        if not self.provider or not isinstance(self.provider, str) or not self.provider.strip():
            raise AgentRegistrationError("provider must be a non-empty string")
        object.__setattr__(self, "provider", self.provider.strip())

        if not isinstance(self.agent_type, AgentType):
            try:
                object.__setattr__(self, "agent_type", AgentType(str(self.agent_type).upper()))
            except ValueError:
                object.__setattr__(self, "agent_type", AgentType.CUSTOM)

        if not isinstance(self.status, AgentStatus):
            try:
                object.__setattr__(self, "status", AgentStatus(str(self.status).upper()))
            except ValueError:
                object.__setattr__(self, "status", AgentStatus.INACTIVE)

        # Enforce that no secrets or API keys are stored in configuration
        _validate_no_secrets(self.configuration)

        object.__setattr__(self, "capabilities", tuple(str(c) for c in self.capabilities))
        object.__setattr__(self, "configuration", _freeze_container(_unfreeze_container(self.configuration)))
        object.__setattr__(self, "supported_interaction_methods", tuple(str(m) for m in self.supported_interaction_methods))

        if not isinstance(self.created_at, datetime):
            raise AgentRegistrationError("created_at must be a datetime object")
        c_utc = (
            self.created_at.replace(tzinfo=timezone.utc)
            if self.created_at.tzinfo is None
            else self.created_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "created_at", c_utc)

        if not isinstance(self.updated_at, datetime):
            raise AgentRegistrationError("updated_at must be a datetime object")
        u_utc = (
            self.updated_at.replace(tzinfo=timezone.utc)
            if self.updated_at.tzinfo is None
            else self.updated_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "updated_at", u_utc)

    def with_status(self, status: AgentStatus) -> "AgentRecord":
        """Return a copy of this record with an updated status."""
        return AgentRecord(
            agent_id=self.agent_id,
            name=self.name,
            provider=self.provider,
            agent_type=self.agent_type,
            capabilities=self.capabilities,
            configuration=self.configuration,
            status=status,
            supported_interaction_methods=self.supported_interaction_methods,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def with_configuration(self, configuration: Mapping[str, Any]) -> "AgentRecord":
        """Return a copy of this record with updated configuration."""
        _validate_no_secrets(configuration)
        return AgentRecord(
            agent_id=self.agent_id,
            name=self.name,
            provider=self.provider,
            agent_type=self.agent_type,
            capabilities=self.capabilities,
            configuration=configuration,
            status=self.status,
            supported_interaction_methods=self.supported_interaction_methods,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "provider": self.provider,
            "agent_type": self.agent_type.value,
            "capabilities": list(self.capabilities),
            "configuration": _unfreeze_container(self.configuration),
            "status": self.status.value,
            "supported_interaction_methods": list(self.supported_interaction_methods),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRecord":
        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            provider=data["provider"],
            agent_type=AgentType(data["agent_type"]),
            capabilities=tuple(data.get("capabilities", ())),
            configuration=data.get("configuration", {}),
            status=AgentStatus(data.get("status", AgentStatus.ACTIVE.value)),
            supported_interaction_methods=tuple(data.get("supported_interaction_methods", ("sync", "prepare_and_complete"))),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AgentRecord":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True, kw_only=True)
class AgentExecutionRequest:
    """
    Standardized invocation request sent to an agent.
    """
    project_id: str
    session_id: str
    content: str
    query: Optional[str] = None
    token_budget: Optional[int] = None
    idempotency_key: Optional[str] = None
    retrieval_method: str = "HYBRID"
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", self.project_id.strip())
        object.__setattr__(self, "session_id", self.session_id.strip())
        object.__setattr__(self, "content", self.content.strip())
        object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))


@dataclass(frozen=True, kw_only=True)
class AgentExecutionResponse:
    """
    Standardized response returned from an agent execution lifecycle.
    """
    agent_id: str
    project_id: str
    session_id: str
    interaction_id: str
    content: str
    context_package: Optional[ContextPackage] = None
    completion_result: Optional[InteractionCompletionResult] = None
    execution_time_ms: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))

