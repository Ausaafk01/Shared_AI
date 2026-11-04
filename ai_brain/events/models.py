"""
Event model for the AI Brain Event System.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence
import uuid

from .exceptions import EventValidationError
from .types import EventType, is_valid_event_type, normalize_event_type


def _json_serializer(obj: Any) -> Any:
    """Helper serializer for JSON conversion of non-standard objects."""
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, EventType):
        return obj.value
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, (MappingProxyType, Mapping)):
        return dict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _freeze_container(data: Any) -> Any:
    """Recursively freeze dicts into MappingProxyType and lists into tuples for deep immutability."""
    if isinstance(data, dict):
        return MappingProxyType({k: _freeze_container(v) for k, v in data.items()})
    elif isinstance(data, list):
        return tuple(_freeze_container(v) for v in data)
    elif isinstance(data, set):
        return frozenset(_freeze_container(v) for v in data)
    return data


def _unfreeze_container(data: Any) -> Any:
    """Recursively convert frozen containers back into standard dicts and lists for serialization."""
    if isinstance(data, (MappingProxyType, dict)):
        return {k: _unfreeze_container(v) for k, v in data.items()}
    elif isinstance(data, (tuple, list, frozenset, set)):
        return [_unfreeze_container(v) for v in data]
    return data


@dataclass(frozen=True, kw_only=True)
class Event:
    """
    Immutable representation of an event recorded in the AI Brain.
    
    All attributes are read-only once instantiated. Attempting to assign
    or delete attributes will raise a FrozenInstanceError (AttributeError).
    Containers like metadata and files_involved are also guarded against mutation.
    """
    event_id: str
    event_type: str
    timestamp: datetime
    project_id: str
    agent_id: Optional[str] = None
    session_id: str
    actor: str
    input: Optional[Any] = None
    output: Optional[Any] = None
    tool_used: Optional[str] = None
    files_involved: tuple[str, ...] = field(default_factory=tuple)
    result: Optional[Any] = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Validate all fields and enforce immutability."""
        # 1. Validate and normalize event_id
        if not self.event_id or not isinstance(self.event_id, str) or not self.event_id.strip():
            raise EventValidationError("event_id must be a non-empty string")
        object.__setattr__(self, "event_id", self.event_id.strip())

        # 2. Validate and normalize event_type
        try:
            normalized_type = normalize_event_type(self.event_type)
            object.__setattr__(self, "event_type", normalized_type)
        except ValueError as e:
            raise EventValidationError(str(e)) from e

        # 3. Validate and enforce UTC timestamp
        if not isinstance(self.timestamp, datetime):
            raise EventValidationError(
                f"timestamp must be a datetime object, got {type(self.timestamp).__name__}"
            )
        
        if self.timestamp.tzinfo is None:
            # Naive datetime: assume UTC
            utc_timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        else:
            # Aware datetime: normalize to UTC
            utc_timestamp = self.timestamp.astimezone(timezone.utc)
        object.__setattr__(self, "timestamp", utc_timestamp)

        # 4. Validate project_id
        if not self.project_id or not isinstance(self.project_id, str) or not self.project_id.strip():
            raise EventValidationError("project_id must be a non-empty string")
        object.__setattr__(self, "project_id", self.project_id.strip())

        # 5. Validate session_id
        if not self.session_id or not isinstance(self.session_id, str) or not self.session_id.strip():
            raise EventValidationError("session_id must be a non-empty string")
        object.__setattr__(self, "session_id", self.session_id.strip())

        # 6. Validate actor
        if not self.actor or not isinstance(self.actor, str) or not self.actor.strip():
            raise EventValidationError("actor must be a non-empty string")
        object.__setattr__(self, "actor", self.actor.strip())

        # 7. Validate agent_id (optional)
        if self.agent_id is not None:
            if not isinstance(self.agent_id, str) or not self.agent_id.strip():
                raise EventValidationError("agent_id must be a non-empty string if provided")
            object.__setattr__(self, "agent_id", self.agent_id.strip())

        # 8. Validate tool_used (optional)
        if self.tool_used is not None:
            if not isinstance(self.tool_used, str) or not self.tool_used.strip():
                raise EventValidationError("tool_used must be a non-empty string if provided")
            object.__setattr__(self, "tool_used", self.tool_used.strip())

        # 9. Normalize files_involved into immutable tuple of strings
        if self.files_involved is None:
            object.__setattr__(self, "files_involved", ())
        elif isinstance(self.files_involved, (list, tuple, set)):
            files = []
            for item in self.files_involved:
                if not isinstance(item, str):
                    raise EventValidationError(
                        f"All files_involved must be strings, got {type(item).__name__}"
                    )
                files.append(item)
            object.__setattr__(self, "files_involved", tuple(files))
        else:
            raise EventValidationError("files_involved must be a sequence of strings")

        # 10. Deep freeze metadata
        if self.metadata is None:
            object.__setattr__(self, "metadata", MappingProxyType({}))
        elif isinstance(self.metadata, (dict, MappingProxyType)):
            frozen_meta = _freeze_container(deepcopy(dict(self.metadata)))
            object.__setattr__(self, "metadata", frozen_meta)
        else:
            raise EventValidationError("metadata must be a dictionary or mapping")

        # 11. Deep freeze input, output, result if they are collections
        if self.input is not None:
            object.__setattr__(self, "input", _freeze_container(deepcopy(self.input)))
        if self.output is not None:
            object.__setattr__(self, "output", _freeze_container(deepcopy(self.output)))
        if self.result is not None:
            object.__setattr__(self, "result", _freeze_container(deepcopy(self.result)))

    @classmethod
    def create(
        cls,
        *,
        event_type: str | EventType,
        project_id: str,
        session_id: str,
        actor: str,
        event_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        agent_id: Optional[str] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        tool_used: Optional[str] = None,
        files_involved: Optional[Sequence[str]] = None,
        result: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "Event":
        """
        Factory method to create a validated immutable Event.
        
        Automatically generates a UUID v4 event_id if not supplied,
        and defaults timestamp to current UTC time.
        """
        return cls(
            event_id=event_id or str(uuid.uuid4()),
            event_type=event_type,
            timestamp=timestamp or datetime.now(timezone.utc),
            project_id=project_id,
            agent_id=agent_id,
            session_id=session_id,
            actor=actor,
            input=input,
            output=output,
            tool_used=tool_used,
            files_involved=tuple(files_involved or ()),
            result=result,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the Event into a standard Python dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "project_id": self.project_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "actor": self.actor,
            "input": _unfreeze_container(self.input),
            "output": _unfreeze_container(self.output),
            "tool_used": self.tool_used,
            "files_involved": list(self.files_involved),
            "result": _unfreeze_container(self.result),
            "metadata": _unfreeze_container(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """
        Reconstruct an Event instance from a dictionary.
        
        Handles ISO format timestamp strings and validates data structure.
        """
        if not isinstance(data, dict):
            raise EventValidationError(f"Expected dict for Event deserialization, got {type(data).__name__}")

        raw_data = dict(data)
        
        # Parse timestamp if it is a string
        ts = raw_data.get("timestamp")
        if isinstance(ts, str):
            try:
                # Handle 'Z' suffix in ISO-8601 strings
                normalized_ts = ts.replace("Z", "+00:00")
                parsed_ts = datetime.fromisoformat(normalized_ts)
                raw_data["timestamp"] = parsed_ts
            except Exception as e:
                raise EventValidationError(f"Invalid timestamp format '{ts}': {e}") from e

        # Ensure files_involved is converted to a tuple if present
        if "files_involved" in raw_data and raw_data["files_involved"] is not None:
            raw_data["files_involved"] = tuple(raw_data["files_involved"])

        try:
            return cls(**raw_data)
        except TypeError as e:
            raise EventValidationError(f"Invalid fields in event dictionary: {e}") from e

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize event to a JSON string."""
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Deserialize an Event from a JSON string."""
        if not isinstance(json_str, (str, bytes)):
            raise EventValidationError(f"Expected JSON string, got {type(json_str).__name__}")
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise EventValidationError(f"Malformed JSON: {e}") from e
        return cls.from_dict(data)

