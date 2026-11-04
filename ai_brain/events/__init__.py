"""
AI Brain Event System package.
"""

from typing import Any, Mapping, Optional, Sequence
from datetime import datetime

from .exceptions import (
    DuplicateEventError,
    EventNotFoundError,
    EventSystemError,
    EventValidationError,
    StorageCorruptionError,
    StorageError,
)
from .models import Event
from .storage.base import EventStore
from .storage.sqlite import SQLiteEventStore
from .types import (
    EventType,
    is_valid_event_type,
    normalize_event_type,
    register_event_type,
    unregister_event_type,
)


def create_event(
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
) -> Event:
    """
    Convenience function to create an immutable Event instance.
    """
    return Event.create(
        event_type=event_type,
        project_id=project_id,
        session_id=session_id,
        actor=actor,
        event_id=event_id,
        timestamp=timestamp,
        agent_id=agent_id,
        input=input,
        output=output,
        tool_used=tool_used,
        files_involved=files_involved,
        result=result,
        metadata=metadata,
    )


__all__ = [
    "Event",
    "EventType",
    "create_event",
    "register_event_type",
    "unregister_event_type",
    "is_valid_event_type",
    "normalize_event_type",
    "EventStore",
    "SQLiteEventStore",
    "EventSystemError",
    "EventValidationError",
    "DuplicateEventError",
    "EventNotFoundError",
    "StorageError",
    "StorageCorruptionError",
]

