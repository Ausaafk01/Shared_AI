"""
Event types and extensible registry for the AI Brain Event System.
"""

from enum import StrEnum
from typing import Set


class EventType(StrEnum):
    """Core categories of events in the AI Brain event stream."""
    USER_MESSAGE = "USER_MESSAGE"
    AGENT_MESSAGE = "AGENT_MESSAGE"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    CODE_CHANGE = "CODE_CHANGE"
    FILE_CHANGE = "FILE_CHANGE"
    ERROR = "ERROR"
    TASK_CREATED = "TASK_CREATED"
    TASK_COMPLETED = "TASK_COMPLETED"
    DECISION = "DECISION"
    AGENT_SWITCH = "AGENT_SWITCH"
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"


# Dynamic registry to allow extensions without modifying the core enum
_CUSTOM_EVENT_TYPES: Set[str] = set()


def register_event_type(name: str) -> str:
    """
    Register a custom event type for future or domain-specific needs.
    
    Args:
        name: Name of the event type (e.g. 'MEMORY_SNAPSHOT', 'CHECKPOINT').
              Will be converted to upper-case stripped string.
              
    Returns:
        The normalized event type name.
        
    Raises:
        ValueError: If the name is empty or contains invalid characters.
    """
    if not isinstance(name, str):
        raise ValueError(f"Event type must be a string, got {type(name).__name__}")
    
    clean_name = name.strip().upper()
    if not clean_name:
        raise ValueError("Event type cannot be empty")
    
    # Must be valid identifier-like (alphanumeric with underscores)
    if not clean_name.replace("_", "").isalnum():
        raise ValueError(
            f"Event type '{clean_name}' must contain only letters, numbers, and underscores"
        )
    
    _CUSTOM_EVENT_TYPES.add(clean_name)
    return clean_name


def unregister_event_type(name: str) -> None:
    """Remove a previously registered custom event type (primarily for testing)."""
    clean_name = name.strip().upper() if isinstance(name, str) else name
    _CUSTOM_EVENT_TYPES.discard(clean_name)


def is_valid_event_type(event_type: str | EventType) -> bool:
    """Check whether a given string or EventType is registered or predefined."""
    if isinstance(event_type, EventType):
        return True
    if not isinstance(event_type, str):
        return False
    
    clean_name = event_type.strip().upper()
    try:
        EventType(clean_name)
        return True
    except ValueError:
        return clean_name in _CUSTOM_EVENT_TYPES


def normalize_event_type(event_type: str | EventType) -> str:
    """
    Normalize and validate an event type.
    
    Returns:
        The string representation of the event type.
        
    Raises:
        ValueError: If event_type is not a recognized or registered event type.
    """
    if isinstance(event_type, EventType):
        return event_type.value
    
    if not isinstance(event_type, str):
        raise ValueError(
            f"Event type must be EventType or str, got {type(event_type).__name__}"
        )
    
    clean = event_type.strip().upper()
    if not clean:
        raise ValueError("Event type cannot be empty")
        
    try:
        return EventType(clean).value
    except ValueError:
        if clean in _CUSTOM_EVENT_TYPES:
            return clean
        raise ValueError(
            f"Unknown event type: '{event_type}'. "
            f"Use a standard EventType or register it via register_event_type()."
        )

