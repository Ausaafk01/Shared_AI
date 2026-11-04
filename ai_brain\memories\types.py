"""
Memory types, statuses, and extensible registry for the AI Brain Memory System.
"""

from enum import StrEnum
from typing import Set


class MemoryType(StrEnum):
    """Core categories of established knowledge in the AI Brain."""
    FACT = "FACT"
    DECISION = "DECISION"
    TASK = "TASK"
    ERROR = "ERROR"
    ATTEMPT = "ATTEMPT"
    CODE_KNOWLEDGE = "CODE_KNOWLEDGE"
    PREFERENCE = "PREFERENCE"
    CONTEXT = "CONTEXT"


class MemoryStatus(StrEnum):
    """Lifecycle states of a memory in the AI Brain."""
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    DISPUTED = "DISPUTED"
    ARCHIVED = "ARCHIVED"


# Dynamic registry to allow extensions without modifying core enum
_CUSTOM_MEMORY_TYPES: Set[str] = set()


def register_memory_type(name: str) -> str:
    """
    Register a custom memory type for domain-specific knowledge representations.
    
    Args:
        name: Name of the memory type (e.g. 'HEURISTIC', 'ASSUMPTION').
              Will be converted to upper-case stripped string.
              
    Returns:
        The normalized memory type name.
        
    Raises:
        ValueError: If name is empty or contains invalid characters.
    """
    if not isinstance(name, str):
        raise ValueError(f"Memory type must be a string, got {type(name).__name__}")
    
    clean_name = name.strip().upper()
    if not clean_name:
        raise ValueError("Memory type cannot be empty")
    
    if not clean_name.replace("_", "").isalnum():
        raise ValueError(
            f"Memory type '{clean_name}' must contain only letters, numbers, and underscores"
        )
    
    _CUSTOM_MEMORY_TYPES.add(clean_name)
    return clean_name


def unregister_memory_type(name: str) -> None:
    """Remove a previously registered custom memory type (primarily for testing)."""
    clean_name = name.strip().upper() if isinstance(name, str) else name
    _CUSTOM_MEMORY_TYPES.discard(clean_name)


def is_valid_memory_type(memory_type: str | MemoryType) -> bool:
    """Check whether a given string or MemoryType is registered or predefined."""
    if isinstance(memory_type, MemoryType):
        return True
    if not isinstance(memory_type, str):
        return False
    
    clean_name = memory_type.strip().upper()
    try:
        MemoryType(clean_name)
        return True
    except ValueError:
        return clean_name in _CUSTOM_MEMORY_TYPES


def normalize_memory_type(memory_type: str | MemoryType) -> str:
    """
    Normalize and validate a memory type.
    
    Returns:
        The uppercase string value of the memory type.
        
    Raises:
        ValueError: If memory_type is not a recognized or registered memory type.
    """
    if isinstance(memory_type, MemoryType):
        return memory_type.value
    
    if not isinstance(memory_type, str):
        raise ValueError(
            f"Memory type must be MemoryType or str, got {type(memory_type).__name__}"
        )
    
    clean = memory_type.strip().upper()
    if not clean:
        raise ValueError("Memory type cannot be empty")
        
    try:
        return MemoryType(clean).value
    except ValueError:
        if clean in _CUSTOM_MEMORY_TYPES:
            return clean
        raise ValueError(
            f"Unknown memory type: '{memory_type}'. "
            f"Use a standard MemoryType or register it via register_memory_type()."
        )


def is_valid_memory_status(status: str | MemoryStatus) -> bool:
    """Check whether a given string or MemoryStatus is a valid status value."""
    if isinstance(status, MemoryStatus):
        return True
    if not isinstance(status, str):
        return False
    try:
        MemoryStatus(status.strip().upper())
        return True
    except ValueError:
        return False


def normalize_memory_status(status: str | MemoryStatus) -> str:
    """
    Normalize and validate a memory status.
    
    Returns:
        The uppercase string value of the memory status.
        
    Raises:
        ValueError: If status is not a valid MemoryStatus.
    """
    if isinstance(status, MemoryStatus):
        return status.value
    
    if not isinstance(status, str):
        raise ValueError(
            f"Memory status must be MemoryStatus or str, got {type(status).__name__}"
        )
    
    clean = status.strip().upper()
    if not clean:
        raise ValueError("Memory status cannot be empty")
        
    try:
        return MemoryStatus(clean).value
    except ValueError:
        valid_statuses = ", ".join(s.value for s in MemoryStatus)
        raise ValueError(
            f"Invalid memory status: '{status}'. Valid statuses are: {valid_statuses}."
        )

