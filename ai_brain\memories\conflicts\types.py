"""
Conflict types, statuses, and validation for AI Brain Conflict Management.
"""

from enum import StrEnum
from typing import Set

from ai_brain.memories.exceptions import ConflictValidationError, InvalidConflictStateError


class ConflictType(StrEnum):
    """Categories of disagreement between Memories."""
    FACTUAL = "FACTUAL"        # Two factual claims disagree
    TEMPORAL = "TEMPORAL"      # Claims differ because they describe different points in time
    DECISION = "DECISION"      # Agents or users record incompatible decisions
    STATUS = "STATUS"          # Memories disagree about whether a task/issue is active or completed
    TECHNICAL = "TECHNICAL"    # Conflicting technical configurations or implementations
    SOURCE = "SOURCE"          # Different sources provide incompatible information


class ConflictStatus(StrEnum):
    """Lifecycle states of a MemoryConflict."""
    DETECTED = "DETECTED"          # Conflict identified
    UNDER_REVIEW = "UNDER_REVIEW"  # Evidence is being evaluated
    RESOLVED = "RESOLVED"          # Sufficient evidence supports a resolution
    UNRESOLVED = "UNRESOLVED"      # Available evidence is insufficient
    DISMISSED = "DISMISSED"        # Disagreement was not a genuine conflict


# Dynamic registry for extending conflict types
_CUSTOM_CONFLICT_TYPES: Set[str] = set()


def register_conflict_type(name: str) -> str:
    """Register a custom conflict type."""
    if not isinstance(name, str):
        raise ValueError(f"Conflict type must be a string, got {type(name).__name__}")
    clean = name.strip().upper()
    if not clean or not clean.replace("_", "").isalnum():
        raise ValueError(f"Invalid conflict type name '{name}'")
    _CUSTOM_CONFLICT_TYPES.add(clean)
    return clean


def unregister_conflict_type(name: str) -> None:
    """Unregister a custom conflict type (for testing)."""
    clean = name.strip().upper() if isinstance(name, str) else name
    _CUSTOM_CONFLICT_TYPES.discard(clean)


def is_valid_conflict_type(conflict_type: str | ConflictType) -> bool:
    """Check if a conflict type is valid."""
    if isinstance(conflict_type, ConflictType):
        return True
    if not isinstance(conflict_type, str):
        return False
    clean = conflict_type.strip().upper()
    try:
        ConflictType(clean)
        return True
    except ValueError:
        return clean in _CUSTOM_CONFLICT_TYPES


def normalize_conflict_type(conflict_type: str | ConflictType) -> str:
    """Normalize and validate a conflict type."""
    if isinstance(conflict_type, ConflictType):
        return conflict_type.value
    if not isinstance(conflict_type, str):
        raise ConflictValidationError(f"Expected ConflictType or str, got {type(conflict_type).__name__}")
    clean = conflict_type.strip().upper()
    try:
        return ConflictType(clean).value
    except ValueError:
        if clean in _CUSTOM_CONFLICT_TYPES:
            return clean
        raise ConflictValidationError(f"Unknown conflict type: '{conflict_type}'")


def normalize_conflict_status(status: str | ConflictStatus) -> str:
    """Normalize and validate a conflict status."""
    if isinstance(status, ConflictStatus):
        return status.value
    if not isinstance(status, str):
        raise ConflictValidationError(f"Expected ConflictStatus or str, got {type(status).__name__}")
    clean = status.strip().upper()
    try:
        return ConflictStatus(clean).value
    except ValueError:
        valid_statuses = ", ".join(s.value for s in ConflictStatus)
        raise ConflictValidationError(f"Invalid conflict status '{status}'. Valid statuses: {valid_statuses}")


# Controlled lifecycle state transitions
VALID_TRANSITIONS = {
    ConflictStatus.DETECTED: {
        ConflictStatus.UNDER_REVIEW,
        ConflictStatus.RESOLVED,
        ConflictStatus.UNRESOLVED,
        ConflictStatus.DISMISSED,
    },
    ConflictStatus.UNDER_REVIEW: {
        ConflictStatus.RESOLVED,
        ConflictStatus.UNRESOLVED,
        ConflictStatus.DISMISSED,
    },
    ConflictStatus.UNRESOLVED: {
        ConflictStatus.UNDER_REVIEW,
        ConflictStatus.RESOLVED,
        ConflictStatus.DISMISSED,
    },
    ConflictStatus.RESOLVED: {
        ConflictStatus.UNDER_REVIEW,  # Can reopen if conflicting evidence emerges
    },
    ConflictStatus.DISMISSED: {
        ConflictStatus.UNDER_REVIEW,
        ConflictStatus.DETECTED,
    },
}


def validate_status_transition(current: str | ConflictStatus, target: str | ConflictStatus) -> None:
    """Ensure that the transition from current status to target status is valid."""
    c_status = ConflictStatus(normalize_conflict_status(current))
    t_status = ConflictStatus(normalize_conflict_status(target))

    if c_status == t_status:
        return

    allowed = VALID_TRANSITIONS.get(c_status, set())
    if t_status not in allowed:
        raise InvalidConflictStateError(
            f"Illegal conflict state transition from '{c_status.value}' to '{t_status.value}'"
        )

