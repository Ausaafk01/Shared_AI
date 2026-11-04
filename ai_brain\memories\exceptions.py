"""
Exceptions for the AI Brain Memory System.
"""


class MemorySystemError(Exception):
    """Base exception for all memory system errors."""
    pass


class MemoryValidationError(MemorySystemError):
    """Raised when a memory fails validation (e.g. invalid type, out-of-range value, inconsistent timestamps)."""
    pass


class MemoryNotFoundError(MemorySystemError):
    """Raised when a specific memory cannot be found."""
    pass


class ExtractionError(MemorySystemError):
    """Base exception for extraction-related errors."""
    pass


class InvalidEventError(ExtractionError):
    """Raised when an invalid or malformed event is provided for extraction."""
    pass


class ExtractionRuleError(ExtractionError):
    """Raised when an extraction rule encounters an unexpected failure."""
    pass


class MemoryUpdateError(MemorySystemError):
    """Base exception for memory update errors."""
    pass


class InvalidUpdateOperationError(MemoryUpdateError):
    """Raised when an update operation is invalid (e.g. mismatched project, invalid version transition)."""
    pass


class VersionConflictError(MemoryUpdateError):
    """Raised when a version conflict occurs during memory update."""
    pass


class ConflictError(MemorySystemError):
    """Base exception for conflict management errors."""
    pass


class ConflictValidationError(ConflictError):
    """Raised when a conflict model or operation fails validation."""
    pass


class ConflictResolutionError(ConflictError):
    """Raised when a conflict resolution operation fails."""
    pass


class InvalidConflictStateError(ConflictError):
    """Raised when an illegal conflict lifecycle state transition is attempted."""
    pass
