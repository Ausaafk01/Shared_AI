"""
Exceptions for the AI Brain Event System.
"""


class EventSystemError(Exception):
    """Base exception for all event system errors."""
    pass


class EventValidationError(EventSystemError):
    """Raised when an event fails validation (e.g. missing required field, invalid type)."""
    pass


class DuplicateEventError(EventSystemError):
    """Raised when attempting to store an event with an ID that already exists."""
    pass


class EventNotFoundError(EventSystemError):
    """Raised when a specific event cannot be found."""
    pass


class StorageError(EventSystemError):
    """Raised when a persistence or database operation fails."""
    pass


class StorageCorruptionError(StorageError):
    """Raised when stored event data is corrupt or cannot be deserialized."""
    pass

