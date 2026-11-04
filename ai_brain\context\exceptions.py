"""
Exception hierarchy for the Context Builder subsystem.
"""


class ContextError(Exception):
    """Base exception for all Context Builder operations."""


class ContextBuilderError(ContextError):
    """Raised when context building cannot be completed."""


class TokenBudgetExceededError(ContextError):
    """Raised when a strict token budget cannot accommodate mandatory context."""


class SelectionError(ContextError):
    """Raised when memory selection criteria are invalid or encounter an error."""


class PrioritizationError(ContextError):
    """Raised when memory prioritization encounters an error."""


class CompressionError(ContextError):
    """Raised when context compression encounters an error."""


class FormattingError(ContextError):
    """Raised when formatting context output fails."""


class TokenizerError(ContextError):
    """Raised when token counting or tokenization fails."""


class ContextProjectIsolationError(ContextError):
    """Raised when cross-project memories are supplied to Context Builder."""

