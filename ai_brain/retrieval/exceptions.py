"""
Exception hierarchy for the AI Brain Retrieval Engine subsystem.
"""


class RetrievalError(Exception):
    """Base exception for all retrieval operations."""


class QueryValidationError(RetrievalError):
    """Raised when query text or filters are invalid, malformed, or empty."""


class ProjectIsolationError(RetrievalError):
    """Raised when cross-project retrieval or invalid project scope is detected."""


class RetrievalExecutionError(RetrievalError):
    """Raised when retrieval execution encounters a system or processing error."""


class GraphRetrievalError(RetrievalExecutionError):
    """Raised when graph-based retrieval fails."""


class VectorRetrievalError(RetrievalExecutionError):
    """Raised when vector-based retrieval fails."""


class HybridFusionError(RetrievalExecutionError):
    """Raised when hybrid ranking or fusion encounters an error."""


class ScoringError(RetrievalExecutionError):
    """Raised when memory relevance scoring encounters an error."""

