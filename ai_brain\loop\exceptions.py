"""
Exception hierarchy for the Brain Loop subsystem.
"""


class BrainLoopError(Exception):
    """Base exception for all Brain Loop orchestration operations."""


class InputValidationError(BrainLoopError):
    """Raised when an incoming agent interaction input is invalid or malformed."""


class ResponseValidationError(BrainLoopError):
    """Raised when an agent response is mismatched, invalid, or malformed."""


class InteractionNotFoundError(BrainLoopError):
    """Raised when a referenced interaction ID does not exist."""


class InteractionStateError(BrainLoopError):
    """Raised when an interaction lifecycle transition is invalid."""


class LoopProjectIsolationError(BrainLoopError):
    """Raised when a cross-project reference or mismatch is detected."""


class EventPersistenceError(BrainLoopError):
    """Raised when event generation or persistence fails."""


class RetrievalStageError(BrainLoopError):
    """Raised when retrieval fails during context preparation."""


class ContextStageError(BrainLoopError):
    """Raised when context building fails during context preparation."""


class ExtractionStageError(BrainLoopError):
    """Raised when memory extraction fails during interaction completion."""


class MemoryUpdateStageError(BrainLoopError):
    """Raised when memory updating or conflict processing fails."""


class GraphIntegrationStageError(BrainLoopError):
    """Raised when knowledge graph integration fails."""


class VectorIntegrationStageError(BrainLoopError):
    """Raised when vector memory indexing fails."""
