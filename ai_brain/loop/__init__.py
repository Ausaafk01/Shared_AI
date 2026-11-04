"""
Brain Loop subsystem for AI Brain.

Orchestrates the repeatable interaction lifecycle across input processing,
retrieval, context assembly, response capture, event logging, memory updating,
and knowledge synchronization.
"""

from ai_brain.loop.event_handler import LoopEventHandler
from ai_brain.loop.exceptions import (
    BrainLoopError,
    ContextStageError,
    EventPersistenceError,
    ExtractionStageError,
    GraphIntegrationStageError,
    InputValidationError,
    InteractionNotFoundError,
    InteractionStateError,
    LoopProjectIsolationError,
    MemoryUpdateStageError,
    ResponseValidationError,
    RetrievalStageError,
    VectorIntegrationStageError,
)
from ai_brain.loop.input_processor import InputProcessor
from ai_brain.loop.memory_handler import LoopMemoryHandler, LoopMemoryProcessingResult
from ai_brain.loop.models import (
    AgentInteractionInput,
    AgentResponseInput,
    InteractionCompletionResult,
    InteractionContextResult,
    InteractionRecord,
    InteractionStatus,
    LoopStage,
)
from ai_brain.loop.orchestrator import BrainLoop
from ai_brain.loop.response_capture import ResponseCapture
from ai_brain.loop.storage.base import InteractionStore
from ai_brain.loop.storage.sqlite import SQLiteInteractionStore

__all__ = [
    "BrainLoop",
    "AgentInteractionInput",
    "AgentResponseInput",
    "InteractionContextResult",
    "InteractionCompletionResult",
    "InteractionRecord",
    "InteractionStatus",
    "LoopStage",
    "InputProcessor",
    "ResponseCapture",
    "LoopEventHandler",
    "LoopMemoryHandler",
    "LoopMemoryProcessingResult",
    "InteractionStore",
    "SQLiteInteractionStore",
    "BrainLoopError",
    "InputValidationError",
    "ResponseValidationError",
    "InteractionNotFoundError",
    "InteractionStateError",
    "LoopProjectIsolationError",
    "EventPersistenceError",
    "RetrievalStageError",
    "ContextStageError",
    "ExtractionStageError",
    "MemoryUpdateStageError",
    "GraphIntegrationStageError",
    "VectorIntegrationStageError",
]
