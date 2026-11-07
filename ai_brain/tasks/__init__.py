"""Background task processing and worker fleet for AI Brain."""

from ai_brain.tasks.exceptions import (
    DuplicateTaskError,
    PermanentTaskError,
    TaskCancelledError,
    TaskClaimError,
    TaskError,
    TaskFencingError,
    TaskNotFoundError,
    TransientTaskError,
    WorkerError,
)
from ai_brain.tasks.fleet import WorkerFleet
from ai_brain.tasks.handlers import (
    handle_interaction_execution,
    handle_memory_maintenance,
    handle_workflow_step,
    register_default_handlers,
)
from ai_brain.tasks.models import (
    RetryPolicy,
    TaskPriority,
    TaskRecord,
    TaskResult,
    TaskStatus,
    WorkerInfo,
    WorkerState,
)
from ai_brain.tasks.registry import TaskHandlerRegistry
from ai_brain.tasks.store import (
    BaseTaskStore,
    PostgreSQLTaskStore,
    SQLiteTaskStore,
    create_task_store,
)
from ai_brain.tasks.worker import BackgroundWorker

__all__ = [
    # Models
    "TaskStatus",
    "TaskPriority",
    "WorkerState",
    "RetryPolicy",
    "TaskRecord",
    "TaskResult",
    "WorkerInfo",
    # Exceptions
    "TaskError",
    "TaskNotFoundError",
    "DuplicateTaskError",
    "TaskClaimError",
    "TaskFencingError",
    "TransientTaskError",
    "PermanentTaskError",
    "TaskCancelledError",
    "WorkerError",
    # Stores
    "BaseTaskStore",
    "PostgreSQLTaskStore",
    "SQLiteTaskStore",
    "create_task_store",
    # Registry & Handlers
    "TaskHandlerRegistry",
    "handle_interaction_execution",
    "handle_workflow_step",
    "handle_memory_maintenance",
    "register_default_handlers",
    # Workers & Fleet
    "BackgroundWorker",
    "WorkerFleet",
]
