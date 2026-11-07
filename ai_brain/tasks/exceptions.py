"""
Domain exceptions for AI Brain Background Worker Fleet (M5.6).
"""

from __future__ import annotations


class TaskError(Exception):
    """Base exception for all background task and worker subsystem errors."""
    pass


class TaskNotFoundError(TaskError):
    """Raised when a referenced task does not exist in the store."""
    pass


class DuplicateTaskError(TaskError):
    """Raised when a task with an identical idempotency key is already registered."""
    pass


class TaskClaimError(TaskError):
    """Raised when an atomic task claim operation fails or is rejected."""
    pass


class TaskFencingError(TaskError):
    """Raised when a worker's lease expired or was reclaimed by another worker."""
    pass


class TaskExecutionError(TaskError):
    """Base exception for errors raised during task handler execution."""
    pass


class TransientTaskError(TaskExecutionError):
    """
    Classified failure that is temporary and eligible for exponential backoff retry
    (e.g. network blips, lock contention, temporary rate limits).
    """
    pass


class PermanentTaskError(TaskExecutionError):
    """
    Classified failure that will not succeed on retry and should transition directly
    to the Dead-Letter Queue (e.g. invalid arguments, unrecoverable domain errors).
    """
    pass


class TaskCancelledError(TaskError):
    """Raised when an active task is explicitly cancelled or aborted."""
    pass


class WorkerError(Exception):
    """Base exception for worker fleet lifecycle errors."""
    pass
