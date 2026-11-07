"""
Domain models and lifecycle enumerations for AI Brain Background Worker Fleet (M5.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, List, Optional
import uuid


class TaskStatus(StrEnum):
    """Lifecycle status of a durable background task."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskPriority(StrEnum):
    """Execution priority levels for background tasks."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class WorkerState(StrEnum):
    """Operational lifecycle state of an individual worker."""
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class RetryPolicy:
    """Configures retry behavior, backoff parameters, and bounds for task failures."""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_factor: float = 2.0

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 60.0,
        backoff_factor: float = 2.0,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
    ) -> None:
        b_delay = base_delay if base_delay is not None else base_delay_seconds
        m_delay = max_delay if max_delay is not None else max_delay_seconds
        object.__setattr__(self, "max_retries", max_retries)
        object.__setattr__(self, "base_delay_seconds", b_delay)
        object.__setattr__(self, "max_delay_seconds", m_delay)
        object.__setattr__(self, "backoff_factor", backoff_factor)

    def compute_delay(self, retry_count: int) -> float:
        """Computes exponential backoff bounded by max_delay_seconds."""
        if retry_count <= 0:
            return self.base_delay_seconds
        exp_delay = self.base_delay_seconds * (self.backoff_factor ** retry_count)
        return min(self.max_delay_seconds, max(self.base_delay_seconds, exp_delay))


@dataclass(frozen=True)
class TaskRecord:
    """Persistent representation of a background task."""
    task_id: str
    project_id: str
    task_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    locked_by: Optional[str] = None
    locked_until: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def idempotency_key(self) -> Optional[str]:
        """Optional idempotency key stored in payload."""
        return self.payload.get("idempotency_key")

    @property
    def is_expired(self) -> bool:
        """Check if active lease has expired."""
        if not self.locked_until:
            return False
        now_utc = datetime.now(timezone.utc)
        exp = self.locked_until if self.locked_until.tzinfo else self.locked_until.replace(tzinfo=timezone.utc)
        return now_utc >= exp

    def to_dict(self) -> Dict[str, Any]:
        """Convert task record to dictionary."""
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "task_type": self.task_type,
            "payload": dict(self.payload),
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "locked_by": self.locked_by,
            "locked_until": self.locked_until.isoformat() if self.locked_until else None,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass(frozen=True)
class TaskResult:
    """Outcome of a task handler execution."""
    success: bool = True
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    is_transient: bool = False
    execution_time_ms: float = 0.0

    def __init__(
        self,
        success: bool = True,
        result_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        is_transient: bool = False,
        execution_time_ms: float = 0.0,
        output: Optional[Dict[str, Any]] = None,
    ) -> None:
        object.__setattr__(self, "success", success)
        data = output if output is not None else (result_data or {})
        object.__setattr__(self, "result_data", data)
        object.__setattr__(self, "error_message", error_message)
        object.__setattr__(self, "is_transient", is_transient)
        object.__setattr__(self, "execution_time_ms", execution_time_ms)

    @property
    def output(self) -> Dict[str, Any]:
        """Convenience alias for result_data."""
        return self.result_data


@dataclass
class WorkerInfo:
    """Diagnostic state and metrics for a registered background worker."""
    worker_id: str
    state: WorkerState = WorkerState.STARTING
    current_task_id: Optional[str] = None
    tasks_claimed: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    heartbeat_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "state": self.state.value if hasattr(self.state, "value") else str(self.state),
            "current_task_id": self.current_task_id,
            "tasks_claimed": self.tasks_claimed,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "heartbeat_at": self.heartbeat_at.isoformat(),
            "started_at": self.started_at.isoformat(),
        }
