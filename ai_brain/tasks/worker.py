"""
BackgroundWorker implementation for durable task execution in AI Brain (M5.6).
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import threading
import time
from typing import Any, Optional, Sequence

from ai_brain.observability.metrics import get_metrics_registry
from ai_brain.tasks.exceptions import (
    PermanentTaskError,
    TaskCancelledError,
    TaskError,
    TaskFencingError,
    TransientTaskError,
    WorkerError,
)
from ai_brain.tasks.models import (
    RetryPolicy,
    TaskRecord,
    TaskResult,
    TaskStatus,
    WorkerInfo,
    WorkerState,
)
from ai_brain.tasks.registry import TaskHandlerRegistry
from ai_brain.tasks.store import BaseTaskStore

logger = logging.getLogger("ai_brain.tasks.worker")


class BackgroundWorker:
    """
    Independent worker executing durable tasks claimed from a BaseTaskStore.
    
    Guarantees:
    - Provider-independent worker interface.
    - Independent periodic lease heartbeats during task execution.
    - Fencing detection: halts immediately and prevents side effect commits if lease is lost.
    - Automatic retry classification (transient with exponential backoff vs permanent DLQ).
    - Graceful startup, shutdown, and cancellation.
    - Completely decoupled from HTTP request lifecycle.
    """

    def __init__(
        self,
        *,
        store: BaseTaskStore,
        project_id: str,
        registry: TaskHandlerRegistry,
        worker_id: Optional[str] = None,
        context: Any = None,
        task_types: Optional[Sequence[str]] = None,
        lease_seconds: float = 30.0,
        poll_interval: float = 0.2,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        self.store = store
        self.project_id = project_id
        self.registry = registry
        self.worker_id = worker_id or f"worker_{threading.get_ident()}_{int(time.time()*1000)%1000000}"
        self.context = context
        self.task_types = list(task_types) if task_types else None
        self.lease_seconds = lease_seconds
        self.poll_interval = poll_interval
        self.retry_policy = retry_policy or RetryPolicy()

        self._state = WorkerState.STARTING
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._active_task_id: Optional[str] = None
        self._tasks_claimed = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._lock = threading.RLock()
        self._started_at = datetime.now(timezone.utc)
        self._heartbeat_at = datetime.now(timezone.utc)

    @property
    def state(self) -> WorkerState:
        with self._lock:
            return self._state

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def info(self) -> WorkerInfo:
        with self._lock:
            return WorkerInfo(
                worker_id=self.worker_id,
                state=self._state,
                current_task_id=self._active_task_id,
                tasks_claimed=self._tasks_claimed,
                tasks_completed=self._tasks_completed,
                tasks_failed=self._tasks_failed,
                heartbeat_at=self._heartbeat_at,
                started_at=self._started_at,
            )

    def start(self) -> None:
        """Start worker in a background daemon thread."""
        with self._lock:
            if self.is_alive:
                return
            self._state = WorkerState.RUNNING
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name=f"worker-{self.worker_id}",
                daemon=True,
            )
            self._thread.start()
            logger.info("Worker '%s' started for project '%s'", self.worker_id, self.project_id)

    def stop(self, timeout: float = 5.0, graceful: bool = True) -> None:
        """Signal worker to stop and wait for completion."""
        with self._lock:
            if self._state == WorkerState.STOPPED:
                return
            self._state = WorkerState.STOPPING
            self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        with self._lock:
            self._state = WorkerState.STOPPED
            logger.info("Worker '%s' stopped", self.worker_id)

    def run_once(self) -> bool:
        """
        Attempt to claim and process a single task synchronously.
        
        Returns True if a task was claimed and processed, False otherwise.
        """
        with self._lock:
            if self._stop_event.is_set():
                return False
            self._heartbeat_at = datetime.now(timezone.utc)

        task = self.store.claim_next_task(
            project_id=self.project_id,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
            task_types=self.task_types,
        )
        if not task:
            return False

        with self._lock:
            self._active_task_id = task.task_id
            self._tasks_claimed += 1

        self._process_claimed_task(task)

        with self._lock:
            self._active_task_id = None
        return True

    def _run_loop(self) -> None:
        """Continuous execution loop until stopped."""
        try:
            while not self._stop_event.is_set():
                claimed = self.run_once()
                if not claimed:
                    # Sleep poll interval if no task was available
                    self._stop_event.wait(timeout=self.poll_interval)
        except Exception as e:
            logger.exception("Worker '%s' encountered fatal unhandled loop exception: %s", self.worker_id, e)
        finally:
            with self._lock:
                self._state = WorkerState.STOPPED
                self._active_task_id = None

    def _process_claimed_task(self, task: TaskRecord) -> None:
        """Execute task with independent background heartbeat and fencing validation."""
        task_id = task.task_id
        project_id = task.project_id
        logger.info("Worker '%s' processing task '%s' (type: %s)", self.worker_id, task_id, task.task_type)

        # Start independent heartbeat thread
        hb_stop = threading.Event()
        hb_lost = threading.Event()
        hb_interval = max(0.05, self.lease_seconds / 3.0)

        def _heartbeat_worker() -> None:
            while not hb_stop.is_set():
                stopped = hb_stop.wait(timeout=hb_interval)
                if stopped:
                    break
                try:
                    ok = self.store.heartbeat(
                        project_id=project_id,
                        task_id=task_id,
                        worker_id=self.worker_id,
                        lease_seconds=self.lease_seconds,
                    )
                    if not ok:
                        logger.warning("Heartbeat failed for task '%s': lease lost or reclaimed", task_id)
                        hb_lost.set()
                        break
                except Exception as ex:
                    logger.warning("Heartbeat exception for task '%s': %s", task_id, ex)

        hb_thread = threading.Thread(
            target=_heartbeat_worker,
            name=f"heartbeat-{task_id}",
            daemon=True,
        )
        hb_thread.start()

        try:
            # Check if shutdown requested right after claim
            if self._stop_event.is_set():
                logger.info("Worker stopping: cancelling claimed task '%s'", task_id)
                self.store.cancel_task(project_id=project_id, task_id=task_id, reason="Worker shutting down")
                return

            # Execute task via registry
            result: TaskResult = self.registry.execute(task, self.context)

            # Check if lease was lost during execution
            if hb_lost.is_set():
                logger.error("Zombie worker '%s': lease lost during task '%s', aborting commit", self.worker_id, task_id)
                raise TaskFencingError(f"Lease lost for task '{task_id}'; worker was fenced.")

            if result.success:
                # Atomically complete task
                completed = self.store.complete_task(
                    project_id=project_id,
                    task_id=task_id,
                    worker_id=self.worker_id,
                    result_payload=result.result_data,
                )
                if not completed:
                    logger.error("Fencing check failed on completion for task '%s'", task_id)
                    raise TaskFencingError(f"Fencing check failed: cannot complete task '{task_id}'.")

                with self._lock:
                    self._tasks_completed += 1
                try:
                    get_metrics_registry().tasks_total.inc(project_id=project_id, task_type=task.task_type, status="COMPLETED")
                except Exception:
                    pass
                logger.info("Task '%s' completed successfully by worker '%s'", task_id, self.worker_id)
            else:
                # Task execution returned failure
                self.store.fail_task(
                    project_id=project_id,
                    task_id=task_id,
                    worker_id=self.worker_id,
                    error_message=result.error_message or "Execution failed without message",
                    is_permanent=not result.is_transient,
                    retry_policy=self.retry_policy,
                )
                with self._lock:
                    self._tasks_failed += 1
                try:
                    reg = get_metrics_registry()
                    if result.is_transient and task.attempts < self.retry_policy.max_retries:
                        reg.task_retries_total.inc(project_id=project_id, task_type=task.task_type)
                    else:
                        reg.task_failures_total.inc(project_id=project_id, task_type=task.task_type, error_class="ExecutionFailed")
                except Exception:
                    pass
                logger.warning("Task '%s' failed (transient=%s): %s", task_id, result.is_transient, result.error_message)

        except TaskFencingError:
            with self._lock:
                self._tasks_failed += 1
        except Exception as e:
            logger.exception("Unexpected exception executing task '%s': %s", task_id, e)
            try:
                if not hb_lost.is_set():
                    self.store.fail_task(
                        project_id=project_id,
                        task_id=task_id,
                        worker_id=self.worker_id,
                        error_message=f"{type(e).__name__}: {e}",
                        is_permanent=isinstance(e, PermanentTaskError),
                        retry_policy=self.retry_policy,
                    )
            except Exception as fail_ex:
                logger.warning("Could not record failure for task '%s': %s", task_id, fail_ex)
            with self._lock:
                self._tasks_failed += 1
            try:
                get_metrics_registry().task_failures_total.inc(project_id=project_id, task_type=task.task_type, error_class=type(e).__name__)
            except Exception:
                pass
        finally:
            hb_stop.set()
            hb_thread.join(timeout=1.0)
