"""Worker Fleet manager for orchestrating pools of background workers."""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

from ai_brain.observability.metrics import get_metrics_registry
from ai_brain.tasks.exceptions import WorkerError
from ai_brain.tasks.models import TaskRecord, WorkerInfo, WorkerState
from ai_brain.tasks.registry import TaskHandlerRegistry
from ai_brain.tasks.store import BaseTaskStore
from ai_brain.tasks.worker import BackgroundWorker

logger = logging.getLogger(__name__)


class WorkerFleet:
    """Manages a pool of concurrent background workers."""

    def __init__(
        self,
        store: BaseTaskStore,
        registry: TaskHandlerRegistry,
        project_id: str = "proj_default",
        concurrency: int = 4,
        poll_interval: float = 0.2,
        lease_timeout: float = 30.0,
        fleet_id: Optional[str] = None,
        context: Any = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {concurrency}")

        self.store = store
        self.registry = registry
        self.project_id = project_id
        self.target_concurrency = concurrency
        self.poll_interval = poll_interval
        self.lease_timeout = lease_timeout
        self.fleet_id = fleet_id or f"fleet_{uuid.uuid4().hex[:8]}"
        self.context = context

        self._workers: List[BackgroundWorker] = []
        self._lock = threading.Lock()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def current_concurrency(self) -> int:
        with self._lock:
            return sum(1 for w in self._workers if w.is_alive)

    def start(self) -> None:
        """Start the worker fleet up to the target concurrency."""
        with self._lock:
            if self._is_running:
                logger.warning("Worker fleet %s is already running", self.fleet_id)
                return

            self._is_running = True
            self._workers.clear()
            logger.info(
                "Starting worker fleet %s with %d workers",
                self.fleet_id,
                self.target_concurrency,
            )

            for i in range(self.target_concurrency):
                worker = self._create_worker(i)
                self._workers.append(worker)
                worker.start()

        try:
            get_metrics_registry().worker_active_count.set(self.current_concurrency, fleet_id=self.fleet_id)
        except Exception:
            pass

    def _create_worker(self, index: int) -> BackgroundWorker:
        worker_id = f"{self.fleet_id}-w{index + 1}-{uuid.uuid4().hex[:6]}"
        return BackgroundWorker(
            store=self.store,
            project_id=self.project_id,
            registry=self.registry,
            worker_id=worker_id,
            poll_interval=self.poll_interval,
            lease_seconds=self.lease_timeout,
            context=self.context,
        )

    def stop(self, timeout: float = 10.0, graceful: bool = True) -> None:
        """Stop all workers in the fleet."""
        with self._lock:
            if not self._is_running:
                return

            logger.info("Stopping worker fleet %s (graceful=%s)", self.fleet_id, graceful)
            self._is_running = False
            workers = list(self._workers)

        # Signal workers outside lock
        for worker in workers:
            worker.stop(timeout=timeout, graceful=graceful)

        try:
            get_metrics_registry().worker_active_count.set(0, fleet_id=self.fleet_id)
        except Exception:
            pass

        logger.info("Worker fleet %s stopped", self.fleet_id)

    def scale(self, new_concurrency: int, timeout: float = 5.0) -> None:
        """Dynamically adjust worker concurrency."""
        if new_concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {new_concurrency}")

        with self._lock:
            self.target_concurrency = new_concurrency
            if not self._is_running:
                return

            current_count = len(self._workers)
            if new_concurrency > current_count:
                to_add = new_concurrency - current_count
                logger.info("Scaling up fleet %s: adding %d workers", self.fleet_id, to_add)
                for i in range(to_add):
                    worker = self._create_worker(current_count + i)
                    self._workers.append(worker)
                    worker.start()
            elif new_concurrency < current_count:
                to_remove = current_count - new_concurrency
                logger.info("Scaling down fleet %s: removing %d workers", self.fleet_id, to_remove)
                workers_to_stop = self._workers[-to_remove:]
                self._workers = self._workers[:-to_remove]
            else:
                return

        # If scaling down, stop removed workers
        if new_concurrency < current_count:
            for w in workers_to_stop:
                w.stop(timeout=timeout, graceful=True)

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate statistics across all active and stopped workers."""
        with self._lock:
            workers = list(self._workers)

        total_claimed = sum(w.info.tasks_claimed for w in workers)
        total_completed = sum(w.info.tasks_completed for w in workers)
        total_failed = sum(w.info.tasks_failed for w in workers)

        return {
            "fleet_id": self.fleet_id,
            "target_concurrency": self.target_concurrency,
            "active_workers": sum(1 for w in workers if w.is_alive),
            "tasks_claimed": total_claimed,
            "tasks_completed": total_completed,
            "tasks_failed": total_failed,
        }

    def get_health(self) -> Dict[str, Any]:
        """Fleet health check including worker states and queue depth."""
        stats = self.get_stats()
        queue_stats = self.store.get_queue_stats(self.project_id)

        healthy_workers = stats["active_workers"]
        if not self._is_running:
            fleet_status = "STOPPED"
        elif healthy_workers == self.target_concurrency:
            fleet_status = "HEALTHY"
        elif healthy_workers > 0:
            fleet_status = "DEGRADED"
        else:
            fleet_status = "UNHEALTHY"

        return {
            "status": fleet_status,
            "fleet_id": self.fleet_id,
            "concurrency": {
                "target": self.target_concurrency,
                "active": healthy_workers,
            },
            "queue": queue_stats,
            "metrics": stats,
        }

    def get_worker_infos(self) -> List[WorkerInfo]:
        """Get diagnostic information for each worker in the fleet."""
        with self._lock:
            return [w.info for w in self._workers]

    def list_dead_letters(
        self,
        project_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TaskRecord]:
        """List tasks currently in the Dead Letter Queue."""
        return self.store.list_dead_letters(project_id, limit=limit, offset=offset)

    def retry_dead_letter(
        self,
        project_id: str,
        task_id: str,
    ) -> Optional[TaskRecord]:
        """Re-queue an exhausted dead-letter task."""
        return self.store.retry_dead_letter(project_id, task_id)

    def __enter__(self) -> WorkerFleet:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
