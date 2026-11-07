"""
Task handler registry and execution dispatcher for AI Brain Background Workers (M5.6).
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import inspect
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional, Union

from ai_brain.tasks.exceptions import (
    PermanentTaskError,
    TaskError,
    TaskExecutionError,
    TransientTaskError,
)
from ai_brain.tasks.models import TaskRecord, TaskResult

logger = logging.getLogger("ai_brain.tasks.registry")

HandlerCallable = Callable[[TaskRecord, Any], Any]


class TaskHandlerRegistry:
    """Central registry mapping task_type to execution handlers."""

    def __init__(self) -> None:
        self._handlers: Dict[str, HandlerCallable] = {}

    def register(self, task_type: str, handler: HandlerCallable) -> None:
        """Register a handler callable for a specific task_type."""
        if not task_type:
            raise ValueError("task_type must not be empty.")
        if not callable(handler):
            raise TypeError(f"Handler for '{task_type}' must be callable.")
        self._handlers[task_type] = handler
        logger.info("Registered background task handler for '%s'", task_type)

    def unregister(self, task_type: str) -> bool:
        """Unregister a handler callable."""
        return self._handlers.pop(task_type, None) is not None

    def has_handler(self, task_type: str) -> bool:
        """Check if a handler is registered for a task_type."""
        return task_type in self._handlers

    def get_handler(self, task_type: str) -> Optional[HandlerCallable]:
        """Retrieve the registered handler callable."""
        return self._handlers.get(task_type)

    def execute(self, task: TaskRecord, context: Any = None) -> TaskResult:
        """
        Execute task synchronously or by running async handler in event loop.
        
        Returns TaskResult with success status, result data, and error classification.
        """
        handler = self._handlers.get(task.task_type)
        if not handler:
            return TaskResult(
                success=False,
                error_message=f"No registered handler for task_type '{task.task_type}'",
                is_transient=False,
            )

        start_time = time.perf_counter()
        try:
            # Check if handler accepts context
            sig = inspect.signature(handler)
            if len(sig.parameters) >= 2:
                call_res = handler(task, context)
            else:
                call_res = handler(task)

            # If coroutine returned, run to completion
            if inspect.iscoroutine(call_res):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # Run in threadpool if already inside an active event loop
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        fut = pool.submit(asyncio.run, call_res)
                        res = fut.result()
                else:
                    res = asyncio.run(call_res)
            else:
                res = call_res

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if isinstance(res, TaskResult):
                return res
            elif isinstance(res, dict):
                return TaskResult(success=True, result_data=res, execution_time_ms=elapsed_ms)
            else:
                return TaskResult(success=True, result_data={"output": res}, execution_time_ms=elapsed_ms)

        except PermanentTaskError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return TaskResult(
                success=False,
                error_message=str(e),
                is_transient=False,
                execution_time_ms=elapsed_ms,
            )
        except TransientTaskError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return TaskResult(
                success=False,
                error_message=str(e),
                is_transient=True,
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            # Default classification: standard unexpected exceptions are transient unless marked permanent
            logger.exception("Task '%s' failed during execution: %s", task.task_id, e)
            return TaskResult(
                success=False,
                error_message=f"{type(e).__name__}: {e}",
                is_transient=True,
                execution_time_ms=elapsed_ms,
            )
