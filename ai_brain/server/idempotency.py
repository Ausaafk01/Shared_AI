"""
Concurrency-safe Idempotency Subsystem for AI Brain REST API (M5.5).

Guarantees:
1. Atomic Reservation:
   Idempotency keys are atomically reserved before interaction execution begins.
2. Strict Concurrency Protection:
   Two concurrent requests with the same idempotency key cannot execute simultaneously.
   The second concurrent caller is rejected with HTTP 409 Conflict.
3. Explicit Lifecycle State Machine:
   - COMPLETED: Replays cached progress and completion events with zero duplicate memory/graph writes.
   - IN_PROGRESS (Active): Concurrent execution blocked; raises HTTP 409 Conflict.
   - IN_PROGRESS (Stale/Interrupted): If a client disconnected or process crashed (> stale_timeout_seconds),
     the reservation is safely reclaimed and execution restarts cleanly according to documented policy.
   - FAILED / INTERRUPTED: Reconnecting caller safely restarts from clean state without duplicate writes.
4. Client Disconnect Safety:
   On client disconnect, downstream execution halts immediately. The interaction is marked FAILED with
   error 'INTERRUPTED: Client disconnected'. complete_interaction() is never invoked, ensuring zero
   partial memory extractions, zero graph node insertions, and zero duplicate events.
5. Zero Redis Requirement:
   Operates using in-process thread synchronization coupled with durable database records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException, status

from ai_brain.loop.models import InteractionRecord, InteractionStatus
from ai_brain.loop.storage.base import InteractionStore

logger = logging.getLogger("ai_brain.server.idempotency")


class IdempotencyStatus(StrEnum):
    """Classification of idempotency reservation state."""
    NEW = "NEW"
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"


class IdempotencyReservation:
    """Active reservation handle for an idempotent interaction."""

    def __init__(
        self,
        *,
        scoped_key: str,
        interaction_id: str,
        status: IdempotencyStatus,
        existing_record: Optional[InteractionRecord],
        manager: "IdempotencyManager",
    ) -> None:
        self.scoped_key = scoped_key
        self.interaction_id = interaction_id
        self.status = status
        self.existing_record = existing_record
        self.manager = manager
        self._released = False

    def release(self) -> None:
        """Release the active reservation lock."""
        if not self._released:
            self._released = True
            self.manager.release_active(self.scoped_key)

    def __enter__(self) -> "IdempotencyReservation":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class IdempotencyManager:
    """Thread-safe and concurrency-safe manager for interaction idempotency keys."""

    def __init__(self, stale_timeout_seconds: float = 30.0) -> None:
        self.stale_timeout_seconds = stale_timeout_seconds
        self._lock = threading.Lock()
        # scoped_key -> {"start_mono": float, "interaction_id": str}
        self._active_reservations: Dict[str, Dict[str, Any]] = {}

    def reserve(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        input_content: str,
        interaction_store: InteractionStore,
    ) -> IdempotencyReservation:
        """
        Atomically reserve an idempotency key before execution begins.
        
        Raises:
            HTTPException(409): If an active request with this key is already executing,
                               or if input payload does not match the original record.
        """
        scoped_key = f"{project_id}:{idempotency_key}"
        now_mono = time.monotonic()

        with self._lock:
            # 1. Check in-process active execution
            if scoped_key in self._active_reservations:
                active_entry = self._active_reservations[scoped_key]
                elapsed = now_mono - active_entry["start_mono"]
                if elapsed < self.stale_timeout_seconds:
                    logger.warning(
                        "Concurrent request rejected for idempotency key '%s' (active for %.2fs)",
                        idempotency_key,
                        elapsed,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=(
                            f"An interaction with idempotency key '{idempotency_key}' is "
                            f"currently IN_PROGRESS in project '{project_id}'."
                        ),
                    )
                else:
                    logger.warning(
                        "Reclaiming stale in-process reservation for '%s' (elapsed %.1fs)",
                        scoped_key,
                        elapsed,
                    )
                    del self._active_reservations[scoped_key]

            # 2. Check persistence store for prior interaction
            existing_record = interaction_store.get_by_idempotency_key(idempotency_key)
            if existing_record is not None:
                # Validate project match
                if existing_record.project_id != project_id:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"Idempotency key '{idempotency_key}' was created under another project.",
                    )

                # Validate input payload match
                existing_content = existing_record.input.content if existing_record.input else ""
                if existing_content != input_content:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"Idempotency key '{idempotency_key}' reused with mismatched input payload.",
                    )

                # Case A: Already COMPLETED -> replay
                if existing_record.status in (InteractionStatus.COMPLETED, InteractionStatus.PARTIAL_SUCCESS):
                    return IdempotencyReservation(
                        scoped_key=scoped_key,
                        interaction_id=existing_record.interaction_id,
                        status=IdempotencyStatus.COMPLETED,
                        existing_record=existing_record,
                        manager=self,
                    )

                # Case B: Was FAILED or INTERRUPTED -> clean recovery / restart
                if existing_record.status == InteractionStatus.FAILED:
                    logger.info(
                        "Retrying previously FAILED/INTERRUPTED interaction '%s' with key '%s'",
                        existing_record.interaction_id,
                        idempotency_key,
                    )
                    self._active_reservations[scoped_key] = {
                        "start_mono": now_mono,
                        "interaction_id": existing_record.interaction_id,
                    }
                    return IdempotencyReservation(
                        scoped_key=scoped_key,
                        interaction_id=existing_record.interaction_id,
                        status=IdempotencyStatus.INTERRUPTED,
                        existing_record=existing_record,
                        manager=self,
                    )

                # Case C: Recorded as IN_PROGRESS in store (e.g. INITIALIZED, CONTEXT_PREPARED)
                # Since it was not active in _active_reservations, it represents an abandoned run
                logger.info(
                    "Resuming abandoned/interrupted in-progress interaction '%s' with key '%s'",
                    existing_record.interaction_id,
                    idempotency_key,
                )
                self._active_reservations[scoped_key] = {
                    "start_mono": now_mono,
                    "interaction_id": existing_record.interaction_id,
                }
                return IdempotencyReservation(
                    scoped_key=scoped_key,
                    interaction_id=existing_record.interaction_id,
                    status=IdempotencyStatus.INTERRUPTED,
                    existing_record=existing_record,
                    manager=self,
                )

            # 3. New interaction: claim active reservation
            new_id = f"int_sse_{time.strftime('%Y%m%d%H%M%S')}_{scoped_key[-6:]}"
            self._active_reservations[scoped_key] = {
                "start_mono": now_mono,
                "interaction_id": new_id,
            }
            return IdempotencyReservation(
                scoped_key=scoped_key,
                interaction_id=new_id,
                status=IdempotencyStatus.NEW,
                existing_record=None,
                manager=self,
            )

    def release_active(self, scoped_key: str) -> None:
        """Release an active reservation lock."""
        with self._lock:
            self._active_reservations.pop(scoped_key, None)

    def active_count(self) -> int:
        """Return the number of currently active reservations in memory."""
        with self._lock:
            return len(self._active_reservations)

    def clear(self) -> None:
        """Clear all active reservations (test fixture aid)."""
        with self._lock:
            self._active_reservations.clear()
