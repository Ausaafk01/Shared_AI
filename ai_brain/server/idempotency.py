"""
Durable, cross-process and distributed Idempotency Subsystem for AI Brain REST API (M5.5).

Guarantees:
1. Cross-Process & Distributed Atomic Reservation:
   Idempotency keys are atomically reserved in PostgreSQL (or SQLite for offline operation)
   before interaction execution begins. Unique constraints on (project_id, idempotency_key)
   and transaction-level advisory locks prevent race conditions across multiple Uvicorn workers
   and container replicas without requiring Redis.
2. Strict Concurrency Protection:
   Two concurrent requests with the same idempotency key cannot execute simultaneously.
   The second concurrent caller is rejected with HTTP 409 Conflict.
3. Explicit Lease Expiration & Periodic Heartbeats:
   Active execution holds a time-bounded lease (e.g. 10s default). Active workers periodically
   renew their lease via heartbeats during progressive streaming chunks.
4. Safe Atomic Stale-Lease Reclamation:
   If a worker crashes or halts without renewing its lease, once the lease expires
   (now_utc >= lease_expires_at), another worker can atomically reclaim ownership.
   A live worker actively renewing its lease can never be reclaimed.
5. Fencing Tokens for Zombie Worker Protection:
   Each lease acquisition or reclamation increments a monotonically increasing fencing_token.
   A worker whose lease has expired or been reclaimed cannot commit completion or write
   extracted memories because its fencing token is rejected by the database.
6. Tenant Isolation & RLS Integrity:
   PostgreSQL storage operates under strict Row Level Security (RLS) within tenant transactions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
import uuid

from fastapi import HTTPException, status

from ai_brain.loop.models import InteractionRecord, InteractionStatus
from ai_brain.loop.storage.base import InteractionStore
from ai_brain.storage.database import DatabaseConnectionManager

logger = logging.getLogger("ai_brain.server.idempotency")

T = TypeVar("T")


def _resolve_existing_record(
    store: InteractionStore, idempotency_key: str, project_id: str, interaction_id: Optional[str] = None
) -> Optional[InteractionRecord]:
    """Helper to safely retrieve an existing interaction record by key or ID with project scope."""
    rec: Optional[InteractionRecord] = None
    try:
        rec = store.get_by_idempotency_key(idempotency_key, project_id=project_id)
    except TypeError:
        rec = store.get_by_idempotency_key(idempotency_key)
    if rec is None and interaction_id:
        try:
            rec = store.get(interaction_id, project_id=project_id)
        except TypeError:
            rec = store.get(interaction_id)
    return rec


class StaleFencingTokenError(Exception):
    """Raised when a worker's fencing token is obsolete or its lease was reclaimed."""
    pass


class IdempotencyStatus(StrEnum):
    """Classification of idempotency reservation state."""
    NEW = "NEW"
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"


class AsyncHeartbeatHandle:
    """Manages asynchronous periodic lease heartbeats running independently while execution is active."""

    def __init__(
        self,
        reservation: "IdempotencyReservation",
        interval: float,
        lease_seconds: float,
    ) -> None:
        self.reservation = reservation
        self.interval = max(0.05, interval)
        self.lease_seconds = lease_seconds
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._is_stopped = False
        self._ownership_lost = False
        self._last_error: Optional[Exception] = None

    @property
    def is_alive(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def ownership_lost(self) -> bool:
        return self._ownership_lost

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval)
                break
            except asyncio.TimeoutError:
                pass

            if self._stop_event.is_set() or self.reservation.is_terminal:
                break

            try:
                await asyncio.to_thread(self.reservation.heartbeat, self.lease_seconds)
            except StaleFencingTokenError as e:
                logger.warning(
                    "Async heartbeat detected lost ownership for key '%s': %s",
                    self.reservation.idempotency_key,
                    e,
                )
                self._ownership_lost = True
                self._last_error = e
                self.reservation.mark_lost(str(e))
                break
            except Exception as e:
                logger.warning(
                    "Async heartbeat renewal error for key '%s': %s",
                    self.reservation.idempotency_key,
                    e,
                )

    def start(self) -> None:
        if self._task is None and not self._is_stopped:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._is_stopped:
            return
        self._is_stopped = True
        self._stop_event.set()
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()


class BackgroundHeartbeatHandle:
    """Manages thread-based periodic lease heartbeats running independently while execution is active."""

    def __init__(
        self,
        reservation: "IdempotencyReservation",
        interval: float,
        lease_seconds: float,
    ) -> None:
        self.reservation = reservation
        self.interval = max(0.05, interval)
        self.lease_seconds = lease_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_stopped = False
        self._ownership_lost = False
        self._last_error: Optional[Exception] = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def ownership_lost(self) -> bool:
        return self._ownership_lost

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            stopped = self._stop_event.wait(timeout=self.interval)
            if stopped or self.reservation.is_terminal:
                break
            try:
                self.reservation.heartbeat(self.lease_seconds)
            except StaleFencingTokenError as e:
                logger.warning(
                    "Thread heartbeat detected lost ownership for key '%s': %s",
                    self.reservation.idempotency_key,
                    e,
                )
                self._ownership_lost = True
                self._last_error = e
                self.reservation.mark_lost(str(e))
                break
            except Exception as e:
                logger.warning(
                    "Thread heartbeat renewal error for key '%s': %s",
                    self.reservation.idempotency_key,
                    e,
                )

    def start(self) -> None:
        if self._thread is None and not self._is_stopped:
            self._thread = threading.Thread(
                target=self._loop,
                name=f"heartbeat_{self.reservation.idempotency_key}",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if self._is_stopped:
            return
        self._is_stopped = True
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)


class IdempotencyReservation:
    """Active reservation handle for an idempotent interaction with fencing token support."""

    def __init__(
        self,
        *,
        scoped_key: str,
        project_id: str,
        idempotency_key: str,
        interaction_id: str,
        owner_id: str,
        fencing_token: int,
        status: IdempotencyStatus,
        existing_record: Optional[InteractionRecord],
        coordinator: "BaseIdempotencyCoordinator",
        lease_seconds: float = 10.0,
    ) -> None:
        self.scoped_key = scoped_key
        self.project_id = project_id
        self.idempotency_key = idempotency_key
        self.interaction_id = interaction_id
        self.owner_id = owner_id
        self.fencing_token = fencing_token
        self.status = status
        self.existing_record = existing_record
        self.coordinator = coordinator
        self.lease_seconds = lease_seconds
        self._released = False
        self._lost = False
        self._lost_reason = ""
        self._lock = threading.RLock()
        self._active_async_heartbeat: Optional[AsyncHeartbeatHandle] = None
        self._active_bg_heartbeat: Optional[BackgroundHeartbeatHandle] = None

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return self._released or self._lost or self.status == IdempotencyStatus.COMPLETED

    def mark_lost(self, reason: str) -> None:
        with self._lock:
            self._lost = True
            self._lost_reason = reason

    def heartbeat(self, lease_seconds: Optional[float] = None) -> None:
        """Renew active lease. Raises StaleFencingTokenError if lease was lost."""
        with self._lock:
            if self._released or self.status == IdempotencyStatus.COMPLETED:
                return
            if self._lost:
                raise StaleFencingTokenError(
                    f"Heartbeat failed for key '{self.idempotency_key}': {self._lost_reason or 'lease lost or reclaimed'}."
                )
            dur = lease_seconds if lease_seconds is not None else self.lease_seconds
            ok = self.coordinator.heartbeat(
                project_id=self.project_id,
                idempotency_key=self.idempotency_key,
                fencing_token=self.fencing_token,
                lease_seconds=dur,
            )
            if not ok:
                self.mark_lost("lease expired or reclaimed by another worker")
                raise StaleFencingTokenError(
                    f"Heartbeat failed for key '{self.idempotency_key}': lease expired or reclaimed by another worker."
                )

    def execute_fenced_completion(self, completion_fn: Callable[[], T]) -> T:
        """Atomically validate fencing token, execute completion side effects, and transition to COMPLETED."""
        with self._lock:
            if self._released:
                raise StaleFencingTokenError("Cannot complete a released reservation.")
            if self._lost:
                raise StaleFencingTokenError(
                    f"Cannot commit completion for key '{self.idempotency_key}': {self._lost_reason or 'fencing token is obsolete'}."
                )
            if self.status == IdempotencyStatus.COMPLETED:
                return completion_fn()

            # Execute transactional fenced completion via coordinator
            result = self.coordinator.execute_fenced_completion(
                project_id=self.project_id,
                idempotency_key=self.idempotency_key,
                fencing_token=self.fencing_token,
                completion_fn=completion_fn,
            )
            self.status = IdempotencyStatus.COMPLETED
            return result

    def commit_completion(self) -> None:
        """Atomically transition state to COMPLETED. Raises StaleFencingTokenError if reclaimed."""
        self.execute_fenced_completion(lambda: None)

    def mark_interrupted(self, reason: str = "") -> None:
        """Atomically mark reservation as INTERRUPTED."""
        with self._lock:
            if self._released or self.status == IdempotencyStatus.COMPLETED:
                return
            self.coordinator.mark_interrupted(
                project_id=self.project_id,
                idempotency_key=self.idempotency_key,
                fencing_token=self.fencing_token,
                reason=reason,
            )
            self.status = IdempotencyStatus.INTERRUPTED

    def release(self) -> None:
        """Release active reservation if not completed."""
        with self._lock:
            if not self._released:
                self._released = True
                if self._active_bg_heartbeat is not None:
                    self._active_bg_heartbeat.stop()
                if self.status != IdempotencyStatus.COMPLETED and not self._lost:
                    self.coordinator.release(
                        project_id=self.project_id,
                        idempotency_key=self.idempotency_key,
                        fencing_token=self.fencing_token,
                    )

    def start_async_heartbeat(
        self,
        interval: Optional[float] = None,
        lease_seconds: Optional[float] = None,
    ) -> AsyncHeartbeatHandle:
        """Start periodic asynchronous heartbeat task running independently while execution is active."""
        dur = lease_seconds if lease_seconds is not None else self.lease_seconds
        intv = interval if interval is not None else max(0.1, dur / 3.0)
        handle = AsyncHeartbeatHandle(self, interval=intv, lease_seconds=dur)
        handle.start()
        self._active_async_heartbeat = handle
        return handle

    def start_background_heartbeat(
        self,
        interval: Optional[float] = None,
        lease_seconds: Optional[float] = None,
    ) -> BackgroundHeartbeatHandle:
        """Start periodic background heartbeat thread running independently while execution is active."""
        dur = lease_seconds if lease_seconds is not None else self.lease_seconds
        intv = interval if interval is not None else max(0.1, dur / 3.0)
        handle = BackgroundHeartbeatHandle(self, interval=intv, lease_seconds=dur)
        handle.start()
        self._active_bg_heartbeat = handle
        return handle

    def __enter__(self) -> "IdempotencyReservation":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class BaseIdempotencyCoordinator(ABC):
    """Abstract base class for durable, cross-process idempotency coordinators."""

    @abstractmethod
    def reserve(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        owner_id: str,
        lease_seconds: float,
        interaction_store: InteractionStore,
        input_content: str,
    ) -> IdempotencyReservation:
        """Atomically reserve or recover an idempotency key."""
        pass

    @abstractmethod
    def heartbeat(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
        lease_seconds: float,
    ) -> bool:
        """Renew active lease if fencing token matches."""
        pass

    @abstractmethod
    def complete(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
    ) -> bool:
        """Mark reservation as COMPLETED if fencing token matches."""
        pass

    @abstractmethod
    def mark_interrupted(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
        reason: str,
    ) -> bool:
        """Mark reservation as INTERRUPTED if fencing token matches."""
        pass

    @abstractmethod
    def release(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
    ) -> None:
        """Release or clean up an abandoned reservation."""
        pass

    @abstractmethod
    def execute_fenced_completion(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
        completion_fn: Callable[[], Any],
    ) -> Any:
        """Atomically validate fencing token, execute completion side effects, and mark COMPLETED."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all reservations (test fixture helper)."""
        pass


class PostgreSQLIdempotencyCoordinator(BaseIdempotencyCoordinator):
    """
    Production-grade distributed idempotency coordinator backed by PostgreSQL.
    
    Guarantees:
    - Transaction-scoped advisory locks serialize concurrent attempts on identical keys.
    - PRIMARY KEY (project_id, idempotency_key) guarantees database-level uniqueness.
    - Tenant isolation enforced via RLS in tenant transactions.
    - Fencing tokens prevent zombie workers from committing stale completions.
    """

    def __init__(self, db_manager: DatabaseConnectionManager) -> None:
        self.db_manager = db_manager

    def reserve(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        owner_id: str,
        lease_seconds: float,
        interaction_store: InteractionStore,
        input_content: str,
    ) -> IdempotencyReservation:
        scoped_key = f"{project_id}:{idempotency_key}"
        now_utc = datetime.now(timezone.utc)
        lease_exp = now_utc + timedelta(seconds=lease_seconds)

        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                # 1. Acquire transaction advisory lock on hashed key to serialize competing workers
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s));", (scoped_key,))

                # 2. Query existing reservation under FOR UPDATE
                cur.execute(
                    """
                    SELECT interaction_id, owner_id, status, fencing_token, lease_expires_at, heartbeat_at
                    FROM idempotency_reservations
                    WHERE project_id = %s AND idempotency_key = %s
                    FOR UPDATE;
                    """,
                    (project_id, idempotency_key),
                )
                row = cur.fetchone()

                if row is None:
                    # Check if interaction store already has an interaction for this key
                    existing_record = _resolve_existing_record(interaction_store, idempotency_key, project_id)
                    if existing_record is not None:
                        new_iid = existing_record.interaction_id
                        res_status = IdempotencyStatus.INTERRUPTED
                    else:
                        new_iid = f"int_sse_{now_utc.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
                        res_status = IdempotencyStatus.NEW

                    cur.execute(
                        """
                        INSERT INTO idempotency_reservations (
                            project_id, idempotency_key, interaction_id, owner_id, status,
                            fencing_token, lease_expires_at, heartbeat_at, created_at, updated_at
                        ) VALUES (
                            %s, %s, %s, %s, 'IN_PROGRESS', 1, %s, %s, %s, %s
                        );
                        """,
                        (project_id, idempotency_key, new_iid, owner_id, lease_exp, now_utc, now_utc, now_utc),
                    )
                    return IdempotencyReservation(
                        scoped_key=scoped_key,
                        project_id=project_id,
                        idempotency_key=idempotency_key,
                        interaction_id=new_iid,
                        owner_id=owner_id,
                        fencing_token=1,
                        status=res_status,
                        existing_record=existing_record,
                        coordinator=self,
                        lease_seconds=lease_seconds,
                    )

                (existing_iid, current_owner, current_status, fencing_token, current_lease_exp, _) = row
                if current_lease_exp.tzinfo is None:
                    current_lease_exp = current_lease_exp.replace(tzinfo=timezone.utc)

                # Validate input payload against existing interaction record
                existing_record = _resolve_existing_record(interaction_store, idempotency_key, project_id, existing_iid)
                if existing_record is not None:
                    if existing_record.project_id != project_id:
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail=f"Idempotency key '{idempotency_key}' was created under another project.",
                        )
                    existing_content = existing_record.input.content if existing_record.input else ""
                    if existing_content != input_content:
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail=f"Idempotency key '{idempotency_key}' reused with mismatched input payload.",
                        )

                # Case A: Already COMPLETED -> replay
                if current_status == "COMPLETED" or (
                    existing_record and existing_record.status in (InteractionStatus.COMPLETED, InteractionStatus.PARTIAL_SUCCESS)
                ):
                    return IdempotencyReservation(
                        scoped_key=scoped_key,
                        project_id=project_id,
                        idempotency_key=idempotency_key,
                        interaction_id=existing_iid,
                        owner_id=current_owner,
                        fencing_token=fencing_token,
                        status=IdempotencyStatus.COMPLETED,
                        existing_record=existing_record,
                        coordinator=self,
                        lease_seconds=lease_seconds,
                    )

                # Case B: Currently IN_PROGRESS
                if current_status == "IN_PROGRESS":
                    if now_utc < current_lease_exp:
                        # Lease is STILL ACTIVE: reject concurrent request
                        logger.warning(
                            "Concurrent request rejected for key '%s' (active lease held by '%s' until %s)",
                            idempotency_key,
                            current_owner,
                            current_lease_exp.isoformat(),
                        )
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail=(
                                f"An interaction with idempotency key '{idempotency_key}' is "
                                f"currently IN_PROGRESS in project '{project_id}'."
                            ),
                        )
                    else:
                        # Lease HAS EXPIRED: reclaim stale reservation with incremented fencing token
                        logger.warning(
                            "Reclaiming expired lease for key '%s' (previous owner '%s' expired at %s)",
                            idempotency_key,
                            current_owner,
                            current_lease_exp.isoformat(),
                        )
                        new_token = fencing_token + 1
                        cur.execute(
                            """
                            UPDATE idempotency_reservations
                            SET owner_id = %s,
                                status = 'IN_PROGRESS',
                                fencing_token = %s,
                                lease_expires_at = %s,
                                heartbeat_at = %s,
                                updated_at = %s
                            WHERE project_id = %s AND idempotency_key = %s;
                            """,
                            (owner_id, new_token, lease_exp, now_utc, now_utc, project_id, idempotency_key),
                        )
                        return IdempotencyReservation(
                            scoped_key=scoped_key,
                            project_id=project_id,
                            idempotency_key=idempotency_key,
                            interaction_id=existing_iid,
                            owner_id=owner_id,
                            fencing_token=new_token,
                            status=IdempotencyStatus.INTERRUPTED,
                            existing_record=existing_record,
                            coordinator=self,
                            lease_seconds=lease_seconds,
                        )

                # Case C: Previous execution was INTERRUPTED or FAILED -> retry
                new_token = fencing_token + 1
                cur.execute(
                    """
                    UPDATE idempotency_reservations
                    SET owner_id = %s,
                        status = 'IN_PROGRESS',
                        fencing_token = %s,
                        lease_expires_at = %s,
                        heartbeat_at = %s,
                        updated_at = %s
                    WHERE project_id = %s AND idempotency_key = %s;
                    """,
                    (owner_id, new_token, lease_exp, now_utc, now_utc, project_id, idempotency_key),
                )
                return IdempotencyReservation(
                    scoped_key=scoped_key,
                    project_id=project_id,
                    idempotency_key=idempotency_key,
                    interaction_id=existing_iid,
                    owner_id=owner_id,
                    fencing_token=new_token,
                    status=IdempotencyStatus.INTERRUPTED,
                    existing_record=existing_record,
                    coordinator=self,
                    lease_seconds=lease_seconds,
                )

    def heartbeat(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
        lease_seconds: float,
    ) -> bool:
        now_utc = datetime.now(timezone.utc)
        new_exp = now_utc + timedelta(seconds=lease_seconds)
        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE idempotency_reservations
                    SET lease_expires_at = %s,
                        heartbeat_at = %s,
                        updated_at = %s
                    WHERE project_id = %s AND idempotency_key = %s
                      AND fencing_token = %s AND status = 'IN_PROGRESS';
                    """,
                    (new_exp, now_utc, now_utc, project_id, idempotency_key, fencing_token),
                )
                return cur.rowcount > 0

    def complete(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
    ) -> bool:
        now_utc = datetime.now(timezone.utc)
        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE idempotency_reservations
                    SET status = 'COMPLETED',
                        updated_at = %s
                    WHERE project_id = %s AND idempotency_key = %s
                      AND fencing_token = %s AND status = 'IN_PROGRESS';
                    """,
                    (now_utc, project_id, idempotency_key, fencing_token),
                )
                return cur.rowcount > 0

    def execute_fenced_completion(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
        completion_fn: Callable[[], T],
    ) -> T:
        scoped_key = f"{project_id}:{idempotency_key}"
        now_utc = datetime.now(timezone.utc)

        with self.db_manager.scoped_transaction(project_id) as conn:
            with conn.cursor() as cur:
                # 1. Acquire transaction-level advisory lock to serialize with competing workers
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s));", (scoped_key,))

                # 2. Verify fencing token and status under FOR UPDATE
                cur.execute(
                    """
                    SELECT fencing_token, status
                    FROM idempotency_reservations
                    WHERE project_id = %s AND idempotency_key = %s
                    FOR UPDATE;
                    """,
                    (project_id, idempotency_key),
                )
                row = cur.fetchone()
                if row is None:
                    raise StaleFencingTokenError(
                        f"Reservation for key '{idempotency_key}' does not exist in project '{project_id}'."
                    )

                current_token, current_status = row
                if current_token != fencing_token:
                    raise StaleFencingTokenError(
                        f"Cannot commit completion for key '{idempotency_key}': fencing token {fencing_token} "
                        f"is obsolete (current token is {current_token})."
                    )

                if current_status not in ("IN_PROGRESS", "COMPLETED"):
                    raise StaleFencingTokenError(
                        f"Cannot commit completion for key '{idempotency_key}': status is '{current_status}'."
                    )

            # 3. Execute completion side effects within the active scoped transaction
            result = completion_fn()

            # 4. Mark reservation COMPLETED only after side effects have succeeded
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE idempotency_reservations
                    SET status = 'COMPLETED',
                        updated_at = %s
                    WHERE project_id = %s AND idempotency_key = %s AND fencing_token = %s;
                    """,
                    (now_utc, project_id, idempotency_key, fencing_token),
                )
                if cur.rowcount == 0:
                    raise StaleFencingTokenError(
                        f"Failed to commit completion for key '{idempotency_key}': fencing token {fencing_token} was modified."
                    )

            return result

    def mark_interrupted(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
        reason: str,
    ) -> bool:
        now_utc = datetime.now(timezone.utc)
        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE idempotency_reservations
                    SET status = 'INTERRUPTED',
                        updated_at = %s
                    WHERE project_id = %s AND idempotency_key = %s AND fencing_token = %s;
                    """,
                    (now_utc, project_id, idempotency_key, fencing_token),
                )
                return cur.rowcount > 0

    def release(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
    ) -> None:
        now_utc = datetime.now(timezone.utc)
        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE idempotency_reservations
                    SET status = 'INTERRUPTED',
                        updated_at = %s
                    WHERE project_id = %s AND idempotency_key = %s
                      AND fencing_token = %s AND status = 'IN_PROGRESS';
                    """,
                    (now_utc, project_id, idempotency_key, fencing_token),
                )

    def clear(self) -> None:
        with self.db_manager.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM idempotency_reservations;")
            conn.commit()


class SQLiteIdempotencyCoordinator(BaseIdempotencyCoordinator):
    """
    Offline/local concurrency coordinator backed by SQLite.
    
    Guarantees:
    - PRIMARY KEY (project_id, idempotency_key) prevents duplicate records.
    - Transactional locks (BEGIN IMMEDIATE) serialize competing workers across processes.
    - Same lease expiration, heartbeat, and fencing token semantics as PostgreSQL.
    """

    def __init__(self, database_path: Union[str, Path] = "interactions.db") -> None:
        self.database_path = str(database_path)
        self._lock = threading.RLock()
        self._shared_conn: Optional[sqlite3.Connection] = None

        if self.database_path != ":memory:":
            db_dir = Path(self.database_path).parent
            if db_dir and not db_dir.exists():
                db_dir.mkdir(parents=True, exist_ok=True)

        # For in-memory databases, keep a persistent connection open so data is not wiped between queries
        if self.database_path == ":memory:":
            self._shared_conn = sqlite3.connect(
                self.database_path,
                check_same_thread=False,
                isolation_level=None,
            )
            self._init_schema(self._shared_conn)
        else:
            with sqlite3.connect(self.database_path, timeout=10.0) as conn:
                self._init_schema(conn)

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_reservations (
                project_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                interaction_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'IN_PROGRESS',
                fencing_token INTEGER NOT NULL DEFAULT 1,
                lease_expires_at TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (project_id, idempotency_key)
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_idemp_res_status ON idempotency_reservations(project_id, status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_idemp_res_lease ON idempotency_reservations(project_id, lease_expires_at);")

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.database_path, timeout=10.0, isolation_level=None)
        return conn

    def close(self) -> None:
        """Close storage resources."""
        if self._shared_conn is not None:
            try:
                self._shared_conn.close()
            except Exception:
                pass
            self._shared_conn = None

    def reserve(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        owner_id: str,
        lease_seconds: float,
        interaction_store: InteractionStore,
        input_content: str,
    ) -> IdempotencyReservation:
        scoped_key = f"{project_id}:{idempotency_key}"
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        lease_exp = (now_utc + timedelta(seconds=lease_seconds)).isoformat()

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT interaction_id, owner_id, status, fencing_token, lease_expires_at, heartbeat_at
                    FROM idempotency_reservations
                    WHERE project_id = ? AND idempotency_key = ?;
                    """,
                    (project_id, idempotency_key),
                )
                row = cur.fetchone()

                if row is None:
                    existing_record = _resolve_existing_record(interaction_store, idempotency_key, project_id)
                    if existing_record is not None:
                        new_iid = existing_record.interaction_id
                        res_status = IdempotencyStatus.INTERRUPTED
                    else:
                        new_iid = f"int_sse_{now_utc.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
                        res_status = IdempotencyStatus.NEW

                    cur.execute(
                        """
                        INSERT INTO idempotency_reservations (
                            project_id, idempotency_key, interaction_id, owner_id, status,
                            fencing_token, lease_expires_at, heartbeat_at, created_at, updated_at
                        ) VALUES (
                            ?, ?, ?, ?, 'IN_PROGRESS', 1, ?, ?, ?, ?
                        );
                        """,
                        (project_id, idempotency_key, new_iid, owner_id, lease_exp, now_iso, now_iso, now_iso),
                    )
                    conn.execute("COMMIT;")
                    return IdempotencyReservation(
                        scoped_key=scoped_key,
                        project_id=project_id,
                        idempotency_key=idempotency_key,
                        interaction_id=new_iid,
                        owner_id=owner_id,
                        fencing_token=1,
                        status=res_status,
                        existing_record=existing_record,
                        coordinator=self,
                        lease_seconds=lease_seconds,
                    )

                (existing_iid, current_owner, current_status, fencing_token, exp_iso, _) = row
                current_lease_exp = datetime.fromisoformat(exp_iso)
                if current_lease_exp.tzinfo is None:
                    current_lease_exp = current_lease_exp.replace(tzinfo=timezone.utc)

                existing_record = _resolve_existing_record(interaction_store, idempotency_key, project_id, existing_iid)
                if existing_record is not None:
                    if existing_record.project_id != project_id:
                        conn.execute("ROLLBACK;")
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail=f"Idempotency key '{idempotency_key}' was created under another project.",
                        )
                    existing_content = existing_record.input.content if existing_record.input else ""
                    if existing_content != input_content:
                        conn.execute("ROLLBACK;")
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail=f"Idempotency key '{idempotency_key}' reused with mismatched input payload.",
                        )

                if current_status == "COMPLETED" or (
                    existing_record and existing_record.status in (InteractionStatus.COMPLETED, InteractionStatus.PARTIAL_SUCCESS)
                ):
                    conn.execute("COMMIT;")
                    return IdempotencyReservation(
                        scoped_key=scoped_key,
                        project_id=project_id,
                        idempotency_key=idempotency_key,
                        interaction_id=existing_iid,
                        owner_id=current_owner,
                        fencing_token=fencing_token,
                        status=IdempotencyStatus.COMPLETED,
                        existing_record=existing_record,
                        coordinator=self,
                        lease_seconds=lease_seconds,
                    )

                if current_status == "IN_PROGRESS":
                    if now_utc < current_lease_exp:
                        conn.execute("ROLLBACK;")
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail=(
                                f"An interaction with idempotency key '{idempotency_key}' is "
                                f"currently IN_PROGRESS in project '{project_id}'."
                            ),
                        )
                    else:
                        new_token = fencing_token + 1
                        cur.execute(
                            """
                            UPDATE idempotency_reservations
                            SET owner_id = ?,
                                status = 'IN_PROGRESS',
                                fencing_token = ?,
                                lease_expires_at = ?,
                                heartbeat_at = ?,
                                updated_at = ?
                            WHERE project_id = ? AND idempotency_key = ?;
                            """,
                            (owner_id, new_token, lease_exp, now_iso, now_iso, project_id, idempotency_key),
                        )
                        conn.execute("COMMIT;")
                        return IdempotencyReservation(
                            scoped_key=scoped_key,
                            project_id=project_id,
                            idempotency_key=idempotency_key,
                            interaction_id=existing_iid,
                            owner_id=owner_id,
                            fencing_token=new_token,
                            status=IdempotencyStatus.INTERRUPTED,
                            existing_record=existing_record,
                            coordinator=self,
                            lease_seconds=lease_seconds,
                        )

                new_token = fencing_token + 1
                cur.execute(
                    """
                    UPDATE idempotency_reservations
                    SET owner_id = ?,
                        status = 'IN_PROGRESS',
                        fencing_token = ?,
                        lease_expires_at = ?,
                        heartbeat_at = ?,
                        updated_at = ?
                    WHERE project_id = ? AND idempotency_key = ?;
                    """,
                    (owner_id, new_token, lease_exp, now_iso, now_iso, project_id, idempotency_key),
                )
                conn.execute("COMMIT;")
                return IdempotencyReservation(
                    scoped_key=scoped_key,
                    project_id=project_id,
                    idempotency_key=idempotency_key,
                    interaction_id=existing_iid,
                    owner_id=owner_id,
                    fencing_token=new_token,
                    status=IdempotencyStatus.INTERRUPTED,
                    existing_record=existing_record,
                    coordinator=self,
                    lease_seconds=lease_seconds,
                )
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                if self._shared_conn is None:
                    conn.close()

    def heartbeat(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
        lease_seconds: float,
    ) -> bool:
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        lease_exp = (now_utc + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE idempotency_reservations
                    SET lease_expires_at = ?,
                        heartbeat_at = ?,
                        updated_at = ?
                    WHERE project_id = ? AND idempotency_key = ?
                      AND fencing_token = ? AND status = 'IN_PROGRESS';
                    """,
                    (lease_exp, now_iso, now_iso, project_id, idempotency_key, fencing_token),
                )
                return cur.rowcount > 0
            finally:
                if self._shared_conn is None:
                    conn.close()

    def complete(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
    ) -> bool:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE idempotency_reservations
                    SET status = 'COMPLETED',
                        updated_at = ?
                    WHERE project_id = ? AND idempotency_key = ?
                      AND fencing_token = ? AND status = 'IN_PROGRESS';
                    """,
                    (now_iso, project_id, idempotency_key, fencing_token),
                )
                return cur.rowcount > 0
            finally:
                if self._shared_conn is None:
                    conn.close()

    def execute_fenced_completion(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
        completion_fn: Callable[[], T],
    ) -> T:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT fencing_token, status
                    FROM idempotency_reservations
                    WHERE project_id = ? AND idempotency_key = ?;
                    """,
                    (project_id, idempotency_key),
                )
                row = cur.fetchone()
                if row is None:
                    conn.execute("ROLLBACK;")
                    raise StaleFencingTokenError(
                        f"Reservation for key '{idempotency_key}' does not exist in project '{project_id}'."
                    )

                current_token, current_status = row
                if current_token != fencing_token:
                    conn.execute("ROLLBACK;")
                    raise StaleFencingTokenError(
                        f"Cannot commit completion for key '{idempotency_key}': fencing token {fencing_token} "
                        f"is obsolete (current token is {current_token})."
                    )

                if current_status not in ("IN_PROGRESS", "COMPLETED"):
                    conn.execute("ROLLBACK;")
                    raise StaleFencingTokenError(
                        f"Cannot commit completion for key '{idempotency_key}': status is '{current_status}'."
                    )

                # Execute completion side effects
                result = completion_fn()

                cur.execute(
                    """
                    UPDATE idempotency_reservations
                    SET status = 'COMPLETED',
                        updated_at = ?
                    WHERE project_id = ? AND idempotency_key = ? AND fencing_token = ?;
                    """,
                    (now_iso, project_id, idempotency_key, fencing_token),
                )
                conn.execute("COMMIT;")
                return result
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                if self._shared_conn is None:
                    conn.close()

    def mark_interrupted(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
        reason: str,
    ) -> bool:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE idempotency_reservations
                    SET status = 'INTERRUPTED',
                        updated_at = ?
                    WHERE project_id = ? AND idempotency_key = ? AND fencing_token = ?;
                    """,
                    (now_iso, project_id, idempotency_key, fencing_token),
                )
                return cur.rowcount > 0
            finally:
                if self._shared_conn is None:
                    conn.close()

    def release(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        fencing_token: int,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    UPDATE idempotency_reservations
                    SET status = 'INTERRUPTED',
                        updated_at = ?
                    WHERE project_id = ? AND idempotency_key = ?
                      AND fencing_token = ? AND status = 'IN_PROGRESS';
                    """,
                    (now_iso, project_id, idempotency_key, fencing_token),
                )
            finally:
                if self._shared_conn is None:
                    conn.close()

    def clear(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM idempotency_reservations;")
            finally:
                if self._shared_conn is None:
                    conn.close()


class IdempotencyManager:
    """Unified entry point managing idempotency via PostgreSQL or SQLite coordinators."""

    def __init__(
        self,
        coordinator: Optional[BaseIdempotencyCoordinator] = None,
        default_lease_seconds: float = 10.0,
    ) -> None:
        self.coordinator: BaseIdempotencyCoordinator = coordinator or SQLiteIdempotencyCoordinator(":memory:")
        self.default_lease_seconds = default_lease_seconds
        self.owner_id = f"worker_{os.getpid()}_{uuid.uuid4().hex[:8]}"

    def reserve(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        input_content: str,
        interaction_store: InteractionStore,
        lease_seconds: Optional[float] = None,
    ) -> IdempotencyReservation:
        return self.coordinator.reserve(
            project_id=project_id,
            idempotency_key=idempotency_key,
            owner_id=self.owner_id,
            lease_seconds=lease_seconds or self.default_lease_seconds,
            interaction_store=interaction_store,
            input_content=input_content,
        )

    def release_active(self, scoped_key: str) -> None:
        """Compatibility helper."""
        parts = scoped_key.split(":", 1)
        if len(parts) == 2:
            self.coordinator.release(project_id=parts[0], idempotency_key=parts[1], fencing_token=0)

    def clear(self) -> None:
        """Clear all active reservations (test fixture helper)."""
        self.coordinator.clear()

    def close(self) -> None:
        """Close coordinator resources."""
        if hasattr(self.coordinator, "close"):
            self.coordinator.close()
