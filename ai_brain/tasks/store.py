"""
Durable PostgreSQL and SQLite Task Store engines for AI Brain Background Workers (M5.6).

Guarantees:
1. Multi-Process Atomic Claiming:
   PostgreSQL uses `SELECT ... FOR UPDATE SKIP LOCKED` inside a short transaction
   to guarantee that multiple concurrent workers never block or claim identical tasks.
2. Zero Long-Running Database Transactions:
   Claims, heartbeats, and acknowledgements use short-lived transactions. Database
   connections are never held open while long-running agent or workflow handlers execute.
3. Visibility Timeouts & Independent Heartbeats:
   Active execution holds a time-bounded lease (`locked_until`). Active workers periodically
   renew their lease. Abandoned or crashed tasks automatically become claimable once the lease expires.
4. Fencing Protection Against Zombie Workers:
   Every status update, heartbeat, and completion validates `locked_by = worker_id`.
   If a worker's lease expired and was reclaimed, the stale worker is rejected and cannot commit side effects.
5. Strict Tenant Isolation & Row Level Security:
   PostgreSQL queries operate under `tenant_context(project_id)` enforcing RLS.
6. Offline Parity:
   SQLiteTaskStore provides identical queuing, leasing, and fencing semantics for local development.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Sequence, Union
import uuid

from ai_brain.storage.database import DatabaseConnectionManager
from ai_brain.tasks.exceptions import (
    DuplicateTaskError,
    TaskClaimError,
    TaskFencingError,
    TaskNotFoundError,
)
from ai_brain.tasks.models import RetryPolicy, TaskRecord, TaskStatus

logger = logging.getLogger("ai_brain.tasks.store")


class BaseTaskStore(ABC):
    """Abstract interface for durable task queue stores."""

    @abstractmethod
    def submit_task(
        self,
        project_id: str,
        task_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        task_id: Optional[str] = None,
        priority: Any = None,
        max_retries: int = 3,
        idempotency_key: Optional[str] = None,
    ) -> TaskRecord:
        """Submit a new task to the queue with optional idempotency de-duplication."""
        pass

    @abstractmethod
    def claim_next_task(
        self,
        project_id: str,
        worker_id: str,
        lease_seconds: float = 30.0,
        task_types: Optional[Sequence[str]] = None,
    ) -> Optional[TaskRecord]:
        """Atomically claim the next claimable task using visibility timeout semantics."""
        pass

    @abstractmethod
    def heartbeat(
        self,
        project_id: str,
        task_id: str,
        worker_id: str,
        lease_seconds: float = 30.0,
    ) -> bool:
        """Renew active lease. Returns False if ownership was lost or reclaimed."""
        pass

    @abstractmethod
    def complete_task(
        self,
        project_id: str,
        task_id: str,
        worker_id: str,
        result: Optional[Any] = None,
        result_payload: Optional[Dict[str, Any]] = None,
    ) -> TaskRecord:
        """Mark task as COMPLETED. Raises TaskFencingError if fencing check failed."""
        pass

    @abstractmethod
    def fail_task(
        self,
        project_id: str,
        task_id: str,
        worker_id: str,
        error_message: str,
        is_permanent: bool = False,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> TaskRecord:
        """Mark task failure, schedule exponential retry, or route to Dead-Letter Queue."""
        pass

    @abstractmethod
    def cancel_task(
        self,
        project_id: str,
        task_id: str,
        reason: str = "Cancelled by user",
    ) -> bool:
        """Cancel a pending or processing task."""
        pass

    @abstractmethod
    def get_task(
        self,
        project_id: str,
        task_id: str,
    ) -> Optional[TaskRecord]:
        """Retrieve a task record by ID within project tenant scope."""
        pass

    @abstractmethod
    def list_tasks(
        self,
        project_id: str,
        status: Optional[TaskStatus] = None,
        task_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[TaskRecord]:
        """List tasks matching filters in the project tenant."""
        pass

    @abstractmethod
    def list_dead_letters(
        self,
        project_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TaskRecord]:
        """Retrieve dead-letter tasks (FAILED status) in the project."""
        pass

    @abstractmethod
    def retry_dead_letter(
        self,
        project_id: str,
        task_id: str,
    ) -> bool:
        """Manually resurrect a dead-letter task back to PENDING status."""
        pass

    @abstractmethod
    def get_queue_stats(
        self,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return diagnostic metrics on queue depths."""
        pass

    @abstractmethod
    def clear(self, *, project_id: Optional[str] = None) -> None:
        """Purge tasks (test helper)."""
        pass

    def close(self) -> None:
        """Release any persistent resources."""
        pass


class PostgreSQLTaskStore(BaseTaskStore):
    """
    Production PostgreSQL implementation of the durable task queue.
    
    Uses `FOR UPDATE SKIP LOCKED` for lock-free parallel worker claiming,
    PostgreSQL RLS for multi-tenant isolation, and explicit lease checks.
    """

    def __init__(self, db_manager: DatabaseConnectionManager) -> None:
        self.db_manager = db_manager

    def submit_task(
        self,
        project_id: str,
        task_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        task_id: Optional[str] = None,
        priority: Any = None,
        max_retries: int = 3,
        idempotency_key: Optional[str] = None,
    ) -> TaskRecord:
        if not project_id:
            raise ValueError("project_id must not be empty.")
        if not task_type:
            raise ValueError("task_type must not be empty.")

        task_payload = dict(payload or {})
        if priority is not None:
            task_payload["priority"] = priority.value if hasattr(priority, "value") else str(priority)
        if idempotency_key:
            task_payload["idempotency_key"] = idempotency_key
            # Deterministic task_id based on project and idempotency key
            derived_id = f"task_{hashlib.sha256(f'{project_id}:{idempotency_key}'.encode()).hexdigest()[:32]}"
            tid = task_id or derived_id
        else:
            tid = task_id or f"task_{uuid.uuid4().hex[:16]}"

        now_utc = datetime.now(timezone.utc)
        meta_json = json.dumps(task_payload)

        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                # Check for duplicate idempotency key
                if idempotency_key:
                    cur.execute(
                        """
                        SELECT task_id, project_id, task_type, payload, status, retry_count,
                               max_retries, locked_by, locked_until, error_message, created_at, updated_at
                        FROM background_tasks
                        WHERE project_id = %s AND (task_id = %s OR payload->>'idempotency_key' = %s);
                        """,
                        (project_id, tid, idempotency_key),
                    )
                    row = cur.fetchone()
                    if row:
                        existing = self._row_to_record(row)
                        if existing.status in (TaskStatus.PENDING, TaskStatus.PROCESSING, TaskStatus.COMPLETED):
                            raise DuplicateTaskError(
                                f"Task with idempotency key '{idempotency_key}' already exists in project '{project_id}' "
                                f"(status: {existing.status})."
                            )

                cur.execute(
                    """
                    INSERT INTO background_tasks (
                        task_id, project_id, task_type, payload, status, retry_count,
                        max_retries, locked_by, locked_until, error_message, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s::jsonb, 'PENDING', 0, %s, NULL, NULL, NULL, %s, %s
                    )
                    ON CONFLICT (task_id) DO NOTHING;
                    """,
                    (tid, project_id, task_type, meta_json, max_retries, now_utc, now_utc),
                )
                if cur.rowcount == 0:
                    raise DuplicateTaskError(
                        f"Task '{tid}' already exists in project '{project_id}'."
                    )

        return TaskRecord(
            task_id=tid,
            project_id=project_id,
            task_type=task_type,
            payload=task_payload,
            status=TaskStatus.PENDING,
            retry_count=0,
            max_retries=max_retries,
            locked_by=None,
            locked_until=None,
            error_message=None,
            created_at=now_utc,
            updated_at=now_utc,
        )

    def claim_next_task(
        self,
        project_id: str,
        worker_id: str,
        lease_seconds: float = 30.0,
        task_types: Optional[Sequence[str]] = None,
    ) -> Optional[TaskRecord]:
        now_utc = datetime.now(timezone.utc)
        lease_until = now_utc + timedelta(seconds=lease_seconds)

        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                # Build type filtering
                type_clause = ""
                params: List[Any] = [now_utc]
                if task_types:
                    type_clause = "AND task_type = ANY(%s)"
                    params.append(list(task_types))

                # FOR UPDATE SKIP LOCKED guarantees lock-free multi-worker concurrency
                query = f"""
                    SELECT task_id
                    FROM background_tasks
                    WHERE (status = 'PENDING' OR (status = 'PROCESSING' AND locked_until < %s))
                    {type_clause}
                    ORDER BY created_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED;
                """
                cur.execute(query, tuple(params))
                row = cur.fetchone()
                if not row:
                    return None

                claimed_id = row[0]
                cur.execute(
                    """
                    UPDATE background_tasks
                    SET status = 'PROCESSING',
                        locked_by = %s,
                        locked_until = %s,
                        retry_count = CASE WHEN status = 'PROCESSING' THEN retry_count + 1 ELSE retry_count END,
                        updated_at = %s
                    WHERE task_id = %s AND project_id = %s
                    RETURNING task_id, project_id, task_type, payload, status, retry_count,
                              max_retries, locked_by, locked_until, error_message, created_at, updated_at;
                    """,
                    (worker_id, lease_until, now_utc, claimed_id, project_id),
                )
                updated_row = cur.fetchone()
                if not updated_row:
                    return None

                return self._row_to_record(updated_row)

    def heartbeat(
        self,
        project_id: str,
        task_id: str,
        worker_id: str,
        lease_seconds: float = 30.0,
    ) -> bool:
        now_utc = datetime.now(timezone.utc)
        new_exp = now_utc + timedelta(seconds=lease_seconds)

        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE background_tasks
                    SET locked_until = %s,
                        updated_at = %s
                    WHERE task_id = %s AND project_id = %s
                      AND locked_by = %s AND status = 'PROCESSING';
                    """,
                    (new_exp, now_utc, task_id, project_id, worker_id),
                )
                return cur.rowcount > 0

    def complete_task(
        self,
        project_id: str,
        task_id: str,
        worker_id: str,
        result: Optional[Any] = None,
        result_payload: Optional[Dict[str, Any]] = None,
    ) -> TaskRecord:
        now_utc = datetime.now(timezone.utc)
        effective_result = result_payload
        if effective_result is None and result is not None:
            if hasattr(result, "result_data"):
                effective_result = result.result_data
            elif isinstance(result, dict):
                effective_result = result

        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                # If result provided, merge into payload
                if effective_result:
                    cur.execute(
                        """
                        UPDATE background_tasks
                        SET status = 'COMPLETED',
                            locked_by = NULL,
                            locked_until = NULL,
                            payload = payload || %s::jsonb,
                            updated_at = %s
                        WHERE task_id = %s AND project_id = %s
                          AND locked_by = %s AND status = 'PROCESSING';
                        """,
                        (json.dumps({"result": effective_result}), now_utc, task_id, project_id, worker_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE background_tasks
                        SET status = 'COMPLETED',
                            locked_by = NULL,
                            locked_until = NULL,
                            updated_at = %s
                        WHERE task_id = %s AND project_id = %s
                          AND locked_by = %s AND status = 'PROCESSING';
                        """,
                        (now_utc, task_id, project_id, worker_id),
                    )
                if cur.rowcount == 0:
                    raise TaskFencingError(
                        f"Cannot complete task '{task_id}': worker '{worker_id}' is not the active owner or task is not in PROCESSING status."
                    )
                cur.execute(
                    """
                    SELECT task_id, project_id, task_type, payload, status, retry_count,
                           max_retries, locked_by, locked_until, error_message, created_at, updated_at
                    FROM background_tasks
                    WHERE task_id = %s AND project_id = %s;
                    """,
                    (task_id, project_id),
                )
                row = cur.fetchone()
                return self._row_to_record(row)

    def fail_task(
        self,
        *,
        project_id: str,
        task_id: str,
        worker_id: str,
        error_message: str,
        is_permanent: bool = False,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> TaskRecord:
        policy = retry_policy or RetryPolicy()
        now_utc = datetime.now(timezone.utc)

        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                # Verify worker holds ownership
                cur.execute(
                    """
                    SELECT task_id, project_id, task_type, payload, status, retry_count,
                           max_retries, locked_by, locked_until, error_message, created_at, updated_at
                    FROM background_tasks
                    WHERE task_id = %s AND project_id = %s
                    FOR UPDATE;
                    """,
                    (task_id, project_id),
                )
                row = cur.fetchone()
                if not row:
                    raise TaskNotFoundError(f"Task '{task_id}' not found in project '{project_id}'.")

                record = self._row_to_record(row)
                if record.locked_by != worker_id or record.status != TaskStatus.PROCESSING:
                    raise TaskFencingError(
                        f"Cannot fail task '{task_id}': worker '{worker_id}' is not the active owner "
                        f"(current owner: '{record.locked_by}', status: {record.status})."
                    )

                next_retry = record.retry_count + 1
                should_dead_letter = is_permanent or (next_retry >= record.max_retries)

                if should_dead_letter:
                    reason = "PERMANENT_ERROR" if is_permanent else f"RETRIES_EXHAUSTED ({next_retry}/{record.max_retries})"
                    formatted_err = f"{reason}: {error_message}"
                    cur.execute(
                        """
                        UPDATE background_tasks
                        SET status = 'FAILED',
                            retry_count = %s,
                            locked_by = NULL,
                            locked_until = NULL,
                            error_message = %s,
                            updated_at = %s
                        WHERE task_id = %s AND project_id = %s
                        RETURNING task_id, project_id, task_type, payload, status, retry_count,
                                  max_retries, locked_by, locked_until, error_message, created_at, updated_at;
                        """,
                        (next_retry, formatted_err, now_utc, task_id, project_id),
                    )
                else:
                    delay_sec = policy.compute_delay(next_retry)
                    retry_visible_at = now_utc + timedelta(seconds=delay_sec)
                    formatted_err = f"ATTEMPT_{next_retry}_FAILED: {error_message}"
                    cur.execute(
                        """
                        UPDATE background_tasks
                        SET status = 'PENDING',
                            retry_count = %s,
                            locked_by = NULL,
                            locked_until = %s,
                            error_message = %s,
                            updated_at = %s
                        WHERE task_id = %s AND project_id = %s
                        RETURNING task_id, project_id, task_type, payload, status, retry_count,
                                  max_retries, locked_by, locked_until, error_message, created_at, updated_at;
                        """,
                        (next_retry, retry_visible_at, formatted_err, now_utc, task_id, project_id),
                    )

                updated_row = cur.fetchone()
                return self._row_to_record(updated_row)

    def cancel_task(
        self,
        project_id: str,
        task_id: str,
        reason: str = "Cancelled by user",
    ) -> bool:
        now_utc = datetime.now(timezone.utc)
        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE background_tasks
                    SET status = 'CANCELLED',
                        locked_by = NULL,
                        locked_until = NULL,
                        error_message = %s,
                        updated_at = %s
                    WHERE task_id = %s AND project_id = %s
                      AND status IN ('PENDING', 'PROCESSING');
                    """,
                    (f"CANCELLED: {reason}", now_utc, task_id, project_id),
                )
                return cur.rowcount > 0

    def get_task(
        self,
        project_id: str,
        task_id: str,
    ) -> Optional[TaskRecord]:
        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT task_id, project_id, task_type, payload, status, retry_count,
                           max_retries, locked_by, locked_until, error_message, created_at, updated_at
                    FROM background_tasks
                    WHERE task_id = %s AND project_id = %s;
                    """,
                    (task_id, project_id),
                )
                row = cur.fetchone()
                return self._row_to_record(row) if row else None

    def list_tasks(
        self,
        project_id: str,
        status: Optional[TaskStatus] = None,
        task_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[TaskRecord]:
        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                conditions = ["project_id = %s"]
                params: List[Any] = [project_id]

                if status:
                    conditions.append("status = %s")
                    params.append(status.value if hasattr(status, "value") else str(status))
                if task_type:
                    conditions.append("task_type = %s")
                    params.append(task_type)

                params.append(limit)
                query = f"""
                    SELECT task_id, project_id, task_type, payload, status, retry_count,
                           max_retries, locked_by, locked_until, error_message, created_at, updated_at
                    FROM background_tasks
                    WHERE {" AND ".join(conditions)}
                    ORDER BY created_at DESC
                    LIMIT %s;
                """
                cur.execute(query, tuple(params))
                return [self._row_to_record(r) for r in cur.fetchall()]

    def list_dead_letters(
        self,
        project_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TaskRecord]:
        return self.list_tasks(project_id=project_id, status=TaskStatus.FAILED, limit=limit)

    def retry_dead_letter(
        self,
        project_id: str,
        task_id: str,
    ) -> bool:
        now_utc = datetime.now(timezone.utc)
        with self.db_manager.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE background_tasks
                    SET status = 'PENDING',
                        retry_count = 0,
                        locked_by = NULL,
                        locked_until = NULL,
                        error_message = NULL,
                        updated_at = %s
                    WHERE task_id = %s AND project_id = %s AND status = 'FAILED';
                    """,
                    (now_utc, task_id, project_id),
                )
                return cur.rowcount > 0

    def get_queue_stats(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        with self.db_manager.tenant_context(project_id) if project_id else self.db_manager.connection() as conn:
            with conn.cursor() as cur:
                if project_id:
                    cur.execute(
                        """
                        SELECT status, COUNT(*)
                        FROM background_tasks
                        WHERE project_id = %s
                        GROUP BY status;
                        """,
                        (project_id,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT status, COUNT(*)
                        FROM background_tasks
                        GROUP BY status;
                        """
                    )
                counts = {row[0]: row[1] for row in cur.fetchall()}
                return {
                    "pending": counts.get("PENDING", 0),
                    "processing": counts.get("PROCESSING", 0),
                    "completed": counts.get("COMPLETED", 0),
                    "failed": counts.get("FAILED", 0),
                    "cancelled": counts.get("CANCELLED", 0),
                    "total": sum(counts.values()),
                }

    def clear(self, *, project_id: Optional[str] = None) -> None:
        if project_id:
            with self.db_manager.tenant_context(project_id) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM background_tasks WHERE project_id = %s;", (project_id,))
        else:
            with self.db_manager.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM background_tasks;")
                conn.commit()

    @staticmethod
    def _row_to_record(row: Any) -> TaskRecord:
        (
            tid,
            pid,
            ttype,
            payload_data,
            st,
            rc,
            mr,
            lb,
            lu,
            err,
            cat,
            uat,
        ) = row

        payload = payload_data if isinstance(payload_data, dict) else (json.loads(payload_data) if payload_data else {})
        status_enum = TaskStatus(st) if st in TaskStatus.__members__ else TaskStatus(st.upper())

        if cat and cat.tzinfo is None:
            cat = cat.replace(tzinfo=timezone.utc)
        if uat and uat.tzinfo is None:
            uat = uat.replace(tzinfo=timezone.utc)
        if lu and lu.tzinfo is None:
            lu = lu.replace(tzinfo=timezone.utc)

        return TaskRecord(
            task_id=tid,
            project_id=pid,
            task_type=ttype,
            payload=payload,
            status=status_enum,
            retry_count=rc,
            max_retries=mr,
            locked_by=lb,
            locked_until=lu,
            error_message=err,
            created_at=cat,
            updated_at=uat,
        )


class SQLiteTaskStore(BaseTaskStore):
    """
    Offline/local SQLite implementation of the durable task queue.
    
    Provides identical status lifecycle, visibility timeout expiration,
    fencing protection, and dead-letter queue behavior for offline operation.
    """

    def __init__(self, database_path: Union[str, Path] = ":memory:") -> None:
        self.database_path = str(database_path)
        self._lock = threading.RLock()
        self._shared_conn: Optional[sqlite3.Connection] = None

        if self.database_path != ":memory:":
            db_dir = Path(self.database_path).parent
            if db_dir and not db_dir.exists():
                db_dir.mkdir(parents=True, exist_ok=True)

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
            CREATE TABLE IF NOT EXISTS background_tasks (
                task_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'PENDING',
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                locked_by TEXT,
                locked_until TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_project_status ON background_tasks(project_id, status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status_lease ON background_tasks(status, locked_until);")

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        return sqlite3.connect(self.database_path, timeout=10.0, isolation_level=None)

    def close(self) -> None:
        if self._shared_conn is not None:
            try:
                self._shared_conn.close()
            except Exception:
                pass
            self._shared_conn = None

    def submit_task(
        self,
        project_id: str,
        task_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        task_id: Optional[str] = None,
        priority: Any = None,
        max_retries: int = 3,
        idempotency_key: Optional[str] = None,
    ) -> TaskRecord:
        if not project_id:
            raise ValueError("project_id must not be empty.")
        if not task_type:
            raise ValueError("task_type must not be empty.")

        task_payload = dict(payload or {})
        if priority is not None:
            task_payload["priority"] = priority.value if hasattr(priority, "value") else str(priority)
        if idempotency_key:
            task_payload["idempotency_key"] = idempotency_key
            derived_id = f"task_{hashlib.sha256(f'{project_id}:{idempotency_key}'.encode()).hexdigest()[:32]}"
            tid = task_id or derived_id
        else:
            tid = task_id or f"task_{uuid.uuid4().hex[:16]}"

        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        meta_json = json.dumps(task_payload)

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                cur = conn.cursor()
                if idempotency_key:
                    cur.execute(
                        """
                        SELECT task_id, status FROM background_tasks
                        WHERE project_id = ? AND (task_id = ? OR json_extract(payload, '$.idempotency_key') = ?);
                        """,
                        (project_id, tid, idempotency_key),
                    )
                    row = cur.fetchone()
                    if row:
                        conn.execute("ROLLBACK;")
                        raise DuplicateTaskError(
                            f"Task with idempotency key '{idempotency_key}' already exists in project '{project_id}' "
                            f"(status: {row[1]})."
                        )

                cur.execute(
                    """
                    INSERT INTO background_tasks (
                        task_id, project_id, task_type, payload, status, retry_count,
                        max_retries, locked_by, locked_until, error_message, created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, 'PENDING', 0, ?, NULL, NULL, NULL, ?, ?
                    );
                    """,
                    (tid, project_id, task_type, meta_json, max_retries, now_iso, now_iso),
                )
                conn.execute("COMMIT;")
            except sqlite3.IntegrityError as e:
                conn.execute("ROLLBACK;")
                raise DuplicateTaskError(f"Task '{tid}' already exists in project '{project_id}': {e}") from e
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                if self._shared_conn is None:
                    conn.close()

        return TaskRecord(
            task_id=tid,
            project_id=project_id,
            task_type=task_type,
            payload=task_payload,
            status=TaskStatus.PENDING,
            retry_count=0,
            max_retries=max_retries,
            locked_by=None,
            locked_until=None,
            error_message=None,
            created_at=now_utc,
            updated_at=now_utc,
        )

    def claim_next_task(
        self,
        project_id: str,
        worker_id: str,
        lease_seconds: float = 30.0,
        task_types: Optional[Sequence[str]] = None,
    ) -> Optional[TaskRecord]:
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        lease_until_iso = (now_utc + timedelta(seconds=lease_seconds)).isoformat()

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                cur = conn.cursor()

                type_clause = ""
                params: List[Any] = [project_id, now_iso]
                if task_types:
                    placeholders = ",".join(["?"] * len(task_types))
                    type_clause = f"AND task_type IN ({placeholders})"
                    params.extend(task_types)

                query = f"""
                    SELECT task_id
                    FROM background_tasks
                    WHERE project_id = ?
                      AND (status = 'PENDING' OR (status = 'PROCESSING' AND locked_until < ?))
                      {type_clause}
                    ORDER BY created_at ASC
                    LIMIT 1;
                """
                cur.execute(query, tuple(params))
                row = cur.fetchone()
                if not row:
                    conn.execute("COMMIT;")
                    return None

                claimed_id = row[0]
                cur.execute(
                    """
                    UPDATE background_tasks
                    SET status = 'PROCESSING',
                        locked_by = ?,
                        locked_until = ?,
                        retry_count = CASE WHEN status = 'PROCESSING' THEN retry_count + 1 ELSE retry_count END,
                        updated_at = ?
                    WHERE task_id = ? AND project_id = ?;
                    """,
                    (worker_id, lease_until_iso, now_iso, claimed_id, project_id),
                )
                cur.execute(
                    """
                    SELECT task_id, project_id, task_type, payload, status, retry_count,
                           max_retries, locked_by, locked_until, error_message, created_at, updated_at
                    FROM background_tasks
                    WHERE task_id = ? AND project_id = ?;
                    """,
                    (claimed_id, project_id),
                )
                updated_row = cur.fetchone()
                conn.execute("COMMIT;")
                return self._row_to_record(updated_row) if updated_row else None
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
        project_id: str,
        task_id: str,
        worker_id: str,
        lease_seconds: float = 30.0,
    ) -> bool:
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        new_exp_iso = (now_utc + timedelta(seconds=lease_seconds)).isoformat()

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE background_tasks
                    SET locked_until = ?,
                        updated_at = ?
                    WHERE task_id = ? AND project_id = ?
                      AND locked_by = ? AND status = 'PROCESSING';
                    """,
                    (new_exp_iso, now_iso, task_id, project_id, worker_id),
                )
                return cur.rowcount > 0
            finally:
                if self._shared_conn is None:
                    conn.close()

    def complete_task(
        self,
        project_id: str,
        task_id: str,
        worker_id: str,
        result: Optional[Any] = None,
        result_payload: Optional[Dict[str, Any]] = None,
    ) -> TaskRecord:
        now_iso = datetime.now(timezone.utc).isoformat()
        effective_result = result_payload
        if effective_result is None and result is not None:
            if hasattr(result, "result_data"):
                effective_result = result.result_data
            elif isinstance(result, dict):
                effective_result = result

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT payload FROM background_tasks
                    WHERE task_id = ? AND project_id = ? AND locked_by = ? AND status = 'PROCESSING';
                    """,
                    (task_id, project_id, worker_id),
                )
                row = cur.fetchone()
                if not row:
                    conn.execute("ROLLBACK;")
                    raise TaskFencingError(
                        f"Cannot complete task '{task_id}': worker '{worker_id}' is not the active owner or task is not in PROCESSING status."
                    )

                current_payload = json.loads(row[0]) if row[0] else {}
                if effective_result:
                    current_payload["result"] = effective_result

                cur.execute(
                    """
                    UPDATE background_tasks
                    SET status = 'COMPLETED',
                        locked_by = NULL,
                        locked_until = NULL,
                        payload = ?,
                        updated_at = ?
                    WHERE task_id = ? AND project_id = ?
                      AND locked_by = ? AND status = 'PROCESSING';
                    """,
                    (json.dumps(current_payload), now_iso, task_id, project_id, worker_id),
                )
                cur.execute(
                    """
                    SELECT task_id, project_id, task_type, payload, status, retry_count,
                           max_retries, locked_by, locked_until, error_message, created_at, updated_at
                    FROM background_tasks
                    WHERE task_id = ? AND project_id = ?;
                    """,
                    (task_id, project_id),
                )
                completed_row = cur.fetchone()
                conn.execute("COMMIT;")
                return self._row_to_record(completed_row)
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                if self._shared_conn is None:
                    conn.close()

    def fail_task(
        self,
        *,
        project_id: str,
        task_id: str,
        worker_id: str,
        error_message: str,
        is_permanent: bool = False,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> TaskRecord:
        policy = retry_policy or RetryPolicy()
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT task_id, project_id, task_type, payload, status, retry_count,
                           max_retries, locked_by, locked_until, error_message, created_at, updated_at
                    FROM background_tasks
                    WHERE task_id = ? AND project_id = ?;
                    """,
                    (task_id, project_id),
                )
                row = cur.fetchone()
                if not row:
                    conn.execute("ROLLBACK;")
                    raise TaskNotFoundError(f"Task '{task_id}' not found in project '{project_id}'.")

                record = self._row_to_record(row)
                if record.locked_by != worker_id or record.status != TaskStatus.PROCESSING:
                    conn.execute("ROLLBACK;")
                    raise TaskFencingError(
                        f"Cannot fail task '{task_id}': worker '{worker_id}' is not the active owner "
                        f"(current owner: '{record.locked_by}', status: {record.status})."
                    )

                next_retry = record.retry_count + 1
                should_dead_letter = is_permanent or (next_retry >= record.max_retries)

                if should_dead_letter:
                    reason = "PERMANENT_ERROR" if is_permanent else f"RETRIES_EXHAUSTED ({next_retry}/{record.max_retries})"
                    formatted_err = f"{reason}: {error_message}"
                    cur.execute(
                        """
                        UPDATE background_tasks
                        SET status = 'FAILED',
                            retry_count = ?,
                            locked_by = NULL,
                            locked_until = NULL,
                            error_message = ?,
                            updated_at = ?
                        WHERE task_id = ? AND project_id = ?;
                        """,
                        (next_retry, formatted_err, now_iso, task_id, project_id),
                    )
                else:
                    delay_sec = policy.compute_delay(next_retry)
                    retry_visible_iso = (now_utc + timedelta(seconds=delay_sec)).isoformat()
                    formatted_err = f"ATTEMPT_{next_retry}_FAILED: {error_message}"
                    cur.execute(
                        """
                        UPDATE background_tasks
                        SET status = 'PENDING',
                            retry_count = ?,
                            locked_by = NULL,
                            locked_until = ?,
                            error_message = ?,
                            updated_at = ?
                        WHERE task_id = ? AND project_id = ?;
                        """,
                        (next_retry, retry_visible_iso, formatted_err, now_iso, task_id, project_id),
                    )

                cur.execute(
                    """
                    SELECT task_id, project_id, task_type, payload, status, retry_count,
                           max_retries, locked_by, locked_until, error_message, created_at, updated_at
                    FROM background_tasks
                    WHERE task_id = ? AND project_id = ?;
                    """,
                    (task_id, project_id),
                )
                updated_row = cur.fetchone()
                conn.execute("COMMIT;")
                return self._row_to_record(updated_row)
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                if self._shared_conn is None:
                    conn.close()

    def cancel_task(
        self,
        project_id: str,
        task_id: str,
        reason: str = "Cancelled by user",
    ) -> bool:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE background_tasks
                    SET status = 'CANCELLED',
                        locked_by = NULL,
                        locked_until = NULL,
                        error_message = ?,
                        updated_at = ?
                    WHERE task_id = ? AND project_id = ?
                      AND status IN ('PENDING', 'PROCESSING');
                    """,
                    (f"CANCELLED: {reason}", now_iso, task_id, project_id),
                )
                return cur.rowcount > 0
            finally:
                if self._shared_conn is None:
                    conn.close()

    def get_task(
        self,
        project_id: str,
        task_id: str,
    ) -> Optional[TaskRecord]:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT task_id, project_id, task_type, payload, status, retry_count,
                           max_retries, locked_by, locked_until, error_message, created_at, updated_at
                    FROM background_tasks
                    WHERE task_id = ? AND project_id = ?;
                    """,
                    (task_id, project_id),
                )
                row = cur.fetchone()
                return self._row_to_record(row) if row else None
            finally:
                if self._shared_conn is None:
                    conn.close()

    def list_tasks(
        self,
        project_id: str,
        status: Optional[TaskStatus] = None,
        task_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[TaskRecord]:
        with self._lock:
            conn = self._get_connection()
            try:
                conditions = ["project_id = ?"]
                params: List[Any] = [project_id]

                if status:
                    conditions.append("status = ?")
                    params.append(status.value if hasattr(status, "value") else str(status))
                if task_type:
                    conditions.append("task_type = ?")
                    params.append(task_type)

                params.append(limit)
                query = f"""
                    SELECT task_id, project_id, task_type, payload, status, retry_count,
                           max_retries, locked_by, locked_until, error_message, created_at, updated_at
                    FROM background_tasks
                    WHERE {" AND ".join(conditions)}
                    ORDER BY created_at DESC
                    LIMIT ?;
                """
                cur = conn.cursor()
                cur.execute(query, tuple(params))
                return [self._row_to_record(r) for r in cur.fetchall()]
            finally:
                if self._shared_conn is None:
                    conn.close()

    def list_dead_letters(
        self,
        project_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TaskRecord]:
        return self.list_tasks(project_id=project_id, status=TaskStatus.FAILED, limit=limit)

    def retry_dead_letter(
        self,
        project_id: str,
        task_id: str,
    ) -> bool:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE background_tasks
                    SET status = 'PENDING',
                        retry_count = 0,
                        locked_by = NULL,
                        locked_until = NULL,
                        error_message = NULL,
                        updated_at = ?
                    WHERE task_id = ? AND project_id = ? AND status = 'FAILED';
                    """,
                    (now_iso, task_id, project_id),
                )
                return cur.rowcount > 0
            finally:
                if self._shared_conn is None:
                    conn.close()

    def get_queue_stats(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                if project_id:
                    cur.execute(
                        """
                        SELECT status, COUNT(*)
                        FROM background_tasks
                        WHERE project_id = ?
                        GROUP BY status;
                        """,
                        (project_id,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT status, COUNT(*)
                        FROM background_tasks
                        GROUP BY status;
                        """
                    )
                counts = {row[0]: row[1] for row in cur.fetchall()}
                return {
                    "pending": counts.get("PENDING", 0),
                    "processing": counts.get("PROCESSING", 0),
                    "completed": counts.get("COMPLETED", 0),
                    "failed": counts.get("FAILED", 0),
                    "cancelled": counts.get("CANCELLED", 0),
                    "total": sum(counts.values()),
                }
            finally:
                if self._shared_conn is None:
                    conn.close()

    def clear(self, *, project_id: Optional[str] = None) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                if project_id:
                    conn.execute("DELETE FROM background_tasks WHERE project_id = ?;", (project_id,))
                else:
                    conn.execute("DELETE FROM background_tasks;")
            finally:
                if self._shared_conn is None:
                    conn.close()

    @staticmethod
    def _row_to_record(row: Any) -> TaskRecord:
        (
            tid,
            pid,
            ttype,
            payload_str,
            st,
            rc,
            mr,
            lb,
            lu_str,
            err,
            cat_str,
            uat_str,
        ) = row

        payload = json.loads(payload_str) if payload_str else {}
        status_enum = TaskStatus(st) if st in TaskStatus.__members__ else TaskStatus(st.upper())

        cat = datetime.fromisoformat(cat_str) if cat_str else datetime.now(timezone.utc)
        if cat.tzinfo is None:
            cat = cat.replace(tzinfo=timezone.utc)

        uat = datetime.fromisoformat(uat_str) if uat_str else datetime.now(timezone.utc)
        if uat.tzinfo is None:
            uat = uat.replace(tzinfo=timezone.utc)

        lu = None
        if lu_str:
            lu = datetime.fromisoformat(lu_str)
            if lu.tzinfo is None:
                lu = lu.replace(tzinfo=timezone.utc)

        return TaskRecord(
            task_id=tid,
            project_id=pid,
            task_type=ttype,
            payload=payload,
            status=status_enum,
            retry_count=rc,
            max_retries=mr,
            locked_by=lb,
            locked_until=lu,
            error_message=err,
            created_at=cat,
            updated_at=uat,
        )


def create_task_store(backend: str, target: Any = None) -> BaseTaskStore:
    """Factory helper to create a BaseTaskStore instance."""
    if backend in ("postgres", "postgresql"):
        return PostgreSQLTaskStore(db_manager=target)
    elif backend in ("sqlite", "sqlite3"):
        db_path = target if isinstance(target, str) else ":memory:"
        return SQLiteTaskStore(db_path=db_path)
    else:
        raise ValueError(f"Unsupported task store backend: {backend}")

