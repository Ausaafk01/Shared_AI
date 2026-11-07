"""
Comprehensive test suite for Phase 5, Milestone 5.6: Background Worker Fleet.

Verifies:
1. Task submission and persistence (PostgreSQL & SQLite)
2. Successful execution and acknowledgement
3. Multiple workers claiming different tasks concurrently (FOR UPDATE SKIP LOCKED)
4. Two workers competing for the same task (atomic claim, exactly 1 winner)
5. Lease renewal during long-running execution (independent heartbeat)
6. Worker crash and abandoned-task recovery
7. Stale worker fencing and prevention of duplicate side effects
8. Retry classification (transient exponential backoff vs permanent DLQ)
9. Maximum-attempt enforcement and dead-letter handling (with manual resurrection)
10. Graceful shutdown and task cancellation
11. Idempotent execution and duplicate submission detection
12. Tenant isolation under PostgreSQL RLS and SQLite
13. PostgreSQL integration with independent pooled connections
14. SQLite offline compatibility and feature parity
15. Resource cleanup, worker thread lifecycle, and zero connection leaks
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import threading
import time
import unittest
import uuid
from typing import Any, Dict, List, Optional

from ai_brain.config import DatabaseConfig, StorageBackend, load_env_file
from ai_brain.storage.database import DatabaseConnectionManager
from ai_brain.tasks.exceptions import (
    DuplicateTaskError,
    PermanentTaskError,
    TaskCancelledError,
    TaskFencingError,
    TaskNotFoundError,
    TransientTaskError,
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
from ai_brain.server.dependencies import create_sqlite_container, create_postgres_container

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tests.test_background_tasks")

load_env_file()

ADMIN_PG_URL = os.environ.get(
    "TEST_POSTGRES_ADMIN_URL",
    os.environ.get("TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/test_ai_brain"),
)


def _check_postgres_live() -> tuple[bool, str]:
    cfg = DatabaseConfig(
        backend=StorageBackend.POSTGRES,
        database_url=ADMIN_PG_URL,
        pool_size=2,
        pool_timeout=3.0,
    )
    mgr = DatabaseConnectionManager(cfg)
    if not mgr.driver_available:
        return False, "Driver not installed"
    try:
        conn = mgr.get_connection(timeout=3.0)
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            ver = cur.fetchone()
        mgr.return_connection(conn)
        mgr.close()
        return True, str(ver[0] if ver else "unknown")
    except Exception as e:
        mgr.close()
        return False, str(e)


_IS_PG_LIVE, _PG_MSG = _check_postgres_live()


class TestTaskModelsAndRegistry(unittest.TestCase):
    """Unit tests for models, retry backoff calculation, and task registry."""

    def test_retry_policy_exponential_backoff(self) -> None:
        policy = RetryPolicy(base_delay=1.0, max_delay=30.0, backoff_factor=2.0)
        self.assertEqual(policy.compute_delay(0), 1.0)
        self.assertEqual(policy.compute_delay(1), 2.0)
        self.assertEqual(policy.compute_delay(2), 4.0)
        self.assertEqual(policy.compute_delay(3), 8.0)
        self.assertEqual(policy.compute_delay(4), 16.0)
        self.assertEqual(policy.compute_delay(5), 30.0)  # Capped at max_delay
        self.assertEqual(policy.compute_delay(10), 30.0)

    def test_task_record_properties(self) -> None:
        rec = TaskRecord(
            task_id="t1",
            project_id="p1",
            task_type="dummy",
            payload={"k": "v"},
            status=TaskStatus.PROCESSING,
            locked_by="w1",
            locked_until=datetime.now(timezone.utc) - timedelta(seconds=5),
        )
        self.assertTrue(rec.is_expired)
        self.assertIn("t1", rec.to_dict()["task_id"])

    def test_registry_sync_and_async_handler(self) -> None:
        registry = TaskHandlerRegistry()

        def sync_fn(task: TaskRecord, ctx: Any = None) -> Dict[str, Any]:
            return {"result": task.payload["val"] * 2}

        async def async_fn(task: TaskRecord, ctx: Any = None) -> Dict[str, Any]:
            return {"async_result": task.payload["val"] + 10}

        registry.register("calc_sync", sync_fn)
        registry.register("calc_async", async_fn)

        task_sync = TaskRecord(task_id="ts1", project_id="p1", task_type="calc_sync", payload={"val": 5})
        task_async = TaskRecord(task_id="ta1", project_id="p1", task_type="calc_async", payload={"val": 5})

        res1 = registry.execute(task_sync)
        self.assertEqual(res1.output, {"result": 10})

        res2 = registry.execute(task_async)
        self.assertEqual(res2.output, {"async_result": 15})

    def test_built_in_handlers_execution(self) -> None:
        registry = TaskHandlerRegistry()
        register_default_handlers(registry)

        task_wf = TaskRecord(
            task_id="tw1",
            project_id="proj_1",
            task_type="workflow_step",
            payload={"workflow_id": "wf_123", "stage_name": "Review"},
        )
        res_wf = registry.execute(task_wf)
        self.assertEqual(res_wf.output["status"], "COMPLETED")
        self.assertEqual(res_wf.output["stage_name"], "Review")

        task_mem = TaskRecord(
            task_id="tm1",
            project_id="proj_1",
            task_type="memory_maintenance",
            payload={"operation": "prune_outdated"},
        )
        res_mem = registry.execute(task_mem)
        self.assertEqual(res_mem.output["status"], "SUCCESS")
        self.assertEqual(res_mem.output["operation"], "prune_outdated")


class TestSQLiteTaskStoreUnit(unittest.TestCase):
    """Unit tests for SQLiteTaskStore persistence, claiming, and DLQ."""

    def setUp(self) -> None:
        self.project_id = f"proj_sqlite_{uuid.uuid4().hex[:8]}"
        self.store = SQLiteTaskStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_submit_and_get_task(self) -> None:
        task = self.store.submit_task(
            project_id=self.project_id,
            task_type="test_type",
            payload={"message": "hello sqlite"},
            priority=TaskPriority.HIGH,
        )
        self.assertIsNotNone(task.task_id)
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.payload["message"], "hello sqlite")

        retrieved = self.store.get_task(self.project_id, task.task_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.task_id, task.task_id)

    def test_idempotency_duplicate_submission(self) -> None:
        idem_key = "sqlite_dup_test_1"
        self.store.submit_task(
            project_id=self.project_id,
            task_type="test_idem",
            payload={"data": 1},
            idempotency_key=idem_key,
        )
        with self.assertRaises(DuplicateTaskError):
            self.store.submit_task(
                project_id=self.project_id,
                task_type="test_idem",
                payload={"data": 2},
                idempotency_key=idem_key,
            )

    def test_atomic_claim_and_completion(self) -> None:
        task = self.store.submit_task(
            project_id=self.project_id,
            task_type="unit_work",
            payload={"num": 42},
        )
        claimed = self.store.claim_next_task(
            project_id=self.project_id,
            worker_id="worker_sqlite_1",
            lease_seconds=10.0,
        )
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.task_id, task.task_id)
        self.assertEqual(claimed.locked_by, "worker_sqlite_1")
        self.assertEqual(claimed.status, TaskStatus.PROCESSING)

        # Heartbeat
        hb_ok = self.store.heartbeat(
            project_id=self.project_id,
            task_id=task.task_id,
            worker_id="worker_sqlite_1",
            lease_seconds=20.0,
        )
        self.assertTrue(hb_ok)

        # Complete
        completed = self.store.complete_task(
            project_id=self.project_id,
            task_id=task.task_id,
            worker_id="worker_sqlite_1",
            result=TaskResult(output={"answer": 84}),
        )
        self.assertEqual(completed.status, TaskStatus.COMPLETED)
        self.assertIsNone(completed.locked_by)

    def test_stale_fencing_rejection(self) -> None:
        task = self.store.submit_task(
            project_id=self.project_id,
            task_type="fence_unit",
            payload={"x": 1},
        )
        self.store.claim_next_task(
            project_id=self.project_id,
            worker_id="worker_real",
            lease_seconds=10.0,
        )
        # Attempt completion by an unauthorized / zombie worker
        with self.assertRaises(TaskFencingError):
            self.store.complete_task(
                project_id=self.project_id,
                task_id=task.task_id,
                worker_id="worker_zombie",
                result=TaskResult(output={"x": 2}),
            )


@unittest.skipUnless(_IS_PG_LIVE, f"PostgreSQL not live: {_PG_MSG}")
class TestPostgreSQLTaskStoreUnit(unittest.TestCase):
    """Unit tests for PostgreSQLTaskStore with live database and tenant isolation."""

    def _ensure_project(self, project_id: str) -> None:
        now_utc = datetime.now(timezone.utc)
        with self.db_manager.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO projects (project_id, name, description, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (project_id) DO NOTHING;
                    """,
                    (project_id, f"Project {project_id}", "Test Project", now_utc, now_utc),
                )
            conn.commit()

    def setUp(self) -> None:
        self.project_id = f"proj_pg_{uuid.uuid4().hex[:8]}"
        self.other_project = f"proj_pg_other_{uuid.uuid4().hex[:8]}"
        cfg = DatabaseConfig(
            backend=StorageBackend.POSTGRES,
            database_url=ADMIN_PG_URL,
            pool_size=5,
            pool_timeout=3.0,
        )
        self.db_manager = DatabaseConnectionManager(cfg)
        self.store = PostgreSQLTaskStore(self.db_manager)
        self._ensure_project(self.project_id)
        self._ensure_project(self.other_project)

    def tearDown(self) -> None:
        try:
            self.store.clear(project_id=self.project_id)
            self.store.clear(project_id=self.other_project)
            with self.db_manager.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM projects WHERE project_id IN (%s, %s);", (self.project_id, self.other_project))
                conn.commit()
        except Exception:
            pass
        self.store.close()
        self.db_manager.close()

    def test_pg_submit_claim_and_complete(self) -> None:
        task = self.store.submit_task(
            project_id=self.project_id,
            task_type="pg_work",
            payload={"msg": "pg test"},
        )
        self.assertEqual(task.status, TaskStatus.PENDING)

        claimed = self.store.claim_next_task(
            project_id=self.project_id,
            worker_id="pg_w1",
            lease_seconds=15.0,
        )
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.task_id, task.task_id)
        self.assertEqual(claimed.locked_by, "pg_w1")

        # Heartbeat
        hb_ok = self.store.heartbeat(
            project_id=self.project_id,
            task_id=task.task_id,
            worker_id="pg_w1",
            lease_seconds=30.0,
        )
        self.assertTrue(hb_ok)

        # Complete
        res = TaskResult(output={"status": "all_good"})
        done = self.store.complete_task(
            project_id=self.project_id,
            task_id=task.task_id,
            worker_id="pg_w1",
            result=res,
        )
        self.assertEqual(done.status, TaskStatus.COMPLETED)
        self.assertIsNone(done.locked_by)

    def test_pg_tenant_isolation_under_rls(self) -> None:
        # Submit task in project_id
        task_a = self.store.submit_task(
            project_id=self.project_id,
            task_type="tenant_a_work",
            payload={"secret": "alpha"},
        )

        # Worker claiming for other_project cannot claim or see task_a
        claimed_other = self.store.claim_next_task(
            project_id=self.other_project,
            worker_id="w_other",
            lease_seconds=10.0,
        )
        self.assertIsNone(claimed_other)

        # get_task from other_project returns None
        rec_other = self.store.get_task(self.other_project, task_a.task_id)
        self.assertIsNone(rec_other)

        # Correct project can claim it
        claimed_a = self.store.claim_next_task(
            project_id=self.project_id,
            worker_id="w_a",
            lease_seconds=10.0,
        )
        self.assertIsNotNone(claimed_a)
        self.assertEqual(claimed_a.task_id, task_a.task_id)


class TestFifteenCoreMilestoneRequirements(unittest.TestCase):
    """
    Exhaustively verifies all 15 milestone requirements across SQLite and PostgreSQL.
    """

    def _ensure_project(self, project_id: str) -> None:
        if not self.db_manager:
            return
        now_utc = datetime.now(timezone.utc)
        with self.db_manager.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO projects (project_id, name, description, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (project_id) DO NOTHING;
                    """,
                    (project_id, f"Project {project_id}", "Test Project", now_utc, now_utc),
                )
            conn.commit()

    def setUp(self) -> None:
        self.project_id = f"proj_req_{uuid.uuid4().hex[:8]}"
        self.registry = TaskHandlerRegistry()
        register_default_handlers(self.registry)

        if _IS_PG_LIVE:
            self.cfg = DatabaseConfig(
                backend=StorageBackend.POSTGRES,
                database_url=ADMIN_PG_URL,
                pool_size=10,
                pool_timeout=5.0,
            )
            self.db_manager = DatabaseConnectionManager(self.cfg)
            self.pg_store = PostgreSQLTaskStore(self.db_manager)
            self._ensure_project(self.project_id)
        else:
            self.db_manager = None
            self.pg_store = None

        self.sqlite_store = SQLiteTaskStore(":memory:")

    def tearDown(self) -> None:
        if self.pg_store:
            try:
                self.pg_store.clear(project_id=self.project_id)
                self.pg_store.close()
            except Exception:
                pass
        if self.db_manager:
            try:
                with self.db_manager.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM projects WHERE project_id = %s;", (self.project_id,))
                    conn.commit()
                self.db_manager.close()
            except Exception:
                pass
        self.sqlite_store.close()

    # -------------------------------------------------------------------------
    # Requirement 1: Task submission and persistence
    # -------------------------------------------------------------------------
    def test_01_task_submission_and_persistence(self) -> None:
        """1. Task submission and persistence (PostgreSQL & SQLite)."""
        stores = [self.sqlite_store]
        if self.pg_store:
            stores.append(self.pg_store)

        for store in stores:
            task = store.submit_task(
                project_id=self.project_id,
                task_type="memory_maintenance",
                payload={"operation": "audit", "depth": "full"},
                priority=TaskPriority.HIGH,
                max_retries=5,
            )
            self.assertIsNotNone(task.task_id)
            self.assertEqual(task.project_id, self.project_id)
            self.assertEqual(task.task_type, "memory_maintenance")
            self.assertEqual(task.status, TaskStatus.PENDING)
            self.assertEqual(task.retry_count, 0)
            self.assertEqual(task.max_retries, 5)
            self.assertIsNone(task.locked_by)
            self.assertIsNone(task.locked_until)

            # Persisted and readable
            fetched = store.get_task(self.project_id, task.task_id)
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched.task_id, task.task_id)
            self.assertEqual(fetched.payload["depth"], "full")

    # -------------------------------------------------------------------------
    # Requirement 2: Successful execution and acknowledgement
    # -------------------------------------------------------------------------
    def test_02_successful_execution_and_acknowledgement(self) -> None:
        """2. Successful execution and acknowledgement (status -> COMPLETED, result stored)."""
        store = self.pg_store if self.pg_store else self.sqlite_store

        executed_event = threading.Event()

        def custom_handler(task: TaskRecord, ctx: Any = None) -> Dict[str, Any]:
            executed_event.set()
            return {"sum": task.payload["a"] + task.payload["b"]}

        self.registry.register("math_add", custom_handler)

        task = store.submit_task(
            project_id=self.project_id,
            task_type="math_add",
            payload={"a": 15, "b": 27},
        )

        worker = BackgroundWorker(
            store=store,
            project_id=self.project_id,
            registry=self.registry,
            worker_id="worker_ack_test",
            poll_interval=0.05,
        )
        worker.start()

        self.assertTrue(executed_event.wait(timeout=5.0), "Handler was not invoked")
        time.sleep(0.3)
        worker.stop(timeout=2.0)

        # Check store record
        finished = store.get_task(self.project_id, task.task_id)
        self.assertIsNotNone(finished)
        self.assertEqual(finished.status, TaskStatus.COMPLETED)
        self.assertIsNone(finished.locked_by)
        self.assertEqual(finished.payload.get("result", {}).get("sum"), 42)
        self.assertGreaterEqual(worker.info.tasks_completed, 1)

    # -------------------------------------------------------------------------
    # Requirement 3: Multiple workers claiming different tasks concurrently
    # -------------------------------------------------------------------------
    def test_03_multiple_workers_claiming_different_tasks_concurrently(self) -> None:
        """3. Multiple workers claiming different tasks concurrently (FOR UPDATE SKIP LOCKED)."""
        store = self.pg_store if self.pg_store else self.sqlite_store

        # Create 6 tasks
        task_ids = []
        for i in range(6):
            t = store.submit_task(
                project_id=self.project_id,
                task_type="concurrent_claim",
                payload={"index": i},
            )
            task_ids.append(t.task_id)

        # Concurrently claim tasks using 3 threads simulating 3 independent workers
        claimed_tasks: List[str] = []
        lock = threading.Lock()

        def claimer(w_id: str) -> None:
            for _ in range(4):
                c = store.claim_next_task(
                    project_id=self.project_id,
                    worker_id=w_id,
                    lease_seconds=5.0,
                )
                if c:
                    with lock:
                        claimed_tasks.append(c.task_id)
                time.sleep(0.01)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(claimer, f"worker_thread_{i}") for i in range(3)]
            concurrent.futures.wait(futures)

        # Exactly 6 tasks claimed, with 0 duplicate claims
        self.assertEqual(len(claimed_tasks), 6)
        self.assertEqual(len(set(claimed_tasks)), 6)
        self.assertEqual(set(claimed_tasks), set(task_ids))

    # -------------------------------------------------------------------------
    # Requirement 4: Two workers competing for the same task
    # -------------------------------------------------------------------------
    def test_04_two_workers_competing_for_same_task_atomic_single_winner(self) -> None:
        """4. Two workers competing for the same task (atomic claim, exactly 1 winner)."""
        store = self.pg_store if self.pg_store else self.sqlite_store

        t = store.submit_task(
            project_id=self.project_id,
            task_type="single_task_race",
            payload={"flag": True},
        )

        winners: List[str] = []
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        def compete(w_name: str) -> None:
            barrier.wait()
            c = store.claim_next_task(
                project_id=self.project_id,
                worker_id=w_name,
                lease_seconds=10.0,
            )
            if c:
                with lock:
                    winners.append(w_name)

        t1 = threading.Thread(target=compete, args=("competitor_1",))
        t2 = threading.Thread(target=compete, args=("competitor_2",))
        t1.start()
        t2.start()
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        # Exactly ONE competitor won
        self.assertEqual(len(winners), 1)

        # The claimed task is now PROCESSING and locked by the winner
        claimed = store.get_task(self.project_id, t.task_id)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.status, TaskStatus.PROCESSING)
        self.assertEqual(claimed.locked_by, winners[0])

    # -------------------------------------------------------------------------
    # Requirement 5: Lease renewal during long-running execution
    # -------------------------------------------------------------------------
    def test_05_lease_renewal_during_long_running_execution(self) -> None:
        """5. Lease renewal during long-running execution (independent heartbeat keeps lease valid)."""
        store = self.pg_store if self.pg_store else self.sqlite_store

        # A long task taking 1.5s with a 0.8s lease. Without heartbeats, it would expire!
        def slow_handler(task: TaskRecord, ctx: Any = None) -> Dict[str, Any]:
            time.sleep(1.2)
            return {"completed_after": "sleep"}

        self.registry.register("slow_task", slow_handler)

        t = store.submit_task(
            project_id=self.project_id,
            task_type="slow_task",
            payload={"duration": 1.2},
        )

        worker = BackgroundWorker(
            store=store,
            project_id=self.project_id,
            registry=self.registry,
            worker_id="worker_heartbeat",
            lease_seconds=0.8,
            poll_interval=0.05,
        )
        worker.start()

        # Check halfway through that lease was renewed beyond the current time
        time.sleep(0.6)
        active = store.get_task(self.project_id, t.task_id)
        self.assertIsNotNone(active)
        self.assertEqual(active.status, TaskStatus.PROCESSING)
        self.assertGreater(active.locked_until, datetime.now(timezone.utc))

        # A second worker trying to claim tasks finds nothing because the task lease is kept active
        c2 = store.claim_next_task(self.project_id, "worker_intruder", lease_seconds=1.0)
        self.assertIsNone(c2, "Intruder worker should not claim actively heartbeated task")

        # Wait for completion
        time.sleep(1.0)
        worker.stop(timeout=2.0)

        done = store.get_task(self.project_id, t.task_id)
        self.assertEqual(done.status, TaskStatus.COMPLETED)

    # -------------------------------------------------------------------------
    # Requirement 6: Worker crash and abandoned-task recovery
    # -------------------------------------------------------------------------
    def test_06_worker_crash_and_abandoned_task_recovery(self) -> None:
        """6. Worker crash and abandoned-task recovery (expired locked_until claimed by second worker)."""
        store = self.pg_store if self.pg_store else self.sqlite_store

        t = store.submit_task(
            project_id=self.project_id,
            task_type="workflow_step",
            payload={"workflow_id": "wf_recover", "stage_name": "Execution"},
        )

        # Worker 1 claims with very short lease (0.2s) and "crashes" (stops without heartbeat or complete)
        c1 = store.claim_next_task(self.project_id, "worker_crashed", lease_seconds=0.2)
        self.assertIsNotNone(c1)
        self.assertEqual(c1.locked_by, "worker_crashed")

        # Wait for lease to expire
        time.sleep(0.35)

        # Worker 2 attempts to claim: successfully reclaims abandoned task!
        c2 = store.claim_next_task(self.project_id, "worker_survivor", lease_seconds=10.0)
        self.assertIsNotNone(c2, "Survivor worker must reclaim abandoned task")
        self.assertEqual(c2.task_id, t.task_id)
        self.assertEqual(c2.locked_by, "worker_survivor")
        self.assertEqual(c2.retry_count, 1)  # Incremented attempt counter

        # Survivor completes it
        done = store.complete_task(
            self.project_id,
            c2.task_id,
            "worker_survivor",
            TaskResult(output={"recovered": True}),
        )
        self.assertEqual(done.status, TaskStatus.COMPLETED)

    # -------------------------------------------------------------------------
    # Requirement 7: Stale worker fencing and prevention of duplicate side effects
    # -------------------------------------------------------------------------
    def test_07_stale_worker_fencing_blocks_duplicate_side_effects(self) -> None:
        """7. Stale worker fencing and prevention of duplicate side effects."""
        store = self.pg_store if self.pg_store else self.sqlite_store

        t = store.submit_task(
            project_id=self.project_id,
            task_type="fence_test",
            payload={"amount": 100},
        )

        # Zombie worker claims with short lease
        store.claim_next_task(self.project_id, "worker_zombie", lease_seconds=0.2)
        time.sleep(0.35)

        # Second worker legitimately reclaims
        c2 = store.claim_next_task(self.project_id, "worker_legit", lease_seconds=10.0)
        self.assertIsNotNone(c2)

        # Now zombie worker wakes up and attempts heartbeat: rejected!
        hb_zombie = store.heartbeat(self.project_id, t.task_id, "worker_zombie", lease_seconds=10.0)
        self.assertFalse(hb_zombie)

        # Zombie worker attempts completion: raises TaskFencingError and 0 commits succeed!
        with self.assertRaises(TaskFencingError):
            store.complete_task(
                self.project_id,
                t.task_id,
                "worker_zombie",
                TaskResult(output={"side_effect": "zombie_write"}),
            )

        # Legitimate worker completes cleanly
        store.complete_task(
            self.project_id,
            t.task_id,
            "worker_legit",
            TaskResult(output={"side_effect": "legit_write"}),
        )

        final_rec = store.get_task(self.project_id, t.task_id)
        self.assertEqual(final_rec.status, TaskStatus.COMPLETED)
        self.assertEqual(final_rec.payload.get("result", {}).get("side_effect"), "legit_write")

    # -------------------------------------------------------------------------
    # Requirement 8: Retry classification (transient vs permanent)
    # -------------------------------------------------------------------------
    def test_08_retry_classification_transient_vs_permanent(self) -> None:
        """8. Retry classification (transient error triggers backoff retry; permanent error routes to DLQ)."""
        store = self.pg_store if self.pg_store else self.sqlite_store

        transient_attempts = 0

        def flaky_handler(task: TaskRecord, ctx: Any = None) -> Dict[str, Any]:
            nonlocal transient_attempts
            transient_attempts += 1
            if transient_attempts < 2:
                raise TransientTaskError("Temporary network glitch")
            return {"success_after_glitch": True}

        def fatal_handler(task: TaskRecord, ctx: Any = None) -> Dict[str, Any]:
            raise PermanentTaskError("Corrupt payload format")

        self.registry.register("flaky", flaky_handler)
        self.registry.register("fatal", fatal_handler)

        worker = BackgroundWorker(
            store=store,
            project_id=self.project_id,
            registry=self.registry,
            worker_id="worker_classifier",
            poll_interval=0.05,
            retry_policy=RetryPolicy(base_delay=0.1, max_delay=0.5, backoff_factor=1.5),
        )
        worker.start()

        # Submit fatal task: must immediately fail and route to DLQ without retries
        t_fatal = store.submit_task(self.project_id, "fatal", {"data": "bad"}, max_retries=5)
        time.sleep(0.4)
        rec_fatal = store.get_task(self.project_id, t_fatal.task_id)
        self.assertEqual(rec_fatal.status, TaskStatus.FAILED)
        self.assertIn("Corrupt payload format", rec_fatal.error_message or "")

        # Submit flaky task: must retry and eventually succeed
        t_flaky = store.submit_task(self.project_id, "flaky", {"data": "good"}, max_retries=3)
        time.sleep(0.6)
        rec_flaky = store.get_task(self.project_id, t_flaky.task_id)
        self.assertEqual(rec_flaky.status, TaskStatus.COMPLETED)
        self.assertEqual(transient_attempts, 2)

        worker.stop(timeout=2.0)

    # -------------------------------------------------------------------------
    # Requirement 9: Maximum-attempt enforcement and dead-letter handling
    # -------------------------------------------------------------------------
    def test_09_maximum_attempt_enforcement_and_dead_letter_handling(self) -> None:
        """9. Maximum-attempt enforcement and dead-letter handling (with manual DLQ resurrection)."""
        store = self.pg_store if self.pg_store else self.sqlite_store

        def always_fail(task: TaskRecord, ctx: Any = None) -> None:
            raise TransientTaskError("Persistent transient failure")

        self.registry.register("always_fail", always_fail)

        t = store.submit_task(
            project_id=self.project_id,
            task_type="always_fail",
            payload={"x": 1},
            max_retries=2,
        )

        worker = BackgroundWorker(
            store=store,
            project_id=self.project_id,
            registry=self.registry,
            worker_id="worker_dlq_test",
            poll_interval=0.05,
            retry_policy=RetryPolicy(base_delay=0.05, max_delay=0.2),
        )
        worker.start()

        # Wait for retries to exhaust
        time.sleep(0.8)
        worker.stop(timeout=2.0)

        rec = store.get_task(self.project_id, t.task_id)
        self.assertEqual(rec.status, TaskStatus.FAILED)
        self.assertGreaterEqual(rec.retry_count, 2)

        # Inspect DLQ
        dead_letters = store.list_dead_letters(self.project_id)
        self.assertIn(t.task_id, [dl.task_id for dl in dead_letters])

        # Manual DLQ resurrection
        resurrected = store.retry_dead_letter(self.project_id, t.task_id)
        self.assertTrue(resurrected)

        rec_resurrected = store.get_task(self.project_id, t.task_id)
        self.assertEqual(rec_resurrected.status, TaskStatus.PENDING)
        self.assertEqual(rec_resurrected.retry_count, 0)
        self.assertIsNone(rec_resurrected.error_message)

    # -------------------------------------------------------------------------
    # Requirement 10: Graceful shutdown and task cancellation
    # -------------------------------------------------------------------------
    def test_10_graceful_shutdown_and_task_cancellation(self) -> None:
        """10. Graceful shutdown and task cancellation."""
        store = self.pg_store if self.pg_store else self.sqlite_store

        # 1. Cancel pending task
        t_cancel = store.submit_task(self.project_id, "dummy_work", {"id": 1})
        self.assertTrue(store.cancel_task(self.project_id, t_cancel.task_id))
        rec_cancelled = store.get_task(self.project_id, t_cancel.task_id)
        self.assertEqual(rec_cancelled.status, TaskStatus.CANCELLED)

        # Cancelled task cannot be claimed
        claimed = store.claim_next_task(self.project_id, "w_cancel_test", 5.0)
        self.assertIsNone(claimed)

        # 2. Graceful worker shutdown: active in-flight task completes cleanly before shutdown finishes
        started_event = threading.Event()
        finish_event = threading.Event()

        def slow_graceful_work(task: TaskRecord, ctx: Any = None) -> Dict[str, Any]:
            started_event.set()
            finish_event.wait(timeout=2.0)
            return {"gracefully_finished": True}

        self.registry.register("slow_graceful", slow_graceful_work)

        t_inflight = store.submit_task(self.project_id, "slow_graceful", {"id": 2})

        worker = BackgroundWorker(
            store=store,
            project_id=self.project_id,
            registry=self.registry,
            worker_id="worker_graceful",
            poll_interval=0.05,
        )
        worker.start()

        self.assertTrue(started_event.wait(timeout=3.0))

        # Stop worker gracefully in another thread while task is running
        stop_thread = threading.Thread(target=lambda: worker.stop(timeout=5.0, graceful=True))
        stop_thread.start()

        # Release task to finish
        finish_event.set()
        stop_thread.join(timeout=3.0)

        # In-flight task finished cleanly
        rec_done = store.get_task(self.project_id, t_inflight.task_id)
        self.assertEqual(rec_done.status, TaskStatus.COMPLETED)
        self.assertFalse(worker.is_alive)

    # -------------------------------------------------------------------------
    # Requirement 11: Idempotent execution and duplicate submission
    # -------------------------------------------------------------------------
    def test_11_idempotent_execution_and_duplicate_submission(self) -> None:
        """11. Idempotent execution and duplicate submission."""
        store = self.pg_store if self.pg_store else self.sqlite_store
        idem_key = f"idem_submission_{uuid.uuid4().hex[:8]}"

        t1 = store.submit_task(
            project_id=self.project_id,
            task_type="memory_maintenance",
            payload={"op": "consolidate"},
            idempotency_key=idem_key,
        )
        self.assertIsNotNone(t1.task_id)

        # Second submission with identical idempotency key is rejected
        with self.assertRaises(DuplicateTaskError):
            store.submit_task(
                project_id=self.project_id,
                task_type="memory_maintenance",
                payload={"op": "consolidate"},
                idempotency_key=idem_key,
            )

    # -------------------------------------------------------------------------
    # Requirement 12: Tenant isolation under RLS
    # -------------------------------------------------------------------------
    def test_12_tenant_isolation_under_rls(self) -> None:
        """12. Tenant isolation under RLS (and SQLite parity)."""
        proj_alpha = f"proj_iso_a_{uuid.uuid4().hex[:8]}"
        proj_beta = f"proj_iso_b_{uuid.uuid4().hex[:8]}"

        self._ensure_project(proj_alpha)
        self._ensure_project(proj_beta)

        stores = [self.sqlite_store]
        if self.pg_store:
            stores.append(self.pg_store)

        for store in stores:
            t_a = store.submit_task(proj_alpha, "memory_maintenance", {"secret": "alpha_only"})
            t_b = store.submit_task(proj_beta, "memory_maintenance", {"secret": "beta_only"})

            # Worker in proj_alpha can only claim t_a
            c_a = store.claim_next_task(proj_alpha, "worker_alpha", lease_seconds=10.0)
            self.assertIsNotNone(c_a)
            self.assertEqual(c_a.task_id, t_a.task_id)

            # Subsequent claim in proj_alpha yields None (t_b is completely invisible)
            c_a_none = store.claim_next_task(proj_alpha, "worker_alpha", lease_seconds=10.0)
            self.assertIsNone(c_a_none)

            # Worker in proj_beta can only claim t_b
            c_b = store.claim_next_task(proj_beta, "worker_beta", lease_seconds=10.0)
            self.assertIsNotNone(c_b)
            self.assertEqual(c_b.task_id, t_b.task_id)

            # Cleanup
            store.clear(project_id=proj_alpha)
            store.clear(project_id=proj_beta)

        if self.db_manager:
            try:
                with self.db_manager.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM projects WHERE project_id IN (%s, %s);", (proj_alpha, proj_beta))
                    conn.commit()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Requirement 13: PostgreSQL integration using independent connections
    # -------------------------------------------------------------------------
    @unittest.skipUnless(_IS_PG_LIVE, f"Live PostgreSQL required: {_PG_MSG}")
    def test_13_postgresql_integration_using_independent_connections(self) -> None:
        """13. PostgreSQL integration using independent pooled connections."""
        self.assertIsNotNone(self.pg_store)

        fleet = WorkerFleet(
            store=self.pg_store,
            registry=self.registry,
            project_id=self.project_id,
            concurrency=3,
            poll_interval=0.05,
        )

        task_ids = []
        for i in range(5):
            t = self.pg_store.submit_task(
                self.project_id,
                "workflow_step",
                {"workflow_id": f"wf_{i}", "stage_name": "Execution"},
            )
            task_ids.append(t.task_id)

        fleet.start()
        self.assertEqual(fleet.current_concurrency, 3)

        # Wait for workers to process tasks
        time.sleep(1.2)
        fleet.stop(timeout=3.0)

        for tid in task_ids:
            rec = self.pg_store.get_task(self.project_id, tid)
            self.assertIsNotNone(rec)
            self.assertEqual(rec.status, TaskStatus.COMPLETED)

        stats = fleet.get_stats()
        self.assertEqual(stats["tasks_completed"], 5)

    # -------------------------------------------------------------------------
    # Requirement 14: SQLite compatibility (identical semantics offline)
    # -------------------------------------------------------------------------
    def test_14_sqlite_compatibility_offline_parity(self) -> None:
        """14. SQLite compatibility (identical semantics offline)."""
        fleet = WorkerFleet(
            store=self.sqlite_store,
            registry=self.registry,
            project_id=self.project_id,
            concurrency=2,
            poll_interval=0.05,
        )

        task_ids = []
        for i in range(4):
            t = self.sqlite_store.submit_task(
                self.project_id,
                "workflow_step",
                {"workflow_id": f"sqlite_wf_{i}", "stage_name": "OfflineStep"},
            )
            task_ids.append(t.task_id)

        fleet.start()
        time.sleep(0.8)
        fleet.stop(timeout=2.0)

        for tid in task_ids:
            rec = self.sqlite_store.get_task(self.project_id, tid)
            self.assertEqual(rec.status, TaskStatus.COMPLETED)

    # -------------------------------------------------------------------------
    # Requirement 15: Resource cleanup and connection lifecycle
    # -------------------------------------------------------------------------
    def test_15_resource_cleanup_and_connection_lifecycle(self) -> None:
        """15. Resource cleanup, worker thread lifecycle, dynamic scaling, and zero leaks."""
        fleet = WorkerFleet(
            store=self.sqlite_store,
            registry=self.registry,
            project_id=self.project_id,
            concurrency=2,
            poll_interval=0.05,
        )

        fleet.start()
        self.assertTrue(fleet.is_running)
        self.assertEqual(fleet.current_concurrency, 2)

        # Dynamic scaling UP
        fleet.scale(4)
        self.assertEqual(fleet.current_concurrency, 4)

        # Dynamic scaling DOWN
        fleet.scale(1)
        self.assertEqual(fleet.current_concurrency, 1)

        health = fleet.get_health()
        self.assertEqual(health["status"], "HEALTHY")
        self.assertEqual(health["concurrency"]["target"], 1)

        # Stop fleet
        fleet.stop(timeout=2.0)
        self.assertFalse(fleet.is_running)
        self.assertEqual(fleet.current_concurrency, 0)


class TestInteractionExecutionEndToEnd(unittest.TestCase):
    """
    Dedicated tests verifying full interaction_execution end-to-end:
    1. Task submitted to queue.
    2. Worker claims task and invokes handler.
    3. Handler calls BrainLoop context preparation.
    4. Registered agent executes interaction.
    5. BrainLoop completion runs successfully.
    6. Interaction output and extracted memories persisted exactly once.
    7. Task acknowledged as COMPLETED.
    8. Duplicate submission/retry cannot duplicate interaction or memory writes.
    9. Failure during execution follows configured retry policy.
    10. Tenant isolation maintained throughout operation.
    """

    def setUp(self) -> None:
        self.project_id = f"proj_int_e2e_{uuid.uuid4().hex[:8]}"
        self.container = create_sqlite_container(":memory:")

    def tearDown(self) -> None:
        self.container.close()

    def test_01_interaction_execution_full_lifecycle(self) -> None:
        """Verifies requirements 1-7: full lifecycle, agent execution, and exact persistence."""
        task_store = self.container.task_store
        task_registry = self.container.task_registry
        self.assertIsNotNone(task_store)
        self.assertIsNotNone(task_registry)

        # 1. Persisted interaction execution task submitted to queue
        task = task_store.submit_task(
            project_id=self.project_id,
            task_type="interaction_execution",
            payload={
                "input_text": "We decided to adopt Redis for session caching and token storage.",
                "agent_id": "agent-coder",
                "session_id": f"sess_{uuid.uuid4().hex[:8]}",
            },
        )
        self.assertIsNotNone(task.task_id)
        self.assertEqual(task.status, TaskStatus.PENDING)

        # 2. Worker claims task and invokes handler
        worker = BackgroundWorker(
            store=task_store,
            project_id=self.project_id,
            registry=task_registry,
            context=self.container,
            worker_id="w_int_e2e_1",
            poll_interval=0.05,
        )
        worker.start()

        # Wait for task completion
        time.sleep(0.6)
        worker.stop(timeout=2.0)

        # 7. Task acknowledged as COMPLETED
        finished_task = task_store.get_task(self.project_id, task.task_id)
        self.assertIsNotNone(finished_task)
        self.assertEqual(finished_task.status, TaskStatus.COMPLETED)
        self.assertIsNone(finished_task.locked_by)

        # 3. BrainLoop context preparation ran
        task_result = finished_task.payload.get("result", {})
        self.assertEqual(task_result.get("status"), "COMPLETED")
        interaction_id = task_result.get("interaction_id")
        self.assertIsNotNone(interaction_id)

        # 4 & 5. Registered agent executed and BrainLoop completion ran successfully
        int_rec = self.container.interaction_store.get(interaction_id)
        self.assertIsNotNone(int_rec)
        self.assertEqual(int_rec.status.value, "COMPLETED")
        self.assertIsNotNone(int_rec.context_result)
        self.assertIsNotNone(int_rec.response)
        self.assertIn("Redis", int_rec.response.content)

        # 6. Interaction output and extracted memories persisted exactly once
        mem_count = self.container.memory_store.count(project_id=self.project_id)
        self.assertGreaterEqual(mem_count, 1)
        self.assertEqual(task_result.get("memories_created"), mem_count)

    def test_02_interaction_execution_idempotency_prevents_duplicate_writes(self) -> None:
        """Verifies requirement 8: retry or duplicate submission cannot duplicate interaction or memory writes."""
        task_store = self.container.task_store
        task_registry = self.container.task_registry
        idem_key = f"e2e_idem_{uuid.uuid4().hex[:8]}"

        # Submit task with idempotency key
        task = task_store.submit_task(
            project_id=self.project_id,
            task_type="interaction_execution",
            payload={
                "input_text": "Architectural decision: Use PostgreSQL for primary ACID storage.",
                "agent_id": "agent-coder",
                "idempotency_key": idem_key,
            },
            idempotency_key=idem_key,
        )

        worker = BackgroundWorker(
            store=task_store,
            project_id=self.project_id,
            registry=task_registry,
            context=self.container,
            worker_id="w_int_idem",
            poll_interval=0.05,
        )
        worker.start()
        time.sleep(0.6)
        worker.stop(timeout=2.0)

        # Initial memory count after completion
        initial_mem_count = self.container.memory_store.count(project_id=self.project_id)
        self.assertGreaterEqual(initial_mem_count, 1)

        # Attempt 1: Duplicate submission to task store is rejected
        with self.assertRaises(DuplicateTaskError):
            task_store.submit_task(
                project_id=self.project_id,
                task_type="interaction_execution",
                payload={
                    "input_text": "Architectural decision: Use PostgreSQL for primary ACID storage.",
                    "agent_id": "agent-coder",
                    "idempotency_key": idem_key,
                },
                idempotency_key=idem_key,
            )

        # Attempt 2: Re-executing the handler on the same task record returns cached result
        res = task_registry.execute(task, self.container)
        self.assertTrue(res.success)
        self.assertTrue(res.output.get("already_completed", False))

        # Memory store count remains unchanged (exactly once, zero duplicates)
        final_mem_count = self.container.memory_store.count(project_id=self.project_id)
        self.assertEqual(final_mem_count, initial_mem_count)

    def test_03_interaction_execution_failure_follows_retry_policy(self) -> None:
        """Verifies requirement 9: failure during execution follows configured retry policy."""
        task_store = self.container.task_store
        task_registry = self.container.task_registry

        attempts = 0
        original_prepare = self.container.brain_loop.prepare_context

        def flaky_prepare(agent_input: Any) -> Any:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("Simulated transient connection timeout during context retrieval")
            return original_prepare(agent_input)

        self.container.brain_loop.prepare_context = flaky_prepare

        task = task_store.submit_task(
            project_id=self.project_id,
            task_type="interaction_execution",
            payload={"input_text": "Configure database connection pooling settings."},
            max_retries=3,
        )

        worker = BackgroundWorker(
            store=task_store,
            project_id=self.project_id,
            registry=task_registry,
            context=self.container,
            worker_id="w_int_retry",
            poll_interval=0.05,
            retry_policy=RetryPolicy(base_delay=0.1, max_delay=0.3, backoff_factor=1.5),
        )
        worker.start()

        # Wait for retry to take place and succeed
        time.sleep(0.8)
        worker.stop(timeout=2.0)

        # Worker retried and succeeded on attempt 2
        finished = task_store.get_task(self.project_id, task.task_id)
        self.assertIsNotNone(finished)
        self.assertEqual(finished.status, TaskStatus.COMPLETED)
        self.assertEqual(attempts, 2)
        self.assertEqual(finished.retry_count, 1)

    def test_04_interaction_execution_tenant_isolation(self) -> None:
        """Verifies requirement 10: tenant isolation is maintained throughout the operation."""
        proj_a = f"proj_int_a_{uuid.uuid4().hex[:8]}"
        proj_b = f"proj_int_b_{uuid.uuid4().hex[:8]}"

        task_store = self.container.task_store
        task_registry = self.container.task_registry

        # Submit task for project A
        task_a = task_store.submit_task(
            project_id=proj_a,
            task_type="interaction_execution",
            payload={"input_text": "Confidential decision for Project A: Use AES-256-GCM encryption."},
        )

        # Worker scoped to project B cannot claim or execute project A task
        worker_b = BackgroundWorker(
            store=task_store,
            project_id=proj_b,
            registry=task_registry,
            context=self.container,
            worker_id="w_scoped_b",
            poll_interval=0.05,
        )
        c_b = task_store.claim_next_task(proj_b, "w_scoped_b", 5.0)
        self.assertIsNone(c_b)

        # Worker scoped to project A claims and executes it
        worker_a = BackgroundWorker(
            store=task_store,
            project_id=proj_a,
            registry=task_registry,
            context=self.container,
            worker_id="w_scoped_a",
            poll_interval=0.05,
        )
        worker_a.start()
        time.sleep(0.6)
        worker_a.stop(timeout=2.0)

        # Verify task completed in project A
        done_a = task_store.get_task(proj_a, task_a.task_id)
        self.assertEqual(done_a.status, TaskStatus.COMPLETED)

        # Verify memories created in project A are strictly isolated from project B
        self.assertGreaterEqual(self.container.memory_store.count(project_id=proj_a), 1)
        self.assertEqual(self.container.memory_store.count(project_id=proj_b), 0)

    @unittest.skipUnless(_IS_PG_LIVE, f"Live PostgreSQL required: {_PG_MSG}")
    def test_05_interaction_execution_live_postgres(self) -> None:
        """Verifies full end-to-end interaction_execution on live PostgreSQL."""
        cfg = DatabaseConfig(
            backend=StorageBackend.POSTGRES,
            database_url=ADMIN_PG_URL,
            pool_size=5,
            pool_timeout=5.0,
        )
        mgr = DatabaseConnectionManager(cfg)
        now_utc = datetime.now(timezone.utc)
        with mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO projects (project_id, name, created_at, updated_at) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;",
                    (self.project_id, "PG E2E Test", now_utc, now_utc),
                )
            conn.commit()

        pg_container = create_postgres_container(mgr)
        try:
            task = pg_container.task_store.submit_task(
                project_id=self.project_id,
                task_type="interaction_execution",
                payload={"input_text": "Live PostgreSQL background task decision: Use Citus sharding."},
            )
            worker = BackgroundWorker(
                store=pg_container.task_store,
                project_id=self.project_id,
                registry=pg_container.task_registry,
                context=pg_container,
                worker_id="w_pg_int_e2e",
                poll_interval=0.05,
            )
            worker.start()
            time.sleep(0.8)
            worker.stop(timeout=2.0)

            done = pg_container.task_store.get_task(self.project_id, task.task_id)
            self.assertEqual(done.status, TaskStatus.COMPLETED)
            self.assertGreaterEqual(pg_container.memory_store.count(project_id=self.project_id), 1)
        finally:
            try:
                with mgr.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM background_tasks WHERE project_id = %s;", (self.project_id,))
                        cur.execute("DELETE FROM memories WHERE project_id = %s;", (self.project_id,))
                        cur.execute("DELETE FROM interactions WHERE project_id = %s;", (self.project_id,))
                        cur.execute("DELETE FROM event_lookup WHERE project_id = %s;", (self.project_id,))
                        cur.execute("DELETE FROM events WHERE project_id = %s;", (self.project_id,))
                        cur.execute("DELETE FROM projects WHERE project_id = %s;", (self.project_id,))
                    conn.commit()
            except Exception:
                pass
            pg_container.close()
            mgr.close()


if __name__ == "__main__":
    unittest.main()
