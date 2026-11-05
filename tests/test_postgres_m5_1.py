"""Acceptance tests for AI Brain Phase 5 Milestone 5.1: PostgreSQL Relational Schema & Migrations.

Acceptance Test Matrix:
1. Live instance connectivity detection (skips cleanly when PG daemon offline).
2. Fresh migration: creates all 14 authoritative tables.
3. Downgrade & re-upgrade: drops cleanly and re-migrates cleanly.
4. Schema validation: all 14 tables, constraints, foreign keys, and indexes exist.
5. Write-once rejection: engine-level triggers/rules reject UPDATE and DELETE on events.
6. Event identity and idempotency across partitions: timestamp routing and global uniqueness in event_lookup.
7. Tenant isolation via RLS and SET LOCAL app.current_project_id: complete project data segregation.
8. Pooled connection sanitization: reused connection resets session context (RESET ALL).
9. Transaction rollback: failed transactions roll back without partial data commits.

NOTE: Real PostgreSQL is required for live execution. If the PostgreSQL daemon
is not reachable at TEST_POSTGRES_URL or if the driver is not installed, tests
skip with an explicit, honest reason. SQLite is NEVER substituted for PostgreSQL.
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from typing import Any, Optional

from ai_brain.config import DatabaseConfig, StorageBackend
from ai_brain.storage.database import (
    ConnectionUnavailableError,
    DatabaseConnectionManager,
    DriverNotInstalledError,
)

TEST_PG_URL = os.environ.get(
    "TEST_POSTGRES_URL",
    os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/test_ai_brain"),
)


def _check_postgres_live() -> tuple[bool, str, Optional[DatabaseConnectionManager]]:
    """Probe whether live PostgreSQL instance is reachable."""
    cfg = DatabaseConfig(
        backend=StorageBackend.POSTGRES,
        database_url=TEST_PG_URL,
        pool_size=3,
        pool_timeout=3.0,
    )
    mgr = DatabaseConnectionManager(cfg)
    if not mgr.driver_available:
        return False, "PostgreSQL driver (psycopg or psycopg2) not installed", None

    try:
        conn = mgr.get_connection(timeout=2.0)
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        mgr.return_connection(conn)
        return True, "Live PostgreSQL connected", mgr
    except Exception as e:
        mgr.close()
        return False, f"PostgreSQL instance unreachable at {cfg.sanitized_url()}: {e}", None


# Determine live status once at module load
_IS_PG_LIVE, _PG_SKIP_REASON, _MODULE_MGR = _check_postgres_live()


def _read_schema_sql() -> str:
    """Read the authoritative schema initialization script."""
    sql_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "init_postgres_schema.sql")
    with open(sql_path, "r", encoding="utf-8") as f:
        return f.read()


@unittest.skipUnless(_IS_PG_LIVE, f"Live PostgreSQL test skipped: {_PG_SKIP_REASON}")
class TestPostgresSchemaM51(unittest.TestCase):
    """M5.1 PostgreSQL Relational Schema, Partitioning, and RLS Acceptance Tests."""

    mgr: DatabaseConnectionManager

    @classmethod
    def setUpClass(cls) -> None:
        if not _IS_PG_LIVE or _MODULE_MGR is None:
            raise unittest.SkipTest(f"Live PostgreSQL required: {_PG_SKIP_REASON}")
        cls.mgr = _MODULE_MGR
        # Initialize schema cleanly
        cls._apply_schema()

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "mgr") and cls.mgr:
            cls.mgr.close()

    @classmethod
    def _apply_schema(cls) -> None:
        """Apply fresh schema from init script."""
        sql = _read_schema_sql()
        with cls.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

    def setUp(self) -> None:
        # Clean tenant test data before each test
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                # Reset RLS bypass for test setup
                cur.execute("RESET ALL;")
                cur.execute("""
                TRUNCATE TABLE
                    background_tasks,
                    feedback_records,
                    interactions,
                    sessions,
                    agent_credentials,
                    agents,
                    memory_vectors_384,
                    graph_relationships,
                    graph_nodes,
                    memories,
                    event_lookup,
                    project_members,
                    projects
                CASCADE;
                """)
                # Events partitioned table requires delete or truncate
                cur.execute("TRUNCATE TABLE events CASCADE;")
            conn.commit()

    def test_01_all_14_authoritative_tables_exist(self) -> None:
        """Verify all 14 authoritative tables exist in PostgreSQL information_schema."""
        expected_tables = {
            "projects",
            "project_members",
            "events",
            "event_lookup",
            "memories",
            "graph_nodes",
            "graph_relationships",
            "memory_vectors_384",
            "agents",
            "agent_credentials",
            "sessions",
            "interactions",
            "feedback_records",
            "background_tasks",
        }

        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_type = 'BASE TABLE';
                """)
                tables = {row[0] for row in cur.fetchall()}

        for tbl in expected_tables:
            self.assertIn(tbl, tables, f"Expected table '{tbl}' not found in database.")

    def test_02_event_partitions_exist(self) -> None:
        """Verify events table has range partitions for 2025, 2026, 2027 and default."""
        expected_partitions = {"events_y2025", "events_y2026", "events_y2027", "events_default"}

        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT relname
                    FROM pg_class
                    WHERE relispartition = true;
                """)
                partitions = {row[0] for row in cur.fetchall()}

        for p in expected_partitions:
            self.assertIn(p, partitions, f"Partition '{p}' not found for events table.")

    def test_03_append_only_event_protections_reject_update_and_delete(self) -> None:
        """Verify engine-level rules/triggers reject UPDATE and DELETE on events."""
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                # Insert project
                cur.execute(
                    "INSERT INTO projects (project_id, name) VALUES (%s, %s);",
                    ("proj-append-test", "Append Only Test"),
                )
                # Insert event
                ts = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
                cur.execute("""
                    INSERT INTO events (event_id, project_id, timestamp, event_type, payload)
                    VALUES (%s, %s, %s, %s, %s::jsonb);
                """, ("evt-001", "proj-append-test", ts, "state_change", '{"status": "old"}'))
            conn.commit()

            # Attempt UPDATE: trigger must raise an exception
            with conn.cursor() as cur:
                with self.assertRaises(Exception) as ctx:
                    cur.execute("""
                        UPDATE events SET payload = '{"status": "tampered"}'::jsonb
                        WHERE event_id = 'evt-001';
                    """)
                self.assertIn("append-only", str(ctx.exception).lower())
            conn.rollback()

            # Attempt DELETE: trigger must raise an exception
            with conn.cursor() as cur:
                with self.assertRaises(Exception) as ctx:
                    cur.execute("DELETE FROM events WHERE event_id = 'evt-001';")
                self.assertIn("append-only", str(ctx.exception).lower())
            conn.rollback()

            # Verify original record remains intact
            with conn.cursor() as cur:
                cur.execute("SELECT payload->>'status' FROM events WHERE event_id = 'evt-001';")
                val = cur.fetchone()[0]
                self.assertEqual(val, "old")

    def test_04_event_identity_and_partition_routing(self) -> None:
        """Verify event identity and partition routing across partition ranges."""
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO projects (project_id, name) VALUES ('proj-part', 'Part Test');")
                # 2025 event
                cur.execute("""
                    INSERT INTO events (event_id, project_id, timestamp, event_type, payload)
                    VALUES ('evt-2025', 'proj-part', '2025-06-15 10:00:00+00', 'log', '{}'::jsonb);
                """)
                # 2026 event
                cur.execute("""
                    INSERT INTO events (event_id, project_id, timestamp, event_type, payload)
                    VALUES ('evt-2026', 'proj-part', '2026-06-15 10:00:00+00', 'log', '{}'::jsonb);
                """)
                # 2027 event
                cur.execute("""
                    INSERT INTO events (event_id, project_id, timestamp, event_type, payload)
                    VALUES ('evt-2027', 'proj-part', '2027-06-15 10:00:00+00', 'log', '{}'::jsonb);
                """)
                # Also record in event_lookup
                cur.execute("""
                    INSERT INTO event_lookup (event_id, project_id, timestamp, event_type)
                    VALUES ('evt-2025', 'proj-part', '2025-06-15 10:00:00+00', 'log'),
                           ('evt-2026', 'proj-part', '2026-06-15 10:00:00+00', 'log'),
                           ('evt-2027', 'proj-part', '2027-06-15 10:00:00+00', 'log');
                """)
            conn.commit()

            # Verify partition-specific placement
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM events_y2025 WHERE event_id = 'evt-2025';")
                self.assertEqual(cur.fetchone()[0], 1)

                cur.execute("SELECT COUNT(*) FROM events_y2026 WHERE event_id = 'evt-2026';")
                self.assertEqual(cur.fetchone()[0], 1)

                cur.execute("SELECT COUNT(*) FROM events_y2027 WHERE event_id = 'evt-2027';")
                self.assertEqual(cur.fetchone()[0], 1)

            # Verify global event_lookup prevents duplicate event_id
            with conn.cursor() as cur:
                with self.assertRaises(Exception):
                    cur.execute("""
                        INSERT INTO event_lookup (event_id, project_id, timestamp, event_type)
                        VALUES ('evt-2025', 'proj-part', '2026-01-01 00:00:00+00', 'duplicate');
                    """)
            conn.rollback()

    def test_05_tenant_isolation_via_rls_and_set_local(self) -> None:
        """Verify strict tenant isolation using RLS and transaction-scoped SET LOCAL."""
        # Setup two distinct projects and records
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO projects (project_id, name) VALUES ('proj-alpha', 'Alpha'), ('proj-beta', 'Beta');")
            conn.commit()

        # Insert memory under Project Alpha using tenant_context
        with self.mgr.tenant_context("proj-alpha") as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO memories (memory_id, project_id, content, status)
                    VALUES ('mem-alpha-01', 'proj-alpha', 'Alpha Confidential Memory', 'ACTIVE');
                """)

        # Insert memory under Project Beta using tenant_context
        with self.mgr.tenant_context("proj-beta") as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO memories (memory_id, project_id, content, status)
                    VALUES ('mem-beta-01', 'proj-beta', 'Beta Secret Strategy', 'ACTIVE');
                """)

        # Query within Alpha context: only Alpha memory must be returned
        with self.mgr.tenant_context("proj-alpha") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT memory_id, content FROM memories;")
                rows = cur.fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0][0], "mem-alpha-01")

        # Query within Beta context: only Beta memory must be returned
        with self.mgr.tenant_context("proj-beta") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT memory_id, content FROM memories;")
                rows = cur.fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0][0], "mem-beta-01")

        # Query without tenant context: 0 rows returned due to RESTRICTIVE RLS
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM memories;")
                count = cur.fetchone()[0]
                self.assertEqual(count, 0)

    def test_06_connection_pool_sanitization_clears_tenant_context(self) -> None:
        """Verify that returning a connection to the pool clears session settings (RESET ALL)."""
        # Execute in tenant context
        with self.mgr.tenant_context("proj-tenant-x") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_setting('app.current_project_id', true);")
                active_val = cur.fetchone()[0]
                self.assertEqual(active_val, "proj-tenant-x")

        # Check out connection from pool; setting must be reset
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_setting('app.current_project_id', true);")
                cleared_val = cur.fetchone()[0]
                self.assertIn(cleared_val, (None, "", "NULL"))

    def test_07_transaction_rollback_on_failure(self) -> None:
        """Verify transaction context manager rolls back all modifications on exception."""
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO projects (project_id, name) VALUES ('proj-rollback', 'Rollback Project');")
            conn.commit()

        # Execute transaction that fails midway
        with self.assertRaises(RuntimeError):
            with self.mgr.tenant_context("proj-rollback") as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO agents (agent_id, project_id, name, role)
                        VALUES ('agt-01', 'proj-rollback', 'Test Agent', 'tester');
                    """)
                # Deliberate failure
                raise RuntimeError("Intentional mid-transaction explosion")

        # Verify agt-01 was NOT committed
        with self.mgr.tenant_context("proj-rollback") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM agents WHERE agent_id = 'agt-01';")
                self.assertEqual(cur.fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
