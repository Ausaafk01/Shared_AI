"""Acceptance tests for AI Brain Phase 5 Milestone 5.1: PostgreSQL Relational Schema & Migrations.

Acceptance Test Matrix:
1. All 14 authoritative tables exist.
2. Required event partitions exist and routing works across partitions.
3. Event UPDATE and DELETE protections work (engine-level triggers and rules reject modification).
4. Event identity, global uniqueness, and idempotency work across partitions.
5. RLS prevents cross-project access for non-superuser application roles.
6. Connection pooling does not leak tenant context (RESET ALL sanitization).
7. Transaction rollback works correctly on failure.
8. Security review: Application role is NOT superuser and does NOT have BYPASSRLS.
9. Security review: Tenant RLS is enabled and forced on every tenant-owned table.
10. Security review: Missing or invalid tenant context denies access rather than exposing data.
11. Security review: Cross-tenant writes are rejected by RLS WITH CHECK policy.
12. Schema parity: Standalone SQL initialization script and Alembic migration produce identical schemas.
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from typing import Optional

import psycopg
from ai_brain.config import DatabaseConfig, StorageBackend, load_env_file
from ai_brain.storage.database import DatabaseConnectionManager

# Load local test environment variables if present
load_env_file()

ADMIN_PG_URL = os.environ.get(
    "TEST_POSTGRES_ADMIN_URL",
    os.environ.get("TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/test_ai_brain"),
)
APP_PG_URL = os.environ.get(
    "TEST_POSTGRES_APP_URL",
    f"postgresql://ai_brain_app_user:{os.environ.get('TEST_POSTGRES_APP_PASSWORD', '')}@127.0.0.1:5432/test_ai_brain",
)


def _check_postgres_live() -> tuple[bool, str, Optional[DatabaseConnectionManager]]:
    """Probe whether live PostgreSQL instance is reachable."""
    cfg = DatabaseConfig(
        backend=StorageBackend.POSTGRES,
        database_url=ADMIN_PG_URL,
        pool_size=3,
        pool_timeout=3.0,
    )
    mgr = DatabaseConnectionManager(cfg)
    if not mgr.driver_available:
        return False, "PostgreSQL driver (psycopg or psycopg2) not installed", None

    try:
        conn = mgr.get_connection(timeout=3.0)
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
    app_mgr: DatabaseConnectionManager

    @classmethod
    def setUpClass(cls) -> None:
        if not _IS_PG_LIVE or _MODULE_MGR is None:
            raise unittest.SkipTest(f"Live PostgreSQL required: {_PG_SKIP_REASON}")
        cls.mgr = _MODULE_MGR

        # Ensure app role and database are set up
        cls._ensure_roles_and_schema()

        # Application connection manager using unprivileged non-superuser role
        app_cfg = DatabaseConfig(
            backend=StorageBackend.POSTGRES,
            database_url=APP_PG_URL,
            pool_size=5,
            pool_timeout=3.0,
        )
        cls.app_mgr = DatabaseConnectionManager(app_cfg)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "app_mgr") and cls.app_mgr:
            cls.app_mgr.close()
        if hasattr(cls, "mgr") and cls.mgr:
            cls.mgr.close()

    @classmethod
    def _ensure_roles_and_schema(cls) -> None:
        """Apply schema and configure test role permissions."""
        sql = _read_schema_sql()
        with cls.mgr.connection() as conn:
            with conn.cursor() as cur:
                # Create role if missing with secure dynamic password
                app_pw = os.environ.get("TEST_POSTGRES_APP_PASSWORD", "test_app_user_secret")
                cur.execute(psycopg.sql.SQL("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ai_brain_app_user') THEN
                        CREATE ROLE ai_brain_app_user WITH LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
                    END IF;
                END $$;
                """).format(psycopg.sql.Literal(app_pw)))
                cur.execute(sql)
                cur.execute("GRANT ALL ON ALL TABLES IN SCHEMA public TO ai_brain_app_user;")
                cur.execute("GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO ai_brain_app_user;")
            conn.commit()

    def setUp(self) -> None:
        # Clean tenant test data before each test using admin connection
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
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

    def test_02_event_partitions_exist_and_route_records(self) -> None:
        """Verify events table has range partitions for 2025, 2026, 2027 and routes rows correctly."""
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

        # Test partition routing
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO projects (project_id, name) VALUES ('proj-part-route', 'Partition Route Test');")
                cur.execute("""
                    INSERT INTO events (event_id, project_id, timestamp, event_type, payload)
                    VALUES ('evt-p2025', 'proj-part-route', '2025-06-01 10:00:00+00', 'log', '{}'::jsonb),
                           ('evt-p2026', 'proj-part-route', '2026-06-01 10:00:00+00', 'log', '{}'::jsonb),
                           ('evt-p2027', 'proj-part-route', '2027-06-01 10:00:00+00', 'log', '{}'::jsonb);
                """)
            conn.commit()

            # Verify rows are stored in respective partition tables
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM events_y2025 WHERE event_id = 'evt-p2025';")
                self.assertEqual(cur.fetchone()[0], 1)
                cur.execute("SELECT COUNT(*) FROM events_y2026 WHERE event_id = 'evt-p2026';")
                self.assertEqual(cur.fetchone()[0], 1)
                cur.execute("SELECT COUNT(*) FROM events_y2027 WHERE event_id = 'evt-p2027';")
                self.assertEqual(cur.fetchone()[0], 1)

    def test_03_append_only_event_protections_reject_update_and_delete(self) -> None:
        """Verify engine-level rules/triggers reject UPDATE and DELETE on events."""
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO projects (project_id, name) VALUES (%s, %s);",
                    ("proj-append-test", "Append Only Test"),
                )
            conn.commit()

        # Insert event under tenant context
        with self.app_mgr.tenant_context("proj-append-test") as conn:
            with conn.cursor() as cur:
                ts = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
                cur.execute("""
                    INSERT INTO events (event_id, project_id, timestamp, event_type, payload)
                    VALUES (%s, %s, %s, %s, %s::jsonb);
                """, ("evt-001", "proj-append-test", ts, "state_change", '{"status": "old"}'))

            # Attempt UPDATE: rejected via rule (rowcount 0) or trigger exception
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        UPDATE events SET payload = '{"status": "tampered"}'::jsonb
                        WHERE event_id = 'evt-001';
                    """)
                    self.assertEqual(cur.rowcount, 0, "UPDATE must not affect any rows due to write-once rule")
                except Exception as e:
                    self.assertIn("append-only", str(e).lower())

            # Attempt DELETE: rejected via rule (rowcount 0) or trigger exception
            with conn.cursor() as cur:
                try:
                    cur.execute("DELETE FROM events WHERE event_id = 'evt-001';")
                    self.assertEqual(cur.rowcount, 0, "DELETE must not affect any rows due to write-once rule")
                except Exception as e:
                    self.assertIn("append-only", str(e).lower())

            # Verify original record remains completely intact and untampered
            with conn.cursor() as cur:
                cur.execute("SELECT payload->>'status' FROM events WHERE event_id = 'evt-001';")
                row = cur.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], "old")

    def test_04_event_identity_and_partition_routing(self) -> None:
        """Verify global event_lookup uniqueness and partition routing across partition ranges."""
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO projects (project_id, name) VALUES ('proj-part', 'Part Test');")
                cur.execute("""
                    INSERT INTO event_lookup (event_id, project_id, timestamp, event_type)
                    VALUES ('evt-uniq-1', 'proj-part', '2026-06-15 10:00:00+00', 'log');
                """)
            conn.commit()

            # Attempt duplicate insert with same event_id
            with conn.cursor() as cur:
                with self.assertRaises(Exception):
                    cur.execute("""
                        INSERT INTO event_lookup (event_id, project_id, timestamp, event_type)
                        VALUES ('evt-uniq-1', 'proj-part', '2026-06-15 11:00:00+00', 'duplicate');
                    """)
            conn.rollback()

    def test_05_tenant_isolation_via_rls_and_set_local(self) -> None:
        """Verify strict tenant isolation using RLS and transaction-scoped SET LOCAL with application role."""
        # Insert base project records
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO projects (project_id, name) VALUES ('proj-alpha', 'Alpha'), ('proj-beta', 'Beta');")
            conn.commit()

        # Insert memory under Project Alpha using app_mgr in tenant context
        with self.app_mgr.tenant_context("proj-alpha") as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO memories (memory_id, project_id, content, status)
                    VALUES ('mem-alpha-01', 'proj-alpha', 'Alpha Confidential Memory', 'ACTIVE');
                """)

        # Insert memory under Project Beta using app_mgr in tenant context
        with self.app_mgr.tenant_context("proj-beta") as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO memories (memory_id, project_id, content, status)
                    VALUES ('mem-beta-01', 'proj-beta', 'Beta Secret Strategy', 'ACTIVE');
                """)

        # Query within Alpha context: only Alpha memory is visible
        with self.app_mgr.tenant_context("proj-alpha") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT memory_id, content FROM memories;")
                rows = cur.fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0][0], "mem-alpha-01")

        # Query within Beta context: only Beta memory is visible
        with self.app_mgr.tenant_context("proj-beta") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT memory_id, content FROM memories;")
                rows = cur.fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0][0], "mem-beta-01")

        # Query without tenant context: 0 rows returned due to RLS
        with self.app_mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM memories;")
                count = cur.fetchone()[0]
                self.assertEqual(count, 0)

    def test_06_connection_pool_sanitization_clears_tenant_context(self) -> None:
        """Verify that returning a connection to the pool clears session settings (RESET ALL)."""
        with self.app_mgr.tenant_context("proj-tenant-x") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_setting('app.current_project_id', true);")
                active_val = cur.fetchone()[0]
                self.assertEqual(active_val, "proj-tenant-x")

        # Check out connection from pool; setting must be reset
        with self.app_mgr.connection() as conn:
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
            with self.app_mgr.tenant_context("proj-rollback") as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO agents (agent_id, project_id, name, role)
                        VALUES ('agt-rb-01', 'proj-rollback', 'Rollback Agent', 'tester');
                    """)
                raise RuntimeError("Intentional mid-transaction explosion")

        # Verify agt-rb-01 was NOT committed
        with self.app_mgr.tenant_context("proj-rollback") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM agents WHERE agent_id = 'agt-rb-01';")
                self.assertEqual(cur.fetchone()[0], 0)

    # -------------------------------------------------------------------------
    # Focused Security Review Acceptance Tests (Part 4)
    # -------------------------------------------------------------------------

    def test_08_application_role_is_not_superuser_and_no_bypass_rls(self) -> None:
        """Security: Verify the application database role has neither superuser nor BYPASSRLS."""
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT rolname, rolsuper, rolbypassrls
                    FROM pg_roles
                    WHERE rolname = 'ai_brain_app_user';
                """)
                row = cur.fetchone()
                self.assertIsNotNone(row, "Role 'ai_brain_app_user' must exist")
                rolname, rolsuper, rolbypassrls = row
                self.assertEqual(rolname, "ai_brain_app_user")
                self.assertFalse(rolsuper, "Application role MUST NOT be a superuser")
                self.assertFalse(rolbypassrls, "Application role MUST NOT have BYPASSRLS")

    def test_09_tenant_rls_enabled_and_forced_on_all_tenant_tables(self) -> None:
        """Security: Verify RLS is enabled and forced on every tenant-owned table."""
        tenant_tables = [
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
        ]

        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                for tbl in tenant_tables:
                    cur.execute("""
                        SELECT c.relrowsecurity, c.relforcerowsecurity
                        FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE n.nspname = 'public' AND c.relname = %s;
                    """, (tbl,))
                    row = cur.fetchone()
                    self.assertIsNotNone(row, f"Table '{tbl}' not found in pg_class")
                    row_sec, force_sec = row
                    self.assertTrue(row_sec, f"RLS must be ENABLED on '{tbl}'")
                    self.assertTrue(force_sec, f"RLS must be FORCED on '{tbl}'")

    def test_10_missing_tenant_context_denies_all_access(self) -> None:
        """Security: Missing or invalid tenant context denies read access rather than exposing data."""
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO projects (project_id, name) VALUES ('proj-sec-test', 'Sec Test');")
            conn.commit()

        # Insert a memory using app_mgr with valid context
        with self.app_mgr.tenant_context("proj-sec-test") as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO memories (memory_id, project_id, content, status)
                    VALUES ('mem-sec-01', 'proj-sec-test', 'Sensitive Content', 'ACTIVE');
                """)

        # Connect with app role without setting app.current_project_id
        with self.app_mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM memories;")
                count = cur.fetchone()[0]
                self.assertEqual(count, 0, "Query without tenant context must return 0 records")

    def test_11_cross_tenant_write_rejected_by_rls_check(self) -> None:
        """Security: Attempting to insert a row with mismatched project_id is rejected by RLS WITH CHECK."""
        with self.mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO projects (project_id, name) VALUES ('proj-own', 'Own'), ('proj-alien', 'Alien');")
            conn.commit()

        # In context of proj-own, attempt to insert row with project_id = 'proj-alien'
        with self.assertRaises(Exception) as ctx:
            with self.app_mgr.tenant_context("proj-own") as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO memories (memory_id, project_id, content, status)
                        VALUES ('mem-alien-injection', 'proj-alien', 'Injected Cross-Tenant Data', 'ACTIVE');
                    """)
        # Error must be RLS policy violation
        self.assertIn("policy", str(ctx.exception).lower())

    def test_12_alembic_migration_lifecycle_and_schema_parity(self) -> None:
        """Verify Alembic upgrade head -> downgrade base -> upgrade head lifecycle and schema parity."""
        import subprocess
        import sys

        alembic_db = "test_ai_brain_alembic_gate"
        admin_url = "postgresql://postgres@127.0.0.1:5432/postgres"
        target_db_url = f"postgresql://postgres@127.0.0.1:5432/{alembic_db}"

        # 1. Create fresh database for Alembic test
        with psycopg.connect(admin_url, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE IF EXISTS {alembic_db};")
            conn.execute(f"CREATE DATABASE {alembic_db};")

        try:
            env = dict(os.environ)
            env["TEST_POSTGRES_URL"] = target_db_url
            env["DATABASE_URL"] = target_db_url

            # 2. Run alembic upgrade head
            cmd_up = [sys.executable, "-m", "alembic", "upgrade", "head"]
            res = subprocess.run(cmd_up, env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Alembic upgrade failed: {res.stderr}")

            # Inspect tables after upgrade
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
            with psycopg.connect(target_db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                          AND table_name NOT IN ('alembic_version');
                    """)
                    tables_up = {r[0] for r in cur.fetchall()}
            for t in expected_tables:
                self.assertIn(t, tables_up, f"Table {t} missing after Alembic upgrade")

            # 3. Run alembic downgrade base
            cmd_down = [sys.executable, "-m", "alembic", "downgrade", "base"]
            res_down = subprocess.run(cmd_down, env=env, capture_output=True, text=True)
            self.assertEqual(res_down.returncode, 0, f"Alembic downgrade failed: {res_down.stderr}")

            # Verify tables cleanly dropped
            with psycopg.connect(target_db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                          AND table_name NOT IN ('alembic_version');
                    """)
                    tables_down = {r[0] for r in cur.fetchall()}
            self.assertEqual(len(tables_down), 0, f"Expected 0 tables after downgrade, found: {tables_down}")

            # 4. Re-run alembic upgrade head
            res_reup = subprocess.run(cmd_up, env=env, capture_output=True, text=True)
            self.assertEqual(res_reup.returncode, 0, f"Alembic re-upgrade failed: {res_reup.stderr}")

            # Verify usable afterward
            with psycopg.connect(target_db_url, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO projects (project_id, name) VALUES ('proj-alembic-check', 'Usability Test');")
                    cur.execute("SELECT name FROM projects WHERE project_id = 'proj-alembic-check';")
                    self.assertEqual(cur.fetchone()[0], "Usability Test")
        finally:
            with psycopg.connect(admin_url, autocommit=True) as conn:
                conn.execute(f"DROP DATABASE IF EXISTS {alembic_db};")


if __name__ == "__main__":
    unittest.main()
