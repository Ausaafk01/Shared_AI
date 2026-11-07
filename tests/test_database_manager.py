"""Unit tests for DatabaseConnectionManager logic and safety guards."""

from concurrent.futures import ThreadPoolExecutor
import os
import unittest
import uuid
from ai_brain.config import DatabaseConfig, StorageBackend, load_env_file
from ai_brain.storage.database import (
    ConnectionUnavailableError,
    DatabaseConnectionManager,
    DriverNotInstalledError,
    PostgreSQLError,
    get_database_manager,
)

load_env_file()
ADMIN_PG_URL = os.environ.get(
    "TEST_POSTGRES_ADMIN_URL",
    os.environ.get("TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/test_ai_brain"),
)


class TestDatabaseManager(unittest.TestCase):
    """Test connection manager initialization, validation, and lifecycle."""

    def test_manager_instantiation_with_custom_config(self) -> None:
        cfg = DatabaseConfig(
            backend=StorageBackend.POSTGRES,
            database_url="postgresql://localhost:5432/test_db",
            pool_size=3,
            max_overflow=2,
        )
        mgr = DatabaseConnectionManager(cfg)
        self.assertEqual(mgr.config.pool_size, 3)
        self.assertEqual(mgr.config.max_overflow, 2)
        # Idempotent close
        mgr.close()
        mgr.close()

    def test_tenant_context_validates_project_id(self) -> None:
        cfg = DatabaseConfig(
            backend=StorageBackend.POSTGRES,
            database_url="postgresql://localhost:5432/test_db",
        )
        mgr = DatabaseConnectionManager(cfg)
        with self.assertRaises(ValueError):
            with mgr.tenant_context(""):
                pass

        with self.assertRaises(ValueError):
            with mgr.tenant_context(None):  # type: ignore
                pass
        mgr.close()

    def test_get_connection_fails_cleanly_when_closed(self) -> None:
        cfg = DatabaseConfig(
            backend=StorageBackend.POSTGRES,
            database_url="postgresql://localhost:5432/test_db",
        )
        mgr = DatabaseConnectionManager(cfg)
        mgr.close()
        with self.assertRaises(PostgreSQLError) as ctx:
            mgr.get_connection()
        self.assertIn("closed", str(ctx.exception))

    def test_missing_driver_or_host_raises_expected_exception(self) -> None:
        cfg = DatabaseConfig(
            backend=StorageBackend.POSTGRES,
            database_url="postgresql://invalid-user:badpass@127.0.0.1:54399/nonexistent",
            pool_timeout=1.0,
        )
        mgr = DatabaseConnectionManager(cfg)
        if not mgr.driver_available:
            with self.assertRaises(DriverNotInstalledError):
                mgr.get_connection()
        else:
            with self.assertRaises(ConnectionUnavailableError):
                mgr.get_connection()
        mgr.close()

    def test_singleton_reset(self) -> None:
        mgr1 = get_database_manager()
        mgr2 = get_database_manager()
        self.assertIs(mgr1, mgr2)
        mgr3 = get_database_manager(reset=True)
        self.assertIsNot(mgr1, mgr3)
        mgr3.close()

    def test_ensure_project_exists_noop_for_falsy(self) -> None:
        cfg = DatabaseConfig(
            backend=StorageBackend.POSTGRES,
            database_url="postgresql://invalid:invalid@127.0.0.1:54399/nonexistent",
        )
        mgr = DatabaseConnectionManager(cfg)
        # Should return immediately without attempting connection checkout
        mgr.ensure_project_exists("")
        mgr.ensure_project_exists(None)  # type: ignore
        mgr.close()

    def test_ensure_project_exists_atomic_and_concurrent(self) -> None:
        cfg = DatabaseConfig(
            backend=StorageBackend.POSTGRES,
            database_url=ADMIN_PG_URL,
            pool_size=10,
            pool_timeout=5.0,
        )
        mgr = DatabaseConnectionManager(cfg)
        if not mgr.driver_available:
            self.skipTest("PostgreSQL driver not available")
        try:
            conn = mgr.get_connection(timeout=2.0)
            mgr.return_connection(conn)
        except Exception as exc:
            mgr.close()
            self.skipTest(f"Live PostgreSQL not reachable: {exc}")

        try:
            test_pid = f"proj_atomic_{uuid.uuid4().hex[:8]}"
            # 1. First insert
            mgr.ensure_project_exists(test_pid)
            with mgr.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT project_id, name FROM projects WHERE project_id = %s;", (test_pid,))
                    row = cur.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], test_pid)
            self.assertEqual(row[1], test_pid)

            # 2. Idempotent re-insert
            mgr.ensure_project_exists(test_pid)

            # 3. Concurrent initialization of same and distinct projects
            concurrent_pid = f"proj_concurrent_{uuid.uuid4().hex[:8]}"
            pids = [concurrent_pid] * 8 + [f"proj_batch_{uuid.uuid4().hex[:6]}" for _ in range(4)]

            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(mgr.ensure_project_exists, p) for p in pids]
                for f in futures:
                    f.result()

            with mgr.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT project_id FROM projects WHERE project_id = %s;", (concurrent_pid,))
                    res = cur.fetchall()
            self.assertEqual(len(res), 1)
        finally:
            mgr.close()


if __name__ == "__main__":
    unittest.main()

