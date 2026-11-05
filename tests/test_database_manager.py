"""Unit tests for DatabaseConnectionManager logic and safety guards."""

import unittest
from ai_brain.config import DatabaseConfig, StorageBackend
from ai_brain.storage.database import (
    ConnectionUnavailableError,
    DatabaseConnectionManager,
    DriverNotInstalledError,
    PostgreSQLError,
    get_database_manager,
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


if __name__ == "__main__":
    unittest.main()

