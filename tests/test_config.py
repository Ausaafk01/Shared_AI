"""Unit tests for AI Brain configuration management."""

import unittest
from ai_brain.config import (
    AppConfig,
    DatabaseConfig,
    SecurityConfig,
    StorageBackend,
    VectorConfig,
    WorkerConfig,
    get_config,
    load_config_from_env,
)


class TestConfig(unittest.TestCase):
    """Test configuration dataclasses, env parsing, and validation."""

    def test_default_config_is_sqlite(self) -> None:
        cfg = load_config_from_env({})
        self.assertEqual(cfg.db.backend, StorageBackend.SQLITE)
        self.assertTrue(cfg.db.is_sqlite)
        self.assertFalse(cfg.db.is_postgres)
        self.assertEqual(cfg.db.database_url, "sqlite:///ai_brain.db")
        self.assertEqual(cfg.security.default_project_id, "default")
        self.assertEqual(cfg.vector.default_dimension, 384)
        self.assertEqual(cfg.worker.worker_concurrency, 4)

    def test_postgres_backend_via_env(self) -> None:
        env = {
            "AI_BRAIN_STORAGE_BACKEND": "postgres",
            "DATABASE_URL": "postgresql://user:secret123@db.example.com:5432/aiprodb",
            "AI_BRAIN_POOL_SIZE": "15",
            "AI_BRAIN_DEFAULT_PROJECT_ID": "proj-corp-01",
            "AI_BRAIN_EMBEDDING_DIMENSION": "768",
        }
        cfg = load_config_from_env(env)
        self.assertEqual(cfg.db.backend, StorageBackend.POSTGRES)
        self.assertTrue(cfg.db.is_postgres)
        self.assertFalse(cfg.db.is_sqlite)
        self.assertEqual(cfg.db.pool_size, 15)
        self.assertEqual(cfg.security.default_project_id, "proj-corp-01")
        self.assertEqual(cfg.vector.default_dimension, 768)

    def test_sanitized_url_masks_password(self) -> None:
        db_cfg = DatabaseConfig(
            database_url="postgresql://admin_user:super_secret_password@db-host:5432/ai_brain"
        )
        sanitized = db_cfg.sanitized_url()
        self.assertNotIn("super_secret_password", sanitized)
        self.assertIn("admin_user:***", sanitized)
        self.assertIn("db-host:5432/ai_brain", sanitized)

    def test_sanitized_url_with_sqlite_returns_raw(self) -> None:
        db_cfg = DatabaseConfig(database_url="sqlite:///var/data/ai_brain.db")
        self.assertEqual(db_cfg.sanitized_url(), "sqlite:///var/data/ai_brain.db")

    def test_get_config_singleton(self) -> None:
        cfg1 = get_config(reload=True)
        cfg2 = get_config()
        self.assertIs(cfg1, cfg2)


if __name__ == "__main__":
    unittest.main()

