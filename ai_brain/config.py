"""Production Configuration Management for AI Brain.

Provides environment-based configuration with safe defaults, supporting
both local embedded SQLite operations (default for offline tests and development)
and production PostgreSQL infrastructure with pgvector and tenant isolation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from urllib.parse import urlparse


class StorageBackend(str, Enum):
    """Supported storage backend engines."""
    SQLITE = "sqlite"
    POSTGRES = "postgres"


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connectivity and connection pool configuration."""
    backend: StorageBackend = StorageBackend.SQLITE
    database_url: str = "sqlite:///ai_brain.db"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 1800
    schema_name: str = "public"
    application_name: str = "ai_brain"

    @property
    def is_postgres(self) -> bool:
        """Return True if backend or URL indicates PostgreSQL."""
        return (
            self.backend == StorageBackend.POSTGRES
            or self.database_url.startswith("postgres://")
            or self.database_url.startswith("postgresql://")
        )

    @property
    def is_sqlite(self) -> bool:
        """Return True if backend or URL indicates SQLite."""
        return not self.is_postgres

    def sanitized_url(self) -> str:
        """Return database URL with password masked for safe logging."""
        if not self.database_url or "://" not in self.database_url:
            return self.database_url
        try:
            parsed = urlparse(self.database_url)
            if parsed.password:
                netloc = f"{parsed.username or ''}:***@{parsed.hostname or ''}"
                if parsed.port:
                    netloc += f":{parsed.port}"
                return parsed._replace(netloc=netloc).geturl()
            return self.database_url
        except Exception:
            return "<masked_url>"


@dataclass(frozen=True)
class SecurityConfig:
    """Security, tenant isolation, and credential handling configuration."""
    default_project_id: str = "default"
    jwt_algorithm: str = "RS256"
    jwt_secret_key: Optional[str] = None
    jwt_public_key: Optional[str] = None
    jwt_issuer: str = "ai-brain"
    jwt_audience: str = "ai-brain-api"
    api_key_prefix: str = "agy"
    token_expire_minutes: int = 60
    enable_rls: bool = True
    mask_credentials_in_logs: bool = True


@dataclass(frozen=True)
class VectorConfig:
    """pgvector and semantic embedding configuration."""
    default_dimension: int = 384
    supported_dimensions: tuple[int, ...] = (384, 768, 1536)
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40


@dataclass(frozen=True)
class WorkerConfig:
    """Background task queue and worker fleet configuration."""
    worker_concurrency: int = 4
    lease_duration_seconds: int = 300
    heartbeat_interval_seconds: int = 30
    max_task_retries: int = 3
    poll_interval_seconds: float = 1.0


@dataclass(frozen=True)
class AppConfig:
    """Master application configuration aggregate."""
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    environment: str = "development"
    debug: bool = False


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except (ValueError, TypeError):
        return default


def _parse_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value.strip())
    except (ValueError, TypeError):
        return default


def load_env_file(file_path: Optional[str] = None) -> None:
    """Load key-value pairs from an ignored .env or .env.test file into os.environ if not set."""
    if file_path is None:
        for candidate in (".env", ".env.local", ".env.test"):
            if os.path.exists(candidate):
                load_env_file(candidate)
        return

    if not os.path.isfile(file_path):
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


def load_config_from_env(env: Optional[Dict[str, str]] = None) -> AppConfig:
    """Load and validate configuration from environment variables or given dict.

    Environment variables:
    - AI_BRAIN_STORAGE_BACKEND / STORAGE_BACKEND: 'sqlite' (default) or 'postgres'
    - DATABASE_URL / AI_BRAIN_DATABASE_URL / TEST_POSTGRES_URL: Database connection string
    - AI_BRAIN_POOL_SIZE: Pool size (default: 5)
    - AI_BRAIN_MAX_OVERFLOW: Pool overflow (default: 10)
    - AI_BRAIN_DEFAULT_PROJECT_ID: Default project ID (default: 'default')
    - AI_BRAIN_EMBEDDING_DIMENSION: Default vector dimension (default: 384)
    - AI_BRAIN_WORKER_CONCURRENCY: Concurrency level (default: 4)
    - AI_BRAIN_ENV / ENVIRONMENT: 'development', 'test', 'production'
    - AI_BRAIN_DEBUG: 'true'/'false'
    """
    if env is None:
        load_env_file()
    e = os.environ if env is None else env

    raw_backend = (e.get("AI_BRAIN_STORAGE_BACKEND") or e.get("STORAGE_BACKEND") or "sqlite").strip().lower()
    raw_db_url = (
        e.get("AI_BRAIN_DATABASE_URL")
        or e.get("DATABASE_URL")
        or ("sqlite:///ai_brain.db" if raw_backend == "sqlite" else "postgresql://postgres:postgres@localhost:5432/ai_brain")
    )

    # Infer backend if postgresql URL provided
    if raw_db_url.startswith(("postgres://", "postgresql://")):
        backend = StorageBackend.POSTGRES
    elif raw_backend == "postgres":
        backend = StorageBackend.POSTGRES
    else:
        backend = StorageBackend.SQLITE

    db_config = DatabaseConfig(
        backend=backend,
        database_url=raw_db_url,
        pool_size=_parse_int(e.get("AI_BRAIN_POOL_SIZE"), 5),
        max_overflow=_parse_int(e.get("AI_BRAIN_MAX_OVERFLOW"), 10),
        pool_timeout=_parse_float(e.get("AI_BRAIN_POOL_TIMEOUT"), 30.0),
        pool_recycle=_parse_int(e.get("AI_BRAIN_POOL_RECYCLE"), 1800),
        schema_name=e.get("AI_BRAIN_SCHEMA", "public").strip(),
        application_name=e.get("AI_BRAIN_APP_NAME", "ai_brain").strip(),
    )

    security_config = SecurityConfig(
        default_project_id=e.get("AI_BRAIN_DEFAULT_PROJECT_ID", "default").strip(),
        jwt_algorithm=e.get("AI_BRAIN_JWT_ALGORITHM", "RS256").strip(),
        jwt_secret_key=e.get("AI_BRAIN_JWT_SECRET"),
        jwt_public_key=e.get("AI_BRAIN_JWT_PUBLIC_KEY"),
        jwt_issuer=e.get("AI_BRAIN_JWT_ISSUER", "ai-brain").strip(),
        jwt_audience=e.get("AI_BRAIN_JWT_AUDIENCE", "ai-brain-api").strip(),
        api_key_prefix=e.get("AI_BRAIN_API_KEY_PREFIX", "agy").strip(),
        token_expire_minutes=_parse_int(e.get("AI_BRAIN_TOKEN_EXPIRE_MINUTES"), 60),
        enable_rls=_parse_bool(e.get("AI_BRAIN_ENABLE_RLS"), True),
        mask_credentials_in_logs=_parse_bool(e.get("AI_BRAIN_MASK_CREDENTIALS"), True),
    )

    vector_config = VectorConfig(
        default_dimension=_parse_int(e.get("AI_BRAIN_EMBEDDING_DIMENSION"), 384),
        hnsw_m=_parse_int(e.get("AI_BRAIN_HNSW_M"), 16),
        hnsw_ef_construction=_parse_int(e.get("AI_BRAIN_HNSW_EF_CONSTRUCTION"), 64),
        hnsw_ef_search=_parse_int(e.get("AI_BRAIN_HNSW_EF_SEARCH"), 40),
    )

    worker_config = WorkerConfig(
        worker_concurrency=_parse_int(e.get("AI_BRAIN_WORKER_CONCURRENCY"), 4),
        lease_duration_seconds=_parse_int(e.get("AI_BRAIN_LEASE_DURATION"), 300),
        heartbeat_interval_seconds=_parse_int(e.get("AI_BRAIN_HEARTBEAT_INTERVAL"), 30),
        max_task_retries=_parse_int(e.get("AI_BRAIN_MAX_TASK_RETRIES"), 3),
        poll_interval_seconds=_parse_float(e.get("AI_BRAIN_POLL_INTERVAL"), 1.0),
    )

    environment = (e.get("AI_BRAIN_ENV") or e.get("ENVIRONMENT") or "development").strip().lower()
    debug = _parse_bool(e.get("AI_BRAIN_DEBUG"), False)

    return AppConfig(
        db=db_config,
        security=security_config,
        vector=vector_config,
        worker=worker_config,
        environment=environment,
        debug=debug,
    )


# Singleton cache
_GLOBAL_CONFIG: Optional[AppConfig] = None


def get_config(reload: bool = False) -> AppConfig:
    """Retrieve global cached AppConfig or initialize from environment."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None or reload:
        _GLOBAL_CONFIG = load_config_from_env()
    return _GLOBAL_CONFIG

