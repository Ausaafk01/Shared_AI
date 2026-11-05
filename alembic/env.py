"""Alembic environment configuration for AI Brain PostgreSQL migrations.

Reads database URL from AI Brain configuration or environment variables,
supporting both offline SQL generation and online execution.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from ai_brain.config import get_config

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name)
    except Exception:
        pass

# Determine target database URL
app_cfg = get_config()
target_url = (
    os.environ.get("TEST_POSTGRES_URL")
    or os.environ.get("DATABASE_URL")
    or os.environ.get("AI_BRAIN_DATABASE_URL")
    or (app_cfg.db.database_url if app_cfg.db.is_postgres else "postgresql+psycopg://postgres@127.0.0.1:5432/test_ai_brain")
)

# Normalize URL scheme for SQLAlchemy + psycopg
if target_url:
    if target_url.startswith("postgres://"):
        target_url = target_url.replace("postgres://", "postgresql+psycopg://", 1)
    elif target_url.startswith("postgresql://") and not target_url.startswith("postgresql+"):
        target_url = target_url.replace("postgresql://", "postgresql+psycopg://", 1)

config.set_main_option("sqlalchemy.url", target_url)

target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Generates raw SQL statements without needing a live database connection.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Creates an Engine and associates a connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
