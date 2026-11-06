"""FastAPI REST Service Layer & Streaming Endpoints package for AI Brain (M5.5)."""

from ai_brain.server.app import create_app
from ai_brain.server.dependencies import (
    BrainServiceContainer,
    create_postgres_container,
    create_sqlite_container,
    get_container,
    get_current_identity,
    require_permission,
)

__all__ = [
    "create_app",
    "BrainServiceContainer",
    "create_sqlite_container",
    "create_postgres_container",
    "get_container",
    "get_current_identity",
    "require_permission",
]
