"""Dependency injection container and authorization providers for AI Brain REST API (M5.5).

Provides unified access to:
- Authentication & Authorization services (JWT, Agent API Keys, RBAC)
- Domain stores & engines (BrainLoop, RetrievalEngine, SessionManager, CollaborationEngine)
- Tenant isolation and connection management
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional
from fastapi import Depends, Header, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ai_brain.agents.collaboration.workflow import CollaborationEngine
from ai_brain.server.idempotency import (
    IdempotencyManager,
    PostgreSQLIdempotencyCoordinator,
    SQLiteIdempotencyCoordinator,
)
from ai_brain.server.limiter import HttpRateLimiter
from ai_brain.tasks.registry import TaskHandlerRegistry
from ai_brain.tasks.handlers import register_default_handlers
from ai_brain.tasks.store import BaseTaskStore, PostgreSQLTaskStore, SQLiteTaskStore
from ai_brain.tasks.fleet import WorkerFleet
from ai_brain.agents.sessions.manager import SessionManager
from ai_brain.agents.sessions.storage.sqlite import SQLiteSessionStore
from ai_brain.auth import (
    AuthenticatedIdentity,
    AuthenticationService,
    AuthorizationService,
    AuthStore,
    KeyManager,
    Permission,
    PermissionDeniedError,
    PrivilegeEscalationError,
    SecurityAuditLogger,
    TokenService,
)
from ai_brain.auth.store import PostgreSQLAuthStore, SQLiteAuthStore
from ai_brain.config import AppConfig, load_config_from_env
from ai_brain.context import ContextBuilder, DeterministicTokenizer
from ai_brain.events import EventStore, SQLiteEventStore
from ai_brain.graph import GraphIntegrator, KnowledgeGraph
from ai_brain.loop import BrainLoop
from ai_brain.loop.storage.sqlite import SQLiteInteractionStore
from ai_brain.memories import MemoryUpdater
from ai_brain.memories.storage import MemoryStore
from ai_brain.memories.storage.sqlite import SQLiteMemoryStore
from ai_brain.retrieval import RetrievalEngine
from ai_brain.storage.database import DatabaseConnectionManager
from ai_brain.vectors import (
    MockEmbeddingProvider,
    SQLiteVectorStore,
    VectorIndex,
    VectorMemoryManager,
    VectorStore,
)

logger = logging.getLogger("ai_brain.server.deps")

# Bearer security scheme for OpenAPI documentation
bearer_scheme = HTTPBearer(auto_error=False)


class BrainServiceContainer:
    """Unified service container maintaining domain components and persistence stores."""

    def __init__(
        self,
        *,
        config: AppConfig,
        db_manager: Optional[DatabaseConnectionManager] = None,
        auth_store: AuthStore,
        key_manager: KeyManager,
        token_service: TokenService,
        audit_logger: SecurityAuditLogger,
        auth_service: AuthenticationService,
        rbac_service: AuthorizationService,
        event_store: EventStore,
        interaction_store: Any,
        memory_store: MemoryStore,
        memory_updater: MemoryUpdater,
        vector_store: VectorStore,
        vector_manager: VectorMemoryManager,
        graph: KnowledgeGraph,
        retrieval_engine: RetrievalEngine,
        context_builder: ContextBuilder,
        brain_loop: BrainLoop,
        session_manager: SessionManager,
        collaboration_engine: CollaborationEngine,
        idempotency_manager: Optional[IdempotencyManager] = None,
        task_store: Optional[BaseTaskStore] = None,
        task_registry: Optional[TaskHandlerRegistry] = None,
        worker_fleet: Optional[WorkerFleet] = None,
    ) -> None:
        self.config = config
        self.db_manager = db_manager
        self.auth_store = auth_store
        self.key_manager = key_manager
        self.token_service = token_service
        self.audit_logger = audit_logger
        self.auth_service = auth_service
        self.rbac_service = rbac_service
        self.event_store = event_store
        self.interaction_store = interaction_store
        self.memory_store = memory_store
        self.memory_updater = memory_updater
        self.vector_store = vector_store
        self.vector_manager = vector_manager
        self.graph = graph
        self.retrieval_engine = retrieval_engine
        self.context_builder = context_builder
        self.brain_loop = brain_loop
        self.session_manager = session_manager
        self.collaboration_engine = collaboration_engine
        self.idempotency_manager = idempotency_manager or IdempotencyManager()
        self.task_store = task_store
        self.task_registry = task_registry
        self.worker_fleet = worker_fleet

    def close(self) -> None:
        """Gracefully release stores, worker fleets, and database connection pools."""
        if self.worker_fleet:
            try:
                self.worker_fleet.stop(timeout=5.0)
            except Exception:
                pass
        if self.task_store and hasattr(self.task_store, "close"):
            try:
                self.task_store.close()
            except Exception:
                pass

        for comp in (
            self.brain_loop,
            self.session_manager,
            self.auth_store,
            self.event_store,
            self.interaction_store,
            self.memory_store,
            self.vector_store,
            self.idempotency_manager,
        ):
            if comp and hasattr(comp, "close"):
                try:
                    comp.close()
                except Exception:
                    pass
        if self.db_manager:
            try:
                self.db_manager.close()
            except Exception:
                pass


def create_sqlite_container(
    db_path: str = ":memory:",
    config: Optional[AppConfig] = None,
    secret_key: str = "default-sqlite-secret-key-32b-length!!",
) -> BrainServiceContainer:
    """Factory creating a self-contained in-memory or on-disk SQLite container."""
    app_cfg = config or load_config_from_env()

    audit_logger = SecurityAuditLogger()
    auth_store = SQLiteAuthStore(db_path)
    key_manager = KeyManager(active_key_id="k_default", active_signing_key=secret_key)
    token_service = TokenService(key_manager=key_manager)
    auth_service = AuthenticationService(auth_store, token_service, audit_logger=audit_logger)
    rbac_service = AuthorizationService(auth_store, audit_logger=audit_logger)

    event_store = SQLiteEventStore(db_path)
    interaction_store = SQLiteInteractionStore(db_path)
    memory_store = SQLiteMemoryStore(db_path)
    memory_updater = MemoryUpdater()

    vector_provider = MockEmbeddingProvider()
    vector_store = SQLiteVectorStore(db_path)
    vector_index = VectorIndex(storage=vector_store, provider=vector_provider)
    vector_manager = VectorMemoryManager(storage=vector_store, provider=vector_provider)

    graph = KnowledgeGraph()
    graph_integrator = GraphIntegrator(graph)

    retrieval_engine = RetrievalEngine(
        vector_index=vector_index,
        graph=graph,
        allow_partial_results=True,
    )
    context_builder = ContextBuilder(
        tokenizer=DeterministicTokenizer(),
        default_max_tokens=1000,
    )

    brain_loop = BrainLoop(
        event_store=event_store,
        interaction_store=interaction_store,
        retrieval_engine=retrieval_engine,
        context_builder=context_builder,
        graph=graph,
        graph_integrator=graph_integrator,
        vector_manager=vector_manager,
        memory_updater=memory_updater,
    )

    session_store = SQLiteSessionStore(db_path)
    session_manager = SessionManager(store=session_store, event_store=event_store)
    collaboration_engine = CollaborationEngine()

    idempotency_coordinator = SQLiteIdempotencyCoordinator(db_path)
    idempotency_manager = IdempotencyManager(coordinator=idempotency_coordinator)

    task_store = SQLiteTaskStore(db_path)
    task_registry = TaskHandlerRegistry()
    register_default_handlers(task_registry)

    return BrainServiceContainer(
        config=app_cfg,
        db_manager=None,
        auth_store=auth_store,
        key_manager=key_manager,
        token_service=token_service,
        audit_logger=audit_logger,
        auth_service=auth_service,
        rbac_service=rbac_service,
        event_store=event_store,
        interaction_store=interaction_store,
        memory_store=memory_store,
        memory_updater=memory_updater,
        vector_store=vector_store,
        vector_manager=vector_manager,
        graph=graph,
        retrieval_engine=retrieval_engine,
        context_builder=context_builder,
        brain_loop=brain_loop,
        session_manager=session_manager,
        collaboration_engine=collaboration_engine,
        idempotency_manager=idempotency_manager,
        task_store=task_store,
        task_registry=task_registry,
    )


def create_postgres_container(
    db_manager: DatabaseConnectionManager,
    config: Optional[AppConfig] = None,
    secret_key: str = "default-postgres-secret-key-32b-length!",
) -> BrainServiceContainer:
    """Factory creating a production-grade PostgreSQL container."""
    from ai_brain.events.storage.postgres import PostgreSQLEventStore
    from ai_brain.graph.storage.postgres import PostgreSQLGraphStore
    from ai_brain.loop.storage.postgres import PostgreSQLInteractionStore
    from ai_brain.memories.storage.postgres import PostgreSQLMemoryStore
    from ai_brain.vectors.storage.postgres import PostgreSQLVectorStore
    from ai_brain.agents.sessions.storage.postgres import PostgreSQLSessionStore

    app_cfg = config or load_config_from_env()

    audit_logger = SecurityAuditLogger()
    auth_store = PostgreSQLAuthStore(db_manager)
    key_manager = KeyManager(active_key_id="k_pg_default", active_signing_key=secret_key)
    token_service = TokenService(key_manager=key_manager)
    auth_service = AuthenticationService(auth_store, token_service, audit_logger=audit_logger)
    rbac_service = AuthorizationService(auth_store, audit_logger=audit_logger)

    event_store = PostgreSQLEventStore(db_manager)
    interaction_store = PostgreSQLInteractionStore(
        db_manager, default_project_id=app_cfg.security.default_project_id
    )
    memory_store = PostgreSQLMemoryStore(db_manager)
    memory_updater = MemoryUpdater()

    vector_provider = MockEmbeddingProvider()
    vector_store = PostgreSQLVectorStore(db_manager, allow_in_memory_fallback=True)
    vector_index = VectorIndex(storage=vector_store, provider=vector_provider)
    vector_manager = VectorMemoryManager(storage=vector_store, provider=vector_provider)

    graph_store = PostgreSQLGraphStore(db_manager)
    graph = KnowledgeGraph()
    graph_integrator = GraphIntegrator(graph)

    retrieval_engine = RetrievalEngine(
        vector_index=vector_index,
        graph=graph,
        allow_partial_results=True,
    )
    context_builder = ContextBuilder(
        tokenizer=DeterministicTokenizer(),
        default_max_tokens=1000,
    )

    brain_loop = BrainLoop(
        event_store=event_store,
        interaction_store=interaction_store,
        retrieval_engine=retrieval_engine,
        context_builder=context_builder,
        graph=graph,
        graph_integrator=graph_integrator,
        vector_manager=vector_manager,
        memory_updater=memory_updater,
    )

    session_store = PostgreSQLSessionStore(db_manager)
    session_manager = SessionManager(store=session_store, event_store=event_store)
    collaboration_engine = CollaborationEngine()

    idempotency_coordinator = PostgreSQLIdempotencyCoordinator(db_manager)
    idempotency_manager = IdempotencyManager(coordinator=idempotency_coordinator)

    task_store = PostgreSQLTaskStore(db_manager)
    task_registry = TaskHandlerRegistry()
    register_default_handlers(task_registry)

    return BrainServiceContainer(
        config=app_cfg,
        db_manager=db_manager,
        auth_store=auth_store,
        key_manager=key_manager,
        token_service=token_service,
        audit_logger=audit_logger,
        auth_service=auth_service,
        rbac_service=rbac_service,
        event_store=event_store,
        interaction_store=interaction_store,
        memory_store=memory_store,
        memory_updater=memory_updater,
        vector_store=vector_store,
        vector_manager=vector_manager,
        graph=graph,
        retrieval_engine=retrieval_engine,
        context_builder=context_builder,
        brain_loop=brain_loop,
        session_manager=session_manager,
        collaboration_engine=collaboration_engine,
        idempotency_manager=idempotency_manager,
        task_store=task_store,
        task_registry=task_registry,
    )


def create_container_from_config(config: Optional[AppConfig] = None) -> BrainServiceContainer:
    """Factory creating the appropriate BrainServiceContainer based on AppConfig."""
    app_cfg = config or load_config_from_env()
    if app_cfg.db.is_postgres:
        db_manager = DatabaseConnectionManager(app_cfg.db)
        return create_postgres_container(
            db_manager=db_manager,
            config=app_cfg,
            secret_key=app_cfg.security.jwt_secret_key or "default-postgres-secret-key-32b-length!",
        )
    else:
        db_path = ":memory:"
        if app_cfg.db.database_url.startswith("sqlite:///"):
            db_path = app_cfg.db.database_url[len("sqlite:///"):]
            if not db_path:
                db_path = ":memory:"
        elif app_cfg.db.database_url == ":memory:":
            db_path = ":memory:"
        return create_sqlite_container(
            db_path=db_path,
            config=app_cfg,
            secret_key=app_cfg.security.jwt_secret_key or "default-sqlite-secret-key-32b-length!!",
        )


# =============================================================================
# FastAPI Dependency Callables
# =============================================================================
def get_container(request: Request) -> BrainServiceContainer:
    """Retrieve BrainServiceContainer from application state."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service container is uninitialized.",
        )
    return container


def get_client_ip(request: Request) -> str:
    """Extract client IP address for brute-force tracking and security audit."""
    return getattr(request.state, "client_ip", None) or (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "127.0.0.1")
    )


async def get_current_identity(
    request: Request,
    auth_header: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    container: BrainServiceContainer = Depends(get_container),
) -> AuthenticatedIdentity:
    """Authenticate caller identity via Bearer JWT token or Agent API key."""
    client_ip = get_client_ip(request)

    # 1. Check X-API-Key header first
    if x_api_key:
        try:
            identity = container.auth_service.authenticate_api_key(x_api_key, ip_address=client_ip)
            container.audit_logger.reset_failures(identity_id=identity.identity_id, ip_address=client_ip)
            return identity
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid API key: {e}",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    # 2. Check Authorization Bearer header
    if auth_header and auth_header.credentials:
        token = auth_header.credentials.strip()

        # Check if caller supplied an API key as a Bearer token
        if token.startswith(("agy_ak_", "agy_")):
            try:
                identity = container.auth_service.authenticate_api_key(token, ip_address=client_ip)
                container.audit_logger.reset_failures(identity_id=identity.identity_id, ip_address=client_ip)
                return identity
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid API key: {e}",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        # Authenticate JWT token
        try:
            identity = container.auth_service.authenticate_token(token, ip_address=client_ip)
            container.audit_logger.reset_failures(identity_id=identity.identity_id, ip_address=client_ip)
            return identity
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid or expired access token: {e}",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # 3. Missing authentication
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication credentials were not provided. Submit Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer, ApiKey"},
    )


def require_permission(permission: Permission) -> Callable[..., Any]:
    """Dependency factory checking that caller has permission in target project_id and enforcing rate limits."""

    async def _authorizer(
        project_id: str,
        request: Request,
        identity: AuthenticatedIdentity = Depends(get_current_identity),
        container: BrainServiceContainer = Depends(get_container),
    ) -> AuthenticatedIdentity:
        # Step 1: Authorize identity against project_id and required permission
        try:
            container.rbac_service.authorize(identity, project_id, permission)
        except (PermissionDeniedError, PrivilegeEscalationError) as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Authorization failure: {e}",
            )

        # Step 2: Rate limit enforcement keyed by validated identity and authorized project
        limiter: Optional[HttpRateLimiter] = getattr(request.app.state, "limiter", None)
        if limiter is not None:
            bucket_key = limiter.build_authenticated_key(identity, project_id)
            allowed, retry_after = limiter.check_bucket(bucket_key, is_authenticated=True)
            if not allowed:
                req_id = getattr(request.state, "request_id", None)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded for project '{project_id}'. Try again in {int(retry_after)} seconds.",
                    headers={"Retry-After": str(int(retry_after))},
                )

        return identity

    return _authorizer
