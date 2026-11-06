"""FastAPI Application Factory for AI Brain REST API (M5.5).

Configures:
- OpenAPI 3.1 specification & interactive documentation (Swagger UI, ReDoc)
- Domain routers: Interactions, Memories, Retrieval, Sessions, Workflows
- System probes: /healthz, /readyz
- Security middleware: Request size limiting (1MB max), Correlation IDs, Brute-force throttling
- Safe CORS policies and sanitized exception handlers
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from ai_brain.auth.exceptions import (
    AuthenticationError,
    CredentialExpiredError,
    CredentialNotFoundError,
    CredentialRevokedError,
    InvalidTokenError,
    MembershipNotFoundError,
    PermissionDeniedError,
    PrivilegeEscalationError,
    ProjectNotFoundError,
    TokenExpiredError,
    UserNotFoundError,
)
from ai_brain.config import AppConfig, load_config_from_env
from ai_brain.server.dependencies import (
    BrainServiceContainer,
    create_sqlite_container,
)
from ai_brain.server.middleware import (
    BruteForceThrottleMiddleware,
    CorrelationIdMiddleware,
    RequestSizeLimitMiddleware,
)
from ai_brain.server.routers import (
    interactions,
    memories,
    retrieval,
    sessions,
    workflows,
)
from ai_brain.server.schemas import (
    ErrorResponse,
    HealthResponse,
    ReadyResponse,
)

logger = logging.getLogger("ai_brain.server.app")


def create_app(
    container: Optional[BrainServiceContainer] = None,
    config: Optional[AppConfig] = None,
) -> FastAPI:
    """Create and configure the production FastAPI application."""
    app_cfg = config or load_config_from_env()

    app = FastAPI(
        title="AI Brain REST API",
        description=(
            "Asynchronous REST API and Server-Sent Events (SSE) streaming service for AI Brain.\n\n"
            "Provides project-isolated cognitive persistence, hybrid retrieval (pgvector + knowledge graph), "
            "interaction lifecycles, and multi-agent coordination."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 1. Initialize and attach service container
    resolved_container = container or create_sqlite_container(config=app_cfg)
    app.state.container = resolved_container

    # 2. Add Security & Tracing Middleware (Order: Last added runs first)
    # Brute-force throttle
    app.add_middleware(BruteForceThrottleMiddleware)

    # Request size limiting (Architecture limit: 1 MB maximum payload)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=1024 * 1024)

    # Correlation ID tracing
    app.add_middleware(CorrelationIdMiddleware)

    # Safe CORS configuration (Localhost default origins; no wildcards with credentials)
    allowed_origins = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # 3. Register Domain Routers
    app.include_router(interactions.router)
    app.include_router(memories.router)
    app.include_router(retrieval.router)
    app.include_router(sessions.router)
    app.include_router(workflows.router)

    # 4. System Health Endpoints
    @app.get(
        "/healthz",
        response_model=HealthResponse,
        tags=["System"],
        summary="Service Liveness Probe",
    )
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok", version="1.0.0")

    @app.get(
        "/readyz",
        response_model=ReadyResponse,
        tags=["System"],
        summary="Service Readiness Probe",
    )
    async def readyz() -> ReadyResponse:
        backend_str = (
            resolved_container.config.db.backend.value
            if hasattr(resolved_container.config.db.backend, "value")
            else str(resolved_container.config.db.backend)
        )
        return ReadyResponse(
            status="ready",
            backend=backend_str,
            database="connected",
        )

    # 5. Global Exception Handlers Mapping Domain Errors to Clean HTTP Responses
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": exc.detail,
                "request_id": req_id,
            },
            headers=exc.headers or {},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "VALIDATION_ERROR",
                "message": "Request payload validation failed.",
                "request_id": req_id,
                "details": {"validation_errors": exc.errors()},
            },
        )

    @app.exception_handler(AuthenticationError)
    @app.exception_handler(InvalidTokenError)
    @app.exception_handler(TokenExpiredError)
    @app.exception_handler(CredentialExpiredError)
    @app.exception_handler(CredentialRevokedError)
    async def auth_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "AUTHENTICATION_FAILED",
                "message": str(exc) or "Missing or invalid authentication credentials.",
                "request_id": req_id,
            },
            headers={"WWW-Authenticate": "Bearer, ApiKey"},
        )

    @app.exception_handler(PermissionDeniedError)
    @app.exception_handler(PrivilegeEscalationError)
    async def permission_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "PERMISSION_DENIED",
                "message": str(exc),
                "request_id": req_id,
            },
        )

    @app.exception_handler(CredentialNotFoundError)
    @app.exception_handler(UserNotFoundError)
    @app.exception_handler(MembershipNotFoundError)
    @app.exception_handler(ProjectNotFoundError)
    async def not_found_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "NOT_FOUND",
                "message": str(exc),
                "request_id": req_id,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        logger.error("Unhandled server exception (Request ID: %s): %s", req_id, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred. Please contact the administrator.",
                "request_id": req_id,
            },
        )

    # 6. Custom OpenAPI 3.1 Schema Customizer
    def custom_openapi() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

        schema["openapi"] = "3.1.0"

        # Define security schemes
        components = schema.setdefault("components", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes["HTTPBearer"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Access Token obtained via authentication service.",
        }
        security_schemes["ApiKeyAuth"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "High-entropy cryptographic Agent API Key (prefix: agy_ak_).",
        }

        # Apply security requirement globally to protected endpoints
        schema["security"] = [
            {"HTTPBearer": []},
            {"ApiKeyAuth": []},
        ]

        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    return app
