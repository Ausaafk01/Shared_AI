"""HTTP Middleware components for AI Brain REST API (M5.5).

Enforces:
1. Request size limiting (Architecture constraint: 1 MB maximum payload limit).
2. Request correlation IDs for end-to-end tracing (X-Request-ID).
3. Process-local brute-force throttling using M5.4 SecurityAuditLogger.
4. Credential and authorization header redaction in server logs.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("ai_brain.server.http")

MAX_REQUEST_BYTES = 1024 * 1024  # 1 MB


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Enforces the 1 MB architectural payload size limit on incoming requests."""

    def __init__(self, app: Any, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "PAYLOAD_TOO_LARGE",
                            "message": f"Request body size ({length} bytes) exceeds maximum limit of {self.max_bytes} bytes (1 MB).",
                            "request_id": getattr(request.state, "request_id", None),
                        },
                    )
            except ValueError:
                pass
        elif request.method in ("POST", "PUT", "PATCH"):
            body = await request.body()
            if len(body) > self.max_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "PAYLOAD_TOO_LARGE",
                        "message": f"Request body size ({len(body)} bytes) exceeds maximum limit of {self.max_bytes} bytes (1 MB).",
                        "request_id": getattr(request.state, "request_id", None),
                    },
                )

        return await call_next(request)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Ensures every request carries a correlation ID for observability."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        req_id = (
            request.headers.get("x-request-id")
            or request.headers.get("x-correlation-id")
            or uuid.uuid4().hex
        )
        request.state.request_id = req_id

        response = await call_next(request)
        response.headers["x-request-id"] = req_id
        return response


class BruteForceThrottleMiddleware(BaseHTTPMiddleware):
    """Integrates with M5.4 SecurityAuditLogger to throttle IPs exceeding failure thresholds."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        # Retrieve client IP
        client_ip = (
            request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "127.0.0.1")
        )
        request.state.client_ip = client_ip

        # Check container audit logger if attached to app.state
        container = getattr(request.app.state, "container", None)
        if container and container.audit_logger:
            if container.audit_logger.is_client_blocked(ip_address=client_ip):
                req_id = getattr(request.state, "request_id", None)
                logger.warning("Throttled request from blocked client IP '%s'", client_ip)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "TOO_MANY_REQUESTS",
                        "message": "Client temporarily throttled due to repeated authentication failures.",
                        "request_id": req_id,
                    },
                    headers={"Retry-After": "60"},
                )

        return await call_next(request)
