"""
In-process bounded HTTP Rate Limiter for AI Brain REST API (M5.5).

Architecture & Deployment Note:
--------------------------------
This rate limiter operates in-process using an in-memory token-bucket algorithm with
bounded state and TTL expiration. In single-worker or development deployments, it provides
project-, identity-, and IP-level traffic bounding without external dependencies.

MULTI-WORKER DEPLOYMENT LIMITATION:
In multi-worker production deployments (e.g. Uvicorn with multiple workers or Kubernetes pods),
in-process memory is isolated per worker process. Global rate limit coordination across
workers requires shared enforcement at a trusted reverse proxy (e.g. Nginx, Envoy, Cloudflare)
or an API gateway. Do NOT introduce Redis into AI Brain core for rate limiting.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple

import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("ai_brain.server.limiter")


class TokenBucket:
    """Thread-safe token bucket rate limit tracker."""

    def __init__(self, capacity: float, refill_rate: float) -> None:
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)  # Tokens added per second
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self.last_accessed = time.time()

    def consume(self, tokens: float = 1.0) -> Tuple[bool, float]:
        """Attempt to consume tokens.
        
        Returns:
            (allowed: bool, retry_after_seconds: float)
        """
        now_mono = time.monotonic()
        now_wall = time.time()
        self.last_accessed = now_wall

        # Refill tokens based on elapsed monotonic time
        elapsed = max(0.0, now_mono - self.last_refill)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now_mono

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0.0

        # Calculate time needed to accumulate the required tokens
        deficit = tokens - self.tokens
        retry_after = deficit / self.refill_rate if self.refill_rate > 0 else 60.0
        return False, max(1.0, math.ceil(retry_after))


class HttpRateLimiter:
    """
    Configurable, bounded HTTP rate limiter tracking authenticated identities and client IPs.
    
    Guarantees:
    - Independent token buckets per project/identity or unauthenticated client IP.
    - Strict boundary on memory usage via periodic TTL pruning and LRU capacity capping.
    - Safe client IP resolution: never trusts X-Forwarded-For unless client is in trusted_proxies.
    - Zero external caching dependencies (Redis-free).
    """

    def __init__(
        self,
        *,
        authenticated_rpm: int = 300,       # 300 requests / min for authenticated clients
        unauthenticated_rpm: int = 60,      # 60 requests / min for unauthenticated requests
        burst_multiplier: float = 1.2,      # Burst allowance above sustained rate
        max_buckets: int = 10000,           # Maximum active buckets in memory
        entry_ttl_seconds: float = 300.0,   # Inactivity purge threshold (5 minutes)
        trusted_proxies: Optional[Set[str]] = None,
    ) -> None:
        self.authenticated_rpm = authenticated_rpm
        self.unauthenticated_rpm = unauthenticated_rpm
        self.burst_multiplier = burst_multiplier
        self.max_buckets = max_buckets
        self.entry_ttl_seconds = entry_ttl_seconds
        self.trusted_proxies = set(trusted_proxies) if trusted_proxies is not None else set()

        self._lock = threading.Lock()
        self._buckets: Dict[str, TokenBucket] = {}
        self._check_counter = 0

    def resolve_client_ip(self, request: Request) -> str:
        """Resolve client IP safely, only trusting forwarded headers from trusted proxies."""
        direct_ip = request.client.host if request.client else "127.0.0.1"

        if direct_ip in self.trusted_proxies:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                client_part = forwarded.split(",")[0].strip()
                if client_part:
                    return client_part

        return direct_ip

    def resolve_bucket_key(self, request: Request) -> Tuple[str, bool]:
        """Extract rate limiting key and determine whether the request carries authentication.
        
        Returns:
            (bucket_key: str, is_authenticated: bool)
        """
        # 1. Check for API key header
        api_key = request.headers.get("x-api-key")
        if api_key and api_key.strip():
            key_hash = hashlib.sha256(api_key.strip().encode()).hexdigest()[:16]
            return f"auth:apikey:{key_hash}", True

        # 2. Check for Bearer token
        auth_header = request.headers.get("authorization", "").strip()
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if token.startswith(("agy_ak_", "agy_")):
                key_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
                return f"auth:apikey:{key_hash}", True
            elif token:
                # Peek JWT subject and path project without validating crypto (crypto is validated in route dependency)
                try:
                    unverified = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
                    sub = unverified.get("sub", "anon")
                    # Extract project from URL if present
                    path_match = re.search(r"/projects/([^/]+)", request.url.path)
                    project_id = path_match.group(1) if path_match else "global"
                    return f"auth:jwt:{project_id}:{sub}", True
                except Exception:
                    token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
                    return f"auth:jwt_raw:{token_hash}", True

        # 3. Unauthenticated request: key by client IP
        client_ip = self.resolve_client_ip(request)
        return f"unauth:ip:{client_ip}", False

    def build_authenticated_key(
        self,
        identity: Any,
        authorized_project_id: str,
    ) -> str:
        """
        Construct rate-limit bucket key incorporating the validated identity and authorized project ID.
        
        Guarantees:
        - Never uses an unvalidated project ID from unauthenticated headers or paths.
        - Keys by identity_type, authorized_project_id, and identity_id.
        - Separate authorized projects under the same identity receive isolated buckets.
        """
        id_type = (
            identity.identity_type.value
            if hasattr(identity.identity_type, "value")
            else str(identity.identity_type)
        )
        if id_type.upper() == "AGENT":
            return f"auth:apikey:{authorized_project_id}:{identity.identity_id}"
        return f"auth:jwt:{authorized_project_id}:{identity.identity_id}"

    def check_bucket(
        self,
        bucket_key: str,
        is_authenticated: bool = True,
        custom_rpm: Optional[int] = None,
    ) -> Tuple[bool, float]:
        """Check and consume rate limit directly using a pre-resolved bucket key."""
        rpm = (
            custom_rpm
            if custom_rpm is not None
            else (self.authenticated_rpm if is_authenticated else self.unauthenticated_rpm)
        )
        refill_rate = max(0.001, rpm / 60.0)
        capacity = max(1.0, float(rpm) * self.burst_multiplier)

        with self._lock:
            self._check_counter += 1
            if self._check_counter % 100 == 0 or len(self._buckets) >= self.max_buckets:
                self._prune_expired()

            bucket = self._buckets.get(bucket_key)
            if bucket is None:
                if len(self._buckets) >= self.max_buckets:
                    self._evict_oldest()
                bucket = TokenBucket(capacity=capacity, refill_rate=refill_rate)
                self._buckets[bucket_key] = bucket

            return bucket.consume(1.0)

    def check(
        self,
        request: Request,
        custom_key: Optional[str] = None,
        custom_rpm: Optional[int] = None,
    ) -> Tuple[bool, float]:
        """Check rate limit for an incoming request.
        
        Returns:
            (allowed: bool, retry_after_seconds: float)
        """
        if custom_key:
            return self.check_bucket(custom_key, is_authenticated=True, custom_rpm=custom_rpm)
        
        key, is_auth = self.resolve_bucket_key(request)
        return self.check_bucket(key, is_authenticated=is_auth, custom_rpm=custom_rpm)

    def _prune_expired(self) -> None:
        """Purge buckets that have been inactive longer than entry_ttl_seconds."""
        now = time.time()
        expired_keys = [
            k for k, b in self._buckets.items()
            if (now - b.last_accessed) > self.entry_ttl_seconds
        ]
        for k in expired_keys:
            del self._buckets[k]

    def _evict_oldest(self) -> None:
        """Evict the least recently accessed bucket to enforce max_buckets upper bound."""
        if not self._buckets:
            return
        oldest_key = min(self._buckets.keys(), key=lambda k: self._buckets[k].last_accessed)
        del self._buckets[oldest_key]

    def reset(self) -> None:
        """Clear all bucket state (testing and diagnostic aid)."""
        with self._lock:
            self._buckets.clear()
            self._check_counter = 0

    def active_bucket_count(self) -> int:
        """Return total active tracked buckets in memory."""
        with self._lock:
            return len(self._buckets)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware enforcing general HTTP rate limits on requests."""

    def __init__(
        self,
        app: Any,
        limiter: Optional[HttpRateLimiter] = None,
    ) -> None:
        super().__init__(app)
        self.limiter = limiter or HttpRateLimiter()

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        # Exclude system probes and interactive documentation from rate limits
        if request.url.path in ("/healthz", "/readyz", "/openapi.json", "/docs", "/redoc"):
            return await call_next(request)

        # If request carries credentials, authenticated rate limiting is enforced in require_permission
        # after identity validation and project authorization.
        # Unauthenticated requests are checked against the IP bucket at this middleware layer.
        has_auth_header = bool(
            request.headers.get("x-api-key")
            or request.headers.get("authorization", "").strip().startswith(("Bearer ", "ApiKey "))
        )
        if not has_auth_header:
            client_ip = self.limiter.resolve_client_ip(request)
            allowed, retry_after = self.limiter.check_bucket(f"unauth:ip:{client_ip}", is_authenticated=False)
            if not allowed:
                req_id = getattr(request.state, "request_id", None)
                logger.warning(
                    "Unauthenticated rate limit exceeded on %s for IP %s (Retry-After: %ds)",
                    request.url.path,
                    client_ip,
                    retry_after,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "TOO_MANY_REQUESTS",
                        "message": f"Rate limit exceeded. Try again in {int(retry_after)} seconds.",
                        "request_id": req_id,
                    },
                    headers={
                        "Retry-After": str(int(retry_after)),
                    },
                )

        return await call_next(request)
