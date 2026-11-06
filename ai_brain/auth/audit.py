"""Structured security audit logging and suspicious activity detection for AI Brain.

Architectural Notice on Brute-Force Failure Tracking:
------------------------------------------------------
The sliding-window failure tracker within `SecurityAuditLogger` is process-local
and in-memory. It maintains recent failure timestamps in thread-safe memory and
resets on process restart.

Operational Scope in Distributed / Multi-Instance Deployments (M5.5+):
- Process-Local Defense: Provides low-latency, zero-dependency protection against rapid
  local brute-force probing without external infrastructure.
- Multi-Instance Scope: In horizontally scaled production deployments with multiple API
  workers, perimeter rate limiting (e.g., token bucket / leaky bucket) should be handled
  at the reverse proxy, load balancer, or API gateway layer (e.g. Nginx, Envoy, Cloudflare).
- No External Dependencies: Strictly avoids external cache/Redis dependencies to preserve
  lightweight offline and embedded execution.

Denial-of-Service (DoS) Mitigation:
- To prevent an attacker from intentionally locking out legitimate users across unrelated
  tenants by submitting malicious failures under a victim's user ID, tracking is keyed by
  composite tuples incorporating client IP address and tenant project scope.
- Blocking is applied to the attacking client/IP address rather than globally locking the
  targeted identity across all sources.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import uuid

from ai_brain.auth.models import SecurityAuditEvent

logger = logging.getLogger("ai_brain.security.audit")

# Sensitive keys that must be masked before recording
_SENSITIVE_KEYS = frozenset({
    "api_key",
    "raw_key",
    "password",
    "token",
    "access_token",
    "secret",
    "private_key",
    "authorization",
})


def sanitize_audit_details(data: Any) -> Any:
    """Recursively mask sensitive values to prevent credential leakage in audit logs."""
    if isinstance(data, dict):
        sanitized: Dict[str, Any] = {}
        for k, v in data.items():
            lower_k = str(k).lower().replace("-", "_")
            if any(s in lower_k for s in _SENSITIVE_KEYS):
                if isinstance(v, str) and v.startswith("agy_ak_"):
                    # Mask all but prefix
                    sanitized[k] = v[:14] + "..." + v[-4:] if len(v) > 18 else "agy_ak_***"
                else:
                    sanitized[k] = "***REDACTED***"
            else:
                sanitized[k] = sanitize_audit_details(v)
        return sanitized
    elif isinstance(data, (list, tuple)):
        return [sanitize_audit_details(item) for item in data]
    return data


class SecurityAuditLogger:
    """Thread-safe structured security audit logger with process-local brute-force anomaly detection."""

    def __init__(
        self,
        persistence_handler: Optional[Callable[[SecurityAuditEvent], None]] = None,
        max_retained_events: int = 1000,
        failure_threshold: int = 5,
        window_seconds: float = 300.0,
    ) -> None:
        self._persistence_handler = persistence_handler
        self._max_retained_events = max_retained_events
        self._failure_threshold = failure_threshold
        self._window_seconds = window_seconds

        self._lock = threading.Lock()
        self._events: List[SecurityAuditEvent] = []
        # Key: composite client tracker key -> list of failure timestamps
        self._failure_history: Dict[str, List[float]] = defaultdict(list)

    @property
    def events(self) -> List[SecurityAuditEvent]:
        """Snapshot of in-memory recorded audit events."""
        with self._lock:
            return list(self._events)

    def _build_tracker_key(
        self,
        identity_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> str:
        """Construct composite tracking key to prevent cross-tenant and cross-identity lockout."""
        parts = []
        if project_id:
            parts.append(f"p:{project_id}")
        if ip_address:
            parts.append(f"ip:{ip_address}")
        if identity_id:
            parts.append(f"id:{identity_id}")
        return "#".join(parts) if parts else "unknown"

    def record_event(
        self,
        event_type: str,
        *,
        identity_id: Optional[str] = None,
        identity_type: Optional[str] = None,
        project_id: Optional[str] = None,
        action: str = "",
        status: str = "SUCCESS",
        details: Optional[Mapping[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> SecurityAuditEvent:
        """Record and dispatch a sanitized structured security audit event."""
        sanitized = sanitize_audit_details(dict(details or {}))

        event = SecurityAuditEvent(
            event_id=f"audit_{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            identity_id=identity_id,
            identity_type=identity_type,
            project_id=project_id,
            action=action,
            status=status,
            details=sanitized,
            ip_address=ip_address,
            timestamp=datetime.now(timezone.utc),
        )

        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_retained_events:
                self._events.pop(0)

        # Emit structured log message
        log_payload = {
            "audit_event": event.event_type,
            "action": event.action,
            "status": event.status,
            "identity_id": event.identity_id,
            "project_id": event.project_id,
            "details": event.details,
            "timestamp": event.timestamp.isoformat(),
        }
        if status == "SUCCESS":
            logger.info("SECURITY_AUDIT: %s", json.dumps(log_payload))
        else:
            logger.warning("SECURITY_AUDIT: %s", json.dumps(log_payload))

        # Check for repeated failures / brute force
        if status == "FAILURE" and (identity_id or ip_address):
            tracker_key = self._build_tracker_key(identity_id, ip_address, project_id)
            self._check_repeated_failures(tracker_key, identity_id, project_id, ip_address)

        # Call persistent storage handler if configured
        if self._persistence_handler:
            try:
                self._persistence_handler(event)
            except Exception as e:
                logger.error("Failed to persist security audit event: %s", e)

        return event

    def _check_repeated_failures(
        self,
        tracker_key: str,
        identity_id: Optional[str],
        project_id: Optional[str],
        ip_address: Optional[str],
    ) -> None:
        """Detect repeated authentication failures exceeding threshold within time window."""
        now = time.time()
        with self._lock:
            failures = self._failure_history[tracker_key]
            cutoff = now - self._window_seconds
            pruned = [t for t in failures if t >= cutoff]
            pruned.append(now)
            self._failure_history[tracker_key] = pruned

            if len(pruned) >= self._failure_threshold:
                # Trigger suspicious repeated failure alert event
                alert_event = SecurityAuditEvent(
                    event_id=f"audit_alert_{uuid.uuid4().hex[:12]}",
                    event_type="BRUTE_FORCE_SUSPECTED",
                    identity_id=identity_id,
                    identity_type="UNKNOWN",
                    project_id=project_id,
                    action="AUTHENTICATE",
                    status="ALERT",
                    details={
                        "failure_count": len(pruned),
                        "window_seconds": self._window_seconds,
                        "tracker_key": tracker_key,
                        "ip_address": ip_address,
                    },
                    ip_address=ip_address,
                    timestamp=datetime.now(timezone.utc),
                )
                self._events.append(alert_event)
                logger.critical(
                    "SECURITY_ALERT: Brute force suspected for '%s' (%d failures in %.0fs)",
                    tracker_key,
                    len(pruned),
                    self._window_seconds,
                )
                if self._persistence_handler:
                    try:
                        self._persistence_handler(alert_event)
                    except Exception:
                        pass

    # -------------------------------------------------------------------------
    # Interfaces for M5.5 HTTP Rate Limiting & Defense Hooks
    # -------------------------------------------------------------------------
    def is_client_blocked(
        self,
        identity_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> bool:
        """Check whether client or IP currently exceeds the failure threshold."""
        now = time.time()
        cutoff = now - self._window_seconds
        tracker_key = self._build_tracker_key(identity_id, ip_address, project_id)

        with self._lock:
            # Check specific composite key
            failures = [t for t in self._failure_history.get(tracker_key, []) if t >= cutoff]
            if len(failures) >= self._failure_threshold:
                return True

            # If IP is provided, also check IP-level threshold to stop distributed identity attacks
            if ip_address:
                ip_key = self._build_tracker_key(ip_address=ip_address, project_id=project_id)
                ip_failures = [t for t in self._failure_history.get(ip_key, []) if t >= cutoff]
                if len(ip_failures) >= self._failure_threshold:
                    return True

        return False

    def reset_failures(
        self,
        identity_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> None:
        """Clear recorded failure history upon successful authentication."""
        tracker_key = self._build_tracker_key(identity_id, ip_address, project_id)
        with self._lock:
            if tracker_key in self._failure_history:
                del self._failure_history[tracker_key]

    def get_failure_count(
        self,
        identity_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> int:
        """Return the active failure count within the current sliding window."""
        now = time.time()
        cutoff = now - self._window_seconds
        tracker_key = self._build_tracker_key(identity_id, ip_address, project_id)
        with self._lock:
            return len([t for t in self._failure_history.get(tracker_key, []) if t >= cutoff])
