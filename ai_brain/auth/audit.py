"""Structured security audit logging and suspicious activity detection for AI Brain."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
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
    """Thread-safe structured security audit logger with brute-force anomaly detection."""

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
        # Key: client_identifier (ip or identity_id) -> list of failure timestamps
        self._failure_history: Dict[str, List[float]] = defaultdict(list)

    @property
    def events(self) -> List[SecurityAuditEvent]:
        """Snapshot of in-memory recorded audit events."""
        with self._lock:
            return list(self._events)

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
            client_key = identity_id or ip_address or "unknown"
            self._check_repeated_failures(client_key, project_id, ip_address)

        # Call persistent storage handler if configured
        if self._persistence_handler:
            try:
                self._persistence_handler(event)
            except Exception as e:
                logger.error("Failed to persist security audit event: %s", e)

        return event

    def _check_repeated_failures(
        self,
        client_key: str,
        project_id: Optional[str],
        ip_address: Optional[str],
    ) -> None:
        """Detect repeated authentication failures exceeding threshold within time window."""
        now = time.time()
        with self._lock:
            failures = self._failure_history[client_key]
            # Prune timestamps outside window
            cutoff = now - self._window_seconds
            pruned = [t for t in failures if t >= cutoff]
            pruned.append(now)
            self._failure_history[client_key] = pruned

            if len(pruned) >= self._failure_threshold:
                # Trigger suspicious repeated failure alert event
                alert_event = SecurityAuditEvent(
                    event_id=f"audit_alert_{uuid.uuid4().hex[:12]}",
                    event_type="BRUTE_FORCE_SUSPECTED",
                    identity_id=client_key if client_key != ip_address else None,
                    identity_type="UNKNOWN",
                    project_id=project_id,
                    action="AUTHENTICATE",
                    status="ALERT",
                    details={
                        "failure_count": len(pruned),
                        "window_seconds": self._window_seconds,
                        "client_identifier": client_key,
                    },
                    ip_address=ip_address,
                    timestamp=datetime.now(timezone.utc),
                )
                self._events.append(alert_event)
                logger.critical(
                    "SECURITY_ALERT: Brute force suspected for '%s' (%d failures in %.0fs)",
                    client_key,
                    len(pruned),
                    self._window_seconds,
                )
                if self._persistence_handler:
                    try:
                        self._persistence_handler(alert_event)
                    except Exception:
                        pass

