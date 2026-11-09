"""Structured JSON Logging Formatter and Setup (M5.7).

Formats application log records as JSON lines, ensuring:
- Standardized ISO-8601 UTC timestamps
- Log level, logger name, and process/thread context
- Request correlation ID extraction
- Automatic credential and sensitive secret redaction
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict

# Secret patterns to redact
_SENSITIVE_PATTERNS = [
    (re.compile(r"(agy_ak_[a-zA-Z0-9_-]{8})[a-zA-Z0-9_-]+"), r"\1...[REDACTED]"),
    (re.compile(r"(Bearer\s+)[a-zA-Z0-9._-]+", re.IGNORECASE), r"\1[REDACTED_TOKEN]"),
    (re.compile(r"(password[:=]\s*)[^\s,]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(postgres(?:ql)?://[^:]+:)([^@]+)(@)", re.IGNORECASE), r"\1***\3"),
]


def redact_sensitive_text(text: str) -> str:
    """Mask credentials, tokens, and database passwords from text."""
    if not text:
        return text
    result = text
    for pattern, replacement in _SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        now = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        raw_message = record.getMessage()
        sanitized_message = redact_sensitive_text(raw_message)

        log_entry: Dict[str, Any] = {
            "timestamp": now,
            "level": record.levelname,
            "logger": record.name,
            "message": sanitized_message,
        }

        # Request / correlation tracing
        req_id = getattr(record, "request_id", None)
        if req_id:
            log_entry["request_id"] = req_id

        # Tenant / identity context if present
        identity_id = getattr(record, "identity_id", None)
        if identity_id:
            log_entry["identity_id"] = identity_id

        project_id = getattr(record, "project_id", None)
        if project_id:
            log_entry["project_id"] = project_id

        # Exception information
        if record.exc_info:
            log_entry["exception"] = redact_sensitive_text(self.formatException(record.exc_info))

        return json.dumps(log_entry, ensure_ascii=False)


def setup_structured_logging(
    level: int = logging.INFO,
    use_json: bool = True,
) -> None:
    """Configure root logger with structured JSON formatting or standard console output."""
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if use_json:
        handler.setFormatter(StructuredJsonFormatter())
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s (RequestID: %(request_id)s): %(message)s"
        handler.setFormatter(logging.Formatter(fmt))

    root.addHandler(handler)
