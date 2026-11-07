"""Observability subsystem for AI Brain (M5.7).

Provides Prometheus metrics collection, standard exposition format,
structured JSON logging, and correlation ID tracing.
"""

from __future__ import annotations

from ai_brain.observability.logging import StructuredJsonFormatter, setup_structured_logging
from ai_brain.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    get_metrics_registry,
)

__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "get_metrics_registry",
    "StructuredJsonFormatter",
    "setup_structured_logging",
]
