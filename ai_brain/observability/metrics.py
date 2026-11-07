"""Zero-dependency Prometheus metrics collector and exposition formatter (M5.7).

Provides standard metric types:
- Counter (monotonically increasing)
- Gauge (arbitrary instantaneous value)
- Histogram (latency / duration distributions with standard le buckets)

Exposes metrics strictly conforming to Prometheus Text-Based Exposition Format (v0.0.4).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


def _escape_label_value(val: Any) -> str:
    """Escape label value for Prometheus format."""
    s = str(val)
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_labels(labels: Dict[str, Any]) -> str:
    """Format label dictionary into `{k="v", ...}` string."""
    if not labels:
        return ""
    sorted_items = sorted(labels.items(), key=lambda x: x[0])
    parts = [f'{k}="{_escape_label_value(v)}"' for k, v in sorted_items]
    return "{" + ",".join(parts) + "}"


class Metric:
    """Base class for Prometheus metrics."""

    def __init__(self, name: str, help_text: str, metric_type: str) -> None:
        self.name = name
        self.help_text = help_text
        self.metric_type = metric_type
        self._lock = threading.Lock()

    def collect_exposition(self) -> List[str]:
        raise NotImplementedError


class Counter(Metric):
    """Monotonically increasing cumulative metric."""

    def __init__(self, name: str, help_text: str) -> None:
        super().__init__(name, help_text, "counter")
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        if amount < 0:
            raise ValueError(f"Counter increment must be non-negative: {amount}")
        key = tuple(sorted((k, str(v)) for k, v in labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def get(self, **labels: Any) -> float:
        key = tuple(sorted((k, str(v)) for k, v in labels.items()))
        with self._lock:
            return self._values.get(key, 0.0)

    def collect_exposition(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            if not self._values:
                lines.append(f"{self.name} 0")
            else:
                for label_tuples, val in sorted(self._values.items()):
                    lbl_dict = dict(label_tuples)
                    lbl_str = _format_labels(lbl_dict)
                    val_str = f"{val:.0f}" if val.is_integer() else f"{val}"
                    lines.append(f"{self.name}{lbl_str} {val_str}")
        return lines


class Gauge(Metric):
    """Metric representing a single numerical value that can arbitrarily go up or down."""

    def __init__(self, name: str, help_text: str) -> None:
        super().__init__(name, help_text, "gauge")
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}

    def set(self, value: float, **labels: Any) -> None:
        key = tuple(sorted((k, str(v)) for k, v in labels.items()))
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        key = tuple(sorted((k, str(v)) for k, v in labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: Any) -> None:
        self.inc(-amount, **labels)

    def get(self, **labels: Any) -> float:
        key = tuple(sorted((k, str(v)) for k, v in labels.items()))
        with self._lock:
            return self._values.get(key, 0.0)

    def collect_exposition(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            if not self._values:
                lines.append(f"{self.name} 0")
            else:
                for label_tuples, val in sorted(self._values.items()):
                    lbl_dict = dict(label_tuples)
                    lbl_str = _format_labels(lbl_dict)
                    val_str = f"{val:.0f}" if val.is_integer() else f"{val}"
                    lines.append(f"{self.name}{lbl_str} {val_str}")
        return lines


class Histogram(Metric):
    """Samples observations and counts them in configurable buckets."""

    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0,
    )

    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: Tuple[float, ...] = DEFAULT_BUCKETS,
    ) -> None:
        super().__init__(name, help_text, "histogram")
        self.buckets = sorted(set(buckets))
        self._data: Dict[Tuple[Tuple[str, str], ...], Dict[str, Any]] = {}

    def observe(self, value: float, **labels: Any) -> None:
        key = tuple(sorted((k, str(v)) for k, v in labels.items()))
        with self._lock:
            if key not in self._data:
                self._data[key] = {
                    "count": 0,
                    "sum": 0.0,
                    "buckets": {b: 0 for b in self.buckets},
                }
            entry = self._data[key]
            entry["count"] += 1
            entry["sum"] += float(value)
            for b in self.buckets:
                if value <= b:
                    entry["buckets"][b] += 1

    def collect_exposition(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            if not self._data:
                lines.append(f"{self.name}_count 0")
                lines.append(f"{self.name}_sum 0.0")
            else:
                for label_tuples, entry in sorted(self._data.items()):
                    base_labels = dict(label_tuples)
                    # Print cumulative buckets
                    for b in self.buckets:
                        bucket_labels = dict(base_labels)
                        bucket_labels["le"] = str(b)
                        lbl_str = _format_labels(bucket_labels)
                        lines.append(f"{self.name}_bucket{lbl_str} {entry['buckets'][b]}")
                    # +Inf bucket
                    inf_labels = dict(base_labels)
                    inf_labels["le"] = "+Inf"
                    lines.append(f"{self.name}_bucket{_format_labels(inf_labels)} {entry['count']}")

                    # sum and count
                    base_lbl_str = _format_labels(base_labels)
                    lines.append(f"{self.name}_sum{base_lbl_str} {entry['sum']:.6f}")
                    lines.append(f"{self.name}_count{base_lbl_str} {entry['count']}")
        return lines


class MetricsRegistry:
    """Central registry of application and system metrics."""

    def __init__(self) -> None:
        self._metrics: Dict[str, Metric] = {}
        self._callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()

        # Initialize core application metrics
        self.http_requests_total = self.register_counter(
            "ai_brain_http_requests_total",
            "Total number of HTTP requests processed by endpoint and status",
        )
        self.http_request_duration_seconds = self.register_histogram(
            "ai_brain_http_request_duration_seconds",
            "HTTP request duration in seconds",
        )
        self.http_errors_total = self.register_counter(
            "ai_brain_http_errors_total",
            "Total number of HTTP errors handled by endpoint and error class",
        )

        # Background Task metrics
        self.task_queue_depth = self.register_gauge(
            "ai_brain_task_queue_depth",
            "Current depth of background tasks by status (PENDING, PROCESSING)",
        )
        self.tasks_total = self.register_counter(
            "ai_brain_tasks_total",
            "Total number of background tasks submitted or transitioned",
        )
        self.task_retries_total = self.register_counter(
            "ai_brain_task_retries_total",
            "Total background task execution retries",
        )
        self.task_failures_total = self.register_counter(
            "ai_brain_task_failures_total",
            "Total background task terminal failures",
        )
        self.task_dead_letter_total = self.register_gauge(
            "ai_brain_task_dead_letter_total",
            "Total tasks moved to dead letter queue",
        )

        # Worker Fleet metrics
        self.worker_active_count = self.register_gauge(
            "ai_brain_worker_active_count",
            "Current number of actively running background worker threads",
        )
        self.worker_state = self.register_gauge(
            "ai_brain_worker_state",
            "Status indicator for background workers (1=active, 0=stopped)",
        )

        # System / DB metrics
        self.database_connected = self.register_gauge(
            "ai_brain_database_connected",
            "Database connectivity status (1=connected, 0=disconnected)",
        )
        self.app_ready = self.register_gauge(
            "ai_brain_ready",
            "Application readiness status (1=ready, 0=unready)",
        )

    def register_counter(self, name: str, help_text: str) -> Counter:
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]  # type: ignore[return-value]
            c = Counter(name, help_text)
            self._metrics[name] = c
            return c

    def register_gauge(self, name: str, help_text: str) -> Gauge:
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]  # type: ignore[return-value]
            g = Gauge(name, help_text)
            self._metrics[name] = g
            return g

    def register_histogram(
        self,
        name: str,
        help_text: str,
        buckets: Tuple[float, ...] = Histogram.DEFAULT_BUCKETS,
    ) -> Histogram:
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]  # type: ignore[return-value]
            h = Histogram(name, help_text, buckets=buckets)
            self._metrics[name] = h
            return h

    def add_callback(self, cb: Callable[[], None]) -> None:
        """Register a callback to be invoked right before exposition generation."""
        with self._lock:
            self._callbacks.append(cb)

    def generate_exposition(self) -> str:
        """Collect all metrics and format as Prometheus exposition text."""
        # Execute callbacks (e.g. queue count updates)
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb()
            except Exception:
                pass

        output_lines: List[str] = []
        with self._lock:
            metrics_list = list(self._metrics.values())

        for m in metrics_list:
            output_lines.extend(m.collect_exposition())

        return "\n".join(output_lines) + "\n"

    def clear(self) -> None:
        """Reset metric values (useful for tests)."""
        with self._lock:
            self._metrics.clear()
            self._callbacks.clear()


# Global Singleton Registry
_GLOBAL_REGISTRY: Optional[MetricsRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_metrics_registry() -> MetricsRegistry:
    """Retrieve global singleton MetricsRegistry."""
    global _GLOBAL_REGISTRY
    with _REGISTRY_LOCK:
        if _GLOBAL_REGISTRY is None:
            _GLOBAL_REGISTRY = MetricsRegistry()
        return _GLOBAL_REGISTRY
