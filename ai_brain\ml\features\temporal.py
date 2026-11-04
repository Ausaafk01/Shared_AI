"""
Temporal and lifecycle feature extractors for AI Brain ML components.
"""

from datetime import datetime, timezone
import math
from typing import Any, Dict, Optional, Tuple

from ai_brain.ml.base import BaseFeatureExtractor


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """Extracts age, recency decay, update frequency, and staleness signals."""

    def __init__(self, half_life_days: float = 30.0) -> None:
        self.half_life_days = max(1.0, half_life_days)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return (
            "age_days",
            "recency_decay",
            "days_since_update",
            "version_count",
            "is_versioned",
            "has_temporal_keywords",
        )

    def extract(self, item: Any, **context: Any) -> Dict[str, float]:
        ref_time = context.get("reference_time") or datetime.now(timezone.utc)
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        # Extract created_at
        created_at = getattr(item, "created_at", None)
        if isinstance(created_at, datetime):
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age_sec = max(0.0, (ref_time - created_at).total_seconds())
            age_days = age_sec / 86400.0
        else:
            age_days = 0.0

        # Extract updated_at
        updated_at = getattr(item, "updated_at", None)
        if isinstance(updated_at, datetime):
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            upd_sec = max(0.0, (ref_time - updated_at).total_seconds())
            upd_days = upd_sec / 86400.0
        else:
            upd_days = age_days

        # Recency decay (half-life exponential)
        decay = math.exp(-age_days / self.half_life_days)

        version = float(getattr(item, "version", 1))
        content = str(getattr(item, "content", item)).lower()
        temporal_kws = ["temporary", "deprecated", "outdated", "until", "migrate", "now uses", "obsolete", "current"]
        has_temporal = 1.0 if any(k in content for k in temporal_kws) else 0.0

        return {
            "age_days": round(age_days, 3),
            "recency_decay": round(decay, 4),
            "days_since_update": round(upd_days, 3),
            "version_count": version,
            "is_versioned": 1.0 if version > 1.0 else 0.0,
            "has_temporal_keywords": has_temporal,
        }
