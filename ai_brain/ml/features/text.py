"""
Textual feature extractors for AI Brain ML components.
"""

import re
from typing import Any, Dict, Tuple

from ai_brain.ml.base import BaseFeatureExtractor


DECISION_MARKERS = [
    "decide", "decision", "chose", "selected", "adopted", "agreed", "resolved",
    "will use", "switched to", "migrated", "policy", "standard"
]

CONSTRAINT_MARKERS = [
    "must", "required", "mandatory", "constraint", "restricted", "limit",
    "never", "forbidden", "disallowed", "only"
]

PREFERENCE_MARKERS = [
    "prefer", "preference", "favor", "recommend", "recommended", "should", "better"
]

TEMPORARY_MARKERS = [
    "temporary", "temp", "wip", "work in progress", "draft", "for now",
    "placeholder", "todo", "fixme", "tentative"
]

NEGATION_MARKERS = [
    "not", "no", "never", "none", "neither", "nor", "cannot", "can't",
    "won't", "don't", "didn't", "is not", "false", "deprecated", "removed"
]

TECH_TERMS = [
    "postgres", "postgresql", "mysql", "sqlite", "redis", "mongodb", "docker",
    "kubernetes", "python", "typescript", "api", "oauth", "jwt", "grpc", "rest",
    "http", "database", "cache", "port", "cluster", "server", "endpoint"
]


class TextFeatureExtractor(BaseFeatureExtractor):
    """Extracts lexical, structural, and semantic markers from text."""

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return (
            "char_count",
            "word_count",
            "avg_word_length",
            "digit_count",
            "uppercase_ratio",
            "decision_score",
            "constraint_score",
            "preference_score",
            "temporary_score",
            "negation_count",
            "tech_term_count",
        )

    def extract(self, item: Any, **context: Any) -> Dict[str, float]:
        text = str(item) if not hasattr(item, "content") else str(item.content)
        lower = text.lower()
        words = re.findall(r"\b\w+\b", lower)
        w_count = max(1, len(words))

        decision_hits = sum(1 for m in DECISION_MARKERS if m in lower)
        constraint_hits = sum(1 for m in CONSTRAINT_MARKERS if m in lower)
        pref_hits = sum(1 for m in PREFERENCE_MARKERS if m in lower)
        temp_hits = sum(1 for m in TEMPORARY_MARKERS if m in lower)
        neg_hits = sum(1 for m in NEGATION_MARKERS if m in lower)
        tech_hits = sum(1 for m in TECH_TERMS if m in lower)

        char_len = len(text)
        digits = sum(1 for c in text if c.isdigit())
        uppers = sum(1 for c in text if c.isupper())

        return {
            "char_count": float(char_len),
            "word_count": float(w_count),
            "avg_word_length": round(char_len / w_count, 3),
            "digit_count": float(digits),
            "uppercase_ratio": round(uppers / max(1, char_len), 4),
            "decision_score": min(1.0, decision_hits * 0.35),
            "constraint_score": min(1.0, constraint_hits * 0.4),
            "preference_score": min(1.0, pref_hits * 0.35),
            "temporary_score": min(1.0, temp_hits * 0.5),
            "negation_count": float(neg_hits),
            "tech_term_count": float(tech_hits),
        }
