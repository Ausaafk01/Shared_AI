"""
Graph structure, vector similarity, and provenance feature extractors.
"""

from typing import Any, Dict, Tuple

from ai_brain.ml.base import BaseFeatureExtractor


class GraphVectorFeatureExtractor(BaseFeatureExtractor):
    """Extracts graph structural degree, vector distance, and provenance corroboration features."""

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return (
            "vector_score",
            "has_vector_match",
            "graph_score",
            "has_graph_match",
            "source_agent_count",
            "source_event_count",
            "relationship_count",
            "status_active",
            "status_disputed",
            "status_superseded",
        )

    def extract(self, item: Any, **context: Any) -> Dict[str, float]:
        v_score = context.get("vector_score")
        if v_score is None:
            v_score = getattr(item, "vector_score", 0.0) or 0.0

        g_score = context.get("graph_score")
        if g_score is None:
            g_score = getattr(item, "graph_score", 0.0) or 0.0

        agents = getattr(item, "source_agent_ids", ())
        events = getattr(item, "source_event_ids", ())
        rels = getattr(item, "relationships", {})

        rel_count = len(rels) if hasattr(rels, "__len__") else 0
        agent_count = len(agents) if hasattr(agents, "__len__") else 0
        event_count = len(events) if hasattr(events, "__len__") else 0

        status = str(getattr(item, "status", "ACTIVE")).upper()

        return {
            "vector_score": float(v_score),
            "has_vector_match": 1.0 if float(v_score) > 0.0 else 0.0,
            "graph_score": float(g_score),
            "has_graph_match": 1.0 if float(g_score) > 0.0 else 0.0,
            "source_agent_count": float(agent_count),
            "source_event_count": float(event_count),
            "relationship_count": float(rel_count),
            "status_active": 1.0 if status == "ACTIVE" else 0.0,
            "status_disputed": 1.0 if status == "DISPUTED" else 0.0,
            "status_superseded": 1.0 if status == "SUPERSEDED" else 0.0,
        }
