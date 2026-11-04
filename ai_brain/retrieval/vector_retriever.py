"""
Vector retrieval integration utilizing the existing Vector Memory subsystem.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import List, Optional, Sequence, Union

from ai_brain.retrieval.exceptions import ProjectIsolationError, VectorRetrievalError
from ai_brain.retrieval.models import ProcessedQuery, QueryFilters
from ai_brain.vectors.exceptions import EmbeddingProviderError, ProjectIsolationError as VecProjectIsolationError
from ai_brain.vectors.index import VectorIndex
from ai_brain.vectors.models import VectorMatch, VectorRecord, VectorSearchResult

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Retrieves semantically similar memories via VectorIndex.
    
    Guarantees:
    - Reuses existing VectorIndex and EmbeddingProvider without duplicating search logic.
    - Strictly enforces project isolation boundaries.
    - Applies status, memory type, agent, and temporal filters.
    - Preserves raw cosine similarity scores.
    - Handles empty index results and provider failures cleanly.
    """

    def __init__(self, index: VectorIndex) -> None:
        self.index = index

    def retrieve(
        self,
        query: ProcessedQuery,
        *,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
    ) -> List[VectorMatch]:
        """
        Execute semantic vector retrieval for the given processed query.
        
        Args:
            query: ProcessedQuery defining target project_id, query text, and filters.
            top_k: Maximum number of matches to retrieve.
            similarity_threshold: Minimum cosine similarity threshold.
            
        Returns:
            List of filtered VectorMatch instances sorted by score descending.
        """
        project_id = query.project_id
        if not project_id:
            raise ProjectIsolationError("project_id must be provided for vector retrieval")

        # 1. Resolve allowed statuses from query filters
        allowed_statuses = (
            query.filters.allowed_statuses
            if query.filters and query.filters.allowed_statuses
            else ("ACTIVE",)
        )

        # 2. Query the vector index
        try:
            # Over-fetch slightly to allow post-filtering on memory type, agent, and time
            fetch_k = top_k * 3 if (query.filters and (query.filters.memory_types or query.filters.source_agent_ids or query.filters.time_range_start)) else top_k
            
            search_result: VectorSearchResult = self.index.search(
                query.original_query,
                project_id=project_id,
                top_k=fetch_k,
                similarity_threshold=similarity_threshold,
                allowed_statuses=allowed_statuses,
            )
        except VecProjectIsolationError as e:
            raise ProjectIsolationError(str(e)) from e
        except EmbeddingProviderError as e:
            raise VectorRetrievalError(f"Embedding provider failure during retrieval: {e}") from e
        except Exception as e:
            raise VectorRetrievalError(f"Vector search failed: {e}") from e

        if not search_result.matches:
            return []

        # 3. Apply post-filtering for criteria not built into VectorIndex
        filtered_matches: List[VectorMatch] = []
        for match in search_result.matches:
            if self._satisfies_filters(match, query.filters):
                filtered_matches.append(match)

        # Sort descending by score, then memory_id ascending for determinism
        filtered_matches.sort(key=lambda m: (-m.score, m.memory_id))
        return filtered_matches[:top_k]

    def _satisfies_filters(self, match: VectorMatch, filters: Optional[QueryFilters]) -> bool:
        """Check if match satisfies status, memory_type, agent, and temporal filters."""
        if not filters:
            return True

        # Status filter
        if filters.allowed_statuses:
            if match.status.upper() not in set(filters.allowed_statuses):
                return False

        metadata = match.metadata or {}

        # Memory type filter
        if filters.memory_types:
            mem_type = (metadata.get("memory_type") or "").upper()
            if mem_type not in set(filters.memory_types):
                return False

        # Agent filter
        if filters.source_agent_ids:
            record_agents: Set[str] = set()
            for ag in (metadata.get("source_agent_ids") or ()):
                record_agents.add(ag)
                if ag.startswith("agent:"):
                    record_agents.add(ag[len("agent:"):])
                else:
                    record_agents.add(f"agent:{ag}")

            filter_agents: Set[str] = set()
            for ag in filters.source_agent_ids:
                filter_agents.add(ag)
                if ag.startswith("agent:"):
                    filter_agents.add(ag[len("agent:"):])
                else:
                    filter_agents.add(f"agent:{ag}")

            if not record_agents.intersection(filter_agents):
                return False

        # Temporal filter
        c_at_val = metadata.get("created_at")
        if c_at_val:
            if isinstance(c_at_val, str):
                c_at = datetime.fromisoformat(c_at_val)
            else:
                c_at = c_at_val
            if filters.time_range_start and c_at < filters.time_range_start:
                return False
            if filters.time_range_end and c_at > filters.time_range_end:
                return False

        return True
