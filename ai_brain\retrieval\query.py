"""
Modular, LLM-independent query processing for AI Brain retrieval.
"""

import re
from typing import Mapping, Optional, Sequence, Set

from ai_brain.retrieval.exceptions import ProjectIsolationError, QueryValidationError
from ai_brain.retrieval.models import ProcessedQuery, QueryFilters

# Common English stopwords to exclude from keyword extraction
_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "can't", "cannot", "could", "couldn't", "did",
    "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few",
    "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "isn't", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "shouldn't", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what",
    "when", "where", "which", "while", "who", "whom", "why", "with", "would", "wouldn't",
    "you", "your", "yours", "yourself", "yourselves",
}

# Regex patterns for identifying entities
_ENTITY_PATTERNS = [
    re.compile(r"\b(proj-[a-zA-Z0-9_\-]+)\b", re.IGNORECASE),
    re.compile(r"\b(mem-[a-zA-Z0-9_\-]+)\b", re.IGNORECASE),
    re.compile(r"\b(evt-[a-zA-Z0-9_\-]+)\b", re.IGNORECASE),
    re.compile(r"\b(agent:[a-zA-Z0-9_\-]+)\b", re.IGNORECASE),
    re.compile(r"\b([A-Z][a-zA-Z0-9_\-]+)\b"),  # Capitalized technical terms / Proper nouns
    re.compile(r"\b([a-z0-9]+(?:_[a-z0-9]+)+)\b", re.IGNORECASE),  # snake_case identifiers
]


class QueryProcessor:
    """
    Analyzes, normalizes, and extracts structured signals from raw natural language queries.
    
    Guarantees:
    - Validates query integrity (rejects empty or whitespace-only inputs).
    - Preserves original query verbatim.
    - Deterministically normalizes whitespace and casing for indexing.
    - Extracts meaningful keywords, discarding high-frequency stopwords.
    - Extracts domain entity candidates (IDs, technical names).
    - Requires and validates explicit project scope.
    - Fully offline and LLM-independent.
    """

    def __init__(self, stopwords: Optional[Set[str]] = None) -> None:
        self._stopwords = stopwords if stopwords is not None else _STOPWORDS

    def process(
        self,
        query: str,
        *,
        project_id: str,
        filters: Optional[QueryFilters] = None,
        seed_memory_ids: Optional[Sequence[str]] = None,
    ) -> ProcessedQuery:
        """
        Process a raw search query into a structured ProcessedQuery.
        
        Args:
            query: Raw natural language query string.
            project_id: Required project scope identifier.
            filters: Optional filter criteria.
            seed_memory_ids: Optional explicit memory IDs for graph traversal.
            
        Returns:
            ProcessedQuery immutable dataclass.
        """
        # 1. Validate query string
        if not isinstance(query, str):
            raise QueryValidationError(f"Query must be a string, got {type(query).__name__}")
        stripped = query.strip()
        if not stripped:
            raise QueryValidationError("Query cannot be empty or whitespace only")

        # 2. Validate project scope
        if not isinstance(project_id, str):
            raise ProjectIsolationError(f"project_id must be a string, got {type(project_id).__name__}")
        p_id = project_id.strip()
        if not p_id:
            raise ProjectIsolationError("project_id cannot be empty or whitespace only")

        # 3. Normalize query text
        # Collapse multiple internal whitespace characters to single spaces
        collapsed = re.sub(r"\s+", " ", stripped)
        normalized = collapsed.lower()

        # 4. Extract keywords
        raw_tokens = re.findall(r"[a-zA-Z0-9_\-:]+", normalized)
        keywords: list[str] = []
        for token in raw_tokens:
            cleaned = token.strip("-_:")
            if len(cleaned) >= 2 and cleaned not in self._stopwords and cleaned not in keywords:
                keywords.append(cleaned)

        # 5. Extract entities from original casing
        entities: list[str] = []
        for pattern in _ENTITY_PATTERNS:
            for match in pattern.finditer(collapsed):
                val = match.group(1).strip()
                if val and val not in entities:
                    entities.append(val)

        # 6. Combine filters
        effective_filters = filters if filters is not None else QueryFilters()
        if seed_memory_ids:
            merged_seeds = tuple(set(list(effective_filters.seed_memory_ids) + list(seed_memory_ids)))
            effective_filters = QueryFilters(
                memory_types=effective_filters.memory_types,
                allowed_statuses=effective_filters.allowed_statuses,
                source_agent_ids=effective_filters.source_agent_ids,
                time_range_start=effective_filters.time_range_start,
                time_range_end=effective_filters.time_range_end,
                seed_memory_ids=merged_seeds,
            )

        return ProcessedQuery(
            original_query=collapsed,
            normalized_query=normalized,
            keywords=tuple(keywords),
            entities=tuple(entities),
            project_id=p_id,
            filters=effective_filters,
        )

