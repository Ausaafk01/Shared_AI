"""
Vector similarity search and indexing engine.
"""

import math
from typing import List, Optional, Sequence, Tuple, Union

from ai_brain.vectors.exceptions import (
    DimensionMismatchError,
    ProjectIsolationError,
    VectorValidationError,
)
from ai_brain.vectors.models import VectorMatch, VectorRecord, VectorSearchResult
from ai_brain.vectors.providers.base import EmbeddingProvider
from ai_brain.vectors.storage.base import VectorStore


def compute_cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """
    Calculate the cosine similarity between two float vectors:
    sim(u, v) = (u . v) / (||u|| * ||v||)
    
    Returns float in [-1.0, 1.0]. Returns 0.0 if either vector has zero magnitude.
    """
    if len(vec_a) != len(vec_b):
        raise DimensionMismatchError(
            f"Cannot compute cosine similarity between mismatched dimensions: {len(vec_a)} vs {len(vec_b)}"
        )

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    sim = dot_product / (norm_a * norm_b)
    # Clamp numerical precision artifacts to [-1.0, 1.0]
    return max(-1.0, min(1.0, sim))


class VectorIndex:
    """
    Semantic similarity search engine over persistent or ephemeral vector records.
    
    Guarantees:
    - Strict project isolation: queries only search records matching project_id.
    - Excludes SUPERSEDED and ARCHIVED memories by default.
    - Deterministic tie-breaking on matching scores.
    - Vector dimension validation.
    """

    def __init__(
        self,
        storage: VectorStore,
        provider: EmbeddingProvider,
    ) -> None:
        self.storage = storage
        self.provider = provider

    @property
    def model_id(self) -> str:
        return self.provider.model_id

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def search(
        self,
        query: Union[str, Sequence[float]],
        *,
        project_id: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        allowed_statuses: Optional[Sequence[str]] = None,
    ) -> VectorSearchResult:
        """
        Execute semantic similarity search against indexed vector records.
        
        Args:
            query: Query text string or precomputed float vector.
            project_id: Required project ID enforcing project isolation.
            top_k: Maximum number of matches to return (default: 10).
            similarity_threshold: Minimum cosine similarity score in [-1.0, 1.0].
            allowed_statuses: Sequence of memory statuses to include. Defaults to ('ACTIVE',).
                              SUPERSEDED and ARCHIVED are excluded by default.
                              
        Returns:
            VectorSearchResult containing sorted VectorMatch items.
        """
        if not project_id or not isinstance(project_id, str) or not project_id.strip():
            raise ProjectIsolationError("project_id must be a non-empty string for vector search")
        p_id = project_id.strip()

        if top_k <= 0:
            raise VectorValidationError(f"top_k must be positive, got {top_k}")

        if similarity_threshold is not None:
            if not isinstance(similarity_threshold, (int, float)) or isinstance(similarity_threshold, bool):
                raise VectorValidationError(f"similarity_threshold must be a float, got {similarity_threshold!r}")
            if not (-1.0 <= float(similarity_threshold) <= 1.0):
                raise VectorValidationError(
                    f"similarity_threshold must be between -1.0 and 1.0, got {similarity_threshold}"
                )

        # 1. Resolve query vector
        query_text = ""
        if isinstance(query, str):
            query_text = self.provider.validate_input_text(query)
            query_vec = self.provider.embed_text(query_text)
        elif isinstance(query, (list, tuple)):
            query_vec = self.provider.validate_embedding_vector(query)
            query_text = f"<precomputed_vector:{len(query_vec)}d>"
        else:
            raise VectorValidationError(f"Query must be text string or vector sequence, got {type(query).__name__}")

        # 2. Status filtering: default to ACTIVE only (exclude SUPERSEDED and ARCHIVED)
        effective_statuses = (
            [s.upper() for s in allowed_statuses]
            if allowed_statuses is not None
            else ["ACTIVE"]
        )

        # 3. Native vector similarity search (e.g. pgvector on PostgreSQL)
        if hasattr(self.storage, "search_similarity"):
            return self.storage.search_similarity(
                query_vector=query_vec,
                project_id=p_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                allowed_statuses=effective_statuses,
                model_id=self.model_id,
                query_text=query_text,
            )

        # 4. Standard in-memory fallback: Retrieve candidates strictly within project boundary
        candidates = self.storage.get_vectors_by_project(
            p_id,
            model_id=self.model_id,
            allowed_statuses=effective_statuses,
        )

        if not candidates:
            return VectorSearchResult(
                query=query_text,
                matches=(),
                project_id=p_id,
                model_id=self.model_id,
                total_candidates=0,
            )

        # 4. Compute similarity and filter by threshold
        scored_matches: List[Tuple[float, str, int, VectorRecord]] = []
        for record in candidates:
            if record.dimensions != self.dimension:
                continue
            sim = compute_cosine_similarity(query_vec, record.embedding)
            if similarity_threshold is not None and sim < similarity_threshold:
                continue
            # Store (score, memory_id, version, record) for deterministic sorting
            scored_matches.append((sim, record.memory_id, record.version, record))

        # 5. Sort deterministically:
        # Highest score first (-sim), memory_id ascending (lexicographical), newest version first (-version)
        scored_matches.sort(key=lambda item: (-item[0], item[1], -item[2]))

        # 6. Apply top-k limit
        top_slice = scored_matches[:top_k]

        matches = tuple(
            VectorMatch(
                memory_id=item[3].memory_id,
                score=item[0],
                project_id=item[3].project_id,
                version=item[3].version,
                status=item[3].status,
                content=item[3].content,
                model_id=item[3].model_id,
                metadata=item[3].metadata,
            )
            for item in top_slice
        )

        return VectorSearchResult(
            query=query_text,
            matches=matches,
            project_id=p_id,
            model_id=self.model_id,
            total_candidates=len(candidates),
        )

