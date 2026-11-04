"""
Main Retrieval Engine coordinating query processing, vector retrieval, graph retrieval,
hybrid fusion, and relevance scoring.
"""

from datetime import datetime, timezone
import logging
import time
from typing import List, Optional, Sequence, Union

from ai_brain.graph.graph import KnowledgeGraph
from ai_brain.retrieval.exceptions import (
    GraphRetrievalError,
    ProjectIsolationError,
    QueryValidationError,
    RetrievalError,
    RetrievalExecutionError,
    VectorRetrievalError,
)
from ai_brain.retrieval.fusion import FusedCandidate, HybridFusion
from ai_brain.retrieval.graph_retriever import GraphMatch, GraphRetriever
from ai_brain.retrieval.models import (
    ProcessedQuery,
    QueryFilters,
    RetrievalItem,
    RetrievalResponse,
)
from ai_brain.retrieval.query import QueryProcessor
from ai_brain.retrieval.scoring import RelevanceScorer
from ai_brain.retrieval.types import RetrievalMethod, normalize_retrieval_method
from ai_brain.retrieval.vector_retriever import VectorRetriever
from ai_brain.vectors.index import VectorIndex
from ai_brain.vectors.models import VectorMatch

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Public interface for memory retrieval across Vector Memory and Knowledge Graph.
    
    Guarantees:
    - Strict project isolation across all query processing, vector search, and graph traversal.
    - Supports vector-only, graph-only, and hybrid retrieval modes.
    - Transparent, deterministic multi-factor relevance scoring.
    - Reciprocal Rank Fusion (RRF) for balanced hybrid ranking.
    - Clear partial failure reporting without silent misleading outcomes.
    - Immutable structured RetrievalResponse objects with provenance and score breakdowns.
    """

    def __init__(
        self,
        *,
        vector_retriever: Optional[VectorRetriever] = None,
        graph_retriever: Optional[GraphRetriever] = None,
        vector_index: Optional[VectorIndex] = None,
        graph: Optional[KnowledgeGraph] = None,
        query_processor: Optional[QueryProcessor] = None,
        fusion: Optional[HybridFusion] = None,
        scorer: Optional[RelevanceScorer] = None,
        allow_partial_results: bool = True,
    ) -> None:
        # Resolve vector retriever
        if vector_retriever is not None:
            self.vector_retriever: Optional[VectorRetriever] = vector_retriever
        elif vector_index is not None:
            self.vector_retriever = VectorRetriever(vector_index)
        else:
            self.vector_retriever = None

        # Resolve graph retriever
        if graph_retriever is not None:
            self.graph_retriever: Optional[GraphRetriever] = graph_retriever
        elif graph is not None:
            self.graph_retriever = GraphRetriever(graph)
        else:
            self.graph_retriever = None

        self.query_processor = query_processor or QueryProcessor()
        self.fusion = fusion or HybridFusion()
        self.scorer = scorer or RelevanceScorer()
        self.allow_partial_results = allow_partial_results

    def retrieve(
        self,
        query: str,
        *,
        project_id: str,
        method: Union[str, RetrievalMethod] = RetrievalMethod.HYBRID,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        max_graph_depth: int = 2,
        filters: Optional[QueryFilters] = None,
        seed_memory_ids: Optional[Sequence[str]] = None,
        vector_weight: Optional[float] = None,
        graph_weight: Optional[float] = None,
    ) -> RetrievalResponse:
        """
        Unified retrieval entrypoint.
        
        Args:
            query: Raw natural language query.
            project_id: Explicit project scope.
            method: VECTOR, GRAPH, or HYBRID.
            top_k: Maximum memories to return.
            similarity_threshold: Minimum cosine similarity threshold.
            max_graph_depth: Traversal depth for graph retrieval.
            filters: Optional QueryFilters instance.
            seed_memory_ids: Optional seed memory IDs for graph traversal.
            vector_weight: Optional override weight for vector stream in fusion.
            graph_weight: Optional override weight for graph stream in fusion.
            
        Returns:
            Immutable RetrievalResponse.
        """
        norm_method = normalize_retrieval_method(method)
        if norm_method == RetrievalMethod.VECTOR:
            return self.retrieve_vector(
                query,
                project_id=project_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filters=filters,
            )
        elif norm_method == RetrievalMethod.GRAPH:
            return self.retrieve_graph(
                query,
                project_id=project_id,
                top_k=top_k,
                max_depth=max_graph_depth,
                filters=filters,
                seed_memory_ids=seed_memory_ids,
            )
        else:
            return self.retrieve_hybrid(
                query,
                project_id=project_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                max_graph_depth=max_graph_depth,
                filters=filters,
                seed_memory_ids=seed_memory_ids,
                vector_weight=vector_weight,
                graph_weight=graph_weight,
            )

    def retrieve_vector(
        self,
        query: str,
        *,
        project_id: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        filters: Optional[QueryFilters] = None,
    ) -> RetrievalResponse:
        """Execute vector-only semantic similarity retrieval."""
        t_start = time.perf_counter()
        if self.vector_retriever is None:
            raise VectorRetrievalError("Vector retriever is not configured in RetrievalEngine")

        p_query = self.query_processor.process(query, project_id=project_id, filters=filters)
        warnings: List[str] = []
        hist_statuses = set(p_query.filters.allowed_statuses) & {"SUPERSEDED", "ARCHIVED"}
        if hist_statuses:
            warnings.append(f"Historical status filtering enabled: includes {sorted(hist_statuses)}")

        matches = self.vector_retriever.retrieve(
            p_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        # Convert VectorMatch to FusedCandidate for uniform scoring
        candidates: List[FusedCandidate] = []
        for vm in matches:
            meta = dict(vm.metadata or {})
            norm_score = max(0.0, min(1.0, (vm.score + 1.0) / 2.0))  # Map [-1, 1] cosine to [0, 1]
            c_at_val = meta.get("created_at")
            if isinstance(c_at_val, str):
                dt = datetime.fromisoformat(c_at_val)
            elif isinstance(c_at_val, datetime):
                dt = c_at_val
            else:
                dt = datetime.now(timezone.utc)

            candidates.append(
                FusedCandidate(
                    memory_id=vm.memory_id,
                    project_id=vm.project_id,
                    content=vm.content,
                    memory_type=meta.get("memory_type", "FACT"),
                    status=vm.status,
                    version=vm.version,
                    root_memory_id=meta.get("root_memory_id"),
                    created_at=dt,
                    vector_score=vm.score,
                    graph_score=None,
                    fused_score=round(norm_score, 4),
                    sources=("vector",),
                    graph_metadata={},
                    metadata=meta,
                    source_event_ids=tuple(meta.get("source_event_ids") or ()),
                    source_agent_ids=tuple(meta.get("source_agent_ids") or ()),
                    importance=float(meta.get("importance", 0.5)),
                    confidence=float(meta.get("confidence", 1.0)),
                    is_disputed=bool(meta.get("is_disputed", vm.status == "DISPUTED")),
                )
            )

        scored_items = [self.scorer.score_candidate(c) for c in candidates]
        scored_items.sort(key=lambda item: (-item.score, item.memory_id))

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        return RetrievalResponse(
            query=p_query.original_query,
            normalized_query=p_query.normalized_query,
            project_id=p_query.project_id,
            method=RetrievalMethod.VECTOR.value,
            results=tuple(scored_items[:top_k]),
            total_count=len(scored_items[:top_k]),
            execution_time_ms=t_elapsed,
            warnings=tuple(warnings),
        )

    def retrieve_graph(
        self,
        query: str,
        *,
        project_id: str,
        top_k: int = 10,
        max_depth: int = 2,
        filters: Optional[QueryFilters] = None,
        seed_memory_ids: Optional[Sequence[str]] = None,
    ) -> RetrievalResponse:
        """Execute graph-only structural traversal retrieval."""
        t_start = time.perf_counter()
        if self.graph_retriever is None:
            raise GraphRetrievalError("Graph retriever is not configured in RetrievalEngine")

        p_query = self.query_processor.process(
            query,
            project_id=project_id,
            filters=filters,
            seed_memory_ids=seed_memory_ids,
        )
        matches = self.graph_retriever.retrieve(
            p_query,
            max_depth=max_depth,
            top_k=top_k,
        )

        warnings: List[str] = []
        hist_statuses = set(p_query.filters.allowed_statuses) & {"SUPERSEDED", "ARCHIVED"}
        if hist_statuses:
            warnings.append(f"Historical status filtering enabled: includes {sorted(hist_statuses)}")

        if not matches:
            warnings.append("No seed nodes or connected memories found in graph for query")

        candidates: List[FusedCandidate] = []
        for gm in matches:
            node = gm.node
            meta = dict(node.metadata or {})
            candidates.append(
                FusedCandidate(
                    memory_id=gm.memory_id,
                    project_id=node.project_id,
                    content=node.label,
                    memory_type=meta.get("memory_type", "FACT"),
                    status=meta.get("status", "ACTIVE"),
                    version=int(meta.get("version", 1)),
                    root_memory_id=meta.get("root_memory_id"),
                    created_at=node.created_at,
                    vector_score=None,
                    graph_score=gm.graph_score,
                    fused_score=gm.graph_score,
                    sources=("graph",),
                    graph_metadata={
                        "seed_memory_id": gm.seed_memory_id,
                        "depth": gm.depth,
                        "path": list(gm.path),
                    },
                    metadata=meta,
                    source_event_ids=tuple(meta.get("source_event_ids") or ()),
                    source_agent_ids=tuple(meta.get("source_agent_ids") or ()),
                    importance=float(meta.get("importance", 0.5)),
                    confidence=float(meta.get("confidence", 1.0)),
                    is_disputed=bool(meta.get("is_disputed", meta.get("status") == "DISPUTED")),
                )
            )

        scored_items = [self.scorer.score_candidate(c) for c in candidates]
        scored_items.sort(key=lambda item: (-item.score, item.memory_id))

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        return RetrievalResponse(
            query=p_query.original_query,
            normalized_query=p_query.normalized_query,
            project_id=p_query.project_id,
            method=RetrievalMethod.GRAPH.value,
            results=tuple(scored_items[:top_k]),
            total_count=len(scored_items[:top_k]),
            execution_time_ms=t_elapsed,
            warnings=tuple(warnings),
        )

    def retrieve_hybrid(
        self,
        query: str,
        *,
        project_id: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        max_graph_depth: int = 2,
        filters: Optional[QueryFilters] = None,
        seed_memory_ids: Optional[Sequence[str]] = None,
        vector_weight: Optional[float] = None,
        graph_weight: Optional[float] = None,
    ) -> RetrievalResponse:
        """
        Execute hybrid retrieval combining Vector Memory and Knowledge Graph.
        """
        t_start = time.perf_counter()
        p_query = self.query_processor.process(
            query,
            project_id=project_id,
            filters=filters,
            seed_memory_ids=seed_memory_ids,
        )

        warnings: List[str] = []
        hist_statuses = set(p_query.filters.allowed_statuses) & {"SUPERSEDED", "ARCHIVED"}
        if hist_statuses:
            warnings.append(f"Historical status filtering enabled: includes {sorted(hist_statuses)}")

        vector_matches: List[VectorMatch] = []
        graph_matches: List[GraphMatch] = []

        # 1. Vector retrieval
        if self.vector_retriever is not None:
            try:
                vector_matches = self.vector_retriever.retrieve(
                    p_query,
                    top_k=top_k * 2,
                    similarity_threshold=similarity_threshold,
                )
            except Exception as e:
                if self.allow_partial_results:
                    logger.warning("Vector retrieval failed in hybrid mode: %s", e)
                    warnings.append(f"Vector retrieval unavailable: {e}")
                else:
                    raise VectorRetrievalError(f"Vector retrieval failed in hybrid mode: {e}") from e
        else:
            warnings.append("Vector retriever not configured; proceeding with graph only")

        # 2. Graph retrieval
        if self.graph_retriever is not None:
            try:
                # Use top vector match IDs as additional seeds for graph exploration if no explicit seeds
                vector_seeds: List[str] = []
                if not p_query.filters.seed_memory_ids and vector_matches:
                    vector_seeds = [vm.memory_id for vm in vector_matches[:3]]

                graph_matches = self.graph_retriever.retrieve(
                    p_query,
                    max_depth=max_graph_depth,
                    vector_seeds=vector_seeds,
                    explicit_seeds=p_query.filters.seed_memory_ids,
                    top_k=top_k * 2,
                )
            except Exception as e:
                if self.allow_partial_results:
                    logger.warning("Graph retrieval failed in hybrid mode: %s", e)
                    warnings.append(f"Graph retrieval unavailable: {e}")
                else:
                    raise GraphRetrievalError(f"Graph retrieval failed in hybrid mode: {e}") from e
        else:
            warnings.append("Graph retriever not configured; proceeding with vector only")

        # 3. Hybrid fusion via RRF
        effective_fusion = self.fusion
        if vector_weight is not None or graph_weight is not None:
            vw = vector_weight if vector_weight is not None else self.fusion.vector_weight
            gw = graph_weight if graph_weight is not None else self.fusion.graph_weight
            effective_fusion = HybridFusion(
                rrf_k=self.fusion.rrf_k,
                vector_weight=vw,
                graph_weight=gw,
            )

        fused_candidates = effective_fusion.fuse(
            vector_matches,
            graph_matches,
            allowed_statuses=p_query.filters.allowed_statuses,
            top_k=top_k * 2,
        )

        # 4. Multi-factor relevance scoring
        scored_items = [self.scorer.score_candidate(c) for c in fused_candidates]
        scored_items.sort(key=lambda item: (-item.score, item.memory_id))

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        return RetrievalResponse(
            query=p_query.original_query,
            normalized_query=p_query.normalized_query,
            project_id=p_query.project_id,
            method=RetrievalMethod.HYBRID.value,
            results=tuple(scored_items[:top_k]),
            total_count=len(scored_items[:top_k]),
            execution_time_ms=t_elapsed,
            warnings=tuple(warnings),
        )
