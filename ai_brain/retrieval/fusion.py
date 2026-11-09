"""
Hybrid retrieval fusion using Reciprocal Rank Fusion (RRF).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ai_brain.retrieval.exceptions import HybridFusionError
from ai_brain.retrieval.graph_retriever import GraphMatch
from ai_brain.vectors.models import VectorMatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusedCandidate:
    """
    Intermediate representation of a memory candidate fused from vector and/or graph retrieval.
    """
    memory_id: str
    project_id: str
    content: str
    memory_type: str
    status: str
    version: int
    root_memory_id: Optional[str]
    created_at: datetime
    vector_score: Optional[float]
    graph_score: Optional[float]
    fused_score: float
    sources: Tuple[str, ...]
    graph_metadata: Mapping[str, Any]
    metadata: Mapping[str, Any]
    source_event_ids: Tuple[str, ...]
    source_agent_ids: Tuple[str, ...]
    importance: float
    confidence: float
    is_disputed: bool


class HybridFusion:
    """
    Merges vector retrieval and graph retrieval candidates using Reciprocal Rank Fusion (RRF).
    
    Formula:
        RRF_score(d) = (w_v / (k + rank_v(d))) + (w_g / (k + rank_g(d)))
    
    Guarantees:
    - Independent testability.
    - Avoids double-counting memories appearing in both retrieval streams.
    - Preserves raw vector similarity and graph relevance scores.
    - Preserves retrieval source attribution ('vector', 'graph', or both).
    - Configurable smoothing constant (k) and stream weights (vector_weight, graph_weight).
    - Normalizes fused scores into [0.0, 1.0].
    - Deterministic tie-breaking on matching scores.
    """

    def __init__(
        self,
        *,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        graph_weight: float = 1.0,
    ) -> None:
        if rrf_k <= 0:
            raise HybridFusionError(f"rrf_k must be a positive integer, got {rrf_k}")
        if vector_weight < 0.0:
            raise HybridFusionError(f"vector_weight must be non-negative, got {vector_weight}")
        if graph_weight < 0.0:
            raise HybridFusionError(f"graph_weight must be non-negative, got {graph_weight}")
        if vector_weight == 0.0 and graph_weight == 0.0:
            raise HybridFusionError("vector_weight and graph_weight cannot both be 0.0")

        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

    def fuse(
        self,
        vector_matches: Sequence[VectorMatch],
        graph_matches: Sequence[GraphMatch],
        *,
        allowed_statuses: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[FusedCandidate]:
        """
        Merge vector matches and graph matches into a unified, rank-fused candidate list.
        
        Args:
            vector_matches: Ranked list of VectorMatch objects.
            graph_matches: Ranked list of GraphMatch objects.
            allowed_statuses: Optional sequence of permitted memory statuses.
            top_k: Optional cutoff limit for returned candidates.
            
        Returns:
            List of FusedCandidate objects sorted by fused_score descending.
        """
        allowed_set = set(s.upper() for s in allowed_statuses) if allowed_statuses else None

        # Map memory_id to 1-based ranks
        vec_ranks: Dict[str, Tuple[int, VectorMatch]] = {}
        for idx, vm in enumerate(vector_matches):
            if allowed_set and vm.status.upper() not in allowed_set:
                continue
            mid = vm.memory_id
            if mid not in vec_ranks:
                vec_ranks[mid] = (idx + 1, vm)

        graph_ranks: Dict[str, Tuple[int, GraphMatch]] = {}
        for idx, gm in enumerate(graph_matches):
            g_status = (gm.node.metadata or {}).get("status", "ACTIVE").upper()
            if allowed_set and g_status not in allowed_set:
                continue
            mid = gm.memory_id
            if mid not in graph_ranks:
                graph_ranks[mid] = (idx + 1, gm)

        all_memory_ids = list(dict.fromkeys(list(vec_ranks.keys()) + list(graph_ranks.keys())))
        if not all_memory_ids:
            return []

        # Determine theoretical max score for normalization
        has_vec = bool(vec_ranks) and self.vector_weight > 0
        has_graph = bool(graph_ranks) and self.graph_weight > 0

        max_possible_score = 0.0
        if has_vec:
            max_possible_score += self.vector_weight / (self.rrf_k + 1)
        if has_graph:
            max_possible_score += self.graph_weight / (self.rrf_k + 1)

        fused_list: List[FusedCandidate] = []

        for mid in all_memory_ids:
            score_rrf = 0.0
            sources: List[str] = []
            vec_score: Optional[float] = None
            graph_score: Optional[float] = None

            # Pull vector data
            vm_entry = vec_ranks.get(mid)
            if vm_entry:
                v_rank, v_match = vm_entry
                score_rrf += self.vector_weight / (self.rrf_k + v_rank)
                vec_score = v_match.score
                sources.append("vector")

            # Pull graph data
            gm_entry = graph_ranks.get(mid)
            if gm_entry:
                g_rank, g_match = gm_entry
                score_rrf += self.graph_weight / (self.rrf_k + g_rank)
                graph_score = g_match.graph_score
                sources.append("graph")

            normalized_score = score_rrf / max_possible_score if max_possible_score > 0 else 0.0
            normalized_score = min(1.0, max(0.0, normalized_score))

            # Synthesize metadata from available sources (prefer vector record if available)
            if vm_entry:
                vm = vm_entry[1]
                project_id = vm.project_id
                content = vm.content
                status = vm.status
                version = vm.version
                meta = dict(vm.metadata or {})
                c_at_val = meta.get("created_at")
                if isinstance(c_at_val, str):
                    created_at = datetime.fromisoformat(c_at_val)
                elif isinstance(c_at_val, datetime):
                    created_at = c_at_val
                else:
                    created_at = datetime.now(timezone.utc)
                memory_type = meta.get("memory_type", "FACT")
                root_mid = meta.get("root_memory_id")
                ev_ids = tuple(meta.get("source_event_ids") or ())
                ag_ids = tuple(meta.get("source_agent_ids") or ())
                importance = float(meta.get("importance", 0.5))
                confidence = float(meta.get("confidence", 1.0))
                is_disputed = bool(meta.get("is_disputed", status == "DISPUTED"))
            else:
                node = gm_entry[1].node
                project_id = node.project_id
                content = node.label
                meta = dict(node.metadata or {})
                status = meta.get("status", "ACTIVE")
                version = int(meta.get("version", 1))
                created_at = node.created_at
                memory_type = meta.get("memory_type", "FACT")
                root_mid = meta.get("root_memory_id")
                ev_ids = tuple(meta.get("source_event_ids") or ())
                ag_ids = tuple(meta.get("source_agent_ids") or ())
                importance = float(meta.get("importance", 0.5))
                confidence = float(meta.get("confidence", 1.0))
                is_disputed = bool(meta.get("is_disputed", status == "DISPUTED"))

            # Graph traversal metadata
            g_meta: Dict[str, Any] = {}
            if gm_entry:
                gm = gm_entry[1]
                g_meta = {
                    "seed_memory_id": gm.seed_memory_id,
                    "depth": gm.depth,
                    "path": list(gm.path),
                    "relationships_traversed": list(gm.relationships_traversed),
                }

            fused_list.append(
                FusedCandidate(
                    memory_id=mid,
                    project_id=project_id,
                    content=content,
                    memory_type=memory_type,
                    status=status,
                    version=version,
                    root_memory_id=root_mid,
                    created_at=created_at,
                    vector_score=vec_score,
                    graph_score=graph_score,
                    fused_score=round(normalized_score, 4),
                    sources=tuple(sources),
                    graph_metadata=MappingProxyType(g_meta),
                    metadata=MappingProxyType(meta),
                    source_event_ids=ev_ids,
                    source_agent_ids=ag_ids,
                    importance=importance,
                    confidence=confidence,
                    is_disputed=is_disputed,
                )
            )

        # Deterministic sorting: fused_score desc, vector_score desc, memory_id asc
        fused_list.sort(
            key=lambda c: (
                -c.fused_score,
                -(c.vector_score if c.vector_score is not None else -1.0),
                c.memory_id,
            )
        )

        return fused_list[:top_k] if top_k is not None else fused_list
