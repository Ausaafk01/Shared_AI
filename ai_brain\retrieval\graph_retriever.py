"""
Graph-based memory retrieval using Knowledge Graph traversal with strict project isolation.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ai_brain.graph.exceptions import NodeNotFoundError, ProjectIsolationError
from ai_brain.graph.graph import KnowledgeGraph
from ai_brain.graph.models import GraphNode
from ai_brain.graph.traversal import GraphTraversal, TraversalResult
from ai_brain.graph.types import NodeType, RelationshipType, TraversalAlgorithm, TraversalDirection
from ai_brain.retrieval.exceptions import GraphRetrievalError
from ai_brain.retrieval.models import ProcessedQuery, QueryFilters
from ai_brain.retrieval.types import SeedStrategy

logger = logging.getLogger(__name__)


def _format_memory_node_id(memory_id: str) -> str:
    """Format canonical node ID for a memory if not already prefixed."""
    if memory_id.startswith("memory:"):
        return memory_id
    return f"memory:{memory_id}"


def _extract_memory_id(node_id: str) -> str:
    """Extract bare memory ID from canonical node ID."""
    if node_id.startswith("memory:"):
        return node_id[len("memory:"):]
    return node_id


def _get_rel_types_for_path(graph: KnowledgeGraph, path: Sequence[str]) -> Tuple[str, ...]:
    """Retrieve the sequence of relationship types traversed between path nodes."""
    rel_types: List[str] = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        found_type = None
        for r in graph.get_outgoing_relationships(u):
            if r.target_id == v:
                found_type = r.relationship_type
                break
        if not found_type:
            for r in graph.get_incoming_relationships(u):
                if r.source_id == v:
                    found_type = f"INCOMING_{r.relationship_type}"
                    break
        rel_types.append(found_type or "CONNECTED_TO")
    return tuple(rel_types)


@dataclass(frozen=True)
class GraphMatch:
    """
    Candidate memory node retrieved via graph traversal with genuine evidence.
    """
    node: GraphNode
    memory_id: str
    graph_score: float
    depth: int
    seed_memory_id: str
    path: Tuple[str, ...]
    relationships_traversed: Tuple[str, ...] = field(default_factory=tuple)


class GraphRetriever:
    """
    Retrieves memories connected through the Knowledge Graph.
    
    Guarantees:
    - Strict project isolation: only explores nodes matching target project_id.
    - Honest graph evidence attribution: vector seeds do not automatically receive depth-0 graph evidence.
    - Graph evidence requires an actual graph relationship traversal (depth >= 1) or explicit graph seed.
    - Configurable traversal depth and direction (OUTGOING, INCOMING, BOTH).
    - Deduplicates visited memories, retaining the shortest path and highest score.
    - Excludes SUPERSEDED and ARCHIVED memories by default before scoring.
    - Preserves traversal explanations (seed node, depth, edges traversed).
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph
        self.traversal = GraphTraversal(graph)

    def find_seed_nodes(
        self,
        query: ProcessedQuery,
        *,
        explicit_seeds: Optional[Sequence[str]] = None,
        max_seeds: int = 5,
    ) -> List[GraphNode]:
        """
        Identify graph-native seed nodes in the Knowledge Graph for traversal.
        
        Strategy:
        1. Explicit seeds from query filters or arguments.
        2. Keyword and entity matching against project MEMORY nodes.
        """
        project_id = query.project_id
        candidate_seed_ids: List[str] = []

        if explicit_seeds:
            for s in explicit_seeds:
                if s and s not in candidate_seed_ids:
                    candidate_seed_ids.append(s)

        if query.filters and query.filters.seed_memory_ids:
            for s in query.filters.seed_memory_ids:
                if s and s not in candidate_seed_ids:
                    candidate_seed_ids.append(s)

        found_seeds: List[GraphNode] = []
        for s_id in candidate_seed_ids:
            canonical_id = _format_memory_node_id(s_id)
            node = self.graph.get_node(canonical_id)
            if node and node.project_id == project_id:
                if node not in found_seeds:
                    found_seeds.append(node)

        if found_seeds:
            return found_seeds[:max_seeds]

        # Keyword / entity matching over project MEMORY nodes
        project_memory_nodes = self.graph.get_nodes_by_type(NodeType.MEMORY, project_id=project_id)
        if not project_memory_nodes:
            return []

        search_tokens = set(query.keywords) | set(e.lower() for e in query.entities)
        if not search_tokens:
            return []

        import re
        scored_candidates: List[Tuple[int, GraphNode]] = []
        for node in project_memory_nodes:
            content_lower = node.label.lower()
            ref_lower = (node.source_ref or "").lower()
            node_words = set(re.findall(r"[a-zA-Z0-9_\-]+", content_lower))
            if ref_lower:
                node_words.add(ref_lower)

            match_count = sum(1 for tok in search_tokens if tok in node_words)
            if match_count > 0:
                scored_candidates.append((match_count, node))

        scored_candidates.sort(key=lambda item: (-item[0], item[1].node_id))
        return [node for _, node in scored_candidates[:max_seeds]]

    def retrieve(
        self,
        query: ProcessedQuery,
        *,
        max_depth: int = 2,
        direction: TraversalDirection = TraversalDirection.BOTH,
        vector_seeds: Optional[Sequence[str]] = None,
        explicit_seeds: Optional[Sequence[str]] = None,
        external_seeds: Optional[Sequence[str]] = None,
        top_k: int = 10,
    ) -> List[GraphMatch]:
        """
        Execute graph retrieval for the given processed query.
        
        Args:
            query: ProcessedQuery defining target project_id, keywords, and filters.
            max_depth: Maximum traversal depth from seed nodes (default: 2).
            direction: Traversal direction across edges (default: BOTH).
            vector_seeds: Optional memory IDs from vector search used strictly for topological expansion.
                          Vector seeds NEVER receive automatic depth-0 graph evidence.
            explicit_seeds: Optional explicit memory IDs requested by caller as graph targets.
            external_seeds: Backward-compatible alias for vector_seeds.
            top_k: Maximum candidate memories to return.
            
        Returns:
            List of genuine GraphMatch objects sorted by graph_score descending.
        """
        project_id = query.project_id
        if not project_id:
            raise ProjectIsolationError("project_id must be provided for graph retrieval")

        if max_depth < 0:
            raise GraphRetrievalError(f"max_depth must be non-negative, got {max_depth}")

        # Resolve vector seeds (support external_seeds alias)
        v_seed_ids: Set[str] = set()
        if vector_seeds:
            v_seed_ids.update(vector_seeds)
        if external_seeds:
            v_seed_ids.update(external_seeds)

        # Resolve explicit seeds
        exp_seed_ids: Set[str] = set()
        if explicit_seeds:
            exp_seed_ids.update(explicit_seeds)
        if query.filters and query.filters.seed_memory_ids:
            exp_seed_ids.update(query.filters.seed_memory_ids)

        # 1. Determine exploration seeds and graph-native seeds
        graph_native_seeds: List[GraphNode] = []
        if exp_seed_ids:
            for sid in exp_seed_ids:
                node = self.graph.get_node(_format_memory_node_id(sid))
                if node and node.project_id == project_id and node not in graph_native_seeds:
                    graph_native_seeds.append(node)
        else:
            # Discover graph-native seeds via query keywords / entities
            graph_native_seeds = self.find_seed_nodes(query)

        # Resolve all seed nodes to explore outward from (graph native + vector seeds)
        all_exploration_seeds: List[GraphNode] = list(graph_native_seeds)
        for vsid in v_seed_ids:
            vnode = self.graph.get_node(_format_memory_node_id(vsid))
            if vnode and vnode.project_id == project_id and vnode not in all_exploration_seeds:
                all_exploration_seeds.append(vnode)

        if not all_exploration_seeds:
            return []

        # 2. Filter criteria (excludes SUPERSEDED and ARCHIVED by default)
        allowed_statuses = (
            set(query.filters.allowed_statuses)
            if query.filters and query.filters.allowed_statuses
            else {"ACTIVE", "DISPUTED"}
        )
        allowed_types = set(query.filters.memory_types) if query.filters and query.filters.memory_types else None
        allowed_agents = set(query.filters.source_agent_ids) if query.filters and query.filters.source_agent_ids else None

        # Track best genuine match per memory node: node_id -> GraphMatch
        best_matches: Dict[str, GraphMatch] = {}

        # 3. Graph-native seeds receive depth 0 evidence ONLY if NOT selected solely via vector seeds
        for seed in graph_native_seeds:
            seed_mem_id = _extract_memory_id(seed.node_id)
            if seed_mem_id in v_seed_ids and seed_mem_id not in exp_seed_ids:
                # Vector-selected seed without explicit graph specification:
                # Must not receive automatic depth-0 graph evidence.
                continue

            if self._satisfies_filters(seed, allowed_statuses, allowed_types, allowed_agents, query.filters):
                best_matches[seed.node_id] = GraphMatch(
                    node=seed,
                    memory_id=seed_mem_id,
                    graph_score=1.0,
                    depth=0,
                    seed_memory_id=seed_mem_id,
                    path=(seed.node_id,),
                    relationships_traversed=(),
                )

        # 4. Traverse outward from ALL exploration seeds (including vector seeds)
        # Use domain knowledge relationships, excluding structural project BELONGS_TO edges
        domain_rel_types = [
            RelationshipType.SUPERSEDES,
            RelationshipType.SUPERSEDED_BY,
            RelationshipType.CONTRADICTS,
            RelationshipType.DERIVED_FROM,
            RelationshipType.CREATED_BY,
            RelationshipType.RELATED_TO,
            RelationshipType.DEPENDS_ON,
            RelationshipType.SUPPORTS,
            RelationshipType.PART_OF,
        ]

        for seed in all_exploration_seeds:
            seed_mem_id = _extract_memory_id(seed.node_id)

            traversal_res: TraversalResult = self.traversal.traverse(
                start_node_id=seed.node_id,
                algorithm=TraversalAlgorithm.BFS,
                direction=direction,
                max_depth=max_depth,
                relationship_type_filter=domain_rel_types,
                project_id=project_id,
            )

            for path in traversal_res.paths:
                # Require actual graph relationship traversal (depth >= 1, path length >= 2)
                if not path or len(path) < 2:
                    continue

                target_node_id = path[-1]
                if not target_node_id.startswith("memory:"):
                    continue

                target_node = self.graph.get_node(target_node_id)
                if not target_node or target_node.project_id != project_id:
                    continue

                if not self._satisfies_filters(target_node, allowed_statuses, allowed_types, allowed_agents, query.filters):
                    continue

                depth = len(path) - 1
                # Distance decay formula: 1.0 / (1.0 + 0.5 * depth)
                score = round(1.0 / (1.0 + 0.5 * depth), 4)

                target_mem_id = _extract_memory_id(target_node_id)
                rel_types = _get_rel_types_for_path(self.graph, path)

                existing = best_matches.get(target_node_id)
                if existing is None or score > existing.graph_score:
                    best_matches[target_node_id] = GraphMatch(
                        node=target_node,
                        memory_id=target_mem_id,
                        graph_score=score,
                        depth=depth,
                        seed_memory_id=seed_mem_id,
                        path=path,
                        relationships_traversed=rel_types,
                    )

        # Sort matches by graph_score descending, then memory_id ascending for determinism
        results = list(best_matches.values())
        results.sort(key=lambda m: (-m.graph_score, m.memory_id))
        return results[:top_k]

    def _satisfies_filters(
        self,
        node: GraphNode,
        allowed_statuses: Set[str],
        allowed_types: Optional[Set[str]],
        allowed_agents: Optional[Set[str]],
        filters: Optional[QueryFilters],
    ) -> bool:
        """Check if a memory graph node satisfies filter constraints."""
        metadata = node.metadata or {}
        
        # Status filter
        status = metadata.get("status", "ACTIVE").upper()
        if status not in allowed_statuses:
            return False

        # Memory type filter
        if allowed_types:
            mem_type = metadata.get("memory_type", "").upper()
            if mem_type not in allowed_types:
                return False

        # Agent filter
        if allowed_agents:
            node_agents: Set[str] = set()
            for ag in (metadata.get("source_agent_ids") or ()):
                node_agents.add(ag)
                if ag.startswith("agent:"):
                    node_agents.add(ag[len("agent:"):])
                else:
                    node_agents.add(f"agent:{ag}")
            for rel in self.graph.get_outgoing_relationships(node.node_id, RelationshipType.CREATED_BY):
                target_id = rel.target_id
                node_agents.add(target_id)
                if target_id.startswith("agent:"):
                    node_agents.add(target_id[len("agent:"):])
                else:
                    node_agents.add(f"agent:{target_id}")

            filter_agents: Set[str] = set()
            for ag in allowed_agents:
                filter_agents.add(ag)
                if ag.startswith("agent:"):
                    filter_agents.add(ag[len("agent:"):])
                else:
                    filter_agents.add(f"agent:{ag}")

            if not node_agents.intersection(filter_agents):
                return False

        # Time range filter
        if filters:
            if filters.time_range_start and node.created_at < filters.time_range_start:
                return False
            if filters.time_range_end and node.created_at > filters.time_range_end:
                return False

        return True
