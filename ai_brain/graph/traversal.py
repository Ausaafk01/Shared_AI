"""
Graph traversal engine supporting BFS, DFS, depth limits, cycle detection, and filtering.
"""

from collections import deque
from dataclasses import dataclass, field
import logging
from typing import List, Optional, Sequence, Set, Tuple

from ai_brain.graph.exceptions import NodeNotFoundError, ProjectIsolationError, TraversalError
from ai_brain.graph.graph import KnowledgeGraph
from ai_brain.graph.models import GraphNode, GraphRelationship
from ai_brain.graph.types import (
    NodeType,
    RelationshipType,
    TraversalAlgorithm,
    TraversalDirection,
    normalize_node_type,
    normalize_relationship_type,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraversalResult:
    """
    Structured outcome of a graph traversal execution.
    """
    start_node_id: str
    visited_nodes: Tuple[GraphNode, ...] = field(default_factory=tuple)
    visited_relationships: Tuple[GraphRelationship, ...] = field(default_factory=tuple)
    paths: Tuple[Tuple[str, ...], ...] = field(default_factory=tuple)
    depth_reached: int = 0
    cycles_detected: int = 0


class GraphTraversal:
    """
    Deterministic traversal engine for KnowledgeGraph.
    
    Guarantees:
    - Supports BFS (level-by-level) and DFS (depth-first branch exploration).
    - Prevents infinite loops and tracks cycles via cycle detection.
    - Enforces project boundaries (rejects cross-project traversal).
    - Supports directional exploration (OUTGOING, INCOMING, BOTH).
    - Supports depth limits and type-based filtering.
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def traverse(
        self,
        start_node_id: str,
        *,
        algorithm: TraversalAlgorithm = TraversalAlgorithm.BFS,
        direction: TraversalDirection = TraversalDirection.OUTGOING,
        max_depth: Optional[int] = None,
        node_type_filter: Optional[Sequence[str | NodeType]] = None,
        relationship_type_filter: Optional[Sequence[str | RelationshipType]] = None,
        project_id: Optional[str] = None,
    ) -> TraversalResult:
        """
        Execute graph traversal from start_node_id.
        
        Args:
            start_node_id: Node ID where traversal begins.
            algorithm: BFS or DFS.
            direction: OUTGOING, INCOMING, or BOTH.
            max_depth: Maximum edge distance from start_node_id (0 = start node only).
            node_type_filter: Restrict returned nodes to specific node types.
            relationship_type_filter: Restrict traversed relationships to specific types.
            project_id: Enforce traversal strictly within this project.
            
        Returns:
            TraversalResult containing visited nodes, relationships, and traversal metrics.
        """
        start_node = self.graph.get_node(start_node_id)
        if not start_node:
            raise NodeNotFoundError(f"Start node '{start_node_id}' does not exist in graph")

        # Project boundary enforcement
        if project_id and start_node.project_id != project_id:
            raise ProjectIsolationError(
                f"Start node project '{start_node.project_id}' does not match requested project '{project_id}'"
            )

        if max_depth is not None and max_depth < 0:
            raise TraversalError("max_depth must be non-negative")

        # Normalize filters
        allowed_node_types = (
            {normalize_node_type(t) for t in node_type_filter} if node_type_filter else None
        )
        allowed_rel_types = (
            {normalize_relationship_type(r) for r in relationship_type_filter}
            if relationship_type_filter
            else None
        )

        if algorithm == TraversalAlgorithm.BFS:
            return self._bfs(
                start_node,
                direction=direction,
                max_depth=max_depth,
                allowed_node_types=allowed_node_types,
                allowed_rel_types=allowed_rel_types,
                project_id=project_id,
            )
        else:
            return self._dfs(
                start_node,
                direction=direction,
                max_depth=max_depth,
                allowed_node_types=allowed_node_types,
                allowed_rel_types=allowed_rel_types,
                project_id=project_id,
            )

    def _get_candidate_edges(
        self,
        current_node_id: str,
        direction: TraversalDirection,
        allowed_rel_types: Optional[Set[str]],
    ) -> List[Tuple[GraphRelationship, str]]:
        """
        Retrieve outgoing/incoming edges from current node, returning list of (relationship, next_node_id).
        """
        candidates: List[Tuple[GraphRelationship, str]] = []

        if direction in (TraversalDirection.OUTGOING, TraversalDirection.BOTH):
            for rel in self.graph.get_outgoing_relationships(current_node_id):
                if allowed_rel_types and rel.relationship_type not in allowed_rel_types:
                    continue
                candidates.append((rel, rel.target_id))

        if direction in (TraversalDirection.INCOMING, TraversalDirection.BOTH):
            for rel in self.graph.get_incoming_relationships(current_node_id):
                if allowed_rel_types and rel.relationship_type not in allowed_rel_types:
                    continue
                candidates.append((rel, rel.source_id))

        return candidates

    def _bfs(
        self,
        start_node: GraphNode,
        direction: TraversalDirection,
        max_depth: Optional[int],
        allowed_node_types: Optional[Set[str]],
        allowed_rel_types: Optional[Set[str]],
        project_id: Optional[str],
    ) -> TraversalResult:
        """Breadth-First Search implementation."""
        # Queue entries: (current_node, current_depth, path_tuple)
        queue = deque([(start_node, 0, (start_node.node_id,))])
        visited_node_ids: Set[str] = {start_node.node_id}
        visited_nodes: List[GraphNode] = []
        visited_rels: List[GraphRelationship] = []
        visited_rel_ids: Set[str] = set()
        recorded_paths: List[Tuple[str, ...]] = []
        max_depth_seen = 0
        cycles_detected = 0

        # Check start node filter
        if not allowed_node_types or start_node.node_type in allowed_node_types:
            visited_nodes.append(start_node)
        recorded_paths.append((start_node.node_id,))

        while queue:
            curr_node, curr_depth, path = queue.popleft()
            if curr_depth > max_depth_seen:
                max_depth_seen = curr_depth

            if max_depth is not None and curr_depth >= max_depth:
                continue

            candidate_edges = self._get_candidate_edges(curr_node.node_id, direction, allowed_rel_types)

            for rel, next_node_id in candidate_edges:
                next_node = self.graph.get_node(next_node_id)
                if not next_node:
                    continue

                # Cross-project boundary check
                if project_id and next_node.project_id != project_id:
                    continue

                # Record relationship if not recorded
                if rel.relationship_id not in visited_rel_ids:
                    visited_rels.append(rel)
                    visited_rel_ids.add(rel.relationship_id)

                # Cycle detection: next_node already in current path
                if next_node_id in path:
                    cycles_detected += 1
                    continue

                new_path = path + (next_node_id,)
                recorded_paths.append(new_path)

                if next_node_id not in visited_node_ids:
                    visited_node_ids.add(next_node_id)
                    if not allowed_node_types or next_node.node_type in allowed_node_types:
                        visited_nodes.append(next_node)
                    queue.append((next_node, curr_depth + 1, new_path))

        return TraversalResult(
            start_node_id=start_node.node_id,
            visited_nodes=tuple(visited_nodes),
            visited_relationships=tuple(visited_rels),
            paths=tuple(recorded_paths),
            depth_reached=max_depth_seen,
            cycles_detected=cycles_detected,
        )

    def _dfs(
        self,
        start_node: GraphNode,
        direction: TraversalDirection,
        max_depth: Optional[int],
        allowed_node_types: Optional[Set[str]],
        allowed_rel_types: Optional[Set[str]],
        project_id: Optional[str],
    ) -> TraversalResult:
        """Depth-First Search implementation."""
        # Stack entries: (current_node, current_depth, path_tuple)
        stack = [(start_node, 0, (start_node.node_id,))]
        visited_node_ids: Set[str] = {start_node.node_id}
        visited_nodes: List[GraphNode] = []
        visited_rels: List[GraphRelationship] = []
        visited_rel_ids: Set[str] = set()
        recorded_paths: List[Tuple[str, ...]] = []
        max_depth_seen = 0
        cycles_detected = 0

        if not allowed_node_types or start_node.node_type in allowed_node_types:
            visited_nodes.append(start_node)
        recorded_paths.append((start_node.node_id,))

        while stack:
            curr_node, curr_depth, path = stack.pop()
            if curr_depth > max_depth_seen:
                max_depth_seen = curr_depth

            if max_depth is not None and curr_depth >= max_depth:
                continue

            candidate_edges = self._get_candidate_edges(curr_node.node_id, direction, allowed_rel_types)

            # Reverse for deterministic DFS traversal order
            for rel, next_node_id in reversed(candidate_edges):
                next_node = self.graph.get_node(next_node_id)
                if not next_node:
                    continue

                if project_id and next_node.project_id != project_id:
                    continue

                if rel.relationship_id not in visited_rel_ids:
                    visited_rels.append(rel)
                    visited_rel_ids.add(rel.relationship_id)

                if next_node_id in path:
                    cycles_detected += 1
                    continue

                new_path = path + (next_node_id,)
                recorded_paths.append(new_path)

                if next_node_id not in visited_node_ids:
                    visited_node_ids.add(next_node_id)
                    if not allowed_node_types or next_node.node_type in allowed_node_types:
                        visited_nodes.append(next_node)
                    stack.append((next_node, curr_depth + 1, new_path))

        return TraversalResult(
            start_node_id=start_node.node_id,
            visited_nodes=tuple(visited_nodes),
            visited_relationships=tuple(visited_rels),
            paths=tuple(recorded_paths),
            depth_reached=max_depth_seen,
            cycles_detected=cycles_detected,
        )

