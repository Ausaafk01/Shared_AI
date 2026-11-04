"""
In-memory KnowledgeGraph implementation with strict project isolation.
"""

from collections import defaultdict
import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ai_brain.graph.exceptions import (
    DuplicateNodeError,
    DuplicateRelationshipError,
    GraphValidationError,
    NodeNotFoundError,
    ProjectIsolationError,
    RelationshipNotFoundError,
)
from ai_brain.graph.models import GraphNode, GraphRelationship
from ai_brain.graph.types import (
    NodeType,
    RelationshipType,
    TraversalDirection,
    normalize_node_type,
    normalize_relationship_type,
)

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    In-memory directed Knowledge Graph for AI Brain.
    
    Guarantees:
    - Strict project isolation: cross-project relationships are prohibited.
    - Integrity enforcement: relationships require both source and target nodes to exist.
    - Deduplication: prevents redundant nodes and duplicate relationship edges.
    - Fast bidirectional traversal and index lookups.
    """

    def __init__(self) -> None:
        # Primary storage
        self._nodes: Dict[str, GraphNode] = {}
        self._relationships: Dict[str, GraphRelationship] = {}

        # Adjacency indexes: node_id -> set of relationship_ids
        self._outgoing: Dict[str, Set[str]] = defaultdict(set)
        self._incoming: Dict[str, Set[str]] = defaultdict(set)

        # Project indices
        self._project_nodes: Dict[str, Set[str]] = defaultdict(set)
        self._project_relationships: Dict[str, Set[str]] = defaultdict(set)

        # Signature deduplication: (project_id, source_id, target_id, relationship_type) -> relationship_id
        self._rel_signatures: Dict[Tuple[str, str, str, str], str] = {}

        # Source ref index: (project_id, node_type, source_ref) -> node_id
        self._source_ref_index: Dict[Tuple[str, str, str], str] = {}

    @property
    def node_count(self) -> int:
        """Total number of nodes currently stored in the graph."""
        return len(self._nodes)

    @property
    def relationship_count(self) -> int:
        """Total number of relationships currently stored in the graph."""
        return len(self._relationships)

    def add_node(self, node: GraphNode, *, fail_if_exists: bool = False) -> GraphNode:
        """
        Add a GraphNode to the graph.
        
        Args:
            node: The GraphNode to add.
            fail_if_exists: If True, raise DuplicateNodeError if node_id already exists.
            
        Returns:
            The added or updated GraphNode.
        """
        if not isinstance(node, GraphNode):
            raise GraphValidationError(f"Expected GraphNode instance, got {type(node).__name__}")

        if node.node_id in self._nodes:
            if fail_if_exists:
                raise DuplicateNodeError(f"Node with id '{node.node_id}' already exists in the graph")
            # If exists and fail_if_exists=False, update if changed
            existing = self._nodes[node.node_id]
            if existing.project_id != node.project_id:
                raise ProjectIsolationError(
                    f"Cannot change project_id of existing node '{node.node_id}' from '{existing.project_id}' to '{node.project_id}'"
                )
            self._nodes[node.node_id] = node
            if node.source_ref:
                self._source_ref_index[(node.project_id, node.node_type, node.source_ref)] = node.node_id
            return node

        # Store node
        self._nodes[node.node_id] = node
        self._project_nodes[node.project_id].add(node.node_id)
        if node.source_ref:
            self._source_ref_index[(node.project_id, node.node_type, node.source_ref)] = node.node_id

        return node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by its node_id, or None if not found."""
        if not node_id:
            return None
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return bool(node_id and node_id in self._nodes)

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all of its incident (incoming and outgoing) relationships.
        
        Args:
            node_id: Unique identifier of the node to remove.
            
        Returns:
            True if the node was removed, False if it was not found.
        """
        if node_id not in self._nodes:
            return False

        node = self._nodes.pop(node_id)
        self._project_nodes[node.project_id].discard(node_id)
        if node.source_ref:
            self._source_ref_index.pop((node.project_id, node.node_type, node.source_ref), None)

        # Remove all incident outgoing relationships
        outgoing_rels = list(self._outgoing.pop(node_id, set()))
        for rel_id in outgoing_rels:
            self._remove_relationship_internal(rel_id)

        # Remove all incident incoming relationships
        incoming_rels = list(self._incoming.pop(node_id, set()))
        for rel_id in incoming_rels:
            self._remove_relationship_internal(rel_id)

        return True

    def add_relationship(
        self,
        relationship: GraphRelationship,
        *,
        fail_if_exists: bool = False,
    ) -> GraphRelationship:
        """
        Add a directed GraphRelationship between two existing nodes.
        
        Enforces:
        - Source and target nodes must exist.
        - Source and target nodes must belong to the same project as the relationship.
        - Prohibits cross-project edges.
        - Detects duplicate relationships.
        """
        if not isinstance(relationship, GraphRelationship):
            raise GraphValidationError(f"Expected GraphRelationship instance, got {type(relationship).__name__}")

        # 1. Verify source node exists
        source_node = self._nodes.get(relationship.source_id)
        if not source_node:
            raise NodeNotFoundError(f"Source node '{relationship.source_id}' does not exist in graph")

        # 2. Verify target node exists
        target_node = self._nodes.get(relationship.target_id)
        if not target_node:
            raise NodeNotFoundError(f"Target node '{relationship.target_id}' does not exist in graph")

        # 3. Enforce strict project isolation
        if source_node.project_id != target_node.project_id:
            raise ProjectIsolationError(
                f"Cross-project relationships are prohibited: source node project is '{source_node.project_id}' "
                f"while target node project is '{target_node.project_id}'"
            )
        if relationship.project_id != source_node.project_id:
            raise ProjectIsolationError(
                f"Relationship project '{relationship.project_id}' does not match nodes project '{source_node.project_id}'"
            )

        # 4. Check for duplicate relationship signature
        sig = (relationship.project_id, relationship.source_id, relationship.target_id, relationship.relationship_type)
        if sig in self._rel_signatures:
            existing_rel_id = self._rel_signatures[sig]
            if fail_if_exists:
                raise DuplicateRelationshipError(
                    f"Relationship '{relationship.relationship_type}' from '{relationship.source_id}' to "
                    f"'{relationship.target_id}' already exists with ID '{existing_rel_id}'"
                )
            return self._relationships[existing_rel_id]

        if relationship.relationship_id in self._relationships:
            if fail_if_exists:
                raise DuplicateRelationshipError(
                    f"Relationship with ID '{relationship.relationship_id}' already exists"
                )
            return self._relationships[relationship.relationship_id]

        # 5. Store relationship
        self._relationships[relationship.relationship_id] = relationship
        self._outgoing[relationship.source_id].add(relationship.relationship_id)
        self._incoming[relationship.target_id].add(relationship.relationship_id)
        self._project_relationships[relationship.project_id].add(relationship.relationship_id)
        self._rel_signatures[sig] = relationship.relationship_id

        return relationship

    def get_relationship(self, relationship_id: str) -> Optional[GraphRelationship]:
        """Retrieve a relationship by its ID, or None if not found."""
        if not relationship_id:
            return None
        return self._relationships.get(relationship_id)

    def has_relationship(self, relationship_id: str) -> bool:
        """Check if a relationship exists in the graph."""
        return bool(relationship_id and relationship_id in self._relationships)

    def remove_relationship(self, relationship_id: str) -> bool:
        """Remove a relationship by ID."""
        if relationship_id not in self._relationships:
            return False
        self._remove_relationship_internal(relationship_id)
        return True

    def _remove_relationship_internal(self, relationship_id: str) -> None:
        """Internal helper to remove relationship and clean up all indexes."""
        rel = self._relationships.pop(relationship_id, None)
        if not rel:
            return
        self._outgoing[rel.source_id].discard(relationship_id)
        self._incoming[rel.target_id].discard(relationship_id)
        self._project_relationships[rel.project_id].discard(relationship_id)
        sig = (rel.project_id, rel.source_id, rel.target_id, rel.relationship_type)
        self._rel_signatures.pop(sig, None)

    def get_outgoing_relationships(
        self,
        node_id: str,
        relationship_type: Optional[str | RelationshipType] = None,
    ) -> List[GraphRelationship]:
        """Get all outgoing relationships originating from node_id, optionally filtered by type."""
        rel_ids = self._outgoing.get(node_id, set())
        norm_type = normalize_relationship_type(relationship_type) if relationship_type else None
        results = [self._relationships[rid] for rid in rel_ids if rid in self._relationships]
        if norm_type:
            results = [r for r in results if r.relationship_type == norm_type]
        return results

    def get_incoming_relationships(
        self,
        node_id: str,
        relationship_type: Optional[str | RelationshipType] = None,
    ) -> List[GraphRelationship]:
        """Get all incoming relationships pointing to node_id, optionally filtered by type."""
        rel_ids = self._incoming.get(node_id, set())
        norm_type = normalize_relationship_type(relationship_type) if relationship_type else None
        results = [self._relationships[rid] for rid in rel_ids if rid in self._relationships]
        if norm_type:
            results = [r for r in results if r.relationship_type == norm_type]
        return results

    def get_neighbors(
        self,
        node_id: str,
        *,
        direction: TraversalDirection = TraversalDirection.BOTH,
        relationship_type: Optional[str | RelationshipType] = None,
    ) -> List[GraphNode]:
        """
        Get all adjacent nodes connected to node_id in the given direction.
        """
        if node_id not in self._nodes:
            return []

        neighbor_ids: Set[str] = set()

        if direction in (TraversalDirection.OUTGOING, TraversalDirection.BOTH):
            for rel in self.get_outgoing_relationships(node_id, relationship_type):
                neighbor_ids.add(rel.target_id)

        if direction in (TraversalDirection.INCOMING, TraversalDirection.BOTH):
            for rel in self.get_incoming_relationships(node_id, relationship_type):
                neighbor_ids.add(rel.source_id)

        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]

    def get_nodes_by_type(
        self,
        node_type: str | NodeType,
        project_id: Optional[str] = None,
    ) -> List[GraphNode]:
        """Retrieve all nodes of a specific type, optionally filtered by project_id."""
        norm_type = normalize_node_type(node_type)
        nodes = self._nodes.values()
        if project_id:
            p_ids = self._project_nodes.get(project_id, set())
            nodes = [self._nodes[nid] for nid in p_ids if nid in self._nodes]
        return [n for n in nodes if n.node_type == norm_type]

    def get_nodes_by_project(self, project_id: str) -> List[GraphNode]:
        """Retrieve all nodes belonging to a project."""
        node_ids = self._project_nodes.get(project_id, set())
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_relationships_by_project(self, project_id: str) -> List[GraphRelationship]:
        """Retrieve all relationships belonging to a project."""
        rel_ids = self._project_relationships.get(project_id, set())
        return [self._relationships[rid] for rid in rel_ids if rid in self._relationships]

    def find_node_by_source_ref(
        self,
        source_ref: str,
        *,
        node_type: Optional[str | NodeType] = None,
        project_id: Optional[str] = None,
    ) -> Optional[GraphNode]:
        """
        Look up a node by its source reference (e.g. original memory_id, event_id, or agent_id).
        
        Args:
            source_ref: The external source reference string.
            node_type: Optional NodeType to distinguish when different entity types share source_ref.
            project_id: Optional Project ID to scope the lookup.
            
        Returns:
            The matching GraphNode, or None if not found.
        """
        if not source_ref:
            return None

        norm_type = normalize_node_type(node_type) if node_type else None

        # 1. Both project_id and node_type provided -> direct O(1) dictionary lookup
        if project_id and norm_type:
            node_id = self._source_ref_index.get((project_id, norm_type, source_ref))
            return self._nodes.get(node_id) if node_id else None

        # 2. project_id provided, no node_type
        if project_id:
            for (p_id, n_type, s_ref), n_id in self._source_ref_index.items():
                if p_id == project_id and s_ref == source_ref:
                    return self._nodes.get(n_id)
            return None

        # 3. node_type provided, no project_id
        if norm_type:
            for (p_id, n_type, s_ref), n_id in self._source_ref_index.items():
                if n_type == norm_type and s_ref == source_ref:
                    return self._nodes.get(n_id)
            return None

        # 4. Neither provided -> scan index
        for (p_id, n_type, s_ref), n_id in self._source_ref_index.items():
            if s_ref == source_ref:
                return self._nodes.get(n_id)

        return None

    def find_nodes_by_source_ref(
        self,
        source_ref: str,
        *,
        node_type: Optional[str | NodeType] = None,
        project_id: Optional[str] = None,
    ) -> List[GraphNode]:
        """Find all nodes that match the given source_ref."""
        if not source_ref:
            return []
        norm_type = normalize_node_type(node_type) if node_type else None
        results: List[GraphNode] = []
        for (p_id, n_type, s_ref), n_id in self._source_ref_index.items():
            if s_ref == source_ref:
                if project_id and p_id != project_id:
                    continue
                if norm_type and n_type != norm_type:
                    continue
                node = self._nodes.get(n_id)
                if node and node not in results:
                    results.append(node)
        return results

    def validate_integrity(self) -> List[str]:
        """
        Check graph integrity: dangling relationships, cross-project violations, index desync.
        
        Returns:
            List of error strings describing any violations found. Empty list if valid.
        """
        errors: List[str] = []

        for rel_id, rel in self._relationships.items():
            # Check source node
            if rel.source_id not in self._nodes:
                errors.append(f"Relationship '{rel_id}' has dangling source_id '{rel.source_id}'")
            # Check target node
            if rel.target_id not in self._nodes:
                errors.append(f"Relationship '{rel_id}' has dangling target_id '{rel.target_id}'")

            # Check cross-project consistency
            src = self._nodes.get(rel.source_id)
            tgt = self._nodes.get(rel.target_id)
            if src and tgt:
                if src.project_id != tgt.project_id:
                    errors.append(
                        f"Cross-project relationship '{rel_id}': source in '{src.project_id}', target in '{tgt.project_id}'"
                    )
                if rel.project_id != src.project_id:
                    errors.append(
                        f"Relationship '{rel_id}' project '{rel.project_id}' does not match source '{src.project_id}'"
                    )

        return errors

    def clear(self) -> None:
        """Reset the graph, clearing all nodes, relationships, and indices."""
        self._nodes.clear()
        self._relationships.clear()
        self._outgoing.clear()
        self._incoming.clear()
        self._project_nodes.clear()
        self._project_relationships.clear()
        self._rel_signatures.clear()
        self._source_ref_index.clear()

