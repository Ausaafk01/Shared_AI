"""
AI Brain Knowledge Graph package.
"""

from .exceptions import (
    DuplicateNodeError,
    DuplicateRelationshipError,
    GraphError,
    GraphValidationError,
    NodeNotFoundError,
    ProjectIsolationError,
    RelationshipNotFoundError,
    TraversalError,
)
from .graph import KnowledgeGraph
from .integration import (
    GraphIntegrator,
    get_agent_node_id,
    get_event_node_id,
    get_memory_node_id,
    get_project_node_id,
)
from .models import GraphNode, GraphRelationship
from .traversal import GraphTraversal, TraversalResult
from .types import (
    NodeType,
    RelationshipType,
    TraversalAlgorithm,
    TraversalDirection,
    is_valid_node_type,
    is_valid_relationship_type,
    normalize_node_type,
    normalize_relationship_type,
    register_node_type,
    register_relationship_type,
)

__all__ = [
    # Types
    "NodeType",
    "RelationshipType",
    "TraversalAlgorithm",
    "TraversalDirection",
    "register_node_type",
    "normalize_node_type",
    "is_valid_node_type",
    "register_relationship_type",
    "normalize_relationship_type",
    "is_valid_relationship_type",
    # Models
    "GraphNode",
    "GraphRelationship",
    # Graph Engine
    "KnowledgeGraph",
    # Traversal Engine
    "GraphTraversal",
    "TraversalResult",
    # Integration Layer
    "GraphIntegrator",
    "get_memory_node_id",
    "get_event_node_id",
    "get_agent_node_id",
    "get_project_node_id",
    # Exceptions
    "GraphError",
    "GraphValidationError",
    "NodeNotFoundError",
    "RelationshipNotFoundError",
    "DuplicateNodeError",
    "DuplicateRelationshipError",
    "ProjectIsolationError",
    "TraversalError",
]

