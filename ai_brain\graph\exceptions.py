"""
Exception hierarchy for the AI Brain Knowledge Graph component.
"""


class GraphError(Exception):
    """Base exception for all graph operations."""


class GraphValidationError(GraphError):
    """Raised when node, relationship, or graph attribute validation fails."""


class NodeNotFoundError(GraphError):
    """Raised when a requested node does not exist in the graph."""


class RelationshipNotFoundError(GraphError):
    """Raised when a requested relationship does not exist in the graph."""


class DuplicateNodeError(GraphError):
    """Raised when attempting to add a node with an existing node_id."""


class DuplicateRelationshipError(GraphError):
    """Raised when attempting to add an identical relationship that already exists."""


class ProjectIsolationError(GraphError):
    """Raised when an operation attempts to link or traverse across different projects."""


class TraversalError(GraphError):
    """Raised when graph traversal encounters an invalid state or configuration."""

