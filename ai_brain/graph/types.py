"""
Node, relationship, and traversal types for AI Brain Knowledge Graph.
"""

from enum import StrEnum
import re
from typing import Set


class NodeType(StrEnum):
    """Initial extensible node categories."""
    MEMORY = "MEMORY"
    EVENT = "EVENT"
    AGENT = "AGENT"
    PROJECT = "PROJECT"
    CONCEPT = "CONCEPT"
    ENTITY = "ENTITY"


class RelationshipType(StrEnum):
    """Initial extensible directed relationship categories."""
    RELATED_TO = "RELATED_TO"
    DERIVED_FROM = "DERIVED_FROM"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"
    SUPERSEDED_BY = "SUPERSEDED_BY"
    CREATED_BY = "CREATED_BY"
    BELONGS_TO = "BELONGS_TO"
    DEPENDS_ON = "DEPENDS_ON"
    PART_OF = "PART_OF"


class TraversalAlgorithm(StrEnum):
    """Supported graph traversal search algorithms."""
    BFS = "BFS"
    DFS = "DFS"


class TraversalDirection(StrEnum):
    """Edge directionality during graph traversal."""
    OUTGOING = "OUTGOING"
    INCOMING = "INCOMING"
    BOTH = "BOTH"


# Extensible registries
_REGISTERED_NODE_TYPES: Set[str] = {t.value for t in NodeType}
_REGISTERED_RELATIONSHIP_TYPES: Set[str] = {t.value for t in RelationshipType}

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_:\-]+$")


def register_node_type(name: str) -> str:
    """
    Dynamically register a new custom node type.
    
    Args:
        name: Custom node type name.
        
    Returns:
        Normalized uppercase node type name.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Node type name must be a non-empty string")
    norm = name.strip().upper()
    if not _IDENTIFIER_PATTERN.match(norm):
        raise ValueError(
            f"Invalid node type name '{name}'. Must contain only alphanumeric characters, underscores, colons, or hyphens."
        )
    _REGISTERED_NODE_TYPES.add(norm)
    return norm


def is_valid_node_type(name: str | NodeType) -> bool:
    """Check whether a node type is recognized in the registry."""
    if isinstance(name, NodeType):
        return True
    if not isinstance(name, str) or not name.strip():
        return False
    return name.strip().upper() in _REGISTERED_NODE_TYPES


def normalize_node_type(name: str | NodeType) -> str:
    """Validate and normalize a node type to its registered string value."""
    if isinstance(name, NodeType):
        return name.value
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Node type must be a non-empty string or NodeType enum")
    norm = name.strip().upper()
    if norm not in _REGISTERED_NODE_TYPES:
        raise ValueError(f"Unknown or unregistered node type '{name}'")
    return norm


def register_relationship_type(name: str) -> str:
    """
    Dynamically register a new custom relationship type.
    
    Args:
        name: Custom relationship type name.
        
    Returns:
        Normalized uppercase relationship type name.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Relationship type name must be a non-empty string")
    norm = name.strip().upper()
    if not _IDENTIFIER_PATTERN.match(norm):
        raise ValueError(
            f"Invalid relationship type name '{name}'. Must contain only alphanumeric characters, underscores, colons, or hyphens."
        )
    _REGISTERED_RELATIONSHIP_TYPES.add(norm)
    return norm


def is_valid_relationship_type(name: str | RelationshipType) -> bool:
    """Check whether a relationship type is recognized in the registry."""
    if isinstance(name, RelationshipType):
        return True
    if not isinstance(name, str) or not name.strip():
        return False
    return name.strip().upper() in _REGISTERED_RELATIONSHIP_TYPES


def normalize_relationship_type(name: str | RelationshipType) -> str:
    """Validate and normalize a relationship type to its registered string value."""
    if isinstance(name, RelationshipType):
        return name.value
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Relationship type must be a non-empty string or RelationshipType enum")
    norm = name.strip().upper()
    if norm not in _REGISTERED_RELATIONSHIP_TYPES:
        raise ValueError(f"Unknown or unregistered relationship type '{name}'")
    return norm

