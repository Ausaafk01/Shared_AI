"""Abstract base class defining the persistent storage contract for Knowledge Graph."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ai_brain.graph.models import GraphNode, GraphRelationship


class GraphStore(ABC):
    """Abstract interface defining required persistent knowledge graph storage operations."""

    @abstractmethod
    def add_node(self, node: GraphNode) -> None:
        """Persist or update a GraphNode within its project boundary."""
        pass

    @abstractmethod
    def get_node(self, project_id: str, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by project_id and node_id."""
        pass

    @abstractmethod
    def get_nodes_by_project(self, project_id: str, node_type: Optional[str] = None) -> List[GraphNode]:
        """List all nodes in a project, optionally filtered by node_type."""
        pass

    @abstractmethod
    def delete_node(self, project_id: str, node_id: str) -> bool:
        """Delete a node and its attached relationships from storage."""
        pass

    @abstractmethod
    def add_relationship(self, relationship: GraphRelationship) -> None:
        """Persist a directed relationship edge between two nodes in the same project."""
        pass

    @abstractmethod
    def get_relationship(self, project_id: str, relationship_id: str) -> Optional[GraphRelationship]:
        """Retrieve a relationship by project_id and relationship_id."""
        pass

    @abstractmethod
    def get_relationships_by_project(self, project_id: str, relation_type: Optional[str] = None) -> List[GraphRelationship]:
        """List all relationships in a project, optionally filtered by relation_type."""
        pass

    @abstractmethod
    def get_outgoing_relationships(self, project_id: str, node_id: str) -> List[GraphRelationship]:
        """List all outgoing relationship edges originating from node_id."""
        pass

    @abstractmethod
    def get_incoming_relationships(self, project_id: str, node_id: str) -> List[GraphRelationship]:
        """List all incoming relationship edges terminating at node_id."""
        pass

    @abstractmethod
    def delete_relationship(self, project_id: str, relationship_id: str) -> bool:
        """Delete a relationship edge."""
        pass

    @abstractmethod
    def count_nodes(self, project_id: Optional[str] = None) -> int:
        """Count graph nodes matching project filter."""
        pass

    @abstractmethod
    def count_relationships(self, project_id: Optional[str] = None) -> int:
        """Count graph relationships matching project filter."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close underlying storage resources. Idempotent."""
        pass

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
