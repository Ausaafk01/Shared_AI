"""
Integration layer converting Memory and Conflict objects into Graph nodes and relationships.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Sequence, Union

from ai_brain.graph.graph import KnowledgeGraph
from ai_brain.graph.models import GraphNode, GraphRelationship
from ai_brain.graph.types import NodeType, RelationshipType
from ai_brain.memories.conflicts.models import MemoryConflict
from ai_brain.memories.conflicts.types import ConflictStatus
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus

logger = logging.getLogger(__name__)


def get_memory_node_id(memory_id: str) -> str:
    """Format canonical node ID for a memory."""
    return f"memory:{memory_id}"


def get_event_node_id(event_id: str) -> str:
    """Format canonical node ID for an event."""
    return f"event:{event_id}"


def get_agent_node_id(agent_id: str) -> str:
    """Format canonical node ID for an agent."""
    return f"agent:{agent_id}"


def get_project_node_id(project_id: str) -> str:
    """Format canonical node ID for a project."""
    return f"project:{project_id}"


class GraphIntegrator:
    """
    Transforms Memory objects and Conflict records into Knowledge Graph elements.
    
    Guarantees:
    - Does not mutate original Memory, Event, or Conflict objects.
    - Preserves project isolation: all created nodes and relationships share project_id.
    - Idempotent: repeated integrations do not produce duplicate nodes or relationships.
    - Preserves version provenance: root_memory_id, parent_memory_id, and supersession.
    - Preserves event provenance: DERIVED_FROM edges to EVENT nodes.
    - Preserves agent provenance: CREATED_BY edges to AGENT nodes.
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def integrate_memory(self, memory: Memory) -> GraphNode:
        """
        Convert a single Memory into a MEMORY node with supporting provenance relationships.
        
        Args:
            memory: The immutable Memory instance to integrate.
            
        Returns:
            The created or updated GraphNode corresponding to the Memory.
        """
        project_id = memory.project_id
        mem_node_id = get_memory_node_id(memory.memory_id)

        # 1. Ensure PROJECT node exists
        proj_node_id = get_project_node_id(project_id)
        if not self.graph.has_node(proj_node_id):
            proj_node = GraphNode.create(
                node_id=proj_node_id,
                node_type=NodeType.PROJECT,
                project_id=project_id,
                label=f"Project {project_id}",
                source_ref=project_id,
                metadata={"project_id": project_id},
                created_at=memory.created_at,
            )
            self.graph.add_node(proj_node)

        # 2. Construct MEMORY node
        node_metadata = {
            "status": memory.status,
            "memory_type": memory.memory_type,
            "importance": memory.importance,
            "confidence": memory.confidence,
            "version": memory.version,
            "root_memory_id": memory.root_memory_id,
            "parent_memory_id": memory.parent_memory_id,
        }
        if memory.metadata:
            node_metadata.update(dict(memory.metadata))

        mem_node = GraphNode.create(
            node_id=mem_node_id,
            node_type=NodeType.MEMORY,
            project_id=project_id,
            label=memory.content,
            source_ref=memory.memory_id,
            metadata=node_metadata,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )
        self.graph.add_node(mem_node)

        # 3. Connect Memory -> Project (BELONGS_TO)
        rel_proj_id = f"rel:belongs_to:{memory.memory_id}:{project_id}"
        self.graph.add_relationship(
            GraphRelationship.create(
                relationship_id=rel_proj_id,
                source_id=mem_node_id,
                target_id=proj_node_id,
                relationship_type=RelationshipType.BELONGS_TO,
                project_id=project_id,
                created_at=memory.created_at,
            )
        )

        # 4. Connect Memory -> Event provenance (DERIVED_FROM)
        for evt_id in memory.source_event_ids:
            if not evt_id or not isinstance(evt_id, str):
                continue
            evt_node_id = get_event_node_id(evt_id)
            if not self.graph.has_node(evt_node_id):
                evt_node = GraphNode.create(
                    node_id=evt_node_id,
                    node_type=NodeType.EVENT,
                    project_id=project_id,
                    label=f"Event {evt_id}",
                    source_ref=evt_id,
                    created_at=memory.created_at,
                )
                self.graph.add_node(evt_node)

            rel_evt_id = f"rel:derived_from:{memory.memory_id}:{evt_id}"
            self.graph.add_relationship(
                GraphRelationship.create(
                    relationship_id=rel_evt_id,
                    source_id=mem_node_id,
                    target_id=evt_node_id,
                    relationship_type=RelationshipType.DERIVED_FROM,
                    project_id=project_id,
                    created_at=memory.created_at,
                )
            )

        # 5. Connect Memory -> Agent provenance (CREATED_BY)
        for agent_id in memory.source_agent_ids:
            if not agent_id or not isinstance(agent_id, str):
                continue
            agent_node_id = get_agent_node_id(agent_id)
            existing_agent_node = self.graph.get_node(agent_node_id)
            if existing_agent_node and existing_agent_node.project_id != project_id:
                agent_node_id = f"agent:{project_id}:{agent_id}"

            if not self.graph.has_node(agent_node_id):
                agent_node = GraphNode.create(
                    node_id=agent_node_id,
                    node_type=NodeType.AGENT,
                    project_id=project_id,
                    label=f"Agent {agent_id}",
                    source_ref=agent_id,
                    created_at=memory.created_at,
                )
                self.graph.add_node(agent_node)

            rel_agent_id = f"rel:created_by:{memory.memory_id}:{agent_id}"
            self.graph.add_relationship(
                GraphRelationship.create(
                    relationship_id=rel_agent_id,
                    source_id=mem_node_id,
                    target_id=agent_node_id,
                    relationship_type=RelationshipType.CREATED_BY,
                    project_id=project_id,
                    created_at=memory.created_at,
                )
            )

        # 6. Version and supersession relationships
        if memory.parent_memory_id:
            parent_node_id = get_memory_node_id(memory.parent_memory_id)
            if self.graph.has_node(parent_node_id):
                # New version SUPERSEDES parent version
                rel_super_id = f"rel:supersedes:{memory.memory_id}:{memory.parent_memory_id}"
                self.graph.add_relationship(
                    GraphRelationship.create(
                        relationship_id=rel_super_id,
                        source_id=mem_node_id,
                        target_id=parent_node_id,
                        relationship_type=RelationshipType.SUPERSEDES,
                        project_id=project_id,
                        provenance={"parent_version": memory.version - 1, "current_version": memory.version},
                        created_at=memory.updated_at,
                    )
                )

                # Parent version is SUPERSEDED_BY new version
                rel_super_by_id = f"rel:superseded_by:{memory.parent_memory_id}:{memory.memory_id}"
                self.graph.add_relationship(
                    GraphRelationship.create(
                        relationship_id=rel_super_by_id,
                        source_id=parent_node_id,
                        target_id=mem_node_id,
                        relationship_type=RelationshipType.SUPERSEDED_BY,
                        project_id=project_id,
                        created_at=memory.updated_at,
                    )
                )

        # 7. Check if any existing memories in graph should point to this as parent
        for other_node in self.graph.get_nodes_by_type(NodeType.MEMORY, project_id=project_id):
            if other_node.node_id != mem_node_id:
                other_parent = other_node.metadata.get("parent_memory_id")
                if other_parent == memory.memory_id:
                    rel_s = f"rel:supersedes:{other_node.source_ref}:{memory.memory_id}"
                    self.graph.add_relationship(
                        GraphRelationship.create(
                            relationship_id=rel_s,
                            source_id=other_node.node_id,
                            target_id=mem_node_id,
                            relationship_type=RelationshipType.SUPERSEDES,
                            project_id=project_id,
                        )
                    )
                    rel_sb = f"rel:superseded_by:{memory.memory_id}:{other_node.source_ref}"
                    self.graph.add_relationship(
                        GraphRelationship.create(
                            relationship_id=rel_sb,
                            source_id=mem_node_id,
                            target_id=other_node.node_id,
                            relationship_type=RelationshipType.SUPERSEDED_BY,
                            project_id=project_id,
                        )
                    )

        return mem_node

    def integrate_memories(self, memories: Sequence[Memory]) -> List[GraphNode]:
        """
        Integrate a batch of Memory instances into the graph.
        
        Args:
            memories: Collection of Memory objects.
            
        Returns:
            List of created or updated GraphNode instances.
        """
        return [self.integrate_memory(m) for m in memories]

    def integrate_conflict(self, conflict: MemoryConflict) -> List[GraphRelationship]:
        """
        Integrate an existing MemoryConflict into the graph.
        
        Creates CONTRADICTS relationships between conflicting memories, enriched
        with conflict status and resolution metadata.
        
        Args:
            conflict: The MemoryConflict instance to integrate.
            
        Returns:
            List of created GraphRelationship edges.
        """
        project_id = conflict.project_id
        created_rels: List[GraphRelationship] = []

        # Find or verify involved memory nodes
        memory_nodes: List[GraphNode] = []
        for mid in conflict.memory_ids:
            node_id = get_memory_node_id(mid)
            node = self.graph.get_node(node_id)
            if node:
                memory_nodes.append(node)

        # Require at least 2 memories in the graph to connect conflict relationships
        if len(memory_nodes) < 2:
            return []

        rel_metadata = {
            "conflict_id": conflict.conflict_id,
            "conflict_type": conflict.conflict_type,
            "status": conflict.status,
            "confidence": conflict.confidence,
        }
        if conflict.resolution:
            rel_metadata["resolution"] = conflict.resolution

        # Create CONTRADICTS edges between pairs
        for i in range(len(memory_nodes)):
            for j in range(i + 1, len(memory_nodes)):
                m1, m2 = memory_nodes[i], memory_nodes[j]

                rel1_id = f"rel:contradicts:{conflict.conflict_id}:{m1.source_ref}:{m2.source_ref}"
                rel1 = self.graph.add_relationship(
                    GraphRelationship.create(
                        relationship_id=rel1_id,
                        source_id=m1.node_id,
                        target_id=m2.node_id,
                        relationship_type=RelationshipType.CONTRADICTS,
                        project_id=project_id,
                        provenance={"conflict_id": conflict.conflict_id},
                        metadata=rel_metadata,
                        created_at=conflict.detected_at,
                    )
                )
                created_rels.append(rel1)

                rel2_id = f"rel:contradicts:{conflict.conflict_id}:{m2.source_ref}:{m1.source_ref}"
                rel2 = self.graph.add_relationship(
                    GraphRelationship.create(
                        relationship_id=rel2_id,
                        source_id=m2.node_id,
                        target_id=m1.node_id,
                        relationship_type=RelationshipType.CONTRADICTS,
                        project_id=project_id,
                        provenance={"conflict_id": conflict.conflict_id},
                        metadata=rel_metadata,
                        created_at=conflict.detected_at,
                    )
                )
                created_rels.append(rel2)

        return created_rels

