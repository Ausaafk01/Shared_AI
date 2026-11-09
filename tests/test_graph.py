"""
Comprehensive test suite for the AI Brain Knowledge Graph component.
"""

from datetime import datetime, timedelta, timezone
import unittest

from ai_brain.graph import (
    DuplicateNodeError,
    DuplicateRelationshipError,
    GraphIntegrator,
    GraphNode,
    GraphRelationship,
    GraphTraversal,
    GraphValidationError,
    KnowledgeGraph,
    NodeType,
    ProjectIsolationError,
    RelationshipType,
    TraversalAlgorithm,
    TraversalDirection,
    get_agent_node_id,
    get_event_node_id,
    get_memory_node_id,
    get_project_node_id,
)
from ai_brain.memories.conflicts.models import MemoryConflict
from ai_brain.memories.conflicts.types import ConflictStatus, ConflictType
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryType


class TestKnowledgeGraph(unittest.TestCase):
    """Unit tests for KnowledgeGraph structure, models, traversal, and integrations."""

    def setUp(self) -> None:
        self.graph = KnowledgeGraph()
        self.traversal = GraphTraversal(self.graph)
        self.integrator = GraphIntegrator(self.graph)
        self.now = datetime.now(timezone.utc)

    # =========================================================================
    # Section 1: Graph Structure & Validation
    # =========================================================================

    def test_node_creation_and_validation(self) -> None:
        """Test valid node creation and attribute enforcement."""
        node = GraphNode.create(
            node_id="node-1",
            node_type=NodeType.CONCEPT,
            project_id="proj-1",
            label="Database Architecture",
            source_ref="ref-1",
            metadata={"domain": "persistence"},
        )
        self.assertEqual(node.node_id, "node-1")
        self.assertEqual(node.node_type, NodeType.CONCEPT.value)
        self.assertEqual(node.project_id, "proj-1")
        self.assertEqual(node.label, "Database Architecture")
        self.assertEqual(node.source_ref, "ref-1")
        self.assertEqual(node.metadata["domain"], "persistence")

        # Immutability
        with self.assertRaises(AttributeError):
            node.label = "New Label"  # type: ignore

    def test_node_validation_errors(self) -> None:
        """Test that invalid node attributes raise GraphValidationError."""
        with self.assertRaises(GraphValidationError):
            GraphNode.create(node_id="", node_type=NodeType.MEMORY, project_id="p1", label="Test")

        with self.assertRaises(GraphValidationError):
            GraphNode.create(node_id="n1", node_type="UNKNOWN_TYPE_123", project_id="p1", label="Test")

        with self.assertRaises(GraphValidationError):
            GraphNode.create(node_id="n1", node_type=NodeType.MEMORY, project_id="", label="Test")

        with self.assertRaises(GraphValidationError):
            GraphNode.create(node_id="n1", node_type=NodeType.MEMORY, project_id="p1", label="")

        # updated_at earlier than created_at
        with self.assertRaises(GraphValidationError):
            GraphNode.create(
                node_id="n1",
                node_type=NodeType.MEMORY,
                project_id="p1",
                label="Test",
                created_at=self.now,
                updated_at=self.now - timedelta(days=1),
            )

    def test_relationship_creation_and_validation(self) -> None:
        """Test valid relationship creation and attribute enforcement."""
        rel = GraphRelationship.create(
            relationship_id="rel-1",
            source_id="src-1",
            target_id="tgt-1",
            relationship_type=RelationshipType.DEPENDS_ON,
            project_id="proj-1",
            weight=2.5,
            metadata={"scope": "compile"},
        )
        self.assertEqual(rel.relationship_id, "rel-1")
        self.assertEqual(rel.source_id, "src-1")
        self.assertEqual(rel.target_id, "tgt-1")
        self.assertEqual(rel.source_node_id, "src-1")
        self.assertEqual(rel.target_node_id, "tgt-1")
        self.assertEqual(rel.relationship_type, RelationshipType.DEPENDS_ON.value)
        self.assertEqual(rel.weight, 2.5)

        # Immutability
        with self.assertRaises(AttributeError):
            rel.weight = 3.0  # type: ignore

    def test_relationship_self_reference_prevention(self) -> None:
        """Verify that self-referencing relationships are rejected."""
        with self.assertRaises(GraphValidationError):
            GraphRelationship.create(
                relationship_id="rel-self",
                source_id="node-same",
                target_id="node-same",
                relationship_type=RelationshipType.RELATED_TO,
                project_id="proj-1",
            )

    def test_node_and_relationship_lookup(self) -> None:
        """Test adding, retrieving, and inspecting nodes and edges."""
        n1 = GraphNode.create(node_id="n1", node_type=NodeType.ENTITY, project_id="p1", label="Service A")
        n2 = GraphNode.create(node_id="n2", node_type=NodeType.ENTITY, project_id="p1", label="Service B")
        self.graph.add_node(n1)
        self.graph.add_node(n2)

        rel = GraphRelationship.create(
            relationship_id="r1",
            source_id="n1",
            target_id="n2",
            relationship_type=RelationshipType.DEPENDS_ON,
            project_id="p1",
        )
        self.graph.add_relationship(rel)

        self.assertTrue(self.graph.has_node("n1"))
        self.assertTrue(self.graph.has_relationship("r1"))
        self.assertEqual(self.graph.get_node("n1"), n1)
        self.assertEqual(self.graph.get_relationship("r1"), rel)

        outgoing = self.graph.get_outgoing_relationships("n1")
        self.assertEqual(len(outgoing), 1)
        self.assertEqual(outgoing[0].relationship_id, "r1")

        incoming = self.graph.get_incoming_relationships("n2")
        self.assertEqual(len(incoming), 1)
        self.assertEqual(incoming[0].relationship_id, "r1")

        neighbors = self.graph.get_neighbors("n1", direction=TraversalDirection.OUTGOING)
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0].node_id, "n2")

    def test_duplicate_prevention(self) -> None:
        """Verify duplicate node and duplicate relationship prevention."""
        n1 = GraphNode.create(node_id="n1", node_type=NodeType.CONCEPT, project_id="p1", label="Concept 1")
        n2 = GraphNode.create(node_id="n2", node_type=NodeType.CONCEPT, project_id="p1", label="Concept 2")
        self.graph.add_node(n1)
        self.graph.add_node(n2)

        # Duplicate node with fail_if_exists=True
        with self.assertRaises(DuplicateNodeError):
            self.graph.add_node(n1, fail_if_exists=True)

        rel = GraphRelationship.create(
            relationship_id="r1",
            source_id="n1",
            target_id="n2",
            relationship_type=RelationshipType.RELATED_TO,
            project_id="p1",
        )
        self.graph.add_relationship(rel)

        # Duplicate relationship with fail_if_exists=True
        with self.assertRaises(DuplicateRelationshipError):
            self.graph.add_relationship(rel, fail_if_exists=True)

        # Adding same signature without fail_if_exists returns existing relationship
        rel_dup = GraphRelationship.create(
            relationship_id="r-diff-id",
            source_id="n1",
            target_id="n2",
            relationship_type=RelationshipType.RELATED_TO,
            project_id="p1",
        )
        retrieved = self.graph.add_relationship(rel_dup, fail_if_exists=False)
        self.assertEqual(retrieved.relationship_id, "r1")
        self.assertEqual(self.graph.relationship_count, 1)

    def test_project_isolation_in_graph(self) -> None:
        """Verify that cross-project relationships are strictly prevented."""
        n_proj_a = GraphNode.create(node_id="na", node_type=NodeType.MEMORY, project_id="proj-A", label="Mem A")
        n_proj_b = GraphNode.create(node_id="nb", node_type=NodeType.MEMORY, project_id="proj-B", label="Mem B")
        self.graph.add_node(n_proj_a)
        self.graph.add_node(n_proj_b)

        # Relationship linking proj-A to proj-B must be rejected
        cross_rel = GraphRelationship.create(
            relationship_id="rel-cross",
            source_id="na",
            target_id="nb",
            relationship_type=RelationshipType.RELATED_TO,
            project_id="proj-A",
        )
        with self.assertRaises(ProjectIsolationError):
            self.graph.add_relationship(cross_rel)

        # Filtering by project
        nodes_a = self.graph.get_nodes_by_project("proj-A")
        self.assertEqual(len(nodes_a), 1)
        self.assertEqual(nodes_a[0].node_id, "na")

    def test_node_removal_safely_cleans_edges(self) -> None:
        """Verify that removing a node cleans up all incident relationships."""
        n1 = GraphNode.create(node_id="n1", node_type=NodeType.ENTITY, project_id="p1", label="N1")
        n2 = GraphNode.create(node_id="n2", node_type=NodeType.ENTITY, project_id="p1", label="N2")
        self.graph.add_node(n1)
        self.graph.add_node(n2)

        rel = GraphRelationship.create(
            relationship_id="r1", source_id="n1", target_id="n2", relationship_type=RelationshipType.RELATED_TO, project_id="p1"
        )
        self.graph.add_relationship(rel)

        self.assertTrue(self.graph.remove_node("n1"))
        self.assertFalse(self.graph.has_node("n1"))
        self.assertFalse(self.graph.has_relationship("r1"))
        self.assertEqual(len(self.graph.get_incoming_relationships("n2")), 0)

    def test_serialization_round_trips(self) -> None:
        """Test to_dict, from_dict, to_json, and from_json for GraphNode and GraphRelationship."""
        node = GraphNode.create(
            node_id="node-ser",
            node_type=NodeType.MEMORY,
            project_id="proj-ser",
            label="Serialized Node",
            source_ref="mem-1",
            metadata={"score": 9.5},
        )
        node_json = node.to_json()
        restored_node = GraphNode.from_json(node_json)
        self.assertEqual(restored_node.node_id, node.node_id)
        self.assertEqual(restored_node.label, node.label)
        self.assertEqual(restored_node.metadata["score"], 9.5)

        rel = GraphRelationship.create(
            relationship_id="rel-ser",
            source_id="node-ser",
            target_id="node-target",
            relationship_type=RelationshipType.SUPERSEDES,
            project_id="proj-ser",
            weight=1.5,
            metadata={"rev": 2},
        )
        rel_json = rel.to_json()
        restored_rel = GraphRelationship.from_json(rel_json)
        self.assertEqual(restored_rel.relationship_id, rel.relationship_id)
        self.assertEqual(restored_rel.weight, 1.5)

    def test_graph_integrity_validation(self) -> None:
        """Verify validate_integrity reports clean on consistent graph."""
        n1 = GraphNode.create(node_id="n1", node_type=NodeType.CONCEPT, project_id="p1", label="N1")
        n2 = GraphNode.create(node_id="n2", node_type=NodeType.CONCEPT, project_id="p1", label="N2")
        self.graph.add_node(n1)
        self.graph.add_node(n2)
        rel = GraphRelationship.create(
            relationship_id="r1", source_id="n1", target_id="n2", relationship_type=RelationshipType.RELATED_TO, project_id="p1"
        )
        self.graph.add_relationship(rel)

        errors = self.graph.validate_integrity()
        self.assertEqual(len(errors), 0)

    # =========================================================================
    # Section 2: Graph Traversal
    # =========================================================================

    def _setup_linear_graph(self) -> None:
        """A -> B -> C -> D"""
        nodes = [
            GraphNode.create(node_id="A", node_type=NodeType.MEMORY, project_id="p1", label="Node A"),
            GraphNode.create(node_id="B", node_type=NodeType.CONCEPT, project_id="p1", label="Node B"),
            GraphNode.create(node_id="C", node_type=NodeType.EVENT, project_id="p1", label="Node C"),
            GraphNode.create(node_id="D", node_type=NodeType.AGENT, project_id="p1", label="Node D"),
        ]
        for n in nodes:
            self.graph.add_node(n)

        rels = [
            GraphRelationship.create(relationship_id="rAB", source_id="A", target_id="B", relationship_type=RelationshipType.RELATED_TO, project_id="p1"),
            GraphRelationship.create(relationship_id="rBC", source_id="B", target_id="C", relationship_type=RelationshipType.DERIVED_FROM, project_id="p1"),
            GraphRelationship.create(relationship_id="rCD", source_id="C", target_id="D", relationship_type=RelationshipType.CREATED_BY, project_id="p1"),
        ]
        for r in rels:
            self.graph.add_relationship(r)

    def test_traversal_bfs(self) -> None:
        """Test BFS traversal exploring level-by-level."""
        self._setup_linear_graph()
        result = self.traversal.traverse("A", algorithm=TraversalAlgorithm.BFS)
        visited_ids = [n.node_id for n in result.visited_nodes]
        self.assertEqual(visited_ids, ["A", "B", "C", "D"])
        self.assertEqual(result.depth_reached, 3)

    def test_traversal_dfs(self) -> None:
        """Test DFS traversal exploring branches to depth first."""
        self._setup_linear_graph()
        result = self.traversal.traverse("A", algorithm=TraversalAlgorithm.DFS)
        visited_ids = [n.node_id for n in result.visited_nodes]
        self.assertEqual(visited_ids, ["A", "B", "C", "D"])

    def test_traversal_with_max_depth(self) -> None:
        """Test restricting traversal to maximum depth."""
        self._setup_linear_graph()
        # Depth 0: start node only
        r0 = self.traversal.traverse("A", max_depth=0)
        self.assertEqual([n.node_id for n in r0.visited_nodes], ["A"])

        # Depth 1: direct neighbors (A -> B)
        r1 = self.traversal.traverse("A", max_depth=1)
        self.assertEqual([n.node_id for n in r1.visited_nodes], ["A", "B"])

        # Depth 2: A -> B -> C
        r2 = self.traversal.traverse("A", max_depth=2)
        self.assertEqual([n.node_id for n in r2.visited_nodes], ["A", "B", "C"])

    def test_traversal_filtered_by_node_type(self) -> None:
        """Test filtering visited nodes by specific node types."""
        self._setup_linear_graph()
        result = self.traversal.traverse("A", node_type_filter=[NodeType.EVENT, NodeType.AGENT])
        visited_ids = [n.node_id for n in result.visited_nodes]
        # Only C (EVENT) and D (AGENT) match the filter
        self.assertEqual(visited_ids, ["C", "D"])

    def test_traversal_filtered_by_relationship_type(self) -> None:
        """Test filtering traversal by specific relationship types."""
        self._setup_linear_graph()
        # Only allow DERIVED_FROM; traversal from A stops because A->B is RELATED_TO
        result = self.traversal.traverse("A", relationship_type_filter=[RelationshipType.DERIVED_FROM])
        self.assertEqual([n.node_id for n in result.visited_nodes], ["A"])

    def test_traversal_incoming_and_outgoing(self) -> None:
        """Test traversal directionality."""
        self._setup_linear_graph()
        # From B outgoing -> C, D
        out_b = self.traversal.traverse("B", direction=TraversalDirection.OUTGOING)
        self.assertEqual([n.node_id for n in out_b.visited_nodes], ["B", "C", "D"])

        # From B incoming -> A
        in_b = self.traversal.traverse("B", direction=TraversalDirection.INCOMING)
        self.assertEqual([n.node_id for n in in_b.visited_nodes], ["B", "A"])

    def test_traversal_cycle_detection_and_termination(self) -> None:
        """Verify cycle detection prevents infinite loops in circular graphs."""
        # Create circular graph: X -> Y -> Z -> X
        for nid in ("X", "Y", "Z"):
            self.graph.add_node(GraphNode.create(node_id=nid, node_type=NodeType.CONCEPT, project_id="p-cyc", label=nid))

        self.graph.add_relationship(GraphRelationship.create(relationship_id="rXY", source_id="X", target_id="Y", relationship_type=RelationshipType.RELATED_TO, project_id="p-cyc"))
        self.graph.add_relationship(GraphRelationship.create(relationship_id="rYZ", source_id="Y", target_id="Z", relationship_type=RelationshipType.RELATED_TO, project_id="p-cyc"))
        self.graph.add_relationship(GraphRelationship.create(relationship_id="rZX", source_id="Z", target_id="X", relationship_type=RelationshipType.RELATED_TO, project_id="p-cyc"))

        # Traversal must terminate cleanly and detect the cycle
        result = self.traversal.traverse("X", algorithm=TraversalAlgorithm.BFS)
        self.assertEqual(len(result.visited_nodes), 3)
        self.assertGreaterEqual(result.cycles_detected, 1)

    def test_cross_project_traversal_rejection(self) -> None:
        """Verify that specifying a project filter prevents traversing nodes in another project."""
        self._setup_linear_graph()  # p1
        with self.assertRaises(ProjectIsolationError):
            self.traversal.traverse("A", project_id="other-project")

    # =========================================================================
    # Section 3: Memory-to-Graph Integration
    # =========================================================================

    def test_single_memory_integration(self) -> None:
        """Test converting a single Memory into corresponding MEMORY and PROJECT nodes."""
        mem = Memory.create(
            memory_id="mem-100",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-db",
            source_event_ids=["evt-1"],
            source_agent_ids=["claude"],
            importance=0.8,
            confidence=0.9,
        )

        node = self.integrator.integrate_memory(mem)
        self.assertEqual(node.node_id, get_memory_node_id("mem-100"))
        self.assertEqual(node.node_type, NodeType.MEMORY.value)
        self.assertEqual(node.source_ref, "mem-100")

        # Verify PROJECT node created and connected
        proj_node = self.graph.get_node(get_project_node_id("proj-db"))
        self.assertIsNotNone(proj_node)

        # Verify EVENT node created and connected
        evt_node = self.graph.get_node(get_event_node_id("evt-1"))
        self.assertIsNotNone(evt_node)
        self.assertEqual(evt_node.source_ref, "evt-1")

        # Verify AGENT node created and connected
        agent_node = self.graph.get_node(get_agent_node_id("claude"))
        self.assertIsNotNone(agent_node)
        self.assertEqual(agent_node.source_ref, "claude")

        # Check relationships
        rels = self.graph.get_outgoing_relationships(node.node_id)
        types = {r.relationship_type for r in rels}
        self.assertIn(RelationshipType.BELONGS_TO.value, types)
        self.assertIn(RelationshipType.DERIVED_FROM.value, types)
        self.assertIn(RelationshipType.CREATED_BY.value, types)

    def test_memory_version_and_supersession_integration(self) -> None:
        """Test that memory version chains generate SUPERSEDES and SUPERSEDED_BY without DERIVED_FROM."""
        mem_v1 = Memory.create(
            memory_id="mem-v1",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-vers",
            source_event_ids=["evt-001"],
            version=1,
            root_memory_id="root-db",
        )
        mem_v2 = Memory.create(
            memory_id="mem-v2",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-vers",
            source_event_ids=["evt-002"],
            version=2,
            root_memory_id="root-db",
            parent_memory_id="mem-v1",
        )

        self.integrator.integrate_memory(mem_v1)
        self.integrator.integrate_memory(mem_v2)

        node_v1_id = get_memory_node_id("mem-v1")
        node_v2_id = get_memory_node_id("mem-v2")

        # 1. Correct relationship directions:
        # v2 SUPERSEDES v1
        out_v2_super = self.graph.get_outgoing_relationships(node_v2_id, RelationshipType.SUPERSEDES)
        self.assertEqual(len(out_v2_super), 1)
        self.assertEqual(out_v2_super[0].target_id, node_v1_id)

        # v1 is SUPERSEDED_BY v2
        out_v1_super_by = self.graph.get_outgoing_relationships(node_v1_id, RelationshipType.SUPERSEDED_BY)
        self.assertEqual(len(out_v1_super_by), 1)
        self.assertEqual(out_v1_super_by[0].target_id, node_v2_id)

        # 2. No DERIVED_FROM relationship between memory versions
        out_v2_derived_to_v1 = [
            r for r in self.graph.get_outgoing_relationships(node_v2_id, RelationshipType.DERIVED_FROM)
            if r.target_id == node_v1_id
        ]
        self.assertEqual(len(out_v2_derived_to_v1), 0)

        out_v1_derived_to_v2 = [
            r for r in self.graph.get_outgoing_relationships(node_v1_id, RelationshipType.DERIVED_FROM)
            if r.target_id == node_v2_id
        ]
        self.assertEqual(len(out_v1_derived_to_v2), 0)

        # 3. Event provenance still uses DERIVED_FROM
        evt_v2_node_id = get_event_node_id("evt-002")
        out_v2_events = [
            r for r in self.graph.get_outgoing_relationships(node_v2_id, RelationshipType.DERIVED_FROM)
            if r.target_id == evt_v2_node_id
        ]
        self.assertEqual(len(out_v2_events), 1)

        # 4. Repeated integration remains idempotent
        rel_count_before = self.graph.relationship_count
        self.integrator.integrate_memory(mem_v2)
        self.assertEqual(self.graph.relationship_count, rel_count_before)

    def test_source_ref_indexing_distinguishes_node_type_and_project(self) -> None:
        """Verify _source_ref_index key distinguishes project_id, node_type, and source_ref."""
        shared_id = "shared-ref-123"

        # 1. An EVENT and an AGENT with the same source reference can coexist in the same project
        node_evt = GraphNode.create(
            node_id=get_event_node_id(shared_id),
            node_type=NodeType.EVENT,
            project_id="proj-idx",
            label="Event with shared ref",
            source_ref=shared_id,
        )
        node_agent = GraphNode.create(
            node_id=get_agent_node_id(shared_id),
            node_type=NodeType.AGENT,
            project_id="proj-idx",
            label="Agent with shared ref",
            source_ref=shared_id,
        )

        self.graph.add_node(node_evt)
        self.graph.add_node(node_agent)

        self.assertTrue(self.graph.has_node(node_evt.node_id))
        self.assertTrue(self.graph.has_node(node_agent.node_id))
        self.assertNotEqual(node_evt.node_id, node_agent.node_id)

        # 2. Both can be retrieved independently
        found_evt = self.graph.find_node_by_source_ref(shared_id, node_type=NodeType.EVENT, project_id="proj-idx")
        self.assertIsNotNone(found_evt)
        self.assertEqual(found_evt.node_id, node_evt.node_id)
        self.assertEqual(found_evt.node_type, NodeType.EVENT.value)

        found_agent = self.graph.find_node_by_source_ref(shared_id, node_type=NodeType.AGENT, project_id="proj-idx")
        self.assertIsNotNone(found_agent)
        self.assertEqual(found_agent.node_id, node_agent.node_id)
        self.assertEqual(found_agent.node_type, NodeType.AGENT.value)

        # 3. Different projects remain isolated
        node_other_proj = GraphNode.create(
            node_id=f"event:other:{shared_id}",
            node_type=NodeType.EVENT,
            project_id="proj-other",
            label="Event in other project",
            source_ref=shared_id,
        )
        self.graph.add_node(node_other_proj)

        found_other = self.graph.find_node_by_source_ref(shared_id, node_type=NodeType.EVENT, project_id="proj-other")
        self.assertIsNotNone(found_other)
        self.assertEqual(found_other.node_id, node_other_proj.node_id)
        self.assertNotEqual(found_other.node_id, node_evt.node_id)

        # 4. Existing memory lookup behavior remains intact
        mem = Memory.create(
            memory_id="mem-unique-99",
            memory_type=MemoryType.FACT,
            content="Unique memory content",
            project_id="proj-idx",
        )
        self.integrator.integrate_memory(mem)

        # Look up by source_ref without explicit node_type
        found_mem = self.graph.find_node_by_source_ref("mem-unique-99", project_id="proj-idx")
        self.assertIsNotNone(found_mem)
        self.assertEqual(found_mem.source_ref, "mem-unique-99")
        self.assertEqual(found_mem.node_type, NodeType.MEMORY.value)

    def test_batch_memory_integration_and_idempotency(self) -> None:
        """Test batch integration and idempotency (duplicate prevention)."""
        mems = [
            Memory.create(memory_id=f"m-{i}", memory_type=MemoryType.FACT, content=f"Fact {i}", project_id="proj-batch")
            for i in range(5)
        ]

        # Integrate batch
        nodes = self.integrator.integrate_memories(mems)
        self.assertEqual(len(nodes), 5)
        initial_node_count = self.graph.node_count

        # Re-integrating the same batch does not inflate node count
        nodes_re = self.integrator.integrate_memories(mems)
        self.assertEqual(len(nodes_re), 5)
        self.assertEqual(self.graph.node_count, initial_node_count)

    def test_original_memory_immutability_preserved(self) -> None:
        """Verify that integration never mutates the original Memory instances."""
        mem = Memory.create(
            memory_id="mem-immut",
            memory_type=MemoryType.PREFERENCE,
            content="Prefer concise code.",
            project_id="proj-immut",
        )
        self.integrator.integrate_memory(mem)
        with self.assertRaises(AttributeError):
            mem.content = "Changed"  # type: ignore

    # =========================================================================
    # Section 4: Conflict Management Integration
    # =========================================================================

    def test_conflict_integration_contradicts_relationships(self) -> None:
        """Test integrating a conflict creates CONTRADICTS relationships with status metadata."""
        m1 = Memory.create(memory_id="c-mem1", memory_type=MemoryType.FACT, content="Uses MySQL", project_id="p-conf")
        m2 = Memory.create(memory_id="c-mem2", memory_type=MemoryType.FACT, content="Uses PostgreSQL", project_id="p-conf")
        self.integrator.integrate_memories([m1, m2])

        conflict = MemoryConflict.create(
            conflict_id="conf-999",
            project_id="p-conf",
            memory_ids=["c-mem1", "c-mem2"],
            conflict_type=ConflictType.FACTUAL,
            description="Database conflict",
            status=ConflictStatus.UNRESOLVED,
            confidence=0.5,
        )

        rels = self.integrator.integrate_conflict(conflict)
        self.assertGreaterEqual(len(rels), 2)
        for r in rels:
            self.assertEqual(r.relationship_type, RelationshipType.CONTRADICTS.value)
            self.assertEqual(r.metadata["status"], ConflictStatus.UNRESOLVED.value)
            self.assertEqual(r.metadata["conflict_id"], "conf-999")

    def test_resolved_conflict_metadata_preservation(self) -> None:
        """Test that resolved conflicts preserve resolution details in relationship metadata."""
        m1 = Memory.create(memory_id="res-m1", memory_type=MemoryType.FACT, content="Uses MySQL", project_id="p-res")
        m2 = Memory.create(memory_id="res-m2", memory_type=MemoryType.FACT, content="Uses PostgreSQL", project_id="p-res")
        self.integrator.integrate_memories([m1, m2])

        conflict = MemoryConflict.create(
            conflict_id="conf-resolved",
            project_id="p-res",
            memory_ids=["res-m1", "res-m2"],
            conflict_type=ConflictType.FACTUAL,
            description="Database conflict resolved",
            status=ConflictStatus.RESOLVED,
            resolution={"winner_id": "res-m2", "reasoning": "Verified with tool execution"},
        )

        rels = self.integrator.integrate_conflict(conflict)
        self.assertGreaterEqual(len(rels), 2)
        for r in rels:
            self.assertEqual(r.metadata["status"], ConflictStatus.RESOLVED.value)
            self.assertEqual(r.metadata["resolution"]["winner_id"], "res-m2")

    def test_no_unsupported_conflict_relationships_without_memories(self) -> None:
        """Verify that conflict integration does not create relationships if memory nodes are absent."""
        conflict = MemoryConflict.create(
            conflict_id="conf-orphan",
            project_id="p-orphan",
            memory_ids=["nonexistent-1", "nonexistent-2"],
            conflict_type=ConflictType.FACTUAL,
            description="Orphan conflict",
        )
        rels = self.integrator.integrate_conflict(conflict)
        self.assertEqual(len(rels), 0)


if __name__ == "__main__":
    unittest.main()

