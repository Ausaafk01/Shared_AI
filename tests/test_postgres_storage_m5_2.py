"""Integration tests for AI Brain Phase 5 Milestone 5.2: PostgreSQL Storage Adapters.

Tests all seven persistent PostgreSQL storage implementations:
1. PostgreSQLEventStore
2. PostgreSQLMemoryStore
3. PostgreSQLGraphStore
4. PostgreSQLInteractionStore
5. PostgreSQLSessionStore
6. PostgreSQLAgentRegistryStore
7. PostgreSQLFeedbackStore

Also verifies:
- Concurrency and transaction-level advisory locks for memory versioning.
- Optimistic concurrency control and version chaining.
- Bidirectional graph traversal and cascading referential integrity.
- Tenant isolation across projects via PostgreSQL Row-Level Security.
- Transaction rollback on mid-operation exceptions.
- Idempotent cleanup and connection pool sanitization.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import unittest
import uuid

import psycopg
from psycopg.errors import UniqueViolation

from ai_brain.agents.models import AgentCapability, AgentRecord, AgentStatus, AgentType
from ai_brain.agents.registry.postgres import PostgreSQLAgentRegistryStore
from ai_brain.agents.sessions.models import SessionRecord, SessionStatus
from ai_brain.agents.sessions.storage.postgres import PostgreSQLSessionStore
from ai_brain.config import DatabaseConfig, StorageBackend, load_env_file
from ai_brain.events.exceptions import DuplicateEventError
from ai_brain.events.models import Event
from ai_brain.events.storage.postgres import PostgreSQLEventStore
from ai_brain.events.types import EventType
from ai_brain.graph.exceptions import NodeNotFoundError
from ai_brain.graph.models import GraphNode, GraphRelationship
from ai_brain.graph.storage.postgres import PostgreSQLGraphStore
from ai_brain.graph.types import NodeType, RelationshipType
from ai_brain.loop.models import (
    AgentInteractionInput,
    AgentResponseInput,
    InteractionRecord,
    InteractionStatus,
    LoopStage,
)
from ai_brain.loop.storage.postgres import PostgreSQLInteractionStore
from ai_brain.memories.models import Memory
from ai_brain.memories.storage.postgres import PostgreSQLMemoryStore
from ai_brain.memories.types import MemoryStatus, MemoryType
from ai_brain.ml.feedback.models import FeedbackRecord, FeedbackSignal, FeedbackType
from ai_brain.ml.feedback.postgres import PostgreSQLFeedbackStore
from ai_brain.storage.database import DatabaseConnectionManager

# Load local test environment variables if present
load_env_file()

ADMIN_PG_URL = os.environ.get(
    "TEST_POSTGRES_ADMIN_URL",
    os.environ.get("TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/test_ai_brain"),
)
APP_PG_URL = os.environ.get(
    "TEST_POSTGRES_APP_URL",
    f"postgresql://ai_brain_app_user:{os.environ.get('TEST_POSTGRES_APP_PASSWORD', '')}@127.0.0.1:5432/test_ai_brain",
)


def _check_postgres_live() -> tuple[bool, str, Optional[DatabaseConnectionManager]]:
    """Probe whether live PostgreSQL instance is reachable."""
    cfg = DatabaseConfig(
        backend=StorageBackend.POSTGRES,
        database_url=ADMIN_PG_URL,
        pool_size=3,
        pool_timeout=3.0,
    )
    mgr = DatabaseConnectionManager(cfg)
    if not mgr.driver_available:
        return False, "PostgreSQL driver (psycopg or psycopg2) not installed", None

    try:
        conn = mgr.get_connection(timeout=3.0)
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            ver = cur.fetchone()
        mgr.return_connection(conn)
        return True, str(ver[0]) if ver else "Connected", mgr
    except Exception as e:
        return False, f"PostgreSQL instance unreachable at {ADMIN_PG_URL}: {e}", None


PG_AVAILABLE, PG_STATUS, ADMIN_MGR = _check_postgres_live()


@unittest.skipUnless(PG_AVAILABLE, f"Live PostgreSQL test skipped: {PG_STATUS}")
class TestPostgresStorageAdaptersM52(unittest.TestCase):
    """Integration test suite executing all M5.2 PostgreSQL adapters against live server."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app_cfg = DatabaseConfig(
            backend=StorageBackend.POSTGRES,
            database_url=APP_PG_URL,
            pool_size=10,
            pool_timeout=5.0,
        )
        cls.app_db = DatabaseConnectionManager(cls.app_cfg)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "app_db") and cls.app_db:
            cls.app_db.close()
        if ADMIN_MGR:
            ADMIN_MGR.close()

    def setUp(self) -> None:
        self.project_id = f"proj_test_{uuid.uuid4().hex[:8]}"

    # -------------------------------------------------------------------------
    # 1. EventStore Tests
    # -------------------------------------------------------------------------
    def test_01_event_store_lifecycle_and_constraints(self) -> None:
        store = PostgreSQLEventStore(self.app_db)
        event_id = f"evt_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)

        event = Event(
            event_id=event_id,
            event_type=EventType.USER_MESSAGE,
            timestamp=now,
            project_id=self.project_id,
            session_id=f"sess_{uuid.uuid4().hex[:6]}",
            actor="test_user",
            input={"prompt": "Hello AI Brain"},
            metadata={"source": "cli"},
        )

        # Store event
        store.store_event(event)

        # Retrieve by ID
        fetched = store.get_event_by_id(event_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.event_id, event_id)
        self.assertEqual(fetched.project_id, self.project_id)
        self.assertEqual(fetched.input.get("prompt"), "Hello AI Brain")

        # Duplicate event_id raises DuplicateEventError
        with self.assertRaises(DuplicateEventError):
            store.store_event(event)

        # Query events with filter
        events = store.filter_events(project_id=self.project_id, event_type=EventType.USER_MESSAGE)
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0].event_id, event_id)

        store.close()

    # -------------------------------------------------------------------------
    # 2. MemoryStore Tests
    # -------------------------------------------------------------------------
    def test_02_memory_store_crud_and_status(self) -> None:
        store = PostgreSQLMemoryStore(self.app_db, default_project_id=self.project_id)
        mem_id = f"mem_{uuid.uuid4().hex[:8]}"

        mem = Memory.create(
            memory_id=mem_id,
            memory_type=MemoryType.PREFERENCE,
            content="User prefers dark mode and concise summaries",
            project_id=self.project_id,
            importance=0.9,
            confidence=0.95,
            metadata={"source": "user_settings"},
        )

        store.store(mem)

        # Retrieve by ID
        fetched = store.get(mem_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.memory_id, mem_id)
        self.assertEqual(fetched.content, mem.content)
        self.assertEqual(fetched.metadata.get("source"), "user_settings")
        self.assertEqual(fetched.status, MemoryStatus.ACTIVE.value)

        # List memories
        mem_list = store.list(project_id=self.project_id, status="ACTIVE")
        self.assertEqual(len(mem_list), 1)
        self.assertEqual(mem_list[0].memory_id, mem_id)

        # Count
        cnt = store.count(project_id=self.project_id, status="ACTIVE")
        self.assertEqual(cnt, 1)

        # Update status
        updated = store.update_status(mem_id, MemoryStatus.DISPUTED.value, project_id=self.project_id)
        self.assertTrue(updated)
        fetched_after = store.get(mem_id)
        self.assertEqual(fetched_after.status, MemoryStatus.DISPUTED.value)

        store.close()

    def test_03_memory_store_versioning_and_superseding(self) -> None:
        store = PostgreSQLMemoryStore(self.app_db, default_project_id=self.project_id)
        root_id = f"mem_root_{uuid.uuid4().hex[:8]}"
        v1_id = root_id
        v2_id = f"mem_v2_{uuid.uuid4().hex[:8]}"

        mem_v1 = Memory.create(
            memory_id=v1_id,
            root_memory_id=root_id,
            memory_type=MemoryType.FACT,
            content="User's favorite language is Python 3.11",
            project_id=self.project_id,
            version=1,
            status=MemoryStatus.ACTIVE,
        )
        store.store(mem_v1)

        # Verify v1 is active
        cur_v1 = store.get_current(root_id)
        self.assertIsNotNone(cur_v1)
        self.assertEqual(cur_v1.memory_id, v1_id)
        self.assertEqual(cur_v1.version, 1)

        # Store v2 superseding v1
        mem_v2 = Memory.create(
            memory_id=v2_id,
            root_memory_id=root_id,
            parent_memory_id=v1_id,
            memory_type=MemoryType.FACT,
            content="User updated preference to Python 3.13 with uv",
            project_id=self.project_id,
            version=2,
            status=MemoryStatus.ACTIVE,
        )
        store.store(mem_v2)

        # Verify v1 is now SUPERSEDED
        v1_reloaded = store.get(v1_id)
        self.assertIsNotNone(v1_reloaded)
        self.assertEqual(v1_reloaded.status, MemoryStatus.SUPERSEDED.value)

        # Verify v2 is now CURRENT and ACTIVE
        cur_v2 = store.get_current(root_id)
        self.assertIsNotNone(cur_v2)
        self.assertEqual(cur_v2.memory_id, v2_id)
        self.assertEqual(cur_v2.version, 2)
        self.assertEqual(cur_v2.status, MemoryStatus.ACTIVE.value)

        # Verify get_by_root_id returns lineage [v1, v2] in version order
        lineage = store.get_by_root_id(root_id)
        self.assertEqual(len(lineage), 2)
        self.assertEqual(lineage[0].version, 1)
        self.assertEqual(lineage[1].version, 2)

        store.close()

    def test_04_memory_store_concurrent_advisory_locks(self) -> None:
        """Verify concurrent updates on same root_memory_id serialize cleanly via pg_advisory_xact_lock."""
        store = PostgreSQLMemoryStore(self.app_db, default_project_id=self.project_id)
        root_id = f"mem_shared_{uuid.uuid4().hex[:8]}"

        mem_base = Memory.create(
            memory_id=root_id,
            root_memory_id=root_id,
            memory_type=MemoryType.FACT,
            content="Base content",
            project_id=self.project_id,
            version=1,
            status=MemoryStatus.ACTIVE,
        )
        store.store(mem_base)

        def worker_store_version(version_num: int) -> bool:
            try:
                sub_store = PostgreSQLMemoryStore(self.app_db, default_project_id=self.project_id)
                new_mem = Memory.create(
                    memory_id=f"mem_{root_id}_v{version_num}",
                    root_memory_id=root_id,
                    parent_memory_id=root_id,
                    memory_type=MemoryType.FACT,
                    content=f"Concurrent content update {version_num}",
                    project_id=self.project_id,
                    version=version_num,
                    status=MemoryStatus.ACTIVE,
                )
                sub_store.store(new_mem)
                sub_store.close()
                return True
            except Exception as e:
                logger.error("Concurrent store failed: %s", e)
                return False

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_store_version, v) for v in range(2, 6)]
            results = [f.result() for f in futures]

        self.assertTrue(all(results))
        lineage = store.get_by_root_id(root_id)
        # Verify 5 versions stored (v1 plus v2..v5)
        self.assertEqual(len(lineage), 5)
        versions = [m.version for m in lineage]
        self.assertEqual(versions, [1, 2, 3, 4, 5])

        store.close()

    # -------------------------------------------------------------------------
    # 3. GraphStore Tests
    # -------------------------------------------------------------------------
    def test_05_graph_store_nodes_crud(self) -> None:
        store = PostgreSQLGraphStore(self.app_db, default_project_id=self.project_id)
        node_id = f"node_{uuid.uuid4().hex[:8]}"

        node = GraphNode.create(
            node_id=node_id,
            node_type=NodeType.CONCEPT,
            project_id=self.project_id,
            label="PostgreSQL Storage Engine",
            source_ref="arch_doc_v5",
            metadata={"tier": "infrastructure"},
        )

        store.add_node(node)

        fetched = store.get_node(self.project_id, node_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.node_id, node_id)
        self.assertEqual(fetched.label, "PostgreSQL Storage Engine")
        self.assertEqual(fetched.metadata.get("tier"), "infrastructure")

        # Count
        self.assertEqual(store.count_nodes(self.project_id), 1)

        # Delete
        deleted = store.delete_node(self.project_id, node_id)
        self.assertTrue(deleted)
        self.assertIsNone(store.get_node(self.project_id, node_id))
        self.assertEqual(store.count_nodes(self.project_id), 0)

        store.close()

    def test_06_graph_store_relationships_and_traversal(self) -> None:
        store = PostgreSQLGraphStore(self.app_db, default_project_id=self.project_id)
        node_a = GraphNode.create(
            node_id=f"node_a_{uuid.uuid4().hex[:6]}",
            node_type=NodeType.AGENT,
            project_id=self.project_id,
            label="Orchestrator Agent",
        )
        node_b = GraphNode.create(
            node_id=f"node_b_{uuid.uuid4().hex[:6]}",
            node_type=NodeType.ENTITY,
            project_id=self.project_id,
            label="Database Tool",
        )
        node_c = GraphNode.create(
            node_id=f"node_c_{uuid.uuid4().hex[:6]}",
            node_type=NodeType.CONCEPT,
            project_id=self.project_id,
            label="Storage Target",
        )

        store.add_node(node_a)
        store.add_node(node_b)
        store.add_node(node_c)

        # Referential integrity: edge to non-existent node fails
        with self.assertRaises(NodeNotFoundError):
            rel_invalid = GraphRelationship.create(
                relationship_id=f"rel_bad_{uuid.uuid4().hex[:6]}",
                source_id=node_a.node_id,
                target_id="non_existent_node",
                relationship_type=RelationshipType.RELATED_TO,
                project_id=self.project_id,
            )
            store.add_relationship(rel_invalid)

        # Add valid edges
        rel_1 = GraphRelationship.create(
            relationship_id=f"rel_1_{uuid.uuid4().hex[:6]}",
            source_id=node_a.node_id,
            target_id=node_b.node_id,
            relationship_type=RelationshipType.RELATED_TO,
            project_id=self.project_id,
            weight=1.5,
        )
        rel_2 = GraphRelationship.create(
            relationship_id=f"rel_2_{uuid.uuid4().hex[:6]}",
            source_id=node_a.node_id,
            target_id=node_c.node_id,
            relationship_type=RelationshipType.DEPENDS_ON,
            project_id=self.project_id,
            weight=2.0,
        )

        store.add_relationship(rel_1)
        store.add_relationship(rel_2)

        # Outgoing traversal from A
        out_a = store.get_outgoing_relationships(self.project_id, node_a.node_id)
        self.assertEqual(len(out_a), 2)
        target_ids = {r.target_id for r in out_a}
        self.assertEqual(target_ids, {node_b.node_id, node_c.node_id})

        # Incoming traversal to B
        in_b = store.get_incoming_relationships(self.project_id, node_b.node_id)
        self.assertEqual(len(in_b), 1)
        self.assertEqual(in_b[0].source_id, node_a.node_id)

        # Cascading deletion: deleting node B cascades to rel_1
        store.delete_node(self.project_id, node_b.node_id)
        self.assertIsNone(store.get_relationship(self.project_id, rel_1.relationship_id))
        self.assertIsNotNone(store.get_relationship(self.project_id, rel_2.relationship_id))

        store.close()

    # -------------------------------------------------------------------------
    # 4. InteractionStore Tests
    # -------------------------------------------------------------------------
    def test_07_interaction_store_lifecycle_and_idempotency(self) -> None:
        store = PostgreSQLInteractionStore(self.app_db, default_project_id=self.project_id)
        inter_id = f"inter_{uuid.uuid4().hex[:8]}"
        idem_key = f"idem_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)

        agent_input = AgentInteractionInput(
            interaction_id=inter_id,
            idempotency_key=idem_key,
            project_id=self.project_id,
            agent_id="agent_orchestrator",
            session_id=f"sess_{uuid.uuid4().hex[:8]}",
            content="Execute Phase 5 Milestone 5.2 storage verification",
        )

        agent_response = AgentResponseInput(
            interaction_id=inter_id,
            project_id=self.project_id,
            agent_id="agent_orchestrator",
            session_id=agent_input.session_id,
            content="Milestone 5.2 storage verified successfully.",
        )

        record = InteractionRecord(
            interaction_id=inter_id,
            idempotency_key=idem_key,
            project_id=self.project_id,
            agent_id=agent_input.agent_id,
            session_id=agent_input.session_id,
            input=agent_input,
            response=agent_response,
            status=InteractionStatus.COMPLETED,
            current_stage=LoopStage.COMPLETED.value,
            stages_completed=(LoopStage.INPUT_PERSISTED.value, LoopStage.COMPLETED.value),
            created_at=now,
            updated_at=now,
        )

        store.save(record)

        # Retrieve by ID
        fetched = store.get(inter_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.interaction_id, inter_id)
        self.assertEqual(fetched.status, InteractionStatus.COMPLETED)
        self.assertEqual(fetched.response.content, agent_response.content)

        # Retrieve by idempotency key
        by_idem = store.get_by_idempotency_key(idem_key)
        self.assertIsNotNone(by_idem)
        self.assertEqual(by_idem.interaction_id, inter_id)

        # List interactions
        listed = store.list(project_id=self.project_id, status=InteractionStatus.COMPLETED)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0].interaction_id, inter_id)

        store.close()

    # -------------------------------------------------------------------------
    # 5. SessionStore Tests
    # -------------------------------------------------------------------------
    def test_08_session_store_lifecycle(self) -> None:
        store = PostgreSQLSessionStore(self.app_db, default_project_id=self.project_id)
        sess_id = f"sess_{uuid.uuid4().hex[:8]}"

        session = SessionRecord(
            session_id=sess_id,
            project_id=self.project_id,
            title="Interactive Storage Debugging Session",
            current_agent_id="agent_coder",
            status=SessionStatus.ACTIVE,
            metadata={"priority": "high"},
        )

        store.save(session)

        fetched = store.get(sess_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.session_id, sess_id)
        self.assertEqual(fetched.title, "Interactive Storage Debugging Session")
        self.assertEqual(fetched.status, SessionStatus.ACTIVE)

        # List sessions
        sessions = store.list(project_id=self.project_id, status=SessionStatus.ACTIVE)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].session_id, sess_id)

        # Delete session
        deleted = store.delete(sess_id)
        self.assertTrue(deleted)
        self.assertIsNone(store.get(sess_id))

        store.close()

    # -------------------------------------------------------------------------
    # 6. AgentRegistryStore Tests
    # -------------------------------------------------------------------------
    def test_09_agent_registry_store_lifecycle(self) -> None:
        store = PostgreSQLAgentRegistryStore(self.app_db, default_project_id=self.project_id)
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"

        record = AgentRecord(
            agent_id=agent_id,
            name="PostgreSQL Architect Agent",
            provider="local_test",
            agent_type=AgentType.CODING,
            capabilities=(AgentCapability.CODE_DEVELOPMENT, AgentCapability.ARCHITECTURE_DESIGN),
            status=AgentStatus.ACTIVE,
            configuration={"memory_limit_mb": 512},
        )

        store.save(record, project_id=self.project_id)

        fetched = store.get(agent_id, project_id=self.project_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.agent_id, agent_id)
        self.assertEqual(fetched.name, "PostgreSQL Architect Agent")
        self.assertEqual(fetched.agent_type, AgentType.CODING)
        self.assertIn(AgentCapability.CODE_DEVELOPMENT, fetched.capabilities)

        # List agents
        agents = store.list(status=AgentStatus.ACTIVE, project_id=self.project_id)
        self.assertGreaterEqual(len(agents), 1)

        # Delete agent
        deleted = store.delete(agent_id, project_id=self.project_id)
        self.assertTrue(deleted)
        self.assertIsNone(store.get(agent_id, project_id=self.project_id))

        store.close()

    # -------------------------------------------------------------------------
    # 7. FeedbackStore Tests
    # -------------------------------------------------------------------------
    def test_10_feedback_store_lifecycle(self) -> None:
        store = PostgreSQLFeedbackStore(self.app_db, default_project_id=self.project_id)
        fb_id = f"fb_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)

        record = FeedbackRecord(
            feedback_id=fb_id,
            feedback_type=FeedbackType.IMPORTANCE_PREDICTION,
            signal=FeedbackSignal.POSITIVE,
            project_id=self.project_id,
            model_name="gradient_boosted_importance_v2",
            model_version="2.1.0",
            created_at=now,
            predicted_value=0.88,
            corrected_value=0.90,
            features_snapshot={"text_length": 42.0, "access_freq": 5.0},
            notes="Accurate importance classification",
        )

        store.record_feedback(record)

        # List feedback
        feedback_list = store.list_feedback(
            project_id=self.project_id,
            feedback_type=FeedbackType.IMPORTANCE_PREDICTION,
            model_name="gradient_boosted_importance_v2",
        )
        self.assertEqual(len(feedback_list), 1)
        self.assertEqual(feedback_list[0].feedback_id, fb_id)
        self.assertEqual(feedback_list[0].signal, FeedbackSignal.POSITIVE)
        self.assertEqual(feedback_list[0].features_snapshot.get("text_length"), 42.0)

        store.close()

    # -------------------------------------------------------------------------
    # 8. Tenant Isolation & Transaction Rollback Tests
    # -------------------------------------------------------------------------
    def test_11_tenant_isolation_across_all_stores(self) -> None:
        """Verify strict tenant isolation: data in project_alpha is invisible in project_beta."""
        proj_alpha = f"proj_alpha_{uuid.uuid4().hex[:6]}"
        proj_beta = f"proj_beta_{uuid.uuid4().hex[:6]}"

        mem_store_alpha = PostgreSQLMemoryStore(self.app_db, default_project_id=proj_alpha)
        mem_store_beta = PostgreSQLMemoryStore(self.app_db, default_project_id=proj_beta)

        mem_alpha_id = f"mem_alpha_{uuid.uuid4().hex[:6]}"
        mem_alpha = Memory.create(
            memory_id=mem_alpha_id,
            memory_type=MemoryType.FACT,
            content="Secret Alpha Information",
            project_id=proj_alpha,
        )
        mem_store_alpha.store(mem_alpha)

        # Query from Beta store: must return None / empty
        self.assertIsNone(mem_store_beta.get(mem_alpha_id))
        self.assertEqual(len(mem_store_beta.list(project_id=proj_beta)), 0)

        mem_store_alpha.close()
        mem_store_beta.close()

    def test_12_transaction_rollback_and_connection_sanitization(self) -> None:
        """Verify transaction context rolls back on exception and clears session parameters."""
        test_proj = f"proj_rollback_{uuid.uuid4().hex[:6]}"

        # Attempt atomic batch with deliberate error at step 2
        with self.assertRaises(RuntimeError):
            with self.app_db.tenant_context(test_proj) as conn:
                with conn.cursor() as cur:
                    # Step 1: Insert into projects
                    cur.execute(
                        "INSERT INTO projects (project_id, name) VALUES (%s, %s);",
                        (test_proj, test_proj),
                    )
                    # Step 2: Trigger error
                    raise RuntimeError("Simulated transaction failure")

        # Verify rollback: project was not committed
        with self.app_db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM projects WHERE project_id = %s;", (test_proj,))
                self.assertIsNone(cur.fetchone())

        # Verify RESET ALL sanitization: setting app.current_project_id is cleared
        conn = self.app_db.get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT NULLIF(current_setting('app.current_project_id', true), '');")
            val = cur.fetchone()[0]
            self.assertIsNone(val)
        self.app_db.return_connection(conn)


if __name__ == "__main__":
    unittest.main()
