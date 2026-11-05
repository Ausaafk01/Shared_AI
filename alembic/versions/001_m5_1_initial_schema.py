"""Initial relational schema migration for AI Brain (M5.1).

Revision ID: 001_m5_1
Revises: None
Create Date: 2026-11-05 12:00:00.000000

Creates authoritative 14-table schema:
1. projects
2. project_members
3. events (RANGE partitioned on timestamp) with write-once protections
4. event_lookup (global uniqueness & routing)
5. memories (versioned semantic/working records)
6. graph_nodes
7. graph_relationships
8. memory_vectors_384 (pgvector HNSW cosine index)
9. agents
10. agent_credentials (hashed API keys & revocations)
11. sessions
12. interactions
13. feedback_records
14. background_tasks (leased worker queue)
Plus Row-Level Security (RLS) on all tenant-owned tables.
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "001_m5_1"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 0. Vector extension (if available)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 1. Projects
    op.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        project_id VARCHAR(64) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)

    # 2. Project Members
    op.execute("""
    CREATE TABLE IF NOT EXISTS project_members (
        member_id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        user_id VARCHAR(64) NOT NULL,
        role VARCHAR(32) NOT NULL DEFAULT 'viewer',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_project_member UNIQUE (project_id, user_id)
    );
    CREATE INDEX IF NOT EXISTS idx_project_members_lookup ON project_members(project_id, user_id);
    """)

    # 3. Events (Partitioned by RANGE on timestamp)
    op.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_id VARCHAR(64) NOT NULL,
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE RESTRICT,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        event_type VARCHAR(64) NOT NULL,
        agent_id VARCHAR(64),
        session_id VARCHAR(64),
        idempotency_key VARCHAR(128),
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (event_id, timestamp)
    ) PARTITION BY RANGE (timestamp);

    CREATE TABLE IF NOT EXISTS events_y2025 PARTITION OF events
        FOR VALUES FROM ('2025-01-01 00:00:00+00') TO ('2026-01-01 00:00:00+00');

    CREATE TABLE IF NOT EXISTS events_y2026 PARTITION OF events
        FOR VALUES FROM ('2026-01-01 00:00:00+00') TO ('2027-01-01 00:00:00+00');

    CREATE TABLE IF NOT EXISTS events_y2027 PARTITION OF events
        FOR VALUES FROM ('2027-01-01 00:00:00+00') TO ('2028-01-01 00:00:00+00');

    CREATE TABLE IF NOT EXISTS events_default PARTITION OF events DEFAULT;

    CREATE INDEX IF NOT EXISTS idx_events_project_timestamp ON events(project_id, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_events_project_type ON events(project_id, event_type);
    CREATE INDEX IF NOT EXISTS idx_events_idempotency ON events(project_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
    """)

    # Append-only triggers and rules for events
    op.execute("""
    CREATE OR REPLACE FUNCTION prevent_event_modification()
    RETURNS TRIGGER AS $$
    BEGIN
        RAISE EXCEPTION 'Event records are append-only and cannot be updated or deleted';
    END;
    $$ LANGUAGE plpgsql;

    DROP TRIGGER IF EXISTS trg_no_update_events ON events;
    CREATE TRIGGER trg_no_update_events
    BEFORE UPDATE ON events
    FOR EACH ROW EXECUTE FUNCTION prevent_event_modification();

    DROP TRIGGER IF EXISTS trg_no_delete_events ON events;
    CREATE TRIGGER trg_no_delete_events
    BEFORE DELETE ON events
    FOR EACH ROW EXECUTE FUNCTION prevent_event_modification();

    CREATE OR REPLACE RULE no_update_events AS ON UPDATE TO events DO INSTEAD NOTHING;
    CREATE OR REPLACE RULE no_delete_events AS ON DELETE TO events DO INSTEAD NOTHING;
    """)

    # 4. Event Lookup
    op.execute("""
    CREATE TABLE IF NOT EXISTS event_lookup (
        event_id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE RESTRICT,
        timestamp TIMESTAMPTZ NOT NULL,
        event_type VARCHAR(64) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_event_lookup_project_event ON event_lookup(project_id, event_id);
    """)

    # 5. Memories
    op.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        memory_id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        version INT NOT NULL DEFAULT 1,
        status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
        tier VARCHAR(32) NOT NULL DEFAULT 'SEMANTIC',
        content TEXT NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        dispute_status VARCHAR(32) NOT NULL DEFAULT 'NONE',
        supersedes_id VARCHAR(64) REFERENCES memories(memory_id) ON DELETE SET NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_memories_project_status ON memories(project_id, status);
    CREATE INDEX IF NOT EXISTS idx_memories_project_tier ON memories(project_id, tier);
    CREATE INDEX IF NOT EXISTS idx_memories_project_dispute ON memories(project_id, dispute_status);
    CREATE INDEX IF NOT EXISTS idx_memories_project_created ON memories(project_id, created_at DESC);
    """)

    # 6. Graph Nodes
    op.execute("""
    CREATE TABLE IF NOT EXISTS graph_nodes (
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        node_id VARCHAR(64) NOT NULL,
        node_type VARCHAR(64) NOT NULL,
        name VARCHAR(255) NOT NULL,
        properties JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (project_id, node_id)
    );
    CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(project_id, node_type);
    CREATE INDEX IF NOT EXISTS idx_graph_nodes_name ON graph_nodes(project_id, name);
    """)

    # 7. Graph Relationships
    op.execute("""
    CREATE TABLE IF NOT EXISTS graph_relationships (
        relationship_id VARCHAR(64) NOT NULL,
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        source_id VARCHAR(64) NOT NULL,
        target_id VARCHAR(64) NOT NULL,
        relation_type VARCHAR(64) NOT NULL,
        weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
        properties JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (project_id, relationship_id),
        FOREIGN KEY (project_id, source_id) REFERENCES graph_nodes(project_id, node_id) ON DELETE CASCADE,
        FOREIGN KEY (project_id, target_id) REFERENCES graph_nodes(project_id, node_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_graph_rel_source ON graph_relationships(project_id, source_id);
    CREATE INDEX IF NOT EXISTS idx_graph_rel_target ON graph_relationships(project_id, target_id);
    CREATE INDEX IF NOT EXISTS idx_graph_rel_type ON graph_relationships(project_id, relation_type);
    """)

    # 8. Memory Vectors 384
    op.execute("""
    CREATE TABLE IF NOT EXISTS memory_vectors_384 (
        vector_id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        memory_id VARCHAR(64) NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
        embedding vector(384),
        model_name VARCHAR(128) NOT NULL DEFAULT 'all-MiniLM-L6-v2',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_memory_vectors_384_lookup ON memory_vectors_384(project_id, memory_id);
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
            CREATE INDEX IF NOT EXISTS idx_memory_vectors_384_hnsw ON memory_vectors_384 USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
        END IF;
    END $$;
    """)

    # 9. Agents
    op.execute("""
    CREATE TABLE IF NOT EXISTS agents (
        agent_id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        name VARCHAR(255) NOT NULL,
        role VARCHAR(128) NOT NULL,
        capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
        status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_agents_project_status ON agents(project_id, status);
    """)

    # 10. Agent Credentials
    op.execute("""
    CREATE TABLE IF NOT EXISTS agent_credentials (
        credential_id VARCHAR(64) PRIMARY KEY,
        agent_id VARCHAR(64) NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        key_hash VARCHAR(128) NOT NULL,
        scopes JSONB NOT NULL DEFAULT '["read", "write"]'::jsonb,
        revoked_at TIMESTAMPTZ,
        expires_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_agent_credentials_lookup ON agent_credentials(project_id, agent_id);
    CREATE INDEX IF NOT EXISTS idx_agent_credentials_hash ON agent_credentials(key_hash);
    CREATE INDEX IF NOT EXISTS idx_agent_credentials_revoked ON agent_credentials(project_id, revoked_at) WHERE revoked_at IS NULL;
    """)

    # 11. Sessions
    op.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        primary_agent_id VARCHAR(64) REFERENCES agents(agent_id) ON DELETE SET NULL,
        status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
        summary TEXT,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_project_status ON sessions(project_id, status);
    CREATE INDEX IF NOT EXISTS idx_sessions_project_agent ON sessions(project_id, primary_agent_id);
    """)

    # 12. Interactions
    op.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        interaction_id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        session_id VARCHAR(64) REFERENCES sessions(session_id) ON DELETE SET NULL,
        agent_id VARCHAR(64) REFERENCES agents(agent_id) ON DELETE SET NULL,
        idempotency_key VARCHAR(128),
        status VARCHAR(32) NOT NULL DEFAULT 'IN_PROGRESS',
        input_text TEXT NOT NULL,
        output_text TEXT,
        decision_records JSONB NOT NULL DEFAULT '[]'::jsonb,
        metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_interactions_project_session ON interactions(project_id, session_id);
    CREATE INDEX IF NOT EXISTS idx_interactions_project_idempotency ON interactions(project_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_interactions_project_created ON interactions(project_id, created_at DESC);
    """)

    # 13. Feedback Records
    op.execute("""
    CREATE TABLE IF NOT EXISTS feedback_records (
        feedback_id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        interaction_id VARCHAR(64) REFERENCES interactions(interaction_id) ON DELETE SET NULL,
        score DOUBLE PRECISION NOT NULL,
        aspect VARCHAR(64) NOT NULL DEFAULT 'general',
        notes TEXT,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_feedback_project_interaction ON feedback_records(project_id, interaction_id);
    CREATE INDEX IF NOT EXISTS idx_feedback_project_aspect ON feedback_records(project_id, aspect);
    """)

    # 14. Background Tasks
    op.execute("""
    CREATE TABLE IF NOT EXISTS background_tasks (
        task_id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        task_type VARCHAR(64) NOT NULL,
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
        retry_count INT NOT NULL DEFAULT 0,
        max_retries INT NOT NULL DEFAULT 3,
        locked_by VARCHAR(64),
        locked_until TIMESTAMPTZ,
        error_message TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_tasks_project_status ON background_tasks(project_id, status);
    CREATE INDEX IF NOT EXISTS idx_tasks_status_lease ON background_tasks(status, locked_until) WHERE status IN ('PENDING', 'PROCESSING');
    """)

    # Row-Level Security Policies
    tenant_tables = [
        "project_members",
        "events",
        "event_lookup",
        "memories",
        "graph_nodes",
        "graph_relationships",
        "memory_vectors_384",
        "agents",
        "agent_credentials",
        "sessions",
        "interactions",
        "feedback_records",
        "background_tasks",
    ]

    for tbl in tenant_tables:
        op.execute(f"ALTER TABLE {tbl} ENABLE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {tbl} FORCE ROW LEVEL SECURITY;")
        op.execute(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE tablename = '{tbl}' AND policyname = '{tbl}_tenant_isolation'
            ) THEN
                CREATE POLICY {tbl}_tenant_isolation ON {tbl}
                    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
                    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));
            END IF;
        END $$;
        """)


def downgrade() -> None:
    tenant_tables = [
        "background_tasks",
        "feedback_records",
        "interactions",
        "sessions",
        "agent_credentials",
        "agents",
        "memory_vectors_384",
        "graph_relationships",
        "graph_nodes",
        "memories",
        "event_lookup",
        "events",
        "project_members",
        "projects",
    ]

    for tbl in tenant_tables:
        op.execute(f"DROP POLICY IF EXISTS {tbl}_tenant_isolation ON {tbl};")

    op.execute("DROP TRIGGER IF EXISTS trg_no_update_events ON events;")
    op.execute("DROP TRIGGER IF EXISTS trg_no_delete_events ON events;")
    op.execute("DROP RULE IF EXISTS no_update_events ON events;")
    op.execute("DROP RULE IF EXISTS no_delete_events ON events;")
    op.execute("DROP FUNCTION IF EXISTS prevent_event_modification();")

    # Drop tables in reverse foreign key order
    drop_order = [
        "background_tasks",
        "feedback_records",
        "interactions",
        "sessions",
        "agent_credentials",
        "agents",
        "memory_vectors_384",
        "graph_relationships",
        "graph_nodes",
        "memories",
        "event_lookup",
        "events_y2025",
        "events_y2026",
        "events_y2027",
        "events_default",
        "events",
        "project_members",
        "projects",
    ]

    for tbl in drop_order:
        op.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE;")
