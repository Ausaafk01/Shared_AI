-- AI Brain Phase 5: Milestone 5.1 Relational Schema Initialization
-- PostgreSQL 16+ with optional pgvector support
-- Authoritative Schema: 14 core tables, partitioned events, append-only rules, and Row-Level Security

-- ============================================================================
-- Extensions (Graceful detection)
-- ============================================================================
DO $$
BEGIN
    BEGIN
        CREATE EXTENSION IF NOT EXISTS vector;
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'pgvector extension not installed on system; memory_vectors_384 will use REAL[] array type';
    END;
END $$;

-- ============================================================================
-- 1. Projects (Root Tenant Boundary)
-- ============================================================================
CREATE TABLE IF NOT EXISTS projects (
    project_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 2. Project Members (RBAC & User Access)
-- ============================================================================
CREATE TABLE IF NOT EXISTS project_members (
    member_id VARCHAR(64) PRIMARY KEY,
    project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    user_id VARCHAR(64) NOT NULL,
    role VARCHAR(32) NOT NULL DEFAULT 'viewer',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_project_member UNIQUE (project_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_project_members_lookup ON project_members(project_id, user_id);

-- ============================================================================
-- 3. Events (Partitioned by RANGE on timestamp)
-- ============================================================================
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

-- Event Partitions
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

-- Append-Only Event Immutability Function & Triggers
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

-- PostgreSQL Write-Once Rules
CREATE OR REPLACE RULE no_update_events AS ON UPDATE TO events DO INSTEAD NOTHING;
CREATE OR REPLACE RULE no_delete_events AS ON DELETE TO events DO INSTEAD NOTHING;

-- ============================================================================
-- 4. Event Lookup (Global Primary Key Uniqueness & Partition Routing)
-- ============================================================================
CREATE TABLE IF NOT EXISTS event_lookup (
    event_id VARCHAR(64) PRIMARY KEY,
    project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE RESTRICT,
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_event_lookup_project_event ON event_lookup(project_id, event_id);

-- ============================================================================
-- 5. Memories (First-Class Versioned Semantic / Working Records)
-- ============================================================================
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

-- ============================================================================
-- 6. Graph Nodes (Knowledge Graph Entities)
-- ============================================================================
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

-- ============================================================================
-- 7. Graph Relationships (Knowledge Graph Edges)
-- ============================================================================
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

-- ============================================================================
-- 8. Memory Vectors (384, 768, and 1536-Dimensional Embeddings)
-- ============================================================================
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        CREATE TABLE IF NOT EXISTS memory_vectors_384 (
            vector_id VARCHAR(64) PRIMARY KEY,
            project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            memory_id VARCHAR(64) NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
            version INT NOT NULL DEFAULT 1,
            embedding vector(384),
            model_name VARCHAR(128) NOT NULL DEFAULT 'all-MiniLM-L6-v2',
            content TEXT NOT NULL DEFAULT '',
            status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_mem_vec_384 UNIQUE (project_id, memory_id, version, model_name)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_384_lookup ON memory_vectors_384(project_id, memory_id);
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_384_status ON memory_vectors_384(project_id, status);
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_384_hnsw ON memory_vectors_384 USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

        CREATE TABLE IF NOT EXISTS memory_vectors_768 (
            vector_id VARCHAR(64) PRIMARY KEY,
            project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            memory_id VARCHAR(64) NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
            version INT NOT NULL DEFAULT 1,
            embedding vector(768),
            model_name VARCHAR(128) NOT NULL DEFAULT 'text-embedding-gecko',
            content TEXT NOT NULL DEFAULT '',
            status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_mem_vec_768 UNIQUE (project_id, memory_id, version, model_name)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_768_lookup ON memory_vectors_768(project_id, memory_id);
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_768_status ON memory_vectors_768(project_id, status);
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_768_hnsw ON memory_vectors_768 USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

        CREATE TABLE IF NOT EXISTS memory_vectors_1536 (
            vector_id VARCHAR(64) PRIMARY KEY,
            project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            memory_id VARCHAR(64) NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
            version INT NOT NULL DEFAULT 1,
            embedding vector(1536),
            model_name VARCHAR(128) NOT NULL DEFAULT 'text-embedding-3-small',
            content TEXT NOT NULL DEFAULT '',
            status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_mem_vec_1536 UNIQUE (project_id, memory_id, version, model_name)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_1536_lookup ON memory_vectors_1536(project_id, memory_id);
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_1536_status ON memory_vectors_1536(project_id, status);
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_1536_hnsw ON memory_vectors_1536 USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
    ELSE
        CREATE TABLE IF NOT EXISTS memory_vectors_384 (
            vector_id VARCHAR(64) PRIMARY KEY,
            project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            memory_id VARCHAR(64) NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
            version INT NOT NULL DEFAULT 1,
            embedding REAL[],
            model_name VARCHAR(128) NOT NULL DEFAULT 'all-MiniLM-L6-v2',
            content TEXT NOT NULL DEFAULT '',
            status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_mem_vec_384 UNIQUE (project_id, memory_id, version, model_name)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_384_lookup ON memory_vectors_384(project_id, memory_id);
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_384_status ON memory_vectors_384(project_id, status);

        CREATE TABLE IF NOT EXISTS memory_vectors_768 (
            vector_id VARCHAR(64) PRIMARY KEY,
            project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            memory_id VARCHAR(64) NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
            version INT NOT NULL DEFAULT 1,
            embedding REAL[],
            model_name VARCHAR(128) NOT NULL DEFAULT 'text-embedding-gecko',
            content TEXT NOT NULL DEFAULT '',
            status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_mem_vec_768 UNIQUE (project_id, memory_id, version, model_name)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_768_lookup ON memory_vectors_768(project_id, memory_id);
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_768_status ON memory_vectors_768(project_id, status);

        CREATE TABLE IF NOT EXISTS memory_vectors_1536 (
            vector_id VARCHAR(64) PRIMARY KEY,
            project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            memory_id VARCHAR(64) NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
            version INT NOT NULL DEFAULT 1,
            embedding REAL[],
            model_name VARCHAR(128) NOT NULL DEFAULT 'text-embedding-3-small',
            content TEXT NOT NULL DEFAULT '',
            status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_mem_vec_1536 UNIQUE (project_id, memory_id, version, model_name)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_1536_lookup ON memory_vectors_1536(project_id, memory_id);
        CREATE INDEX IF NOT EXISTS idx_memory_vectors_1536_status ON memory_vectors_1536(project_id, status);
    END IF;
END $$;

-- ============================================================================
-- 9. Agents (Registered Agent Identities)
-- ============================================================================
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

-- ============================================================================
-- 10. Agent Credentials (Cryptographic API Key Hashes & Revocations)
-- ============================================================================
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

-- ============================================================================
-- 11. Sessions (Session Continuity & Agent Coordination State)
-- ============================================================================
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

-- ============================================================================
-- 12. Interactions (BrainLoop Execution Lifecycle & Idempotency)
-- ============================================================================
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

-- ============================================================================
-- 13. Feedback Records (ML Feedback & Calibration)
-- ============================================================================
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

-- ============================================================================
-- 14. Background Tasks (Durable Asynchronous Job Queue)
-- ============================================================================
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

-- ============================================================================
-- Row-Level Security (RLS) Configuration
-- ============================================================================

-- Enable and Force RLS on all tenant-owned tables
ALTER TABLE project_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_members FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS project_members_tenant_isolation ON project_members;
CREATE POLICY project_members_tenant_isolation ON project_members
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE events ENABLE ROW LEVEL SECURITY;
ALTER TABLE events FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS events_tenant_isolation ON events;
CREATE POLICY events_tenant_isolation ON events
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE event_lookup ENABLE ROW LEVEL SECURITY;
ALTER TABLE event_lookup FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS event_lookup_tenant_isolation ON event_lookup;
CREATE POLICY event_lookup_tenant_isolation ON event_lookup
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS memories_tenant_isolation ON memories;
CREATE POLICY memories_tenant_isolation ON memories
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE graph_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_nodes FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS graph_nodes_tenant_isolation ON graph_nodes;
CREATE POLICY graph_nodes_tenant_isolation ON graph_nodes
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE graph_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_relationships FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS graph_relationships_tenant_isolation ON graph_relationships;
CREATE POLICY graph_relationships_tenant_isolation ON graph_relationships
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE memory_vectors_384 ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_vectors_384 FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS memory_vectors_384_tenant_isolation ON memory_vectors_384;
CREATE POLICY memory_vectors_384_tenant_isolation ON memory_vectors_384
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE memory_vectors_768 ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_vectors_768 FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS memory_vectors_768_tenant_isolation ON memory_vectors_768;
CREATE POLICY memory_vectors_768_tenant_isolation ON memory_vectors_768
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE memory_vectors_1536 ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_vectors_1536 FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS memory_vectors_1536_tenant_isolation ON memory_vectors_1536;
CREATE POLICY memory_vectors_1536_tenant_isolation ON memory_vectors_1536
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE agents FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agents_tenant_isolation ON agents;
CREATE POLICY agents_tenant_isolation ON agents
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE agent_credentials ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_credentials FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_credentials_tenant_isolation ON agent_credentials;
CREATE POLICY agent_credentials_tenant_isolation ON agent_credentials
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS sessions_tenant_isolation ON sessions;
CREATE POLICY sessions_tenant_isolation ON sessions
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE interactions FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS interactions_tenant_isolation ON interactions;
CREATE POLICY interactions_tenant_isolation ON interactions
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE feedback_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback_records FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS feedback_records_tenant_isolation ON feedback_records;
CREATE POLICY feedback_records_tenant_isolation ON feedback_records
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

ALTER TABLE background_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE background_tasks FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS background_tasks_tenant_isolation ON background_tasks;
CREATE POLICY background_tasks_tenant_isolation ON background_tasks
    USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
    WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));
