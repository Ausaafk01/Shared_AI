"""Add pgvector 768 and 1536 dimensional tables and enhance 384 table (M5.3).

Revision ID: 002_m5_3
Revises: 001_m5_1
Create Date: 2026-11-05 21:00:00.000000

Expands vector storage support across:
- memory_vectors_384 (all-MiniLM-L6-v2)
- memory_vectors_768 (text-embedding-gecko / bert-base)
- memory_vectors_1536 (text-embedding-3-small / ada-002)

Adds HNSW cosine indexes, version tracking, content preservation,
and Row-Level Security on all vector tables.
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "002_m5_3"
down_revision: Union[str, None] = "001_m5_1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable pgvector extension if available
    op.execute("""
    DO $$
    BEGIN
        BEGIN
            CREATE EXTENSION IF NOT EXISTS vector;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'pgvector extension not installed; falling back to REAL[] arrays';
        END;
    END $$;
    """)

    # 2. Enhance memory_vectors_384 with version, content, status, metadata columns if missing
    op.execute("""
    ALTER TABLE memory_vectors_384 ADD COLUMN IF NOT EXISTS version INT NOT NULL DEFAULT 1;
    ALTER TABLE memory_vectors_384 ADD COLUMN IF NOT EXISTS content TEXT NOT NULL DEFAULT '';
    ALTER TABLE memory_vectors_384 ADD COLUMN IF NOT EXISTS status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE';
    ALTER TABLE memory_vectors_384 ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
    CREATE INDEX IF NOT EXISTS idx_memory_vectors_384_status ON memory_vectors_384(project_id, status);
    """)

    # If pgvector extension is present, convert embedding to vector(384) and add HNSW index
    op.execute("""
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
            BEGIN
                ALTER TABLE memory_vectors_384 ALTER COLUMN embedding TYPE vector(384) USING embedding::vector;
            EXCEPTION WHEN OTHERS THEN
                NULL;
            END;
            CREATE INDEX IF NOT EXISTS idx_memory_vectors_384_hnsw
                ON memory_vectors_384 USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
        END IF;
    END $$;
    """)

    # 3. Create memory_vectors_768
    op.execute("""
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
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
            CREATE INDEX IF NOT EXISTS idx_memory_vectors_768_hnsw
                ON memory_vectors_768 USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
        ELSE
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
        END IF;
    END $$;
    """)

    # 4. Create memory_vectors_1536
    op.execute("""
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
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
            CREATE INDEX IF NOT EXISTS idx_memory_vectors_1536_hnsw
                ON memory_vectors_1536 USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
        ELSE
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
    """)

    # 5. RLS Policies on 768 and 1536
    for tbl in ["memory_vectors_768", "memory_vectors_1536"]:
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
    for tbl in ["memory_vectors_1536", "memory_vectors_768"]:
        op.execute(f"DROP POLICY IF EXISTS {tbl}_tenant_isolation ON {tbl};")
        op.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE;")

    op.execute("DROP INDEX IF EXISTS idx_memory_vectors_384_hnsw;")
    op.execute("DROP INDEX IF EXISTS idx_memory_vectors_384_status;")
    op.execute("ALTER TABLE memory_vectors_384 DROP COLUMN IF EXISTS version;")
    op.execute("ALTER TABLE memory_vectors_384 DROP COLUMN IF EXISTS content;")
    op.execute("ALTER TABLE memory_vectors_384 DROP COLUMN IF EXISTS status;")
    op.execute("ALTER TABLE memory_vectors_384 DROP COLUMN IF EXISTS metadata;")
