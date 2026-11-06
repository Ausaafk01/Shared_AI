"""Authentication and Role-Based Access Control (RBAC) schema enhancements (M5.4).

Revision ID: 003_m5_4
Revises: 002_m5_3
Create Date: 2026-11-06 17:00:00.000000

Adds:
- users table for human/user identities
- key_prefix, name, and last_used_at columns to agent_credentials
- security_audit_events table for structured audit logging
- Appropriate indexes and grants for application database role
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "003_m5_4"
down_revision: Union[str, None] = "002_m5_3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Users table (Human / User identities)
    op.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(64) PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        name VARCHAR(255) NOT NULL,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    """)

    # 2. Agent credentials enhancements
    op.execute("""
    ALTER TABLE agent_credentials ADD COLUMN IF NOT EXISTS key_prefix VARCHAR(16);
    ALTER TABLE agent_credentials ADD COLUMN IF NOT EXISTS name VARCHAR(255) DEFAULT '';
    ALTER TABLE agent_credentials ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMPTZ;
    CREATE INDEX IF NOT EXISTS idx_agent_credentials_prefix ON agent_credentials(key_prefix);
    """)

    # 3. Security Audit Events table
    op.execute("""
    CREATE TABLE IF NOT EXISTS security_audit_events (
        event_id VARCHAR(64) PRIMARY KEY,
        event_type VARCHAR(64) NOT NULL,
        identity_id VARCHAR(64),
        identity_type VARCHAR(32),
        project_id VARCHAR(64),
        action VARCHAR(64) NOT NULL,
        status VARCHAR(32) NOT NULL,
        details JSONB NOT NULL DEFAULT '{}'::jsonb,
        ip_address VARCHAR(45),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_security_audit_project ON security_audit_events(project_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_security_audit_type ON security_audit_events(event_type, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_security_audit_identity ON security_audit_events(identity_id, created_at DESC);
    """)

    # 4. Authentication lookup function with SECURITY DEFINER
    # Safe search_path, fully-qualified table names, minimum credential information returned.
    op.execute("""
    CREATE OR REPLACE FUNCTION public.authenticate_agent_key(p_key_hash VARCHAR)
    RETURNS TABLE (
        credential_id VARCHAR(64),
        agent_id VARCHAR(64),
        project_id VARCHAR(64),
        name VARCHAR(255),
        key_hash VARCHAR(128),
        key_prefix VARCHAR(16),
        scopes JSONB,
        expires_at TIMESTAMPTZ,
        revoked_at TIMESTAMPTZ
    )
    LANGUAGE sql
    STABLE
    SECURITY DEFINER
    SET search_path = pg_catalog, public
    AS $$
        SELECT ac.credential_id, ac.agent_id, ac.project_id, ac.name, ac.key_hash, ac.key_prefix,
               ac.scopes, ac.expires_at, ac.revoked_at
        FROM public.agent_credentials ac
        WHERE ac.key_hash = p_key_hash;
    $$;

    REVOKE ALL ON FUNCTION public.authenticate_agent_key(VARCHAR) FROM PUBLIC;
    """)

    # 5. Grants for ai_brain_app_user if role exists
    op.execute("""
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ai_brain_app_user') THEN
            GRANT ALL ON TABLE users TO ai_brain_app_user;
            GRANT ALL ON TABLE security_audit_events TO ai_brain_app_user;
            GRANT ALL ON TABLE agent_credentials TO ai_brain_app_user;
            GRANT EXECUTE ON FUNCTION public.authenticate_agent_key(VARCHAR) TO ai_brain_app_user;
        END IF;
    END $$;
    """)


def downgrade() -> None:
    # 0. Drop authentication lookup function
    op.execute("DROP FUNCTION IF EXISTS public.authenticate_agent_key(VARCHAR);")

    # 1. Drop security audit events table
    op.execute("DROP TABLE IF EXISTS security_audit_events;")

    # 2. Revert agent_credentials columns
    op.execute("""
    DROP INDEX IF EXISTS idx_agent_credentials_prefix;
    ALTER TABLE agent_credentials DROP COLUMN IF EXISTS last_used_at;
    ALTER TABLE agent_credentials DROP COLUMN IF EXISTS name;
    ALTER TABLE agent_credentials DROP COLUMN IF EXISTS key_prefix;
    """)

    # 3. Drop users table
    op.execute("DROP TABLE IF EXISTS users;")

