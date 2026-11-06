"""Distributed idempotency reservations and leasing schema (M5.5).

Revision ID: 004_m5_5
Revises: 003_m5_4
Create Date: 2026-11-06 23:50:00.000000

Adds:
- idempotency_reservations table with primary key (project_id, idempotency_key)
- explicit lease expiration, heartbeats, and fencing tokens
- row-level security (RLS) enforcement and policy
- grants for ai_brain_app_user
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "004_m5_5"
down_revision: Union[str, None] = "003_m5_4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
    CREATE TABLE IF NOT EXISTS idempotency_reservations (
        project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        idempotency_key VARCHAR(128) NOT NULL,
        interaction_id VARCHAR(64) NOT NULL,
        owner_id VARCHAR(128) NOT NULL,
        status VARCHAR(32) NOT NULL DEFAULT 'IN_PROGRESS',
        fencing_token INT NOT NULL DEFAULT 1,
        lease_expires_at TIMESTAMPTZ NOT NULL,
        heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (project_id, idempotency_key)
    );
    CREATE INDEX IF NOT EXISTS idx_idempotency_res_status ON idempotency_reservations(project_id, status);
    CREATE INDEX IF NOT EXISTS idx_idempotency_res_lease ON idempotency_reservations(project_id, lease_expires_at);

    ALTER TABLE idempotency_reservations ENABLE ROW LEVEL SECURITY;
    ALTER TABLE idempotency_reservations FORCE ROW LEVEL SECURITY;
    DROP POLICY IF EXISTS idempotency_reservations_tenant_isolation ON idempotency_reservations;
    CREATE POLICY idempotency_reservations_tenant_isolation ON idempotency_reservations
        USING (project_id = NULLIF(current_setting('app.current_project_id', true), ''))
        WITH CHECK (project_id = NULLIF(current_setting('app.current_project_id', true), ''));

    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ai_brain_app_user') THEN
            GRANT ALL ON TABLE idempotency_reservations TO ai_brain_app_user;
        END IF;
    END $$;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS idempotency_reservations CASCADE;")
