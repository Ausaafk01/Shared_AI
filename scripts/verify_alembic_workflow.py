"""Verification script for Alembic migration workflow and schema parity."""

import os
import subprocess
import sys
import psycopg

TEST_DB_URL = "postgresql://postgres@127.0.0.1:5432/test_ai_brain"

def get_tables(conn) -> set[str]:
    cur = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
          AND table_name NOT IN ('alembic_version');
    """)
    return {row[0] for row in cur.fetchall()}


def run_alembic(command: list[str]) -> None:
    env = dict(os.environ)
    env["TEST_POSTGRES_URL"] = TEST_DB_URL
    env["DATABASE_URL"] = TEST_DB_URL
    cmd = [sys.executable, "-m", "alembic"] + command
    print(f"[ALEMBIC-RUN] {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)


def main() -> None:
    print("=== Step 1: Clean database for fresh Alembic test ===")
    with psycopg.connect(TEST_DB_URL, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        conn.execute("GRANT ALL ON SCHEMA public TO postgres; GRANT ALL ON SCHEMA public TO ai_brain_app_user;")
    print("Clean public schema prepared.")

    print("\n=== Step 2: Run 'alembic upgrade head' ===")
    run_alembic(["upgrade", "head"])

    expected_tables = {
        "projects",
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
    }

    with psycopg.connect(TEST_DB_URL) as conn:
        tables = get_tables(conn)
        print(f"Discovered tables after upgrade ({len(tables)}): {sorted(tables)}")
        for t in expected_tables:
            assert t in tables, f"Missing table {t} after upgrade"
        print("✓ All 14 authoritative tables verified after 'alembic upgrade head'.")

    print("\n=== Step 3: Run 'alembic downgrade base' ===")
    run_alembic(["downgrade", "base"])

    with psycopg.connect(TEST_DB_URL) as conn:
        tables_after_downgrade = get_tables(conn)
        print(f"Tables after downgrade: {tables_after_downgrade}")
        assert len(tables_after_downgrade) == 0, f"Expected 0 tables after downgrade, found: {tables_after_downgrade}"
        print("✓ All tables cleanly removed after 'alembic downgrade base'.")

    print("\n=== Step 4: Re-run 'alembic upgrade head' ===")
    run_alembic(["upgrade", "head"])

    with psycopg.connect(TEST_DB_URL) as conn:
        tables_reupgrade = get_tables(conn)
        for t in expected_tables:
            assert t in tables_reupgrade, f"Missing table {t} after re-upgrade"
        print("✓ All 14 tables verified after re-upgrade.")

        # Test inserting and reading a record to verify database usability
        conn.execute("INSERT INTO projects (project_id, name) VALUES ('proj-usable', 'Usability Verification Project');")
        cur = conn.execute("SELECT name FROM projects WHERE project_id = 'proj-usable';")
        assert cur.fetchone()[0] == "Usability Verification Project"
        print("✓ Database is fully usable after re-upgrade.")

    print("\n=== All Alembic Migration Lifecycle Tests PASSED ===")


if __name__ == "__main__":
    main()

