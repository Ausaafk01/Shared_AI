"""Rotate PostgreSQL test user credentials and save to unversioned local environment."""

import os
import secrets
import psycopg

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_TEST_FILE = os.path.join(BASE_DIR, ".env.test")


def rotate() -> str:
    # Generate high-entropy 32-character random secret
    new_password = secrets.token_urlsafe(24)

    # Connect as postgres admin and alter role
    admin_url = "postgresql://postgres@127.0.0.1:5432/test_ai_brain"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(psycopg.sql.SQL("ALTER ROLE ai_brain_app_user WITH PASSWORD {};").format(psycopg.sql.Literal(new_password)))
    print("[ROTATE] PostgreSQL role 'ai_brain_app_user' password successfully rotated.")

    # Write to ignored .env.test
    with open(ENV_TEST_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Local unversioned test credentials (ignored by git)\n")
        f.write(f"TEST_POSTGRES_APP_PASSWORD={new_password}\n")
        f.write(f"TEST_POSTGRES_URL=postgresql://postgres@127.0.0.1:5432/test_ai_brain\n")
        f.write(f"TEST_POSTGRES_APP_URL=postgresql://ai_brain_app_user:{new_password}@127.0.0.1:5432/test_ai_brain\n")

    print(f"[ROTATE] Updated unversioned {ENV_TEST_FILE}")

    # Verify connection with new credentials
    test_url = f"postgresql://ai_brain_app_user:{new_password}@127.0.0.1:5432/test_ai_brain"
    with psycopg.connect(test_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            assert cur.fetchone()[0] == 1
    print("[ROTATE] Verified connection with rotated credentials.")
    return new_password


if __name__ == "__main__":
    rotate()
