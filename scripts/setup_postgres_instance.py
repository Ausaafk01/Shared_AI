"""Setup and manage a local, isolated PostgreSQL test instance for AI Brain."""

import os
import subprocess
import sys
import time
import urllib.request
import zipfile

PG_VERSION = "16.15-5"
ZIP_URL = f"https://get.enterprisedb.com/postgresql/postgresql-{PG_VERSION}-windows-x64-binaries.zip"
BASE_DIR = r"C:\Users\Ausaaf\AppData\Local\Programs\pgsql"
BIN_DIR = os.path.join(BASE_DIR, "pgsql", "bin")
DATA_DIR = os.path.join(BASE_DIR, "data")
ZIP_FILE = os.path.join(BASE_DIR, "pg16.zip")
LOG_FILE = os.path.join(BASE_DIR, "postgres.log")
PORT = 5432


def download_and_extract() -> None:
    if os.path.exists(os.path.join(BIN_DIR, "postgres.exe")):
        print(f"[PG-SETUP] Binaries already present at {BIN_DIR}")
        return

    os.makedirs(BASE_DIR, exist_ok=True)
    if not os.path.exists(ZIP_FILE):
        print(f"[PG-SETUP] Downloading PostgreSQL {PG_VERSION} from {ZIP_URL}...")
        req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(ZIP_FILE, "wb") as out:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = downloaded * 100 // total_size
                    print(f"\r[PG-SETUP] Downloading: {pct}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="", flush=True)
        print("\n[PG-SETUP] Download complete.")

    print(f"[PG-SETUP] Extracting {ZIP_FILE} to {BASE_DIR}...")
    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        z.extractall(BASE_DIR)
    print("[PG-SETUP] Extraction complete.")
    if os.path.exists(ZIP_FILE):
        try:
            os.remove(ZIP_FILE)
        except Exception:
            pass


def init_database() -> None:
    if os.path.exists(os.path.join(DATA_DIR, "PG_VERSION")):
        print(f"[PG-SETUP] Cluster already initialized at {DATA_DIR}")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    initdb = os.path.join(BIN_DIR, "initdb.exe")
    cmd = [initdb, "-D", DATA_DIR, "-U", "postgres", "-A", "trust", "--encoding=UTF8", "--locale=C"]
    print(f"[PG-SETUP] Running initdb: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def start_server() -> None:
    pg_ctl = os.path.join(BIN_DIR, "pg_ctl.exe")
    status_cmd = [pg_ctl, "-D", DATA_DIR, "status"]
    try:
        out = subprocess.check_output(status_cmd, stderr=subprocess.STDOUT, text=True)
        if "server is running" in out:
            print("[PG-SETUP] PostgreSQL server is already running.")
            return
    except subprocess.CalledProcessError:
        pass

    start_cmd = [pg_ctl, "-D", DATA_DIR, "-l", LOG_FILE, "-o", f"-p {PORT}", "start"]
    print(f"[PG-SETUP] Starting PostgreSQL on port {PORT}...")
    subprocess.check_call(start_cmd)

    # Wait for server to become responsive
    time.sleep(2)
    for _ in range(10):
        try:
            out = subprocess.check_output(status_cmd, stderr=subprocess.STDOUT, text=True)
            if "server is running" in out:
                print(f"[PG-SETUP] PostgreSQL successfully started on port {PORT}.")
                return
        except subprocess.CalledProcessError:
            time.sleep(1)
    raise RuntimeError("PostgreSQL failed to start within timeout.")


def create_test_databases_and_roles() -> None:
    psql = os.path.join(BIN_DIR, "psql.exe")
    app_pw = os.environ.get("TEST_POSTGRES_APP_PASSWORD", "test_app_user_secret")
    sql = f"""
    -- Create test database if not exists
    SELECT 'CREATE DATABASE test_ai_brain'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'test_ai_brain')\\gexec

    -- Create non-superuser application role for testing RLS and security boundaries
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ai_brain_app_user') THEN
            CREATE ROLE ai_brain_app_user WITH LOGIN PASSWORD '{app_pw}' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
        END IF;
    END $$;

    -- Grant permissions on test database
    GRANT ALL PRIVILEGES ON DATABASE test_ai_brain TO ai_brain_app_user;
    """
    print("[PG-SETUP] Configuring databases and roles...")
    subprocess.run([psql, "-U", "postgres", "-p", str(PORT), "-c", sql], check=True)

    # Also grant schema permissions inside test_ai_brain
    db_sql = """
    GRANT ALL ON SCHEMA public TO ai_brain_app_user;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ai_brain_app_user;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ai_brain_app_user;
    """
    subprocess.run([psql, "-U", "postgres", "-d", "test_ai_brain", "-p", str(PORT), "-c", db_sql], check=True)
    print("[PG-SETUP] Test database 'test_ai_brain' and role 'ai_brain_app_user' configured.")


def stop_server() -> None:
    pg_ctl = os.path.join(BIN_DIR, "pg_ctl.exe")
    stop_cmd = [pg_ctl, "-D", DATA_DIR, "-m", "fast", "stop"]
    try:
        subprocess.check_call(stop_cmd)
        print("[PG-SETUP] PostgreSQL server stopped.")
    except Exception as e:
        print(f"[PG-SETUP] Warning stopping server: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop_server()
    else:
        download_and_extract()
        init_database()
        start_server()
        create_test_databases_and_roles()

