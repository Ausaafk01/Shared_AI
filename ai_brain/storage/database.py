"""PostgreSQL Database Connection Manager and Pool Abstraction for AI Brain.

Provides:
- Transaction management with safe commit/rollback semantics.
- Tenant isolation context manager using `SET LOCAL app.current_project_id = ...`.
- Connection pool sanitization (`RESET ALL`) on checkout/return to prevent tenant bleed.
- Dynamic driver loading supporting psycopg (v3) and psycopg2 without hard dependencies.
- Clean idempotent lifecycle cleanup.
"""

from __future__ import annotations

import contextvars
import logging
import queue
import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional, Tuple, Type
from urllib.parse import urlparse

from ai_brain.config import AppConfig, DatabaseConfig, get_config

logger = logging.getLogger(__name__)


class PostgreSQLError(Exception):
    """Base exception for PostgreSQL database errors."""
    pass


class DriverNotInstalledError(PostgreSQLError):
    """Raised when PostgreSQL operations are requested but no driver is available."""
    pass


class ConnectionUnavailableError(PostgreSQLError):
    """Raised when connection cannot be established or pool is exhausted."""
    pass


def _load_pg_driver() -> Tuple[Optional[Any], str]:
    """Dynamically attempt to import psycopg (v3) or psycopg2.

    Returns (driver_module, driver_name) or (None, '').
    """
    try:
        import psycopg
        return psycopg, "psycopg"
    except ImportError:
        pass

    try:
        import psycopg2
        return psycopg2, "psycopg2"
    except ImportError:
        pass

    return None, ""


class DatabaseConnectionManager:
    """Manages PostgreSQL connections, pooling, and transaction-safe tenant context."""

    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        self.config = config or get_config().db
        self._driver, self._driver_name = _load_pg_driver()
        self._pool: queue.Queue[Any] = queue.Queue(maxsize=self.config.pool_size + self.config.max_overflow)
        self._all_connections: list[Any] = []
        self._lock = threading.Lock()
        self._closed = False
        self._active_tenant_conn: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
            f"_active_tenant_conn_{id(self)}", default=None
        )

    @property
    def driver_available(self) -> bool:
        """Return True if a suitable PostgreSQL driver is installed."""
        return self._driver is not None

    @property
    def driver_name(self) -> str:
        """Return the name of the active driver ('psycopg', 'psycopg2', or '')."""
        return self._driver_name

    def _create_raw_connection(self) -> Any:
        """Create a new raw connection using available driver."""
        if not self.driver_available:
            raise DriverNotInstalledError(
                "PostgreSQL driver is not installed. Install 'psycopg[binary]' or 'psycopg2-binary' "
                "to enable PostgreSQL database operations."
            )

        parsed = urlparse(self.config.database_url)
        db_name = parsed.path.lstrip("/") or "postgres"
        user = parsed.username or "postgres"
        password = parsed.password or ""
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432

        connect_timeout = max(1, int(self.config.pool_timeout))

        try:
            if self._driver_name == "psycopg":
                conn = self._driver.connect(
                    dbname=db_name,
                    user=user,
                    password=password,
                    host=host,
                    port=port,
                    autocommit=False,
                    application_name=self.config.application_name,
                    connect_timeout=connect_timeout,
                )
            else:  # psycopg2
                conn = self._driver.connect(
                    dbname=db_name,
                    user=user,
                    password=password,
                    host=host,
                    port=port,
                    application_name=self.config.application_name,
                    connect_timeout=connect_timeout,
                )
                conn.autocommit = False
            return conn
        except Exception as e:
            raise ConnectionUnavailableError(
                f"Failed to connect to PostgreSQL at {self.config.sanitized_url()}: {e}"
            ) from e

    def get_connection(self, timeout: Optional[float] = None) -> Any:
        """Check out a sanitized connection from the pool or create a new one."""
        if self._closed:
            raise PostgreSQLError("DatabaseConnectionManager is closed.")

        timeout = timeout or self.config.pool_timeout

        # Try to get from pool
        try:
            conn = self._pool.get_nowait()
            # Verify connection is still alive and sanitized
            if not self._is_connection_alive(conn):
                self._safely_close_raw(conn)
                conn = self._create_raw_connection()
                with self._lock:
                    self._all_connections.append(conn)
            else:
                self._sanitize_connection(conn)
            return conn
        except queue.Empty:
            pass

        # Pool is empty: check if we can create a new connection within max capacity
        with self._lock:
            total = len(self._all_connections)
            max_allowed = self.config.pool_size + self.config.max_overflow
            if total < max_allowed:
                conn = self._create_raw_connection()
                self._all_connections.append(conn)
                return conn

        # Must wait for connection returned to pool
        try:
            conn = self._pool.get(timeout=timeout)
            if not self._is_connection_alive(conn):
                self._safely_close_raw(conn)
                conn = self._create_raw_connection()
                with self._lock:
                    self._all_connections.append(conn)
            else:
                self._sanitize_connection(conn)
            return conn
        except queue.Empty:
            raise ConnectionUnavailableError(
                f"PostgreSQL connection pool exhausted ({total} connections in use, timeout {timeout}s)"
            )

    def return_connection(self, conn: Any) -> None:
        """Sanitize connection and return it to the pool."""
        if self._closed:
            self._safely_close_raw(conn)
            return

        try:
            # Ensure any dangling transaction is rolled back and session state reset
            self._sanitize_connection(conn)
            self._pool.put_nowait(conn)
        except Exception as e:
            logger.warning("Error returning connection to pool, discarding: %s", e)
            self._safely_close_raw(conn)
            with self._lock:
                if conn in self._all_connections:
                    self._all_connections.remove(conn)

    def _sanitize_connection(self, conn: Any) -> None:
        """Sanitize connection state to prevent cross-request and cross-tenant leakage."""
        try:
            # Rollback any uncommitted transaction
            conn.rollback()
            # Reset session settings to server defaults (clears app.current_project_id and SET statements)
            with conn.cursor() as cur:
                cur.execute("RESET ALL;")
            conn.commit()
        except Exception:
            # If sanitization fails, rollback what we can
            try:
                conn.rollback()
            except Exception:
                pass

    def _is_connection_alive(self, conn: Any) -> bool:
        """Check if connection is open and responsive."""
        try:
            if hasattr(conn, "closed") and conn.closed:
                return False
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            return True
        except Exception:
            return False

    def _safely_close_raw(self, conn: Any) -> None:
        """Close raw connection without throwing exceptions."""
        try:
            if hasattr(conn, "close") and not getattr(conn, "closed", False):
                conn.close()
        except Exception:
            pass

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        """Context manager to borrow a connection and automatically return it."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)

    @contextmanager
    def transaction(self, conn: Optional[Any] = None) -> Generator[Any, None, None]:
        """Context manager providing ACID transaction semantics (COMMIT on success, ROLLBACK on error)."""
        manage_checkout = conn is None
        active_conn = self.get_connection() if manage_checkout else conn

        try:
            yield active_conn
            active_conn.commit()
        except Exception:
            try:
                active_conn.rollback()
            except Exception:
                pass
            raise
        finally:
            if manage_checkout:
                self.return_connection(active_conn)

    @contextmanager
    def tenant_context(self, project_id: str, conn: Optional[Any] = None) -> Generator[Any, None, None]:
        """Context manager scoping execution to a specific project tenant.

        Executes `SET LOCAL app.current_project_id = %s;` within a transaction.
        In PostgreSQL, `SET LOCAL` is automatically reverted upon COMMIT or ROLLBACK.
        Upon return to pool, connection sanitization runs `RESET ALL` as defense-in-depth.
        If an ambient scoped_transaction is active, reuses that connection and defers commit.
        """
        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string.")

        ambient_conn = self._active_tenant_conn.get()
        if conn is not None:
            active_conn = conn
            manage_checkout = False
            defer_commit = False
        elif ambient_conn is not None:
            active_conn = ambient_conn
            manage_checkout = False
            defer_commit = True
        else:
            active_conn = self.get_connection()
            manage_checkout = True
            defer_commit = False

        try:
            with active_conn.cursor() as cur:
                # Use parameterized execution to prevent SQL injection in session setting
                cur.execute("SELECT set_config('app.current_project_id', %s, true);", (project_id,))
            yield active_conn
            if not defer_commit:
                active_conn.commit()
        except Exception:
            if not defer_commit:
                try:
                    active_conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            if manage_checkout:
                self.return_connection(active_conn)

    @contextmanager
    def scoped_transaction(self, project_id: str, conn: Optional[Any] = None) -> Generator[Any, None, None]:
        """Context manager providing an atomic multi-operation transaction scoped to a tenant.

        All calls to tenant_context() within this block will reuse this connection,
        participating in a single transaction that commits atomically on completion
        or rolls back entirely on any error.
        """
        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string.")

        manage_checkout = conn is None
        active_conn = self.get_connection() if manage_checkout else conn
        token = self._active_tenant_conn.set(active_conn)

        try:
            with active_conn.cursor() as cur:
                cur.execute("SELECT set_config('app.current_project_id', %s, true);", (project_id,))
            yield active_conn
            active_conn.commit()
        except Exception:
            try:
                active_conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self._active_tenant_conn.reset(token)
            if manage_checkout:
                self.return_connection(active_conn)

    def ensure_project_exists(self, project_id: str) -> None:
        """Ensure parent project exists in projects table to satisfy FK constraints."""
        if not project_id:
            return
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO projects (project_id, name)
                    VALUES (%s, %s)
                    ON CONFLICT (project_id) DO NOTHING;
                    """,
                    (project_id, project_id),
                )
            conn.commit()

    def execute_script(self, sql_script: str, conn: Optional[Any] = None) -> None:
        """Execute a multi-statement SQL script within a single transaction."""
        with self.transaction(conn=conn) as c:
            with c.cursor() as cur:
                cur.execute(sql_script)

    def close(self) -> None:
        """Close all connections in the pool. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

            # Drain pool
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    self._safely_close_raw(conn)
                except queue.Empty:
                    break

            # Close all tracked connections
            for conn in self._all_connections:
                self._safely_close_raw(conn)
            self._all_connections.clear()

    def __enter__(self) -> "DatabaseConnectionManager":
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Any) -> None:
        self.close()


# Global manager singleton
_GLOBAL_DB_MANAGER: Optional[DatabaseConnectionManager] = None
_GLOBAL_DB_LOCK = threading.Lock()


def get_database_manager(config: Optional[DatabaseConfig] = None, reset: bool = False) -> DatabaseConnectionManager:
    """Retrieve global database manager singleton or initialize new one."""
    global _GLOBAL_DB_MANAGER
    with _GLOBAL_DB_LOCK:
        if _GLOBAL_DB_MANAGER is None or reset:
            if _GLOBAL_DB_MANAGER is not None:
                _GLOBAL_DB_MANAGER.close()
            _GLOBAL_DB_MANAGER = DatabaseConnectionManager(config)
        return _GLOBAL_DB_MANAGER


@contextmanager
def tenant_context(project_id: str, conn: Optional[Any] = None) -> Generator[Any, None, None]:
    """Convenience helper for tenant-scoped execution using the global database manager."""
    mgr = get_database_manager()
    with mgr.tenant_context(project_id, conn=conn) as c:
        yield c

