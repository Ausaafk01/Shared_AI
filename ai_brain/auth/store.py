"""Persistence storage interfaces and implementations for users, memberships, and credentials."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import logging
import sqlite3
import threading
from typing import Any, List, Optional

from ai_brain.auth.exceptions import UserNotFoundError
from ai_brain.auth.models import (
    AgentCredentialRecord,
    ProjectMemberRecord,
    ProjectRole,
    SecurityAuditEvent,
    UserRecord,
)
from ai_brain.storage.database import DatabaseConnectionManager, get_database_manager

logger = logging.getLogger(__name__)


class AuthStore(ABC):
    """Abstract interface defining required storage operations for authentication and RBAC."""

    # Users
    @abstractmethod
    def create_user(self, user: UserRecord) -> UserRecord:
        """Create and persist a new user identity."""
        pass

    @abstractmethod
    def get_user(self, user_id: str) -> Optional[UserRecord]:
        """Retrieve user identity by user_id."""
        pass

    @abstractmethod
    def get_user_by_email(self, email: str) -> Optional[UserRecord]:
        """Retrieve user identity by email address."""
        pass

    @abstractmethod
    def update_user(self, user: UserRecord) -> UserRecord:
        """Update an existing user identity."""
        pass

    @abstractmethod
    def list_users(self) -> List[UserRecord]:
        """List all registered users."""
        pass

    # Project Memberships
    @abstractmethod
    def add_project_member(self, member: ProjectMemberRecord) -> ProjectMemberRecord:
        """Add a user to a project with a specific role."""
        pass

    @abstractmethod
    def get_project_member(self, project_id: str, user_id: str) -> Optional[ProjectMemberRecord]:
        """Retrieve membership record for a given project and user."""
        pass

    @abstractmethod
    def list_project_members(self, project_id: str) -> List[ProjectMemberRecord]:
        """List all members of a specific project."""
        pass

    @abstractmethod
    def update_member_role(self, project_id: str, user_id: str, role: ProjectRole) -> bool:
        """Update a member's assigned role within a project."""
        pass

    @abstractmethod
    def remove_project_member(self, project_id: str, user_id: str) -> bool:
        """Remove a user from a project."""
        pass

    @abstractmethod
    def list_user_projects(self, user_id: str) -> List[str]:
        """List all project IDs where user has membership."""
        pass

    # Agent Credentials
    @abstractmethod
    def create_credential(self, cred: AgentCredentialRecord) -> AgentCredentialRecord:
        """Persist a new cryptographic agent credential."""
        pass

    @abstractmethod
    def get_credential(
        self,
        credential_id: str,
        project_id: Optional[str] = None,
    ) -> Optional[AgentCredentialRecord]:
        """Retrieve a credential record by credential_id."""
        pass

    @abstractmethod
    def get_credential_by_hash(self, key_hash: str) -> Optional[AgentCredentialRecord]:
        """Retrieve active or revoked credential by its SHA-256 hash."""
        pass

    @abstractmethod
    def get_credential_by_prefix(
        self,
        key_prefix: str,
        project_id: Optional[str] = None,
    ) -> Optional[AgentCredentialRecord]:
        """Retrieve credential by public key prefix."""
        pass

    @abstractmethod
    def list_agent_credentials(
        self,
        project_id: str,
        agent_id: Optional[str] = None,
    ) -> List[AgentCredentialRecord]:
        """List all credentials associated with a project or agent."""
        pass

    @abstractmethod
    def revoke_credential(
        self,
        credential_id: str,
        project_id: Optional[str] = None,
    ) -> bool:
        """Explicitly revoke a credential record."""
        pass

    @abstractmethod
    def update_credential_last_used(
        self,
        credential_id: str,
        project_id: Optional[str] = None,
    ) -> None:
        """Update the last_used_at timestamp for a credential."""
        pass

    # Security Audit Events
    @abstractmethod
    def record_audit_event(self, event: SecurityAuditEvent) -> None:
        """Persist a security audit log event."""
        pass

    @abstractmethod
    def list_audit_events(
        self,
        project_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[SecurityAuditEvent]:
        """Retrieve audit log events matching filters."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleanly close underlying resources."""
        pass


class PostgreSQLAuthStore(AuthStore):
    """PostgreSQL implementation of AuthStore backed by DatabaseConnectionManager."""

    def __init__(self, db_manager: Optional[DatabaseConnectionManager] = None) -> None:
        self.db = db_manager or get_database_manager()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PostgreSQLAuthStore is closed.")

    # -------------------------------------------------------------------------
    # Users
    # -------------------------------------------------------------------------
    def create_user(self, user: UserRecord) -> UserRecord:
        self._ensure_open()
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (user_id, email, name, is_active, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE
                    SET email = EXCLUDED.email, name = EXCLUDED.name, is_active = EXCLUDED.is_active, updated_at = NOW();
                    """,
                    (user.user_id, user.email, user.name, user.is_active, user.created_at, user.updated_at),
                )
            conn.commit()
        return user

    def get_user(self, user_id: str) -> Optional[UserRecord]:
        self._ensure_open()
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, email, name, is_active, created_at, updated_at
                    FROM users WHERE user_id = %s;
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                u_id, email, name, is_active, created_at, updated_at = row
                return UserRecord(
                    user_id=u_id,
                    email=email,
                    name=name,
                    is_active=is_active,
                    created_at=created_at,
                    updated_at=updated_at,
                )

    def get_user_by_email(self, email: str) -> Optional[UserRecord]:
        self._ensure_open()
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, email, name, is_active, created_at, updated_at
                    FROM users WHERE LOWER(email) = LOWER(%s);
                    """,
                    (email,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                u_id, em, name, is_active, created_at, updated_at = row
                return UserRecord(
                    user_id=u_id,
                    email=em,
                    name=name,
                    is_active=is_active,
                    created_at=created_at,
                    updated_at=updated_at,
                )

    def update_user(self, user: UserRecord) -> UserRecord:
        self._ensure_open()
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE users
                    SET email = %s, name = %s, is_active = %s, updated_at = NOW()
                    WHERE user_id = %s;
                    """,
                    (user.email, user.name, user.is_active, user.user_id),
                )
                if cur.rowcount == 0:
                    raise UserNotFoundError(f"User '{user.user_id}' not found.")
            conn.commit()
        return user

    def list_users(self) -> List[UserRecord]:
        self._ensure_open()
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, email, name, is_active, created_at, updated_at
                    FROM users ORDER BY created_at ASC;
                    """
                )
                rows = cur.fetchall()
                return [
                    UserRecord(
                        user_id=r[0],
                        email=r[1],
                        name=r[2],
                        is_active=r[3],
                        created_at=r[4],
                        updated_at=r[5],
                    )
                    for r in rows
                ]

    # -------------------------------------------------------------------------
    # Project Memberships
    # -------------------------------------------------------------------------
    def _ensure_project(self, project_id: str, conn: Any) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO projects (project_id, name)
                VALUES (%s, %s)
                ON CONFLICT (project_id) DO NOTHING;
                """,
                (project_id, project_id),
            )

    def add_project_member(self, member: ProjectMemberRecord) -> ProjectMemberRecord:
        self._ensure_open()
        with self.db.tenant_context(member.project_id) as conn:
            self._ensure_project(member.project_id, conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO project_members (member_id, project_id, user_id, role, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (project_id, user_id) DO UPDATE
                    SET role = EXCLUDED.role;
                    """,
                    (
                        member.member_id,
                        member.project_id,
                        member.user_id,
                        member.role.value,
                        member.created_at,
                    ),
                )
            conn.commit()
        return member

    def get_project_member(self, project_id: str, user_id: str) -> Optional[ProjectMemberRecord]:
        self._ensure_open()
        with self.db.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT member_id, project_id, user_id, role, created_at
                    FROM project_members
                    WHERE project_id = %s AND user_id = %s;
                    """,
                    (project_id, user_id),
                )
                row = cur.fetchone()
                if not row:
                    return None
                m_id, p_id, u_id, role_str, created_at = row
                return ProjectMemberRecord(
                    member_id=m_id,
                    project_id=p_id,
                    user_id=u_id,
                    role=ProjectRole.from_str(role_str),
                    created_at=created_at,
                )

    def list_project_members(self, project_id: str) -> List[ProjectMemberRecord]:
        self._ensure_open()
        with self.db.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT member_id, project_id, user_id, role, created_at
                    FROM project_members
                    WHERE project_id = %s
                    ORDER BY created_at ASC;
                    """,
                    (project_id,),
                )
                rows = cur.fetchall()
                return [
                    ProjectMemberRecord(
                        member_id=r[0],
                        project_id=r[1],
                        user_id=r[2],
                        role=ProjectRole.from_str(r[3]),
                        created_at=r[4],
                    )
                    for r in rows
                ]

    def update_member_role(self, project_id: str, user_id: str, role: ProjectRole) -> bool:
        self._ensure_open()
        with self.db.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE project_members
                    SET role = %s
                    WHERE project_id = %s AND user_id = %s;
                    """,
                    (role.value, project_id, user_id),
                )
                updated = cur.rowcount > 0
            conn.commit()
            return updated

    def remove_project_member(self, project_id: str, user_id: str) -> bool:
        self._ensure_open()
        with self.db.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM project_members
                    WHERE project_id = %s AND user_id = %s;
                    """,
                    (project_id, user_id),
                )
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted

    def list_user_projects(self, user_id: str) -> List[str]:
        self._ensure_open()
        # Admin or direct query to find all project memberships for user
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT project_id
                    FROM project_members
                    WHERE user_id = %s;
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()
                return [r[0] for r in rows]

    # -------------------------------------------------------------------------
    # Agent Credentials
    # -------------------------------------------------------------------------
    def _ensure_agent(self, project_id: str, agent_id: str, conn: Any) -> None:
        self._ensure_project(project_id, conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agents (agent_id, project_id, name, role)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (agent_id) DO NOTHING;
                """,
                (agent_id, project_id, agent_id, "GENERAL"),
            )

    def create_credential(self, cred: AgentCredentialRecord) -> AgentCredentialRecord:
        self._ensure_open()
        with self.db.tenant_context(cred.project_id) as conn:
            self._ensure_agent(cred.project_id, cred.agent_id, conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_credentials (
                        credential_id, agent_id, project_id, name, key_hash, key_prefix,
                        scopes, expires_at, revoked_at, last_used_at, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s);
                    """,
                    (
                        cred.credential_id,
                        cred.agent_id,
                        cred.project_id,
                        cred.name,
                        cred.key_hash,
                        cred.key_prefix,
                        json.dumps(list(cred.scopes)),
                        cred.expires_at,
                        cred.revoked_at,
                        cred.last_used_at,
                        cred.created_at,
                    ),
                )
            conn.commit()
        return cred

    def _row_to_cred(self, row: tuple[Any, ...]) -> AgentCredentialRecord:
        if len(row) == 9:
            (
                c_id,
                a_id,
                p_id,
                name,
                k_hash,
                k_prefix,
                scopes_raw,
                exp_at,
                rev_at,
            ) = row
            last_used = None
            created_at = datetime.now(timezone.utc)
        else:
            (
                c_id,
                a_id,
                p_id,
                name,
                k_hash,
                k_prefix,
                scopes_raw,
                exp_at,
                rev_at,
                last_used,
                created_at,
            ) = row
        scopes = tuple(scopes_raw if isinstance(scopes_raw, list) else json.loads(scopes_raw or "[]"))
        return AgentCredentialRecord(
            credential_id=c_id,
            agent_id=a_id,
            project_id=p_id,
            name=name or "",
            key_hash=k_hash,
            key_prefix=k_prefix or "",
            scopes=scopes,
            expires_at=exp_at,
            revoked_at=rev_at,
            last_used_at=last_used,
            created_at=created_at,
        )

    def get_credential(
        self,
        credential_id: str,
        project_id: Optional[str] = None,
    ) -> Optional[AgentCredentialRecord]:
        self._ensure_open()
        if project_id:
            with self.db.tenant_context(project_id) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT credential_id, agent_id, project_id, name, key_hash, key_prefix,
                               scopes, expires_at, revoked_at, last_used_at, created_at
                        FROM agent_credentials
                        WHERE credential_id = %s AND project_id = %s;
                        """,
                        (credential_id, project_id),
                    )
                    row = cur.fetchone()
                    return self._row_to_cred(row) if row else None

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT credential_id, agent_id, project_id, name, key_hash, key_prefix,
                           scopes, expires_at, revoked_at, last_used_at, created_at
                    FROM agent_credentials
                    WHERE credential_id = %s;
                    """,
                    (credential_id,),
                )
                row = cur.fetchone()
                return self._row_to_cred(row) if row else None

    def get_credential_by_hash(self, key_hash: str) -> Optional[AgentCredentialRecord]:
        self._ensure_open()
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT credential_id, agent_id, project_id, name, key_hash, key_prefix,
                               scopes, expires_at, revoked_at
                        FROM public.authenticate_agent_key(%s);
                        """,
                        (key_hash,),
                    )
                except Exception:
                    conn.rollback()
                    cur.execute(
                        """
                        SELECT credential_id, agent_id, project_id, name, key_hash, key_prefix,
                               scopes, expires_at, revoked_at
                        FROM agent_credentials
                        WHERE key_hash = %s;
                        """,
                        (key_hash,),
                    )
                row = cur.fetchone()
                return self._row_to_cred(row) if row else None

    def get_credential_by_prefix(
        self,
        key_prefix: str,
        project_id: Optional[str] = None,
    ) -> Optional[AgentCredentialRecord]:
        self._ensure_open()
        if project_id:
            with self.db.tenant_context(project_id) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT credential_id, agent_id, project_id, name, key_hash, key_prefix,
                               scopes, expires_at, revoked_at, last_used_at, created_at
                        FROM agent_credentials
                        WHERE key_prefix = %s AND project_id = %s;
                        """,
                        (key_prefix, project_id),
                    )
                    row = cur.fetchone()
                    return self._row_to_cred(row) if row else None

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT credential_id, agent_id, project_id, name, key_hash, key_prefix,
                           scopes, expires_at, revoked_at, last_used_at, created_at
                    FROM agent_credentials
                    WHERE key_prefix = %s;
                    """,
                    (key_prefix,),
                )
                row = cur.fetchone()
                return self._row_to_cred(row) if row else None

    def list_agent_credentials(
        self,
        project_id: str,
        agent_id: Optional[str] = None,
    ) -> List[AgentCredentialRecord]:
        self._ensure_open()
        with self.db.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT credential_id, agent_id, project_id, name, key_hash, key_prefix,
                           scopes, expires_at, revoked_at, last_used_at, created_at
                    FROM agent_credentials
                    WHERE project_id = %s
                """
                params: List[Any] = [project_id]
                if agent_id:
                    sql += " AND agent_id = %s"
                    params.append(agent_id)
                sql += " ORDER BY created_at DESC;"
                cur.execute(sql, params)
                return [self._row_to_cred(r) for r in cur.fetchall()]

    def revoke_credential(
        self,
        credential_id: str,
        project_id: Optional[str] = None,
    ) -> bool:
        self._ensure_open()
        if project_id:
            with self.db.tenant_context(project_id) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE agent_credentials
                        SET revoked_at = NOW()
                        WHERE credential_id = %s AND project_id = %s AND revoked_at IS NULL;
                        """,
                        (credential_id, project_id),
                    )
                    revoked = cur.rowcount > 0
                conn.commit()
                return revoked

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE agent_credentials
                    SET revoked_at = NOW()
                    WHERE credential_id = %s AND revoked_at IS NULL;
                    """,
                    (credential_id,),
                )
                revoked = cur.rowcount > 0
            conn.commit()
            return revoked

    def update_credential_last_used(
        self,
        credential_id: str,
        project_id: Optional[str] = None,
    ) -> None:
        self._ensure_open()
        if project_id:
            with self.db.tenant_context(project_id) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE agent_credentials
                        SET last_used_at = NOW()
                        WHERE credential_id = %s AND project_id = %s;
                        """,
                        (credential_id, project_id),
                    )
                conn.commit()
                return

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE agent_credentials
                    SET last_used_at = NOW()
                    WHERE credential_id = %s;
                    """,
                    (credential_id,),
                )
            conn.commit()

    # -------------------------------------------------------------------------
    # Security Audit Events
    # -------------------------------------------------------------------------
    def record_audit_event(self, event: SecurityAuditEvent) -> None:
        self._ensure_open()
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO security_audit_events (
                        event_id, event_type, identity_id, identity_type, project_id,
                        action, status, details, ip_address, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s);
                    """,
                    (
                        event.event_id,
                        event.event_type,
                        event.identity_id,
                        event.identity_type,
                        event.project_id,
                        event.action,
                        event.status,
                        json.dumps(dict(event.details)),
                        event.ip_address,
                        event.timestamp,
                    ),
                )
            conn.commit()

    def list_audit_events(
        self,
        project_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[SecurityAuditEvent]:
        self._ensure_open()
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT event_id, event_type, identity_id, identity_type, project_id,
                           action, status, details, ip_address, created_at
                    FROM security_audit_events
                    WHERE 1=1
                """
                params: List[Any] = []
                if project_id:
                    sql += " AND project_id = %s"
                    params.append(project_id)
                if event_type:
                    sql += " AND event_type = %s"
                    params.append(event_type)
                sql += " ORDER BY created_at DESC LIMIT %s;"
                params.append(limit)

                cur.execute(sql, params)
                rows = cur.fetchall()
                events: List[SecurityAuditEvent] = []
                for r in rows:
                    details = r[7] if isinstance(r[7], dict) else json.loads(r[7] or "{}")
                    events.append(
                        SecurityAuditEvent(
                            event_id=r[0],
                            event_type=r[1],
                            identity_id=r[2],
                            identity_type=r[3],
                            project_id=r[4],
                            action=r[5],
                            status=r[6],
                            details=details,
                            ip_address=r[8],
                            timestamp=r[9],
                        )
                    )
                return events

    def close(self) -> None:
        self._closed = True


class SQLiteAuthStore(AuthStore):
    """In-memory or local SQLite persistent implementation for offline compatibility."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._closed = False
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS project_members (
                    member_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'MEMBER',
                    created_at TEXT NOT NULL,
                    UNIQUE (project_id, user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_pm_lookup ON project_members(project_id, user_id);

                CREATE TABLE IF NOT EXISTS agent_credentials (
                    credential_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    key_hash TEXT NOT NULL,
                    key_prefix TEXT,
                    scopes TEXT NOT NULL DEFAULT '["read", "write"]',
                    expires_at TEXT,
                    revoked_at TEXT,
                    last_used_at TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ac_hash ON agent_credentials(key_hash);
                CREATE INDEX IF NOT EXISTS idx_ac_prefix ON agent_credentials(key_prefix);

                CREATE TABLE IF NOT EXISTS security_audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    identity_id TEXT,
                    identity_type TEXT,
                    project_id TEXT,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '{}',
                    ip_address TEXT,
                    created_at TEXT NOT NULL
                );
            """)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SQLiteAuthStore is closed.")

    # -------------------------------------------------------------------------
    # Users
    # -------------------------------------------------------------------------
    def create_user(self, user: UserRecord) -> UserRecord:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO users (user_id, email, name, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (user_id) DO UPDATE SET
                    email = excluded.email,
                    name = excluded.name,
                    is_active = excluded.is_active,
                    updated_at = excluded.updated_at;
                """,
                (
                    user.user_id,
                    user.email,
                    user.name,
                    1 if user.is_active else 0,
                    user.created_at.isoformat(),
                    user.updated_at.isoformat(),
                ),
            )
        return user

    def get_user(self, user_id: str) -> Optional[UserRecord]:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM users WHERE user_id = ?;", (user_id,))
            row = cur.fetchone()
            if not row:
                return None
            return UserRecord(
                user_id=row["user_id"],
                email=row["email"],
                name=row["name"],
                is_active=bool(row["is_active"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def get_user_by_email(self, email: str) -> Optional[UserRecord]:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(?);", (email,))
            row = cur.fetchone()
            if not row:
                return None
            return UserRecord(
                user_id=row["user_id"],
                email=row["email"],
                name=row["name"],
                is_active=bool(row["is_active"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def update_user(self, user: UserRecord) -> UserRecord:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE users
                SET email = ?, name = ?, is_active = ?, updated_at = ?
                WHERE user_id = ?;
                """,
                (
                    user.email,
                    user.name,
                    1 if user.is_active else 0,
                    datetime.now(timezone.utc).isoformat(),
                    user.user_id,
                ),
            )
            if cur.rowcount == 0:
                raise UserNotFoundError(f"User '{user.user_id}' not found.")
        return user

    def list_users(self) -> List[UserRecord]:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM users ORDER BY created_at ASC;")
            rows = cur.fetchall()
            return [
                UserRecord(
                    user_id=r["user_id"],
                    email=r["email"],
                    name=r["name"],
                    is_active=bool(r["is_active"]),
                    created_at=datetime.fromisoformat(r["created_at"]),
                    updated_at=datetime.fromisoformat(r["updated_at"]),
                )
                for r in rows
            ]

    # -------------------------------------------------------------------------
    # Project Memberships
    # -------------------------------------------------------------------------
    def add_project_member(self, member: ProjectMemberRecord) -> ProjectMemberRecord:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO project_members (member_id, project_id, user_id, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (project_id, user_id) DO UPDATE SET
                    role = excluded.role;
                """,
                (
                    member.member_id,
                    member.project_id,
                    member.user_id,
                    member.role.value,
                    member.created_at.isoformat(),
                ),
            )
        return member

    def get_project_member(self, project_id: str, user_id: str) -> Optional[ProjectMemberRecord]:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM project_members WHERE project_id = ? AND user_id = ?;",
                (project_id, user_id),
            )
            row = cur.fetchone()
            if not row:
                return None
            return ProjectMemberRecord(
                member_id=row["member_id"],
                project_id=row["project_id"],
                user_id=row["user_id"],
                role=ProjectRole.from_str(row["role"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    def list_project_members(self, project_id: str) -> List[ProjectMemberRecord]:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM project_members WHERE project_id = ? ORDER BY created_at ASC;",
                (project_id,),
            )
            rows = cur.fetchall()
            return [
                ProjectMemberRecord(
                    member_id=r["member_id"],
                    project_id=r["project_id"],
                    user_id=r["user_id"],
                    role=ProjectRole.from_str(r["role"]),
                    created_at=datetime.fromisoformat(r["created_at"]),
                )
                for r in rows
            ]

    def update_member_role(self, project_id: str, user_id: str, role: ProjectRole) -> bool:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE project_members SET role = ? WHERE project_id = ? AND user_id = ?;",
                (role.value, project_id, user_id),
            )
            return cur.rowcount > 0

    def remove_project_member(self, project_id: str, user_id: str) -> bool:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "DELETE FROM project_members WHERE project_id = ? AND user_id = ?;",
                (project_id, user_id),
            )
            return cur.rowcount > 0

    def list_user_projects(self, user_id: str) -> List[str]:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT project_id FROM project_members WHERE user_id = ?;", (user_id,))
            return [r["project_id"] for r in cur.fetchall()]

    # -------------------------------------------------------------------------
    # Agent Credentials
    # -------------------------------------------------------------------------
    def _row_to_cred(self, row: sqlite3.Row) -> AgentCredentialRecord:
        scopes = tuple(json.loads(row["scopes"] or "[]"))
        exp = datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
        rev = datetime.fromisoformat(row["revoked_at"]) if row["revoked_at"] else None
        last = datetime.fromisoformat(row["last_used_at"]) if row["last_used_at"] else None
        return AgentCredentialRecord(
            credential_id=row["credential_id"],
            agent_id=row["agent_id"],
            project_id=row["project_id"],
            name=row["name"] or "",
            key_hash=row["key_hash"],
            key_prefix=row["key_prefix"] or "",
            scopes=scopes,
            expires_at=exp,
            revoked_at=rev,
            last_used_at=last,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def create_credential(self, cred: AgentCredentialRecord) -> AgentCredentialRecord:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO agent_credentials (
                    credential_id, agent_id, project_id, name, key_hash, key_prefix,
                    scopes, expires_at, revoked_at, last_used_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    cred.credential_id,
                    cred.agent_id,
                    cred.project_id,
                    cred.name,
                    cred.key_hash,
                    cred.key_prefix,
                    json.dumps(list(cred.scopes)),
                    cred.expires_at.isoformat() if cred.expires_at else None,
                    cred.revoked_at.isoformat() if cred.revoked_at else None,
                    cred.last_used_at.isoformat() if cred.last_used_at else None,
                    cred.created_at.isoformat(),
                ),
            )
        return cred

    def get_credential(
        self,
        credential_id: str,
        project_id: Optional[str] = None,
    ) -> Optional[AgentCredentialRecord]:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            if project_id:
                cur.execute(
                    "SELECT * FROM agent_credentials WHERE credential_id = ? AND project_id = ?;",
                    (credential_id, project_id),
                )
            else:
                cur.execute("SELECT * FROM agent_credentials WHERE credential_id = ?;", (credential_id,))
            row = cur.fetchone()
            return self._row_to_cred(row) if row else None

    def get_credential_by_hash(self, key_hash: str) -> Optional[AgentCredentialRecord]:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM agent_credentials WHERE key_hash = ?;", (key_hash,))
            row = cur.fetchone()
            return self._row_to_cred(row) if row else None

    def get_credential_by_prefix(
        self,
        key_prefix: str,
        project_id: Optional[str] = None,
    ) -> Optional[AgentCredentialRecord]:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            if project_id:
                cur.execute(
                    "SELECT * FROM agent_credentials WHERE key_prefix = ? AND project_id = ?;",
                    (key_prefix, project_id),
                )
            else:
                cur.execute("SELECT * FROM agent_credentials WHERE key_prefix = ?;", (key_prefix,))
            row = cur.fetchone()
            return self._row_to_cred(row) if row else None

    def list_agent_credentials(
        self,
        project_id: str,
        agent_id: Optional[str] = None,
    ) -> List[AgentCredentialRecord]:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            if agent_id:
                cur.execute(
                    "SELECT * FROM agent_credentials WHERE project_id = ? AND agent_id = ? ORDER BY created_at DESC;",
                    (project_id, agent_id),
                )
            else:
                cur.execute(
                    "SELECT * FROM agent_credentials WHERE project_id = ? ORDER BY created_at DESC;",
                    (project_id,),
                )
            return [self._row_to_cred(r) for r in cur.fetchall()]

    def revoke_credential(
        self,
        credential_id: str,
        project_id: Optional[str] = None,
    ) -> bool:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            if project_id:
                cur.execute(
                    """
                    UPDATE agent_credentials
                    SET revoked_at = ?
                    WHERE credential_id = ? AND project_id = ? AND revoked_at IS NULL;
                    """,
                    (datetime.now(timezone.utc).isoformat(), credential_id, project_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE agent_credentials
                    SET revoked_at = ?
                    WHERE credential_id = ? AND revoked_at IS NULL;
                    """,
                    (datetime.now(timezone.utc).isoformat(), credential_id),
                )
            return cur.rowcount > 0

    def update_credential_last_used(
        self,
        credential_id: str,
        project_id: Optional[str] = None,
    ) -> None:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            if project_id:
                cur.execute(
                    "UPDATE agent_credentials SET last_used_at = ? WHERE credential_id = ? AND project_id = ?;",
                    (datetime.now(timezone.utc).isoformat(), credential_id, project_id),
                )
            else:
                cur.execute(
                    "UPDATE agent_credentials SET last_used_at = ? WHERE credential_id = ?;",
                    (datetime.now(timezone.utc).isoformat(), credential_id),
                )

    # -------------------------------------------------------------------------
    # Security Audit Events
    # -------------------------------------------------------------------------
    def record_audit_event(self, event: SecurityAuditEvent) -> None:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO security_audit_events (
                    event_id, event_type, identity_id, identity_type, project_id,
                    action, status, details, ip_address, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    event.event_id,
                    event.event_type,
                    event.identity_id,
                    event.identity_type,
                    event.project_id,
                    event.action,
                    event.status,
                    json.dumps(dict(event.details)),
                    event.ip_address,
                    event.timestamp.isoformat(),
                ),
            )

    def list_audit_events(
        self,
        project_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[SecurityAuditEvent]:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            sql = "SELECT * FROM security_audit_events WHERE 1=1"
            params: List[Any] = []
            if project_id:
                sql += " AND project_id = ?"
                params.append(project_id)
            if event_type:
                sql += " AND event_type = ?"
                params.append(event_type)
            sql += " ORDER BY created_at DESC LIMIT ?;"
            params.append(limit)

            cur.execute(sql, params)
            rows = cur.fetchall()
            return [
                SecurityAuditEvent(
                    event_id=r["event_id"],
                    event_type=r["event_type"],
                    identity_id=r["identity_id"],
                    identity_type=r["identity_type"],
                    project_id=r["project_id"],
                    action=r["action"],
                    status=r["status"],
                    details=json.loads(r["details"] or "{}"),
                    ip_address=r["ip_address"],
                    timestamp=datetime.fromisoformat(r["created_at"]),
                )
                for r in rows
            ]

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._conn.close()
