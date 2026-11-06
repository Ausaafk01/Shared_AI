"""Immutable domain models and enums for Authentication & Role-Based Access Control (RBAC)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple
import uuid


class IdentityType(StrEnum):
    """Classification of principal identity."""
    USER = "USER"
    AGENT = "AGENT"
    SYSTEM = "SYSTEM"


class ProjectRole(StrEnum):
    """Project-scoped roles for authorization."""
    OWNER = "OWNER"
    ADMIN = "ADMIN"
    MEMBER = "MEMBER"
    READ_ONLY = "READ_ONLY"
    AGENT = "AGENT"

    @classmethod
    def from_str(cls, val: str) -> ProjectRole:
        """Parse role case-insensitively with alias support."""
        v = (val or "").strip().upper()
        if v in ("VIEWER", "READONLY", "READ-ONLY", "READ_ONLY"):
            return cls.READ_ONLY
        if v == "OWNER":
            return cls.OWNER
        if v == "ADMIN":
            return cls.ADMIN
        if v == "MEMBER":
            return cls.MEMBER
        if v == "AGENT":
            return cls.AGENT
        raise ValueError(f"Unknown project role: {val!r}")


class Permission(StrEnum):
    """Fine-grained domain permissions for AI Brain."""
    # Project lifecycle
    PROJECT_READ = "PROJECT_READ"
    PROJECT_UPDATE = "PROJECT_UPDATE"
    PROJECT_DELETE = "PROJECT_DELETE"

    # Membership & RBAC administration
    MEMBERSHIP_READ = "MEMBERSHIP_READ"
    MEMBERSHIP_MANAGE = "MEMBERSHIP_MANAGE"

    # Agent registry
    AGENT_READ = "AGENT_READ"
    AGENT_REGISTER = "AGENT_REGISTER"
    AGENT_MANAGE = "AGENT_MANAGE"

    # Credential lifecycle
    CREDENTIAL_READ = "CREDENTIAL_READ"
    CREDENTIAL_CREATE = "CREDENTIAL_CREATE"
    CREDENTIAL_ROTATE = "CREDENTIAL_ROTATE"
    CREDENTIAL_REVOKE = "CREDENTIAL_REVOKE"

    # Memory storage
    MEMORY_READ = "MEMORY_READ"
    MEMORY_WRITE = "MEMORY_WRITE"
    MEMORY_UPDATE = "MEMORY_UPDATE"
    MEMORY_DELETE = "MEMORY_DELETE"

    # Event stream
    EVENT_READ = "EVENT_READ"
    EVENT_WRITE = "EVENT_WRITE"

    # Knowledge graph
    GRAPH_READ = "GRAPH_READ"
    GRAPH_WRITE = "GRAPH_WRITE"

    # Vector store & search
    VECTOR_SEARCH = "VECTOR_SEARCH"
    VECTOR_WRITE = "VECTOR_WRITE"

    # Session continuity
    SESSION_READ = "SESSION_READ"
    SESSION_WRITE = "SESSION_WRITE"

    # BrainLoop execution
    INTERACTION_READ = "INTERACTION_READ"
    INTERACTION_EXECUTE = "INTERACTION_EXECUTE"

    # Feedback & calibration
    FEEDBACK_READ = "FEEDBACK_READ"
    FEEDBACK_WRITE = "FEEDBACK_WRITE"


@dataclass(frozen=True)
class UserRecord:
    """Human / user identity record."""
    user_id: str
    email: str
    name: str
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.user_id or not isinstance(self.user_id, str):
            raise ValueError("user_id must be a non-empty string")
        if not self.email or "@" not in self.email:
            raise ValueError("Valid email address is required")
        if not self.name:
            raise ValueError("name must be a non-empty string")


@dataclass(frozen=True)
class ProjectMemberRecord:
    """Association of a user with a project and an assigned role."""
    member_id: str
    project_id: str
    user_id: str
    role: ProjectRole = ProjectRole.MEMBER
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.member_id:
            raise ValueError("member_id cannot be empty")
        if not self.project_id:
            raise ValueError("project_id cannot be empty")
        if not self.user_id:
            raise ValueError("user_id cannot be empty")


@dataclass(frozen=True)
class AgentCredentialRecord:
    """Stored cryptographic credential for an agent identity."""
    credential_id: str
    agent_id: str
    project_id: str
    key_hash: str
    key_prefix: str
    name: str = ""
    scopes: Tuple[str, ...] = ("read", "write")
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_revoked(self) -> bool:
        """Return True if credential has been explicitly revoked."""
        return self.revoked_at is not None

    @property
    def is_expired(self) -> bool:
        """Return True if credential has passed its expiration timestamp."""
        if self.expires_at is None:
            return False
        now = datetime.now(timezone.utc)
        exp = self.expires_at if self.expires_at.tzinfo else self.expires_at.replace(tzinfo=timezone.utc)
        return now >= exp

    @property
    def is_active(self) -> bool:
        """Return True if credential is valid, unrevoked, and unexpired."""
        return not self.is_revoked and not self.is_expired


@dataclass(frozen=True)
class CreatedApiKey:
    """Container returned ONLY once upon API key generation or rotation."""
    credential_id: str
    raw_key: str
    key_prefix: str
    agent_id: str
    project_id: str
    name: str = ""
    scopes: Tuple[str, ...] = ("read", "write")
    expires_at: Optional[datetime] = None

    def __repr__(self) -> str:
        # Never leak raw_key in string representation
        return (
            f"CreatedApiKey(credential_id={self.credential_id!r}, "
            f"key_prefix={self.key_prefix!r}, agent_id={self.agent_id!r}, "
            f"project_id={self.project_id!r}, raw_key='***REDACTED***')"
        )


@dataclass(frozen=True)
class AuthenticatedIdentity:
    """Resolved and authenticated principal identity."""
    identity_id: str
    identity_type: IdentityType
    name: str
    project_id: Optional[str] = None
    role: Optional[ProjectRole] = None
    scopes: Tuple[str, ...] = field(default_factory=tuple)
    is_authenticated: bool = True
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def has_scope(self, scope: str) -> bool:
        """Check whether identity carries a specific scope."""
        s = scope.lower().strip()
        return any(x.lower().strip() == s for x in self.scopes)


@dataclass(frozen=True)
class SecurityAuditEvent:
    """Structured audit trail record for security-sensitive operations."""
    event_id: str
    event_type: str
    identity_id: Optional[str] = None
    identity_type: Optional[str] = None
    project_id: Optional[str] = None
    action: str = ""
    status: str = "SUCCESS"
    details: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    ip_address: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

