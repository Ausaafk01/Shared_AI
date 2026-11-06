"""Authentication, Authorization, and Role-Based Access Control (RBAC) package for AI Brain."""

from ai_brain.auth.audit import SecurityAuditLogger, sanitize_audit_details
from ai_brain.auth.crypto import (
    KeyManager,
    KeyType,
    TokenService,
    generate_api_key,
    hash_api_key,
    verify_api_key,
)
from ai_brain.auth.exceptions import (
    AlgorithmMismatchError,
    AuthenticationError,
    AuthError,
    CredentialExpiredError,
    CredentialNotFoundError,
    CredentialRevokedError,
    InvalidTokenError,
    MembershipNotFoundError,
    PermissionDeniedError,
    PrivilegeEscalationError,
    ProjectNotFoundError,
    TokenExpiredError,
    UserNotFoundError,
)
from ai_brain.auth.models import (
    AgentCredentialRecord,
    AuthenticatedIdentity,
    CreatedApiKey,
    IdentityType,
    Permission,
    ProjectMemberRecord,
    ProjectRole,
    SecurityAuditEvent,
    UserRecord,
)
from ai_brain.auth.permissions import (
    ROLE_PERMISSIONS,
    check_scope_allows_permission,
    has_permission,
    validate_agent_scopes,
    validate_membership_transition,
)
from ai_brain.auth.service import (
    AuthenticationService,
    AuthorizationService,
    trusted_tenant_context,
)
from ai_brain.auth.store import (
    AuthStore,
    PostgreSQLAuthStore,
    SQLiteAuthStore,
)

__all__ = [
    # Models
    "IdentityType",
    "ProjectRole",
    "Permission",
    "UserRecord",
    "ProjectMemberRecord",
    "AgentCredentialRecord",
    "CreatedApiKey",
    "AuthenticatedIdentity",
    "SecurityAuditEvent",
    # Exceptions
    "AuthError",
    "AuthenticationError",
    "InvalidTokenError",
    "TokenExpiredError",
    "AlgorithmMismatchError",
    "CredentialRevokedError",
    "CredentialExpiredError",
    "PermissionDeniedError",
    "PrivilegeEscalationError",
    "UserNotFoundError",
    "ProjectNotFoundError",
    "MembershipNotFoundError",
    "CredentialNotFoundError",
    # Permissions
    "ROLE_PERMISSIONS",
    "has_permission",
    "check_scope_allows_permission",
    "validate_membership_transition",
    # Cryptography
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
    "KeyManager",
    "TokenService",
    # Audit
    "SecurityAuditLogger",
    "sanitize_audit_details",
    # Storage
    "AuthStore",
    "PostgreSQLAuthStore",
    "SQLiteAuthStore",
    # Services
    "AuthenticationService",
    "AuthorizationService",
    "trusted_tenant_context",
]

