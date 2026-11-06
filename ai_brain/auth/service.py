"""Unified Authentication, Authorization (RBAC), and Tenant Context Services."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Generator, List, Mapping, Optional, Sequence, Tuple
import uuid

from ai_brain.auth.audit import SecurityAuditLogger
from ai_brain.auth.crypto import KeyManager, TokenService, generate_api_key, hash_api_key
from ai_brain.auth.exceptions import (
    AuthenticationError,
    CredentialExpiredError,
    CredentialNotFoundError,
    CredentialRevokedError,
    MembershipNotFoundError,
    PermissionDeniedError,
    PrivilegeEscalationError,
    ProjectNotFoundError,
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
    UserRecord,
)
from ai_brain.auth.permissions import (
    has_permission,
    validate_agent_scopes,
    validate_membership_transition,
)
from ai_brain.auth.store import AuthStore, PostgreSQLAuthStore, SQLiteAuthStore
from ai_brain.storage.database import DatabaseConnectionManager

logger = logging.getLogger(__name__)


class AuthenticationService:
    """Core service for managing and validating API keys and JWT access tokens."""

    def __init__(
        self,
        store: AuthStore,
        token_service: TokenService,
        audit_logger: Optional[SecurityAuditLogger] = None,
    ) -> None:
        self.store = store
        self.token_service = token_service
        self.audit = audit_logger or SecurityAuditLogger(
            persistence_handler=lambda e: self.store.record_audit_event(e)
        )

    # -------------------------------------------------------------------------
    # API Key Lifecycle
    # -------------------------------------------------------------------------
    def create_agent_api_key(
        self,
        agent_id: str,
        project_id: str,
        *,
        name: str = "",
        scopes: Sequence[str] = ("read", "write"),
        expires_in_seconds: Optional[int] = None,
        ip_address: Optional[str] = None,
    ) -> CreatedApiKey:
        """Create a new agent API key. Raw key is returned once and never stored."""
        if not agent_id or not project_id:
            raise ValueError("agent_id and project_id are required")

        valid_scopes = validate_agent_scopes(scopes)
        raw_key, key_prefix, key_hash = generate_api_key("agy")
        cred_id = f"cred_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=expires_in_seconds) if expires_in_seconds else None

        cred_record = AgentCredentialRecord(
            credential_id=cred_id,
            agent_id=agent_id,
            project_id=project_id,
            name=name or f"Key for {agent_id}",
            key_hash=key_hash,
            key_prefix=key_prefix,
            scopes=valid_scopes,
            expires_at=expires_at,
            created_at=now,
        )

        self.store.create_credential(cred_record)

        self.audit.record_event(
            "CREDENTIAL_CREATED",
            identity_id=agent_id,
            identity_type=IdentityType.AGENT.value,
            project_id=project_id,
            action="CREATE_KEY",
            status="SUCCESS",
            details={
                "credential_id": cred_id,
                "key_prefix": key_prefix,
                "scopes": list(scopes),
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
            ip_address=ip_address,
        )

        return CreatedApiKey(
            credential_id=cred_id,
            raw_key=raw_key,
            key_prefix=key_prefix,
            agent_id=agent_id,
            project_id=project_id,
            name=name,
            scopes=tuple(scopes),
            expires_at=expires_at,
        )

    def rotate_agent_api_key(
        self,
        credential_id: str,
        *,
        expires_in_seconds: Optional[int] = None,
        ip_address: Optional[str] = None,
    ) -> CreatedApiKey:
        """Atomically rotate an agent API key: revoke old key, generate new key for same agent/project."""
        old_cred = self.store.get_credential(credential_id)
        if not old_cred:
            raise CredentialNotFoundError(f"Credential '{credential_id}' not found.")

        # Revoke old credential
        self.store.revoke_credential(credential_id, project_id=old_cred.project_id)

        # Generate replacement
        new_key = self.create_agent_api_key(
            agent_id=old_cred.agent_id,
            project_id=old_cred.project_id,
            name=old_cred.name,
            scopes=old_cred.scopes,
            expires_in_seconds=expires_in_seconds,
            ip_address=ip_address,
        )

        self.audit.record_event(
            "CREDENTIAL_ROTATED",
            identity_id=old_cred.agent_id,
            identity_type=IdentityType.AGENT.value,
            project_id=old_cred.project_id,
            action="ROTATE_KEY",
            status="SUCCESS",
            details={
                "old_credential_id": credential_id,
                "new_credential_id": new_key.credential_id,
                "new_key_prefix": new_key.key_prefix,
            },
            ip_address=ip_address,
        )

        return new_key

    def revoke_agent_api_key(
        self,
        credential_id: str,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Explicitly revoke an active API key."""
        old_cred = self.store.get_credential(credential_id)
        if not old_cred:
            raise CredentialNotFoundError(f"Credential '{credential_id}' not found.")

        revoked = self.store.revoke_credential(credential_id, project_id=old_cred.project_id)
        if revoked:
            self.audit.record_event(
                "CREDENTIAL_REVOKED",
                identity_id=old_cred.agent_id,
                identity_type=IdentityType.AGENT.value,
                project_id=old_cred.project_id,
                action="REVOKE_KEY",
                status="SUCCESS",
                details={
                    "credential_id": credential_id,
                    "key_prefix": old_cred.key_prefix,
                },
                ip_address=ip_address,
            )
        return revoked

    # -------------------------------------------------------------------------
    # Authentication Operations
    # -------------------------------------------------------------------------
    def authenticate_api_key(
        self,
        raw_key: str,
        ip_address: Optional[str] = None,
    ) -> AuthenticatedIdentity:
        """Verify an API key, check revocation/expiration, and return an AuthenticatedIdentity."""
        if not raw_key or not isinstance(raw_key, str):
            self.audit.record_event(
                "AUTH_FAILURE",
                action="AUTHENTICATE_KEY",
                status="FAILURE",
                details={"reason": "malformed_key"},
                ip_address=ip_address,
            )
            raise AuthenticationError()

        key_hash = hash_api_key(raw_key)
        cred = self.store.get_credential_by_hash(key_hash)

        if not cred:
            self.audit.record_event(
                "AUTH_FAILURE",
                action="AUTHENTICATE_KEY",
                status="FAILURE",
                details={"reason": "unknown_key"},
                ip_address=ip_address,
            )
            raise AuthenticationError()

        if cred.is_revoked:
            self.audit.record_event(
                "AUTH_FAILURE",
                identity_id=cred.agent_id,
                identity_type=IdentityType.AGENT.value,
                project_id=cred.project_id,
                action="AUTHENTICATE_KEY",
                status="FAILURE",
                details={"reason": "revoked_credential", "credential_id": cred.credential_id},
                ip_address=ip_address,
            )
            raise CredentialRevokedError()

        if cred.is_expired:
            self.audit.record_event(
                "AUTH_FAILURE",
                identity_id=cred.agent_id,
                identity_type=IdentityType.AGENT.value,
                project_id=cred.project_id,
                action="AUTHENTICATE_KEY",
                status="FAILURE",
                details={"reason": "expired_credential", "credential_id": cred.credential_id},
                ip_address=ip_address,
            )
            raise CredentialExpiredError()

        # Update last used timestamp
        self.store.update_credential_last_used(cred.credential_id, project_id=cred.project_id)

        self.audit.record_event(
            "AUTH_SUCCESS",
            identity_id=cred.agent_id,
            identity_type=IdentityType.AGENT.value,
            project_id=cred.project_id,
            action="AUTHENTICATE_KEY",
            status="SUCCESS",
            details={"credential_id": cred.credential_id, "key_prefix": cred.key_prefix},
            ip_address=ip_address,
        )

        return AuthenticatedIdentity(
            identity_id=cred.agent_id,
            identity_type=IdentityType.AGENT,
            name=cred.name or cred.agent_id,
            project_id=cred.project_id,
            role=ProjectRole.AGENT,
            scopes=cred.scopes,
            is_authenticated=True,
            metadata={"credential_id": cred.credential_id},
        )

    def authenticate_token(
        self,
        token: str,
        ip_address: Optional[str] = None,
    ) -> AuthenticatedIdentity:
        """Validate a signed JWT token and return an AuthenticatedIdentity."""
        try:
            payload = self.token_service.decode_and_validate_token(token)
        except Exception as e:
            self.audit.record_event(
                "AUTH_FAILURE",
                action="AUTHENTICATE_TOKEN",
                status="FAILURE",
                details={"error": str(e)},
                ip_address=ip_address,
            )
            raise

        sub = payload["sub"]
        typ_str = payload.get("typ", IdentityType.USER.value).upper()
        identity_type = IdentityType(typ_str) if typ_str in IdentityType.__members__ else IdentityType.USER
        name = payload.get("name", sub)
        scopes = tuple(payload.get("scopes", []))

        # If user identity, ensure user exists and is active in store
        if identity_type == IdentityType.USER:
            user = self.store.get_user(sub)
            if not user or not user.is_active:
                self.audit.record_event(
                    "AUTH_FAILURE",
                    identity_id=sub,
                    identity_type=IdentityType.USER.value,
                    action="AUTHENTICATE_TOKEN",
                    status="FAILURE",
                    details={"reason": "inactive_or_missing_user"},
                    ip_address=ip_address,
                )
                raise AuthenticationError("User identity is inactive or does not exist.")
            name = user.name

        self.audit.record_event(
            "AUTH_SUCCESS",
            identity_id=sub,
            identity_type=identity_type.value,
            action="AUTHENTICATE_TOKEN",
            status="SUCCESS",
            details={"jti": payload.get("jti")},
            ip_address=ip_address,
        )

        return AuthenticatedIdentity(
            identity_id=sub,
            identity_type=identity_type,
            name=name,
            project_id=None,  # Project roles are dynamically evaluated per-project
            role=None,
            scopes=scopes,
            is_authenticated=True,
            metadata={"jti": payload.get("jti")},
        )


class AuthorizationService:
    """Evaluates project-scoped RBAC permissions and prevents privilege escalation."""

    def __init__(
        self,
        store: AuthStore,
        audit_logger: Optional[SecurityAuditLogger] = None,
    ) -> None:
        self.store = store
        self.audit = audit_logger or SecurityAuditLogger()

    def get_user_role(self, project_id: str, user_id: str) -> Optional[ProjectRole]:
        """Look up member role for a user within a project."""
        member = self.store.get_project_member(project_id, user_id)
        return member.role if member else None

    def authorize(
        self,
        identity: AuthenticatedIdentity,
        project_id: str,
        permission: Permission,
    ) -> None:
        """Verify that identity is authorized to execute permission within project_id.

        Raises:
            AuthenticationError: if identity is not authenticated.
            PermissionDeniedError: if insufficient permissions or cross-project violation.
        """
        if not identity.is_authenticated:
            raise AuthenticationError("Principal is not authenticated.")

        if not project_id:
            raise ValueError("project_id must be provided for authorization.")

        # 1. System identity has super-authority
        if identity.identity_type == IdentityType.SYSTEM:
            return

        # 2. Agent identity authorization
        if identity.identity_type == IdentityType.AGENT:
            # Agents are strictly bounded to their assigned project
            if identity.project_id != project_id:
                self.audit.record_event(
                    "ACCESS_DENIED",
                    identity_id=identity.identity_id,
                    identity_type=IdentityType.AGENT.value,
                    project_id=project_id,
                    action=permission.value,
                    status="DENIED",
                    details={
                        "reason": "cross_project_agent_violation",
                        "assigned_project": identity.project_id,
                    },
                )
                raise PermissionDeniedError(
                    f"Agent '{identity.identity_id}' is bounded to project '{identity.project_id}' "
                    f"and cannot access project '{project_id}'."
                )

            # Evaluate permission against AGENT role and credential scopes
            allowed = has_permission(ProjectRole.AGENT, permission, scopes=identity.scopes)
            if not allowed:
                self.audit.record_event(
                    "ACCESS_DENIED",
                    identity_id=identity.identity_id,
                    identity_type=IdentityType.AGENT.value,
                    project_id=project_id,
                    action=permission.value,
                    status="DENIED",
                    details={
                        "reason": "insufficient_agent_scopes",
                        "granted_scopes": list(identity.scopes),
                    },
                )
                raise PermissionDeniedError(
                    f"Agent '{identity.identity_id}' lacks permission '{permission.value}' with scopes {list(identity.scopes)}."
                )
            return

        # 3. User identity authorization
        if identity.identity_type == IdentityType.USER:
            member = self.store.get_project_member(project_id, identity.identity_id)
            if not member:
                self.audit.record_event(
                    "ACCESS_DENIED",
                    identity_id=identity.identity_id,
                    identity_type=IdentityType.USER.value,
                    project_id=project_id,
                    action=permission.value,
                    status="DENIED",
                    details={"reason": "not_project_member"},
                )
                raise PermissionDeniedError(
                    f"User '{identity.identity_id}' is not a member of project '{project_id}'."
                )

            allowed = has_permission(member.role, permission)
            if not allowed:
                self.audit.record_event(
                    "ACCESS_DENIED",
                    identity_id=identity.identity_id,
                    identity_type=IdentityType.USER.value,
                    project_id=project_id,
                    action=permission.value,
                    status="DENIED",
                    details={"reason": "role_lacks_permission", "role": member.role.value},
                )
                raise PermissionDeniedError(
                    f"Role '{member.role.value}' lacks permission '{permission.value}' in project '{project_id}'."
                )
            return

        raise PermissionDeniedError("Unknown principal identity type.")

    def check_membership_change(
        self,
        actor: AuthenticatedIdentity,
        project_id: str,
        target_role: ProjectRole,
        existing_target_role: Optional[ProjectRole] = None,
    ) -> None:
        """Validate that actor has authority to assign or modify membership roles without privilege escalation."""
        # Must have MEMBERSHIP_MANAGE permission
        self.authorize(actor, project_id, Permission.MEMBERSHIP_MANAGE)

        # Look up actor's role
        actor_role = self.get_user_role(project_id, actor.identity_id)
        if not actor_role:
            raise PermissionDeniedError("Actor has no assigned role in this project.")

        validate_membership_transition(actor_role, target_role, existing_target_role)


@contextmanager
def trusted_tenant_context(
    auth_service: AuthenticationService,
    rbac_service: AuthorizationService,
    identity: AuthenticatedIdentity,
    project_id: str,
    required_permission: Permission,
    db_manager: DatabaseConnectionManager,
) -> Generator[Any, None, None]:
    """Scoped transaction context manager enforcing authentication, RBAC, and PostgreSQL RLS.

    Lifecycle:
    1. Authorization verification runs BEFORE acquiring any database connection or setting context.
    2. Opens tenant context setting `SET LOCAL app.current_project_id = %s;` within transaction.
    3. Guarantees connection sanitization (`RESET ALL`) on checkout and return.
    """
    # 1. Server-side authorization check first
    rbac_service.authorize(identity, project_id, required_permission)

    # 2. Enter database tenant context
    with db_manager.tenant_context(project_id) as conn:
        yield conn
