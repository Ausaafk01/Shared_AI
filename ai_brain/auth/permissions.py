"""Role-Based Access Control (RBAC) permission matrix and enforcement engine."""

from __future__ import annotations

from typing import FrozenSet, Mapping, Optional, Sequence, Set
from ai_brain.auth.exceptions import PermissionDeniedError, PrivilegeEscalationError
from ai_brain.auth.models import Permission, ProjectRole


# Authoritative Role-Permission Matrix
ROLE_PERMISSIONS: Mapping[ProjectRole, FrozenSet[Permission]] = {
    ProjectRole.OWNER: frozenset(Permission),
    ProjectRole.ADMIN: frozenset({
        Permission.PROJECT_READ,
        Permission.PROJECT_UPDATE,
        Permission.MEMBERSHIP_READ,
        Permission.MEMBERSHIP_MANAGE,
        Permission.AGENT_READ,
        Permission.AGENT_REGISTER,
        Permission.AGENT_MANAGE,
        Permission.CREDENTIAL_READ,
        Permission.CREDENTIAL_CREATE,
        Permission.CREDENTIAL_ROTATE,
        Permission.CREDENTIAL_REVOKE,
        Permission.MEMORY_READ,
        Permission.MEMORY_WRITE,
        Permission.MEMORY_UPDATE,
        Permission.MEMORY_DELETE,
        Permission.EVENT_READ,
        Permission.EVENT_WRITE,
        Permission.GRAPH_READ,
        Permission.GRAPH_WRITE,
        Permission.VECTOR_SEARCH,
        Permission.VECTOR_WRITE,
        Permission.SESSION_READ,
        Permission.SESSION_WRITE,
        Permission.INTERACTION_READ,
        Permission.INTERACTION_EXECUTE,
        Permission.FEEDBACK_READ,
        Permission.FEEDBACK_WRITE,
    }),
    ProjectRole.MEMBER: frozenset({
        Permission.PROJECT_READ,
        Permission.MEMBERSHIP_READ,
        Permission.AGENT_READ,
        Permission.MEMORY_READ,
        Permission.MEMORY_WRITE,
        Permission.MEMORY_UPDATE,
        Permission.EVENT_READ,
        Permission.EVENT_WRITE,
        Permission.GRAPH_READ,
        Permission.GRAPH_WRITE,
        Permission.VECTOR_SEARCH,
        Permission.VECTOR_WRITE,
        Permission.SESSION_READ,
        Permission.SESSION_WRITE,
        Permission.INTERACTION_READ,
        Permission.INTERACTION_EXECUTE,
        Permission.FEEDBACK_READ,
        Permission.FEEDBACK_WRITE,
    }),
    ProjectRole.READ_ONLY: frozenset({
        Permission.PROJECT_READ,
        Permission.MEMBERSHIP_READ,
        Permission.AGENT_READ,
        Permission.MEMORY_READ,
        Permission.EVENT_READ,
        Permission.GRAPH_READ,
        Permission.VECTOR_SEARCH,
        Permission.SESSION_READ,
        Permission.INTERACTION_READ,
        Permission.FEEDBACK_READ,
    }),
    ProjectRole.AGENT: frozenset({
        Permission.PROJECT_READ,
        Permission.AGENT_READ,
        Permission.MEMORY_READ,
        Permission.MEMORY_WRITE,
        Permission.MEMORY_UPDATE,
        Permission.EVENT_READ,
        Permission.EVENT_WRITE,
        Permission.GRAPH_READ,
        Permission.GRAPH_WRITE,
        Permission.VECTOR_SEARCH,
        Permission.VECTOR_WRITE,
        Permission.SESSION_READ,
        Permission.SESSION_WRITE,
        Permission.INTERACTION_READ,
        Permission.INTERACTION_EXECUTE,
        Permission.FEEDBACK_WRITE,
    }),
}

# Granular Scope to Permission Mapping for Machine/Agent Scopes
READ_PERMISSIONS: FrozenSet[Permission] = frozenset({
    Permission.PROJECT_READ,
    Permission.AGENT_READ,
    Permission.MEMORY_READ,
    Permission.EVENT_READ,
    Permission.GRAPH_READ,
    Permission.VECTOR_SEARCH,
    Permission.SESSION_READ,
    Permission.INTERACTION_READ,
    Permission.FEEDBACK_READ,
})

WRITE_PERMISSIONS: FrozenSet[Permission] = frozenset({
    Permission.MEMORY_WRITE,
    Permission.MEMORY_UPDATE,
    Permission.EVENT_WRITE,
    Permission.GRAPH_WRITE,
    Permission.VECTOR_WRITE,
    Permission.SESSION_WRITE,
    Permission.INTERACTION_EXECUTE,
    Permission.FEEDBACK_WRITE,
})


def check_scope_allows_permission(scopes: Sequence[str], permission: Permission) -> bool:
    """Verify whether a sequence of agent credential scopes satisfies the given permission."""
    normalized_scopes = {s.lower().strip() for s in scopes}

    # Universal wildcard scope
    if "*" in normalized_scopes or "admin" in normalized_scopes:
        return True

    # High-level read/write umbrellas
    if "read" in normalized_scopes and permission in READ_PERMISSIONS:
        return True
    if "write" in normalized_scopes and permission in WRITE_PERMISSIONS:
        return True

    # Fine-grained namespace scopes (e.g. "memory:read", "vector:search")
    perm_name = permission.value.lower()
    if perm_name in normalized_scopes:
        return True

    # e.g. "memory:*" allows all MEMORY_* permissions
    for s in normalized_scopes:
        if s.endswith(":*"):
            prefix = s[:-2]
            if perm_name.startswith(prefix):
                return True

    return False


def has_permission(
    role: ProjectRole,
    permission: Permission,
    scopes: Optional[Sequence[str]] = None,
) -> bool:
    """Evaluate whether a role (and optional agent scopes) satisfies a permission."""
    allowed_for_role = ROLE_PERMISSIONS.get(role, frozenset())
    if permission not in allowed_for_role:
        return False

    # Agents are additionally constrained by their credential scopes
    if role == ProjectRole.AGENT and scopes is not None:
        return check_scope_allows_permission(scopes, permission)

    return True


def validate_membership_transition(
    actor_role: ProjectRole,
    target_role: ProjectRole,
    existing_target_role: Optional[ProjectRole] = None,
) -> None:
    """Validate whether actor_role has authority to grant or revoke target_role without privilege escalation."""
    # Only OWNER and ADMIN have membership management
    if actor_role not in (ProjectRole.OWNER, ProjectRole.ADMIN):
        raise PermissionDeniedError("Actor lacks membership management permissions.")

    # Only OWNER can grant OWNER or modify an existing OWNER
    if target_role == ProjectRole.OWNER and actor_role != ProjectRole.OWNER:
        raise PrivilegeEscalationError("Only a project OWNER can grant the OWNER role.")

    if existing_target_role == ProjectRole.OWNER and actor_role != ProjectRole.OWNER:
        raise PrivilegeEscalationError("Only a project OWNER can modify or remove another OWNER.")

    # ADMIN cannot grant higher permissions than themselves
    if actor_role == ProjectRole.ADMIN and target_role not in (
        ProjectRole.ADMIN,
        ProjectRole.MEMBER,
        ProjectRole.READ_ONLY,
    ):
        raise PrivilegeEscalationError(f"ADMIN cannot assign role: {target_role.value}")

