"""Role-Based Access Control (RBAC) permission matrix and enforcement engine.

Authoritative Permission Inventory:
Exactly 28 granular permissions across 10 functional domains:
- Project lifecycle (3): PROJECT_READ, PROJECT_UPDATE, PROJECT_DELETE
- Membership management (2): MEMBERSHIP_READ, MEMBERSHIP_MANAGE
- Agent registry (3): AGENT_READ, AGENT_REGISTER, AGENT_MANAGE
- Credential lifecycle (4): CREDENTIAL_READ, CREDENTIAL_CREATE, CREDENTIAL_ROTATE, CREDENTIAL_REVOKE
- Memory storage (4): MEMORY_READ, MEMORY_WRITE, MEMORY_UPDATE, MEMORY_DELETE
- Event stream (2): EVENT_READ, EVENT_WRITE
- Knowledge graph (2): GRAPH_READ, GRAPH_WRITE
- Vector search (2): VECTOR_SEARCH, VECTOR_WRITE
- Session continuity (2): SESSION_READ, SESSION_WRITE
- BrainLoop execution (2): INTERACTION_READ, INTERACTION_EXECUTE
- Feedback & calibration (2): FEEDBACK_READ, FEEDBACK_WRITE
"""

from __future__ import annotations

from typing import FrozenSet, Mapping, Optional, Sequence, Set, Tuple
from ai_brain.auth.exceptions import PermissionDeniedError, PrivilegeEscalationError
from ai_brain.auth.models import Permission, ProjectRole

# Authoritative Role-Permission Matrix (All 28 Permissions mapped)
ROLE_PERMISSIONS: Mapping[ProjectRole, FrozenSet[Permission]] = {
    ProjectRole.OWNER: frozenset(Permission),  # All 28 permissions
    ProjectRole.ADMIN: frozenset({             # 27 permissions (all except PROJECT_DELETE)
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
    ProjectRole.MEMBER: frozenset({            # 18 permissions
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
    ProjectRole.READ_ONLY: frozenset({         # 10 permissions
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
    ProjectRole.AGENT: frozenset({             # 16 permissions
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

# Permissions explicitly forbidden for AGENT identities to prevent scope elevation
AGENT_DISALLOWED_PERMISSIONS: FrozenSet[Permission] = frozenset({
    Permission.PROJECT_DELETE,
    Permission.PROJECT_UPDATE,
    Permission.MEMBERSHIP_READ,
    Permission.MEMBERSHIP_MANAGE,
    Permission.AGENT_REGISTER,
    Permission.AGENT_MANAGE,
    Permission.CREDENTIAL_READ,
    Permission.CREDENTIAL_CREATE,
    Permission.CREDENTIAL_ROTATE,
    Permission.CREDENTIAL_REVOKE,
    Permission.MEMORY_DELETE,
    Permission.FEEDBACK_READ,
})


def validate_agent_scopes(scopes: Sequence[str]) -> Tuple[str, ...]:
    """Validate that agent credential scopes do not attempt to grant permissions beyond the AGENT role.

    Raises:
        PrivilegeEscalationError: If scopes include wildcard admin or permissions forbidden to agents.
    """
    clean_scopes = []
    for s in scopes:
        norm = s.lower().strip()
        if not norm:
            continue
        if norm in ("*", "admin", "owner", "root"):
            raise PrivilegeEscalationError(
                f"Agent scopes cannot contain administrator or wildcard scope: {s!r}"
            )
        # Check against explicitly forbidden namespaces/permissions
        for forbidden in AGENT_DISALLOWED_PERMISSIONS:
            forbidden_name = forbidden.value.lower()
            if norm == forbidden_name or norm == forbidden_name.replace("_", ":"):
                raise PrivilegeEscalationError(
                    f"Agent scope {s!r} exceeds permissions assigned to the AGENT role"
                )
        if norm.startswith("credential:") or norm.startswith("membership:"):
            raise PrivilegeEscalationError(
                f"Agent scope {s!r} exceeds permissions assigned to the AGENT role"
            )
        clean_scopes.append(norm)
    return tuple(clean_scopes)


def check_scope_allows_permission(scopes: Sequence[str], permission: Permission) -> bool:
    """Verify whether a sequence of agent credential scopes satisfies the given permission."""
    normalized_scopes = {s.lower().strip() for s in scopes}

    # Universal wildcard scope (note: has_permission will still bound this to role)
    if "*" in normalized_scopes or "admin" in normalized_scopes:
        return True

    # High-level read/write umbrellas
    if "read" in normalized_scopes and permission in READ_PERMISSIONS:
        return True
    if "write" in normalized_scopes and permission in WRITE_PERMISSIONS:
        return True

    # Fine-grained namespace scopes (e.g. "memory:read", "vector:search", "memory_read")
    perm_name = permission.value.lower()
    if perm_name in normalized_scopes or perm_name.replace("_", ":") in normalized_scopes:
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
    """Evaluate whether a role (and optional agent scopes) satisfies a permission.

    Strictly guarantees that:
    1. The permission must be contained in the role's allowed set.
    2. For AGENT identities, scopes cannot grant any permission outside ROLE_PERMISSIONS[AGENT].
    """
    allowed_for_role = ROLE_PERMISSIONS.get(role, frozenset())
    if permission not in allowed_for_role:
        return False

    # Agents are additionally constrained by their credential scopes
    if role == ProjectRole.AGENT:
        if scopes is None:
            return False
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
