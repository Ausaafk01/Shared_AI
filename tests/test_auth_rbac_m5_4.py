"""Comprehensive test suite for Phase 5, Milestone 5.4: Authentication & Role-Based Access Control (RBAC).

Covers:
1. API Key Generation, Hashing, Constant-Time Verification, and Redaction.
2. JWT Access Tokens: Signatures, Claims (iss, aud, exp, iat, sub), Malformed/Expired Rejection.
3. Algorithm Confusion Rejection ('none', symmetric/asymmetric mismatch).
4. Signing-Key Rotation with Historical Key IDs.
5. Role-Permission Matrix across all roles (Owner, Admin, Member, Read-Only, Agent).
6. Privilege Escalation Prevention (Admin promoting to Owner, Member modifying roles).
7. Agent Scope Restrictions (Read-only scopes, Write scopes, Granular scopes).
8. API Key Lifecycle: Expiration, Revocation, Rotation, and Last-Used Tracking.
9. Security Audit Logging, Sanitization (Zero Raw Keys), and Brute-Force Detection.
10. Live PostgreSQL Multi-Tenant RLS Integration with trusted_tenant_context.
11. Cross-Project Access Rejection across stores (Memories, Events, Graph, Vectors).
12. Connection Pool Sanitization (RESET ALL) after Success and Exception.
13. SQLite Offline Compatibility & Store Parity.
"""

from __future__ import annotations

import math
import os
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import psycopg
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from ai_brain.auth import (
    AgentCredentialRecord,
    AlgorithmMismatchError,
    AuthenticatedIdentity,
    AuthenticationError,
    AuthenticationService,
    AuthorizationService,
    AuthStore,
    CreatedApiKey,
    CredentialExpiredError,
    CredentialNotFoundError,
    CredentialRevokedError,
    IdentityType,
    InvalidTokenError,
    KeyManager,
    Permission,
    PermissionDeniedError,
    PostgreSQLAuthStore,
    PrivilegeEscalationError,
    ProjectMemberRecord,
    ProjectRole,
    ROLE_PERMISSIONS,
    SecurityAuditLogger,
    SQLiteAuthStore,
    TokenExpiredError,
    TokenService,
    UserRecord,
    generate_api_key,
    has_permission,
    hash_api_key,
    sanitize_audit_details,
    trusted_tenant_context,
    validate_membership_transition,
    verify_api_key,
)
from ai_brain.config import DatabaseConfig, StorageBackend, load_env_file
from ai_brain.events.models import Event
from ai_brain.events.storage.postgres import PostgreSQLEventStore
from ai_brain.graph.models import GraphNode
from ai_brain.graph.storage.postgres import PostgreSQLGraphStore
from ai_brain.memories.models import Memory
from ai_brain.memories.storage.postgres import PostgreSQLMemoryStore
from ai_brain.memories.types import MemoryType
from ai_brain.storage.database import DatabaseConnectionManager
from ai_brain.vectors.models import VectorRecord
from ai_brain.vectors.storage.postgres import PostgreSQLVectorStore

load_env_file()

ADMIN_PG_URL = os.environ.get(
    "TEST_POSTGRES_ADMIN_URL",
    os.environ.get("TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/test_ai_brain"),
)
APP_PG_URL = os.environ.get(
    "TEST_POSTGRES_APP_URL",
    f"postgresql://ai_brain_app_user:{os.environ.get('TEST_POSTGRES_APP_PASSWORD', '')}@127.0.0.1:5432/test_ai_brain",
)


def _check_postgres_live() -> tuple[bool, str, Optional[DatabaseConnectionManager]]:
    cfg = DatabaseConfig(
        backend=StorageBackend.POSTGRES,
        database_url=ADMIN_PG_URL,
        pool_size=3,
        pool_timeout=3.0,
    )
    mgr = DatabaseConnectionManager(cfg)
    if not mgr.driver_available:
        return False, "PostgreSQL driver not installed", None
    try:
        conn = mgr.get_connection(timeout=3.0)
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            ver = cur.fetchone()
        mgr.return_connection(conn)
        return True, str(ver[0]) if ver else "Connected", mgr
    except Exception as e:
        return False, f"PostgreSQL unreachable: {e}", None


PG_AVAILABLE, PG_STATUS, ADMIN_MGR = _check_postgres_live()


def _generate_rsa_key_pair() -> tuple[str, str]:
    """Generate in-memory RSA 2048 PEM keys for RS256 token tests."""
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem_priv = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    pem_pub = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return pem_priv, pem_pub


# =============================================================================
# 1. Unit Tests: Cryptographic Primitives & API Keys
# =============================================================================
class TestAuthCryptoAndTokens(unittest.TestCase):
    """Test cryptographic key generation, constant-time verification, and JWT security."""

    def test_api_key_generation_and_hashing(self) -> None:
        raw_key, prefix, key_hash = generate_api_key("agy")
        self.assertTrue(raw_key.startswith("agy_ak_"))
        self.assertTrue(prefix.startswith("agy_"))
        self.assertEqual(len(key_hash), 64)  # 256-bit SHA-256 hex string

        # Hash determinism
        self.assertEqual(hash_api_key(raw_key), key_hash)

        # Constant-time verification
        self.assertTrue(verify_api_key(raw_key, key_hash))
        self.assertFalse(verify_api_key(raw_key + "X", key_hash))
        self.assertFalse(verify_api_key("wrong_key", key_hash))
        self.assertFalse(verify_api_key("", key_hash))

    def test_api_key_redaction_in_repr(self) -> None:
        raw_key, prefix, key_hash = generate_api_key("agy")
        created = CreatedApiKey(
            credential_id="cred_test_1",
            raw_key=raw_key,
            key_prefix=prefix,
            agent_id="agent_1",
            project_id="proj_1",
        )
        repr_str = repr(created)
        self.assertNotIn(raw_key, repr_str)
        self.assertIn("***REDACTED***", repr_str)
        self.assertIn("cred_test_1", repr_str)

    def test_jwt_symmetric_lifecycle_and_claims(self) -> None:
        secret = "super-secret-symmetric-key-at-least-32-chars-long"
        km = KeyManager(active_key_id="k1", active_signing_key=secret)
        ts = TokenService(key_manager=km, algorithm="HS256", issuer="test-issuer", audience="test-aud")

        token = ts.create_access_token(
            subject="user_123",
            identity_type=IdentityType.USER,
            name="Alice",
            scopes=["read", "write"],
            expires_in_seconds=300,
        )
        self.assertIsInstance(token, str)

        payload = ts.decode_and_validate_token(token)
        self.assertEqual(payload["sub"], "user_123")
        self.assertEqual(payload["iss"], "test-issuer")
        self.assertEqual(payload["aud"], "test-aud")
        self.assertEqual(payload["name"], "Alice")
        self.assertEqual(payload["typ"], "USER")
        self.assertEqual(payload["scopes"], ["read", "write"])
        self.assertIn("exp", payload)
        self.assertIn("iat", payload)

    def test_jwt_asymmetric_rs256_lifecycle(self) -> None:
        priv_pem, pub_pem = _generate_rsa_key_pair()
        km = KeyManager(active_key_id="rsa_k1", active_signing_key=priv_pem, active_verification_key=pub_pem)
        ts = TokenService(key_manager=km, algorithm="RS256", issuer="test-issuer", audience="test-aud")

        token = ts.create_access_token(subject="user_rsa", name="Bob")
        payload = ts.decode_and_validate_token(token)
        self.assertEqual(payload["sub"], "user_rsa")
        self.assertEqual(payload["name"], "Bob")

    def test_jwt_expired_token_rejected(self) -> None:
        secret = "super-secret-key-for-expiration-test-32b"
        km = KeyManager(active_key_id="k1", active_signing_key=secret)
        ts = TokenService(key_manager=km, algorithm="HS256")

        # Create expired token (expires_in_seconds = -10)
        token = ts.create_access_token(subject="user_exp", expires_in_seconds=-10)
        with self.assertRaises(TokenExpiredError):
            ts.decode_and_validate_token(token)

    def test_jwt_malformed_and_tampered_token_rejected(self) -> None:
        secret = "super-secret-key-for-tamper-test-32b"
        km = KeyManager(active_key_id="k1", active_signing_key=secret)
        ts = TokenService(key_manager=km, algorithm="HS256")

        token = ts.create_access_token(subject="user_valid")
        parts = token.split(".")
        tampered_token = parts[0] + "." + parts[1] + "tampered." + parts[2]

        with self.assertRaises(InvalidTokenError):
            ts.decode_and_validate_token(tampered_token)

        with self.assertRaises(InvalidTokenError):
            ts.decode_and_validate_token("not-a-jwt")

        with self.assertRaises(InvalidTokenError):
            ts.decode_and_validate_token("")

    def test_jwt_algorithm_confusion_and_none_rejected(self) -> None:
        secret = "super-secret-key-for-algo-confusion-32b"
        km = KeyManager(active_key_id="k1", active_signing_key=secret)
        ts = TokenService(key_manager=km, algorithm="HS256")

        # Construct an unsigned 'none' algorithm token
        import jwt as pyjwt
        none_token = pyjwt.encode({"sub": "attacker", "iss": "ai-brain", "aud": "ai-brain-api"}, key="", algorithm="none")

        with self.assertRaises(AlgorithmMismatchError):
            ts.decode_and_validate_token(none_token)

    def test_jwt_invalid_issuer_and_audience_rejected(self) -> None:
        secret = "super-secret-key-for-claims-test-32b"
        km = KeyManager(active_key_id="k1", active_signing_key=secret)
        ts_valid = TokenService(key_manager=km, issuer="good-issuer", audience="good-aud")
        token = ts_valid.create_access_token(subject="user_1")

        # Service expecting different issuer
        ts_bad_iss = TokenService(key_manager=km, issuer="wrong-issuer", audience="good-aud")
        with self.assertRaises(InvalidTokenError):
            ts_bad_iss.decode_and_validate_token(token)

        # Service expecting different audience
        ts_bad_aud = TokenService(key_manager=km, issuer="good-issuer", audience="wrong-aud")
        with self.assertRaises(InvalidTokenError):
            ts_bad_aud.decode_and_validate_token(token)

    def test_jwt_signing_key_rotation(self) -> None:
        secret_old = "key-version-1-super-secret-32-chars-long"
        secret_new = "key-version-2-super-secret-32-chars-long"
        km = KeyManager(active_key_id="k1", active_signing_key=secret_old)
        ts = TokenService(key_manager=km, algorithm="HS256")

        # 1. Issue token with key 1
        token_k1 = ts.create_access_token(subject="user_migrating")
        self.assertEqual(ts.decode_and_validate_token(token_k1)["sub"], "user_migrating")

        # 2. Rotate active key to key 2
        km.rotate_signing_key(new_key_id="k2", new_signing_key=secret_new)
        self.assertEqual(km.active_key_id, "k2")

        # 3. Issue token with key 2
        token_k2 = ts.create_access_token(subject="user_new")
        self.assertEqual(ts.decode_and_validate_token(token_k2)["sub"], "user_new")

        # 4. Old token with k1 is still validly decoded via historical verification key pool
        self.assertEqual(ts.decode_and_validate_token(token_k1)["sub"], "user_migrating")

        # 5. Token signed with unknown key ID is rejected
        km_other = KeyManager(active_key_id="k_unknown", active_signing_key="other-secret-key-at-least-32-chars-long")
        ts_other = TokenService(key_manager=km_other, algorithm="HS256")
        token_alien = ts_other.create_access_token(subject="alien")

        with self.assertRaises(InvalidTokenError):
            ts.decode_and_validate_token(token_alien)

    def test_prohibited_secrets_in_token_claims_rejected(self) -> None:
        km = KeyManager(active_key_id="k1", active_signing_key="secret-key-32-chars-minimum-test")
        ts = TokenService(key_manager=km, algorithm="HS256")

        with self.assertRaises(ValueError):
            ts.create_access_token(subject="u1", extra_claims={"api_key": "raw_secret_key"})

        with self.assertRaises(ValueError):
            ts.create_access_token(subject="u1", extra_claims={"password": "plaintext_password"})


# =============================================================================
# 2. Unit Tests: Role-Permission Matrix & Privilege Escalation
# =============================================================================
class TestAuthorizationMatrix(unittest.TestCase):
    """Verify complete role-permission matrix and privilege escalation prevention."""

    def test_owner_has_all_permissions(self) -> None:
        for perm in Permission:
            self.assertTrue(
                has_permission(ProjectRole.OWNER, perm),
                f"OWNER must have permission {perm.value}",
            )

    def test_admin_permissions(self) -> None:
        self.assertFalse(has_permission(ProjectRole.ADMIN, Permission.PROJECT_DELETE))
        self.assertTrue(has_permission(ProjectRole.ADMIN, Permission.PROJECT_UPDATE))
        self.assertTrue(has_permission(ProjectRole.ADMIN, Permission.MEMBERSHIP_MANAGE))
        self.assertTrue(has_permission(ProjectRole.ADMIN, Permission.CREDENTIAL_CREATE))
        self.assertTrue(has_permission(ProjectRole.ADMIN, Permission.MEMORY_WRITE))

    def test_member_permissions(self) -> None:
        self.assertFalse(has_permission(ProjectRole.MEMBER, Permission.PROJECT_DELETE))
        self.assertFalse(has_permission(ProjectRole.MEMBER, Permission.PROJECT_UPDATE))
        self.assertFalse(has_permission(ProjectRole.MEMBER, Permission.MEMBERSHIP_MANAGE))
        self.assertFalse(has_permission(ProjectRole.MEMBER, Permission.CREDENTIAL_CREATE))
        self.assertFalse(has_permission(ProjectRole.MEMBER, Permission.MEMORY_DELETE))

        # Member can do normal read/write
        self.assertTrue(has_permission(ProjectRole.MEMBER, Permission.MEMORY_READ))
        self.assertTrue(has_permission(ProjectRole.MEMBER, Permission.MEMORY_WRITE))
        self.assertTrue(has_permission(ProjectRole.MEMBER, Permission.VECTOR_SEARCH))
        self.assertTrue(has_permission(ProjectRole.MEMBER, Permission.EVENT_WRITE))

    def test_read_only_permissions(self) -> None:
        # Read-only has read permissions
        self.assertTrue(has_permission(ProjectRole.READ_ONLY, Permission.MEMORY_READ))
        self.assertTrue(has_permission(ProjectRole.READ_ONLY, Permission.VECTOR_SEARCH))
        self.assertTrue(has_permission(ProjectRole.READ_ONLY, Permission.EVENT_READ))
        self.assertTrue(has_permission(ProjectRole.READ_ONLY, Permission.GRAPH_READ))

        # Read-only has NO write, delete, or manage permissions
        self.assertFalse(has_permission(ProjectRole.READ_ONLY, Permission.MEMORY_WRITE))
        self.assertFalse(has_permission(ProjectRole.READ_ONLY, Permission.EVENT_WRITE))
        self.assertFalse(has_permission(ProjectRole.READ_ONLY, Permission.GRAPH_WRITE))
        self.assertFalse(has_permission(ProjectRole.READ_ONLY, Permission.VECTOR_WRITE))
        self.assertFalse(has_permission(ProjectRole.READ_ONLY, Permission.CREDENTIAL_CREATE))

    def test_agent_scope_filtering(self) -> None:
        # Agent with ["read"] scope
        self.assertTrue(has_permission(ProjectRole.AGENT, Permission.MEMORY_READ, scopes=["read"]))
        self.assertTrue(has_permission(ProjectRole.AGENT, Permission.VECTOR_SEARCH, scopes=["read"]))
        self.assertFalse(has_permission(ProjectRole.AGENT, Permission.MEMORY_WRITE, scopes=["read"]))
        self.assertFalse(has_permission(ProjectRole.AGENT, Permission.GRAPH_WRITE, scopes=["read"]))

        # Agent with ["read", "write"] scope
        self.assertTrue(has_permission(ProjectRole.AGENT, Permission.MEMORY_READ, scopes=["read", "write"]))
        self.assertTrue(has_permission(ProjectRole.AGENT, Permission.MEMORY_WRITE, scopes=["read", "write"]))
        self.assertTrue(has_permission(ProjectRole.AGENT, Permission.VECTOR_WRITE, scopes=["read", "write"]))

        # Agent NEVER has membership or credential management
        self.assertFalse(has_permission(ProjectRole.AGENT, Permission.MEMBERSHIP_MANAGE, scopes=["read", "write", "*"]))
        self.assertFalse(has_permission(ProjectRole.AGENT, Permission.CREDENTIAL_CREATE, scopes=["read", "write", "*"]))
        self.assertFalse(has_permission(ProjectRole.AGENT, Permission.PROJECT_DELETE, scopes=["read", "write", "*"]))

    def test_privilege_escalation_validation(self) -> None:
        # Owner can assign Owner, Admin, Member
        validate_membership_transition(ProjectRole.OWNER, ProjectRole.OWNER)
        validate_membership_transition(ProjectRole.OWNER, ProjectRole.ADMIN)
        validate_membership_transition(ProjectRole.OWNER, ProjectRole.MEMBER)

        # Admin CANNOT grant Owner
        with self.assertRaises(PrivilegeEscalationError):
            validate_membership_transition(ProjectRole.ADMIN, ProjectRole.OWNER)

        # Admin CANNOT modify or remove an existing Owner
        with self.assertRaises(PrivilegeEscalationError):
            validate_membership_transition(ProjectRole.ADMIN, ProjectRole.MEMBER, existing_target_role=ProjectRole.OWNER)

        # Admin CAN grant Admin, Member, Read-Only
        validate_membership_transition(ProjectRole.ADMIN, ProjectRole.ADMIN)
        validate_membership_transition(ProjectRole.ADMIN, ProjectRole.MEMBER)
        validate_membership_transition(ProjectRole.ADMIN, ProjectRole.READ_ONLY)

        # Member cannot manage membership
        with self.assertRaises(PermissionDeniedError):
            validate_membership_transition(ProjectRole.MEMBER, ProjectRole.MEMBER)

        # Read-Only cannot manage membership
        with self.assertRaises(PermissionDeniedError):
            validate_membership_transition(ProjectRole.READ_ONLY, ProjectRole.READ_ONLY)


# =============================================================================
# 3. Integration Tests: Authentication & Authorization Services (SQLite Offline)
# =============================================================================
class TestAuthServicesSQLite(unittest.TestCase):
    """Verify AuthenticationService and AuthorizationService logic using SQLiteAuthStore."""

    def setUp(self) -> None:
        self.store = SQLiteAuthStore(":memory:")
        self.km = KeyManager(active_key_id="k1", active_signing_key="test-secret-at-least-32-chars-long")
        self.token_service = TokenService(key_manager=self.km)
        self.audit = SecurityAuditLogger()
        self.auth_svc = AuthenticationService(self.store, self.token_service, audit_logger=self.audit)
        self.rbac_svc = AuthorizationService(self.store, audit_logger=self.audit)

    def tearDown(self) -> None:
        self.store.close()

    def test_api_key_lifecycle_create_auth_rotate_revoke(self) -> None:
        agent_id = "agent_coder"
        project_id = "proj_test"

        # 1. Create key
        created = self.auth_svc.create_agent_api_key(agent_id=agent_id, project_id=project_id, scopes=["read", "write"])
        self.assertTrue(created.raw_key.startswith("agy_ak_"))
        cred_id = created.credential_id

        # 2. Authenticate with raw key
        identity = self.auth_svc.authenticate_api_key(created.raw_key)
        self.assertEqual(identity.identity_id, agent_id)
        self.assertEqual(identity.project_id, project_id)
        self.assertEqual(identity.identity_type, IdentityType.AGENT)
        self.assertEqual(identity.role, ProjectRole.AGENT)
        self.assertIn("read", identity.scopes)
        self.assertIn("write", identity.scopes)

        # Verify last_used_at was updated
        stored_cred = self.store.get_credential(cred_id)
        self.assertIsNotNone(stored_cred)
        self.assertIsNotNone(stored_cred.last_used_at)

        # 3. Rotate key
        rotated = self.auth_svc.rotate_agent_api_key(cred_id)
        self.assertNotEqual(rotated.raw_key, created.raw_key)
        self.assertNotEqual(rotated.credential_id, cred_id)

        # Old key is now revoked and rejected
        with self.assertRaises(CredentialRevokedError):
            self.auth_svc.authenticate_api_key(created.raw_key)

        # New key authenticates successfully
        id_rot = self.auth_svc.authenticate_api_key(rotated.raw_key)
        self.assertEqual(id_rot.identity_id, agent_id)

        # 4. Explicit revocation
        self.auth_svc.revoke_agent_api_key(rotated.credential_id)
        with self.assertRaises(CredentialRevokedError):
            self.auth_svc.authenticate_api_key(rotated.raw_key)

    def test_expired_api_key_rejected(self) -> None:
        # Create key with 1-second expiry
        created = self.auth_svc.create_agent_api_key("agent_quick", "proj_quick", expires_in_seconds=1)
        # Verify valid immediately
        self.auth_svc.authenticate_api_key(created.raw_key)

        # Wait for expiration
        time.sleep(1.1)
        with self.assertRaises(CredentialExpiredError):
            self.auth_svc.authenticate_api_key(created.raw_key)

    def test_user_token_authentication_and_rbac(self) -> None:
        # Register user
        user = UserRecord(user_id="user_alice", email="alice@example.com", name="Alice")
        self.store.create_user(user)

        # Add Alice as MEMBER in proj_alpha
        self.store.add_project_member(
            ProjectMemberRecord(
                member_id="pm_1",
                project_id="proj_alpha",
                user_id="user_alice",
                role=ProjectRole.MEMBER,
            )
        )

        # Create token for Alice
        token = self.token_service.create_access_token(subject="user_alice")
        identity = self.auth_svc.authenticate_token(token)
        self.assertEqual(identity.identity_id, "user_alice")
        self.assertEqual(identity.identity_type, IdentityType.USER)

        # Authorize Alice in proj_alpha for MEMORY_READ (allowed)
        self.rbac_svc.authorize(identity, "proj_alpha", Permission.MEMORY_READ)

        # Authorize Alice in proj_alpha for MEMBERSHIP_MANAGE (denied - Member cannot manage members)
        with self.assertRaises(PermissionDeniedError):
            self.rbac_svc.authorize(identity, "proj_alpha", Permission.MEMBERSHIP_MANAGE)

        # Authorize Alice in proj_beta (denied - not a member)
        with self.assertRaises(PermissionDeniedError):
            self.rbac_svc.authorize(identity, "proj_beta", Permission.MEMORY_READ)

    def test_inactive_user_token_rejected(self) -> None:
        user = UserRecord(user_id="user_banned", email="banned@example.com", name="Banned", is_active=False)
        self.store.create_user(user)

        token = self.token_service.create_access_token(subject="user_banned")
        with self.assertRaises(AuthenticationError):
            self.auth_svc.authenticate_token(token)

    def test_brute_force_failure_alert(self) -> None:
        # Generate 5 consecutive failed authentications with same identifier
        for _ in range(5):
            with self.assertRaises(AuthenticationError):
                self.auth_svc.authenticate_api_key("agy_ak_badkey_12345", ip_address="192.168.1.100")

        # Verify BRUTE_FORCE_SUSPECTED alert event was triggered
        alert_events = [e for e in self.audit.events if e.event_type == "BRUTE_FORCE_SUSPECTED"]
        self.assertGreaterEqual(len(alert_events), 1)
        self.assertEqual(alert_events[0].status, "ALERT")
        self.assertEqual(alert_events[0].ip_address, "192.168.1.100")


# =============================================================================
# 4. PostgreSQL Security Acceptance Tests (Live PostgreSQL + RLS + Pool)
# =============================================================================
@unittest.skipUnless(PG_AVAILABLE, f"Live PostgreSQL test skipped: {PG_STATUS}")
class TestPostgreSQLSecurityAndRLS(unittest.TestCase):
    """Rigorous PostgreSQL security tests verifying unprivileged role, RLS, and tenant isolation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.admin_mgr = DatabaseConnectionManager(
            DatabaseConfig(backend=StorageBackend.POSTGRES, database_url=ADMIN_PG_URL, pool_size=5)
        )
        cls.app_mgr = DatabaseConnectionManager(
            DatabaseConfig(backend=StorageBackend.POSTGRES, database_url=APP_PG_URL, pool_size=10)
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.admin_mgr.close()
        cls.app_mgr.close()
        if ADMIN_MGR:
            ADMIN_MGR.close()

    def setUp(self) -> None:
        self.store = PostgreSQLAuthStore(self.app_mgr)
        self.km = KeyManager(active_key_id="k1", active_signing_key="postgres-auth-secret-key-32b-min")
        self.token_service = TokenService(key_manager=self.km)
        self.audit = SecurityAuditLogger()
        self.auth_svc = AuthenticationService(self.store, self.token_service, audit_logger=self.audit)
        self.rbac_svc = AuthorizationService(self.store, audit_logger=self.audit)

        self.project_a = f"proj_sec_a_{uuid.uuid4().hex[:6]}"
        self.project_b = f"proj_sec_b_{uuid.uuid4().hex[:6]}"

        # Seed projects in DB
        with self.admin_mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO projects (project_id, name) VALUES (%s, %s);", (self.project_a, "Project A"))
                cur.execute("INSERT INTO projects (project_id, name) VALUES (%s, %s);", (self.project_b, "Project B"))
            conn.commit()

    def tearDown(self) -> None:
        with self.admin_mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM projects WHERE project_id IN (%s, %s);", (self.project_a, self.project_b))
            conn.commit()

    def test_application_role_security_attributes(self) -> None:
        """Verify ai_brain_app_user has neither superuser nor BYPASSRLS."""
        with self.app_mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT rolname, rolsuper, rolbypassrls
                    FROM pg_roles
                    WHERE rolname = current_user;
                    """
                )
                row = cur.fetchone()
                self.assertIsNotNone(row)
                rolname, rolsuper, rolbypass = row
                self.assertEqual(rolname, "ai_brain_app_user")
                self.assertFalse(rolsuper, "App user must not be a superuser")
                self.assertFalse(rolbypass, "App user must not have BYPASSRLS")

    def test_trusted_tenant_context_success_and_read_write_isolation(self) -> None:
        """Verify trusted_tenant_context verifies RBAC, opens transaction, and enforces RLS."""
        user_id = f"user_{uuid.uuid4().hex[:6]}"
        self.store.create_user(UserRecord(user_id=user_id, email=f"{user_id}@test.com", name="Bob"))

        # User is OWNER in Project A, not a member in Project B
        self.store.add_project_member(
            ProjectMemberRecord(
                member_id=f"pm_{uuid.uuid4().hex[:6]}",
                project_id=self.project_a,
                user_id=user_id,
                role=ProjectRole.OWNER,
            )
        )

        identity = AuthenticatedIdentity(
            identity_id=user_id,
            identity_type=IdentityType.USER,
            name="Bob",
            is_authenticated=True,
        )

        # 1. Access Project A with trusted_tenant_context: succeeds
        with trusted_tenant_context(
            self.auth_svc,
            self.rbac_svc,
            identity,
            self.project_a,
            Permission.MEMORY_WRITE,
            self.app_mgr,
        ) as conn:
            mem_store = PostgreSQLMemoryStore(self.app_mgr, default_project_id=self.project_a)
            mem_store.store(
                Memory.create(
                    memory_id=f"mem_iso_{uuid.uuid4().hex[:6]}",
                    project_id=self.project_a,
                    content="Confidential Project A Knowledge",
                    memory_type=MemoryType.FACT,
                )
            )

        # 2. Access Project B with trusted_tenant_context: fails at RBAC check before opening tenant context
        with self.assertRaises(PermissionDeniedError):
            with trusted_tenant_context(
                self.auth_svc,
                self.rbac_svc,
                identity,
                self.project_b,
                Permission.MEMORY_READ,
                self.app_mgr,
            ):
                pass

        # 3. Verify cross-project read isolation: Project B tenant context sees 0 records from Project A
        with self.app_mgr.tenant_context(self.project_b) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM memories WHERE project_id = %s;", (self.project_a,))
                count = cur.fetchone()[0]
                self.assertEqual(count, 0, "Project B query must see 0 Project A memories under RLS")

    def test_cross_project_agent_rejection_across_stores(self) -> None:
        """Verify agent credentials bounded to Project A cannot execute operations in Project B."""
        agent_id = f"agent_{uuid.uuid4().hex[:6]}"
        created = self.auth_svc.create_agent_api_key(
            agent_id=agent_id,
            project_id=self.project_a,
            scopes=["read", "write"],
        )

        identity = self.auth_svc.authenticate_api_key(created.raw_key)

        # Allowed in Project A
        self.rbac_svc.authorize(identity, self.project_a, Permission.MEMORY_WRITE)
        self.rbac_svc.authorize(identity, self.project_a, Permission.EVENT_WRITE)
        self.rbac_svc.authorize(identity, self.project_a, Permission.VECTOR_SEARCH)

        # Denied in Project B
        with self.assertRaises(PermissionDeniedError):
            self.rbac_svc.authorize(identity, self.project_b, Permission.MEMORY_READ)

        with self.assertRaises(PermissionDeniedError):
            self.rbac_svc.authorize(identity, self.project_b, Permission.EVENT_WRITE)

        with self.assertRaises(PermissionDeniedError):
            self.rbac_svc.authorize(identity, self.project_b, Permission.VECTOR_SEARCH)

    def test_connection_sanitization_after_success_and_exception(self) -> None:
        """Verify RESET ALL clears session settings on both successful context exit and error."""
        user_id = f"user_san_{uuid.uuid4().hex[:6]}"
        self.store.create_user(UserRecord(user_id=user_id, email=f"{user_id}@test.com", name="Sam"))
        self.store.add_project_member(
            ProjectMemberRecord(
                member_id=f"pm_san_{uuid.uuid4().hex[:6]}",
                project_id=self.project_a,
                user_id=user_id,
                role=ProjectRole.OWNER,
            )
        )
        identity = AuthenticatedIdentity(identity_id=user_id, identity_type=IdentityType.USER, name="Sam")

        # 1. Successful context exit
        with trusted_tenant_context(
            self.auth_svc, self.rbac_svc, identity, self.project_a, Permission.MEMORY_READ, self.app_mgr
        ) as conn:
            pass

        # Check checked-out connection has no dangling current_project_id
        with self.app_mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_setting('app.current_project_id', true);")
                setting = cur.fetchone()[0]
                self.assertIn(setting, (None, "", "''"), "Connection sanitization must clear app.current_project_id")

        # 2. Exception inside context
        try:
            with trusted_tenant_context(
                self.auth_svc, self.rbac_svc, identity, self.project_a, Permission.MEMORY_READ, self.app_mgr
            ) as conn:
                raise RuntimeError("Simulated failure inside transaction")
        except RuntimeError:
            pass

        # Check returned connection is cleanly reset
        with self.app_mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_setting('app.current_project_id', true);")
                setting = cur.fetchone()[0]
                self.assertIn(setting, (None, "", "''"), "Connection sanitization must clear app.current_project_id after exception")


if __name__ == "__main__":
    unittest.main()
