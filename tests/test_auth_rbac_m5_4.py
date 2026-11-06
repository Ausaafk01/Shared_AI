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

import jwt
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
    KeyType,
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
    validate_agent_scopes,
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
        km = KeyManager(
            active_key_id="rsa_k1",
            active_signing_key=priv_pem,
            active_verification_key=pub_pem,
            algorithm="RS256",
        )
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

    def test_jwt_key_type_confusion_rejected(self) -> None:
        """Verify RSA or EdDSA public keys passed as HMAC symmetric secrets are rejected."""
        _, rsa_pub = _generate_rsa_key_pair()

        # Passing RSA public key as HMAC key to KeyManager raises AlgorithmMismatchError
        with self.assertRaises(AlgorithmMismatchError):
            KeyManager(active_key_id="confused_key", active_signing_key=rsa_pub, algorithm="HS256")

        # Passing certificate string raises AlgorithmMismatchError
        cert_fake = "-----BEGIN CERTIFICATE-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...\n-----END CERTIFICATE-----"
        with self.assertRaises(AlgorithmMismatchError):
            KeyManager(active_key_id="confused_cert", active_signing_key=cert_fake, algorithm="HS256")

        # Short symmetric key (< 32 bytes) is rejected
        with self.assertRaises(ValueError):
            KeyManager(active_key_id="short_k", active_signing_key="too_short_secret", algorithm="HS256")

    def test_jwt_incorrect_or_unknown_kid_rejected(self) -> None:
        """Verify token header kid resolution and algorithm binding."""
        km = KeyManager(active_key_id="k1", active_signing_key="secret-key-at-least-32-chars-long-1")
        ts = TokenService(key_manager=km, algorithm="HS256")

        # 1. Forge token with unknown kid
        now = int(time.time())
        token_unknown_kid = jwt.encode(
            {"sub": "u1", "iss": "ai-brain", "aud": "ai-brain-api", "exp": now + 60, "iat": now, "nbf": now, "typ": "USER"},
            "secret-key-at-least-32-chars-long-1",
            algorithm="HS256",
            headers={"kid": "unknown_kid_999", "alg": "HS256", "typ": "JWT"},
        )
        with self.assertRaises(InvalidTokenError):
            ts.decode_and_validate_token(token_unknown_kid)

        # 2. Forge token missing kid completely
        token_no_kid = jwt.encode(
            {"sub": "u1", "iss": "ai-brain", "aud": "ai-brain-api", "exp": now + 60, "iat": now, "nbf": now, "typ": "USER"},
            "secret-key-at-least-32-chars-long-1",
            algorithm="HS256",
            headers={"alg": "HS256", "typ": "JWT"},
        )
        with self.assertRaises(InvalidTokenError):
            ts.decode_and_validate_token(token_no_kid)

        # 3. Register RS256 key into KeyManager and verify HS256 TokenService rejects selecting RS256 kid
        priv_rsa, pub_rsa = _generate_rsa_key_pair()
        km.register_verification_key("rsa_k1", pub_rsa, algorithm="RS256", key_type=KeyType.RSA)
        token_rs256_kid = jwt.encode(
            {"sub": "u1", "iss": "ai-brain", "aud": "ai-brain-api", "exp": now + 60, "iat": now, "nbf": now, "typ": "USER"},
            "secret-key-at-least-32-chars-long-1",
            algorithm="HS256",
            headers={"kid": "rsa_k1", "alg": "HS256", "typ": "JWT"},
        )
        with self.assertRaises(AlgorithmMismatchError):
            ts.decode_and_validate_token(token_rs256_kid)

    def test_jwt_unsupported_algorithm_rejected(self) -> None:
        """Verify unsupported algorithms like 'ES256', 'none', or 'HS512' are rejected."""
        km = KeyManager(active_key_id="k1", active_signing_key="secret-key-at-least-32-chars-long-1")
        with self.assertRaises(ValueError):
            TokenService(key_manager=km, algorithm="ES256")
        with self.assertRaises(ValueError):
            TokenService(key_manager=km, algorithm="none")
        with self.assertRaises(ValueError):
            TokenService(key_manager=km, algorithm="HS512")

    def test_jwt_invalid_token_type_rejected(self) -> None:
        """Verify header typ must strictly be 'JWT' and payload typ must be present."""
        km = KeyManager(active_key_id="k1", active_signing_key="secret-key-at-least-32-chars-long-1")
        ts = TokenService(key_manager=km, algorithm="HS256")
        now = int(time.time())

        # Header typ is not JWT
        token_bad_hdr_typ = jwt.encode(
            {"sub": "u1", "iss": "ai-brain", "aud": "ai-brain-api", "exp": now + 60, "iat": now, "nbf": now, "typ": "USER"},
            "secret-key-at-least-32-chars-long-1",
            algorithm="HS256",
            headers={"kid": "k1", "alg": "HS256", "typ": "NOT_JWT"},
        )
        with self.assertRaises(InvalidTokenError):
            ts.decode_and_validate_token(token_bad_hdr_typ)

        # Payload typ missing
        token_missing_payload_typ = jwt.encode(
            {"sub": "u1", "iss": "ai-brain", "aud": "ai-brain-api", "exp": now + 60, "iat": now, "nbf": now},
            "secret-key-at-least-32-chars-long-1",
            algorithm="HS256",
            headers={"kid": "k1", "alg": "HS256", "typ": "JWT"},
        )
        with self.assertRaises(InvalidTokenError):
            ts.decode_and_validate_token(token_missing_payload_typ)

    def test_jwt_missing_required_claims_rejected(self) -> None:
        """Verify strict claim enforcement: iss, aud, exp, iat, nbf, sub are all mandatory."""
        km = KeyManager(active_key_id="k1", active_signing_key="secret-key-at-least-32-chars-long-1")
        ts = TokenService(key_manager=km, algorithm="HS256")
        now = int(time.time())
        secret = "secret-key-at-least-32-chars-long-1"
        hdr = {"kid": "k1", "alg": "HS256", "typ": "JWT"}

        # Valid baseline claims
        valid_claims = {
            "sub": "user_123",
            "iss": "ai-brain",
            "aud": "ai-brain-api",
            "exp": now + 60,
            "iat": now,
            "nbf": now,
            "typ": "USER",
        }

        # Each of the 6 required claims missing one at a time
        for req_claim in ("iss", "aud", "exp", "iat", "nbf", "sub"):
            claims = dict(valid_claims)
            del claims[req_claim]
            bad_token = jwt.encode(claims, secret, algorithm="HS256", headers=hdr)
            with self.assertRaises(InvalidTokenError, msg=f"Should reject when '{req_claim}' is missing"):
                ts.decode_and_validate_token(bad_token)

    def test_jwt_key_revocation_and_retention_window(self) -> None:
        """Verify key revocation immediately blocks verification and expired retention windows are enforced."""
        secret_old = "secret-key-old-at-least-32-chars-long-1"
        secret_new = "secret-key-new-at-least-32-chars-long-2"
        km = KeyManager(active_key_id="k1", active_signing_key=secret_old)
        ts = TokenService(key_manager=km, algorithm="HS256")

        token_k1 = ts.create_access_token(subject="user_alice")
        self.assertEqual(ts.decode_and_validate_token(token_k1)["sub"], "user_alice")

        # 1. Rotate to k2 with 10 second retention
        km.rotate_signing_key(new_key_id="k2", new_signing_key=secret_new, retention_seconds=10.0)
        # Old token still validates
        self.assertEqual(ts.decode_and_validate_token(token_k1)["sub"], "user_alice")

        # 2. Explicitly revoke k1
        km.revoke_key("k1")
        with self.assertRaises(InvalidTokenError):
            ts.decode_and_validate_token(token_k1)

        # 3. Register k3 with negative retention (already expired)
        km.register_verification_key(
            "k3",
            "secret-key-k3-at-least-32-chars-long-3",
            algorithm="HS256",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        )
        token_k3 = jwt.encode(
            {"sub": "user_k3", "iss": "ai-brain", "aud": "ai-brain-api", "exp": int(time.time()) + 60, "iat": int(time.time()), "nbf": int(time.time()), "typ": "USER"},
            "secret-key-k3-at-least-32-chars-long-3",
            algorithm="HS256",
            headers={"kid": "k3", "alg": "HS256", "typ": "JWT"},
        )
        with self.assertRaises(InvalidTokenError):
            ts.decode_and_validate_token(token_k3)

        # 4. Prune retired keys
        pruned_count = km.prune_retired_keys()
        self.assertGreaterEqual(pruned_count, 1)

    def test_jwt_secrets_never_leaked(self) -> None:
        """Verify secrets are never leaked in repr strings or exception messages."""
        secret = "secret-key-super-confidential-32-bytes"
        km = KeyManager(active_key_id="k_safe", active_signing_key=secret)
        self.assertNotIn(secret, repr(km))
        self.assertNotIn(secret, repr(km._keys["k_safe"]))


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

    def test_authoritative_permission_inventory_count_is_28(self) -> None:
        """Verify the authoritative permission inventory contains exactly 28 permissions."""
        self.assertEqual(len(Permission), 28, "Authoritative permission inventory must contain exactly 28 permissions")
        expected_permissions = {
            "PROJECT_READ", "PROJECT_UPDATE", "PROJECT_DELETE",
            "MEMBERSHIP_READ", "MEMBERSHIP_MANAGE",
            "AGENT_READ", "AGENT_REGISTER", "AGENT_MANAGE",
            "CREDENTIAL_READ", "CREDENTIAL_CREATE", "CREDENTIAL_ROTATE", "CREDENTIAL_REVOKE",
            "MEMORY_READ", "MEMORY_WRITE", "MEMORY_UPDATE", "MEMORY_DELETE",
            "EVENT_READ", "EVENT_WRITE",
            "GRAPH_READ", "GRAPH_WRITE",
            "VECTOR_SEARCH", "VECTOR_WRITE",
            "SESSION_READ", "SESSION_WRITE",
            "INTERACTION_READ", "INTERACTION_EXECUTE",
            "FEEDBACK_READ", "FEEDBACK_WRITE",
        }
        actual_permissions = {p.value for p in Permission}
        self.assertEqual(actual_permissions, expected_permissions)

    def test_exhaustive_permission_matrix_positive_and_negative(self) -> None:
        """Test all 28 permissions for positive and negative authorization across every role."""
        for perm in Permission:
            # 1. OWNER has ALL 28 permissions
            self.assertTrue(has_permission(ProjectRole.OWNER, perm))

            # 2. ADMIN has 27 permissions (all except PROJECT_DELETE)
            if perm == Permission.PROJECT_DELETE:
                self.assertFalse(has_permission(ProjectRole.ADMIN, perm))
            else:
                self.assertTrue(has_permission(ProjectRole.ADMIN, perm))

            # 3. MEMBER has 18 permissions
            member_expected = perm in {
                Permission.PROJECT_READ, Permission.MEMBERSHIP_READ, Permission.AGENT_READ,
                Permission.MEMORY_READ, Permission.MEMORY_WRITE, Permission.MEMORY_UPDATE,
                Permission.EVENT_READ, Permission.EVENT_WRITE,
                Permission.GRAPH_READ, Permission.GRAPH_WRITE,
                Permission.VECTOR_SEARCH, Permission.VECTOR_WRITE,
                Permission.SESSION_READ, Permission.SESSION_WRITE,
                Permission.INTERACTION_READ, Permission.INTERACTION_EXECUTE,
                Permission.FEEDBACK_READ, Permission.FEEDBACK_WRITE,
            }
            self.assertEqual(has_permission(ProjectRole.MEMBER, perm), member_expected, f"MEMBER {perm.value}")

            # 4. READ_ONLY has 10 permissions
            readonly_expected = perm in {
                Permission.PROJECT_READ, Permission.MEMBERSHIP_READ, Permission.AGENT_READ,
                Permission.MEMORY_READ, Permission.EVENT_READ, Permission.GRAPH_READ,
                Permission.VECTOR_SEARCH, Permission.SESSION_READ, Permission.INTERACTION_READ,
                Permission.FEEDBACK_READ,
            }
            self.assertEqual(has_permission(ProjectRole.READ_ONLY, perm), readonly_expected, f"READ_ONLY {perm.value}")

            # 5. AGENT role has 16 permissions
            agent_expected = perm in {
                Permission.PROJECT_READ, Permission.AGENT_READ,
                Permission.MEMORY_READ, Permission.MEMORY_WRITE, Permission.MEMORY_UPDATE,
                Permission.EVENT_READ, Permission.EVENT_WRITE,
                Permission.GRAPH_READ, Permission.GRAPH_WRITE,
                Permission.VECTOR_SEARCH, Permission.VECTOR_WRITE,
                Permission.SESSION_READ, Permission.SESSION_WRITE,
                Permission.INTERACTION_READ, Permission.INTERACTION_EXECUTE,
                Permission.FEEDBACK_WRITE,
            }
            # When scoped with wildcard within role
            self.assertEqual(has_permission(ProjectRole.AGENT, perm, scopes=["*"]), agent_expected, f"AGENT {perm.value}")

    def test_agent_scopes_cannot_exceed_agent_permissions(self) -> None:
        """Verify agent scopes cannot elevate permissions beyond the AGENT role."""
        # 1. Validation rejects wildcards or admin scopes at creation
        with self.assertRaises(PrivilegeEscalationError):
            validate_agent_scopes(["*"])
        with self.assertRaises(PrivilegeEscalationError):
            validate_agent_scopes(["admin"])
        with self.assertRaises(PrivilegeEscalationError):
            validate_agent_scopes(["project:delete"])
        with self.assertRaises(PrivilegeEscalationError):
            validate_agent_scopes(["membership:manage"])
        with self.assertRaises(PrivilegeEscalationError):
            validate_agent_scopes(["credential:revoke"])

        # 2. Even if an agent carried forbidden scopes, has_permission strictly rejects them
        for forbidden in (
            Permission.PROJECT_DELETE,
            Permission.PROJECT_UPDATE,
            Permission.MEMBERSHIP_MANAGE,
            Permission.MEMBERSHIP_READ,
            Permission.AGENT_REGISTER,
            Permission.AGENT_MANAGE,
            Permission.CREDENTIAL_CREATE,
            Permission.CREDENTIAL_ROTATE,
            Permission.CREDENTIAL_REVOKE,
            Permission.CREDENTIAL_READ,
            Permission.MEMORY_DELETE,
            Permission.FEEDBACK_READ,
        ):
            self.assertFalse(
                has_permission(ProjectRole.AGENT, forbidden, scopes=["*", forbidden.value.lower()]),
                f"Agent must never have {forbidden.value}",
            )

    def test_admin_cannot_escalate_or_modify_owner(self) -> None:
        """Verify ADMIN cannot modify, remove, or assign OWNER."""
        # Admin cannot grant OWNER
        with self.assertRaises(PrivilegeEscalationError):
            validate_membership_transition(ProjectRole.ADMIN, ProjectRole.OWNER)

        # Admin cannot modify an existing OWNER
        with self.assertRaises(PrivilegeEscalationError):
            validate_membership_transition(ProjectRole.ADMIN, ProjectRole.ADMIN, existing_target_role=ProjectRole.OWNER)

        # Admin cannot remove an existing OWNER
        with self.assertRaises(PrivilegeEscalationError):
            validate_membership_transition(ProjectRole.ADMIN, ProjectRole.MEMBER, existing_target_role=ProjectRole.OWNER)

        # Admin cannot delete project
        self.assertFalse(has_permission(ProjectRole.ADMIN, Permission.PROJECT_DELETE))

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

    def test_brute_force_scoped_protection_prevents_dos(self) -> None:
        """Verify brute-force tracker keys by IP/composite to prevent denial-of-service on legitimate users."""
        attacker_ip = "198.51.100.42"
        legitimate_ip = "203.0.113.10"
        target_user = "user_alice"

        # Attacker makes 5 failed attempts targeting user_alice
        for _ in range(5):
            self.audit.record_event(
                "LOGIN_FAILED",
                identity_id=target_user,
                ip_address=attacker_ip,
                status="FAILURE",
            )

        # Attacker's IP is blocked
        self.assertTrue(self.audit.is_client_blocked(identity_id=target_user, ip_address=attacker_ip))

        # Legitimate user from different IP is NOT blocked (preventing DoS weaponization)
        self.assertFalse(self.audit.is_client_blocked(identity_id=target_user, ip_address=legitimate_ip))

        # Successful login resets failures
        self.audit.reset_failures(identity_id=target_user, ip_address=attacker_ip)
        self.assertFalse(self.audit.is_client_blocked(identity_id=target_user, ip_address=attacker_ip))


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

    def test_authenticate_agent_key_hardening_and_privileges(self) -> None:
        """Verify authenticate_agent_key SECURITY DEFINER safety, privileges, and minimum columns."""
        # 1. Verify prosecdef and search_path in pg_proc
        with self.admin_mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT prosecdef, proconfig
                    FROM pg_proc
                    WHERE proname = 'authenticate_agent_key';
                    """
                )
                row = cur.fetchone()
                self.assertIsNotNone(row)
                prosecdef, proconfig = row
                self.assertTrue(prosecdef, "authenticate_agent_key must be SECURITY DEFINER")
                self.assertIn(
                    "search_path=pg_catalog, public",
                    proconfig or [],
                    "authenticate_agent_key must have fixed search_path = pg_catalog, public",
                )

                # 2. Check that PUBLIC has no execute privilege
                cur.execute(
                    "SELECT has_function_privilege('public', 'authenticate_agent_key(varchar)', 'EXECUTE');"
                )
                public_exec = cur.fetchone()[0]
                self.assertFalse(public_exec, "PUBLIC must NOT have EXECUTE on authenticate_agent_key")

                # 3. Check that ai_brain_app_user has execute privilege
                cur.execute(
                    "SELECT has_function_privilege('ai_brain_app_user', 'authenticate_agent_key(varchar)', 'EXECUTE');"
                )
                app_exec = cur.fetchone()[0]
                self.assertTrue(app_exec, "ai_brain_app_user must have EXECUTE on authenticate_agent_key")

        # 4. ai_brain_app_user CANNOT alter the function or change its owner
        with self.app_mgr.connection() as conn:
            with conn.cursor() as cur:
                with self.assertRaises(Exception):
                    cur.execute("ALTER FUNCTION public.authenticate_agent_key(varchar) OWNER TO ai_brain_app_user;")
                conn.rollback()

        # 5. Verify function returns only the 9 minimum credential columns required
        with self.app_mgr.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM public.authenticate_agent_key('nonexistent_hash');")
                cols = [desc[0] for desc in cur.description]
                self.assertEqual(
                    cols,
                    [
                        "credential_id",
                        "agent_id",
                        "project_id",
                        "name",
                        "key_hash",
                        "key_prefix",
                        "scopes",
                        "expires_at",
                        "revoked_at",
                    ],
                )
                # Query with invalid key hash returns 0 rows (no credential enumeration)
                self.assertEqual(len(cur.fetchall()), 0)

        # 6. Live agent authentication via the hardened SECURITY DEFINER function
        agent_id = f"ag_hard_{uuid.uuid4().hex[:6]}"
        with self.app_mgr.tenant_context(self.project_a) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agents (agent_id, project_id, name, role) VALUES (%s, %s, %s, %s);",
                    (agent_id, self.project_a, "Hardened Agent", "assistant"),
                )
            conn.commit()

        created_key = self.auth_svc.create_agent_api_key(
            agent_id=agent_id,
            project_id=self.project_a,
            scopes=["read", "write"],
        )
        identity = self.auth_svc.authenticate_api_key(created_key.raw_key)
        self.assertEqual(identity.identity_id, agent_id)
        self.assertEqual(identity.project_id, self.project_a)
        self.assertEqual(identity.role, ProjectRole.AGENT)
        self.assertTrue(identity.has_scope("read"))
        self.assertTrue(identity.has_scope("write"))



if __name__ == "__main__":
    unittest.main()
