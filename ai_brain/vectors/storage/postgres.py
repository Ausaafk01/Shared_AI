"""Persistent PostgreSQL pgvector storage backend for AI Brain."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import psycopg
from psycopg.errors import UniqueViolation

from ai_brain.storage.database import DatabaseConnectionManager, get_database_manager
from ai_brain.vectors.exceptions import (
    DimensionMismatchError,
    DuplicateVectorError,
    ProjectIsolationError,
    VectorNotFoundError,
    VectorStorageError,
    VectorValidationError,
)
from ai_brain.vectors.models import (
    VectorMatch,
    VectorRecord,
    VectorSearchResult,
    _freeze_container,
    _unfreeze_container,
)
from ai_brain.vectors.storage.base import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED_DIMENSIONS = (384, 768, 1536)
DIMENSION_TABLE_MAP = {
    384: "memory_vectors_384",
    768: "memory_vectors_768",
    1536: "memory_vectors_1536",
}


def _get_table_for_dimension(dimension: int) -> str:
    """Resolve the authoritative PostgreSQL vector table for a given embedding dimension."""
    if dimension not in DIMENSION_TABLE_MAP:
        raise DimensionMismatchError(
            f"Unsupported vector dimension {dimension}. Supported dimensions are {SUPPORTED_DIMENSIONS}."
        )
    return DIMENSION_TABLE_MAP[dimension]


def _parse_embedding(val: Any) -> tuple[float, ...]:
    """Parse raw database embedding into a tuple of floats."""
    if isinstance(val, (list, tuple)):
        return tuple(float(x) for x in val)
    if isinstance(val, str):
        cleaned = val.strip("[]")
        if not cleaned:
            return ()
        return tuple(float(x.strip()) for x in cleaned.split(","))
    raise VectorValidationError(f"Cannot parse embedding of type {type(val).__name__}")


def _format_vector_literal(embedding: Sequence[float]) -> str:
    """Format float sequence as a PostgreSQL pgvector literal string."""
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


class PostgreSQLVectorStore(VectorStore):
    """PostgreSQL vector storage engine with native pgvector HNSW indexing.

    Guarantees:
    - Persistence of high-dimensional vector embeddings across 384, 768, and 1536 dimensions.
    - Transaction-safe CRUD operations with strict project-level tenant isolation via RLS.
    - Native nearest-neighbor cosine similarity search using pgvector `<=>` operator and HNSW indexes.
    - Zero silent truncation, padding, or reinterpretation of vectors.
    - Deterministic tie-breaking on similarity queries (score DESC, memory_id ASC, version DESC).
    - Full backward compatibility with SQLiteVectorStore contract.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseConnectionManager] = None,
        default_project_id: Optional[str] = None,
    ) -> None:
        self.db = db_manager or get_database_manager()
        self.default_project_id = default_project_id
        self._closed = False
        self._known_projects: set[str] = set()
        if default_project_id:
            self._known_projects.add(default_project_id)
        self._has_vector_ext: Optional[bool] = None

    def has_pgvector(self) -> bool:
        """Check if pgvector extension is installed and active in PostgreSQL."""
        if self._has_vector_ext is not None:
            return self._has_vector_ext

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
                self._has_vector_ext = bool(cur.fetchone())
        return self._has_vector_ext

    def _ensure_open(self) -> None:
        if self._closed:
            raise VectorStorageError("PostgreSQLVectorStore is closed.")

    def _resolve_project(self, project_id: Optional[str]) -> str:
        proj = project_id or self.default_project_id
        if not proj:
            raise ProjectIsolationError(
                "project_id must be provided or default_project_id set on PostgreSQLVectorStore."
            )
        return proj

    def _validate_vector(self, embedding: Sequence[float], dimensions: int) -> tuple[float, ...]:
        """Validate vector elements, rejecting NaN, Inf, non-numbers, and mismatched lengths."""
        if not embedding or not isinstance(embedding, (list, tuple)):
            raise VectorValidationError("Embedding must be a non-empty sequence of floats")
        if len(embedding) != dimensions:
            raise DimensionMismatchError(
                f"Embedding dimension mismatch: expected {dimensions}, got {len(embedding)}"
            )

        validated: List[float] = []
        for i, x in enumerate(embedding):
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise VectorValidationError(f"Vector component at index {i} is not a valid number: {x!r}")
            f_val = float(x)
            if math.isnan(f_val) or math.isinf(f_val):
                raise VectorValidationError(f"Vector component at index {i} cannot be NaN or Infinite")
            validated.append(f_val)
        return tuple(validated)

    def _ensure_project_and_memory(
        self,
        conn: psycopg.Connection,
        project_id: str,
        memory_id: str,
        version: int,
        content: str,
        status: str,
    ) -> None:
        """Ensure parent project and memory records exist to satisfy foreign keys."""
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO projects (project_id, name)
                VALUES (%s, %s)
                ON CONFLICT (project_id) DO NOTHING;
                """,
                (project_id, project_id),
            )
            cur.execute(
                """
                INSERT INTO memories (memory_id, project_id, version, content, status)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (memory_id) DO NOTHING;
                """,
                (memory_id, project_id, version, content, status),
            )

    def add_vector(self, record: VectorRecord, *, fail_if_exists: bool = False) -> VectorRecord:
        """Store a VectorRecord with strict dimension and constraint validation."""
        self._ensure_open()
        if not isinstance(record, VectorRecord):
            raise VectorValidationError(f"Expected VectorRecord, got {type(record).__name__}")

        proj_id = record.project_id
        self._known_projects.add(proj_id)

        # Validate dimension and resolve destination table
        table_name = _get_table_for_dimension(record.dimensions)
        validated_emb = self._validate_vector(record.embedding, record.dimensions)
        vec_literal = _format_vector_literal(validated_emb)

        created_utc = (
            record.created_at.replace(tzinfo=timezone.utc)
            if record.created_at.tzinfo is None
            else record.created_at.astimezone(timezone.utc)
        )

        with self.db.tenant_context(proj_id) as conn:
            # Satisfy FK constraints within transaction
            self._ensure_project_and_memory(
                conn, proj_id, record.memory_id, record.version, record.content, record.status
            )

            with conn.cursor() as cur:
                # 1. Check for existing record by unique constraint (project_id, memory_id, version, model_name)
                cur.execute(
                    f"""
                    SELECT vector_id FROM {table_name}
                    WHERE project_id = %s AND memory_id = %s AND version = %s AND model_name = %s;
                    """,
                    (proj_id, record.memory_id, record.version, record.model_id),
                )
                existing = cur.fetchone()
                if existing:
                    if fail_if_exists:
                        raise DuplicateVectorError(
                            f"Vector record already exists for memory '{record.memory_id}' "
                            f"(version {record.version}, model '{record.model_id}')"
                        )
                    # Return existing record
                    return self.get_vector(existing[0], project_id=proj_id)  # type: ignore

                # 2. Insert vector record
                use_vector_type = self.has_pgvector()
                emb_param = vec_literal if use_vector_type else list(validated_emb)
                sql_insert = f"""
                    INSERT INTO {table_name} (
                        vector_id, project_id, memory_id, version,
                        embedding, model_name, content, status, metadata, created_at
                    ) VALUES (
                        %s, %s, %s, %s,
                        {'%s::vector' if use_vector_type else '%s'},
                        %s, %s, %s, %s::jsonb, %s
                    )
                    ON CONFLICT (vector_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        status = EXCLUDED.status,
                        metadata = EXCLUDED.metadata;
                """
                try:
                    cur.execute(
                        sql_insert,
                        (
                            record.record_id,
                            proj_id,
                            record.memory_id,
                            record.version,
                            emb_param,
                            record.model_id,
                            record.content,
                            record.status,
                            json.dumps(dict(_unfreeze_container(record.metadata))),
                            created_utc,
                        ),
                    )
                    return record
                except UniqueViolation as e:
                    if fail_if_exists:
                        raise DuplicateVectorError(f"Duplicate vector insertion: {e}") from e
                    return self.get_vector(record.record_id, project_id=proj_id)  # type: ignore
                except Exception as e:
                    raise VectorStorageError(f"Failed to persist vector record: {e}") from e

    def _row_to_record(self, row: Sequence[Any], dimensions: int) -> VectorRecord:
        (
            vector_id,
            project_id,
            memory_id,
            version,
            embedding_raw,
            model_name,
            content,
            status,
            metadata_raw,
            created_at,
        ) = row

        embedding = _parse_embedding(embedding_raw)
        meta = metadata_raw if isinstance(metadata_raw, dict) else json.loads(metadata_raw or "{}")

        return VectorRecord.create(
            record_id=vector_id,
            memory_id=memory_id,
            project_id=project_id,
            version=version,
            model_id=model_name,
            embedding=embedding,
            content=content,
            status=status,
            metadata=meta,
            created_at=created_at,
        )

    def get_vector(self, record_id: str, project_id: Optional[str] = None) -> Optional[VectorRecord]:
        """Retrieve a vector record by record_id across supported dimension tables."""
        self._ensure_open()
        if not record_id:
            return None

        candidate_projects = [project_id] if project_id else (
            [self.default_project_id] if self.default_project_id else list(self._known_projects)
        )

        for proj in candidate_projects:
            if not proj:
                continue
            with self.db.tenant_context(proj) as conn:
                with conn.cursor() as cur:
                    for dim, tbl in DIMENSION_TABLE_MAP.items():
                        cur.execute(
                            f"""
                            SELECT vector_id, project_id, memory_id, version,
                                   embedding, model_name, content, status, metadata, created_at
                            FROM {tbl}
                            WHERE vector_id = %s AND project_id = %s;
                            """,
                            (record_id, proj),
                        )
                        row = cur.fetchone()
                        if row:
                            return self._row_to_record(row, dim)
        return None

    def get_vector_by_memory(
        self,
        project_id: str,
        memory_id: str,
        version: int,
        model_id: str,
    ) -> Optional[VectorRecord]:
        """Retrieve vector record by project, memory_id, version, and model_id."""
        self._ensure_open()
        proj = self._resolve_project(project_id)

        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                for dim, tbl in DIMENSION_TABLE_MAP.items():
                    cur.execute(
                        f"""
                        SELECT vector_id, project_id, memory_id, version,
                               embedding, model_name, content, status, metadata, created_at
                        FROM {tbl}
                        WHERE project_id = %s AND memory_id = %s AND version = %s AND model_name = %s;
                        """,
                        (proj, memory_id, version, model_id),
                    )
                    row = cur.fetchone()
                    if row:
                        return self._row_to_record(row, dim)
        return None

    def update_vector(self, record: VectorRecord) -> VectorRecord:
        """Update status, metadata, and content of an existing vector record."""
        self._ensure_open()
        if not isinstance(record, VectorRecord):
            raise VectorValidationError(f"Expected VectorRecord, got {type(record).__name__}")

        proj_id = self._resolve_project(record.project_id)
        table_name = _get_table_for_dimension(record.dimensions)

        with self.db.tenant_context(proj_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {table_name}
                    SET status = %s, metadata = %s::jsonb, content = %s
                    WHERE vector_id = %s AND project_id = %s;
                    """,
                    (
                        record.status,
                        json.dumps(dict(_unfreeze_container(record.metadata))),
                        record.content,
                        record.record_id,
                        proj_id,
                    ),
                )
                if cur.rowcount == 0:
                    raise VectorNotFoundError(f"Vector record '{record.record_id}' not found for update")
                return record

    def delete_vector(self, record_id: str, project_id: Optional[str] = None) -> bool:
        """Delete a vector record by record_id. Returns True if deleted."""
        self._ensure_open()
        candidate_projects = [project_id] if project_id else (
            [self.default_project_id] if self.default_project_id else list(self._known_projects)
        )

        for proj in candidate_projects:
            if not proj:
                continue
            with self.db.tenant_context(proj) as conn:
                with conn.cursor() as cur:
                    for tbl in DIMENSION_TABLE_MAP.values():
                        cur.execute(
                            f"DELETE FROM {tbl} WHERE vector_id = %s AND project_id = %s;",
                            (record_id, proj),
                        )
                        if cur.rowcount > 0:
                            return True
        return False

    def delete_vectors_by_memory(self, project_id: str, memory_id: str) -> int:
        """Delete all vector records for a memory_id. Returns count deleted."""
        self._ensure_open()
        proj = self._resolve_project(project_id)
        total_deleted = 0

        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                for tbl in DIMENSION_TABLE_MAP.values():
                    cur.execute(
                        f"DELETE FROM {tbl} WHERE project_id = %s AND memory_id = %s;",
                        (proj, memory_id),
                    )
                    total_deleted += cur.rowcount
        return total_deleted

    def get_vectors_by_project(
        self,
        project_id: str,
        *,
        model_id: Optional[str] = None,
        allowed_statuses: Optional[Sequence[str]] = None,
    ) -> List[VectorRecord]:
        """Retrieve all vector records for a project, optionally filtered by model_id and statuses."""
        self._ensure_open()
        proj = self._resolve_project(project_id)
        records: List[VectorRecord] = []

        statuses = [s.upper() for s in allowed_statuses] if allowed_statuses else None

        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                for dim, tbl in DIMENSION_TABLE_MAP.items():
                    sql = f"""
                        SELECT vector_id, project_id, memory_id, version,
                               embedding, model_name, content, status, metadata, created_at
                        FROM {tbl}
                        WHERE project_id = %s
                    """
                    params: List[Any] = [proj]
                    if model_id:
                        sql += " AND model_name = %s"
                        params.append(model_id)
                    if statuses:
                        sql += " AND status = ANY(%s)"
                        params.append(statuses)
                    sql += " ORDER BY created_at ASC;"

                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    for r in rows:
                        records.append(self._row_to_record(r, dim))

        return records

    def search_similarity(
        self,
        query_vector: Sequence[float],
        *,
        project_id: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        allowed_statuses: Optional[Sequence[str]] = None,
        model_id: Optional[str] = None,
        query_text: str = "",
    ) -> VectorSearchResult:
        """Execute nearest-neighbor cosine similarity search via pgvector `<=>` operator."""
        self._ensure_open()
        proj = self._resolve_project(project_id)
        dim = len(query_vector)
        table_name = _get_table_for_dimension(dim)
        validated_query = self._validate_vector(query_vector, dim)
        vec_literal = _format_vector_literal(validated_query)

        statuses = [s.upper() for s in allowed_statuses] if allowed_statuses else ["ACTIVE"]

        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                if self.has_pgvector():
                    # Native pgvector HNSW-accelerated similarity search
                    sql = f"""
                        SELECT vector_id, memory_id, project_id, version,
                               status, content, model_name, metadata,
                               1.0 - (embedding <=> %s::vector) AS similarity
                        FROM {table_name}
                        WHERE project_id = %s
                          AND status = ANY(%s)
                    """
                    params: List[Any] = [vec_literal, proj, statuses]

                    if model_id:
                        sql += " AND model_name = %s"
                        params.append(model_id)
                    if similarity_threshold is not None:
                        sql += " AND (1.0 - (embedding <=> %s::vector)) >= %s"
                        params.extend([vec_literal, float(similarity_threshold)])

                    # Deterministic ordering: highest similarity first, then memory_id ASC, version DESC
                    sql += " ORDER BY (embedding <=> %s::vector) ASC, memory_id ASC, version DESC"
                    params.append(vec_literal)

                    if top_k > 0:
                        sql += " LIMIT %s;"
                        params.append(top_k)

                    cur.execute(sql, params)
                    rows = cur.fetchall()

                    matches: List[VectorMatch] = []
                    for r in rows:
                        v_id, m_id, p_id, ver, stat, cont, mod_name, meta_raw, sim = r
                        meta = meta_raw if isinstance(meta_raw, dict) else json.loads(meta_raw or "{}")
                        matches.append(
                            VectorMatch(
                                memory_id=m_id,
                                score=float(sim),
                                project_id=p_id,
                                version=ver,
                                status=stat,
                                content=cont,
                                model_id=mod_name,
                                metadata=meta,
                            )
                        )

                    # Get candidate count
                    count_sql = f"SELECT COUNT(*) FROM {table_name} WHERE project_id = %s AND status = ANY(%s)"
                    count_params: List[Any] = [proj, statuses]
                    if model_id:
                        count_sql += " AND model_name = %s"
                        count_params.append(model_id)
                    cur.execute(count_sql + ";", count_params)
                    count_row = cur.fetchone()
                    total_candidates = int(count_row[0]) if count_row else len(matches)

                    return VectorSearchResult(
                        query=query_text or f"<vector:{dim}d>",
                        matches=tuple(matches),
                        project_id=proj,
                        model_id=model_id or "",
                        total_candidates=total_candidates,
                    )
                else:
                    # Fallback for systems without pgvector extension
                    candidates = self.get_vectors_by_project(
                        proj, model_id=model_id, allowed_statuses=statuses
                    )
                    scored: List[Tuple[float, str, int, VectorRecord]] = []
                    for rec in candidates:
                        if rec.dimensions != dim:
                            continue
                        # compute cosine similarity in Python
                        dot = sum(a * b for a, b in zip(validated_query, rec.embedding))
                        norm_q = math.sqrt(sum(a * a for a in validated_query))
                        norm_r = math.sqrt(sum(b * b for b in rec.embedding))
                        sim = dot / (norm_q * norm_r) if norm_q > 1e-12 and norm_r > 1e-12 else 0.0
                        sim = max(-1.0, min(1.0, sim))
                        if similarity_threshold is not None and sim < similarity_threshold:
                            continue
                        scored.append((sim, rec.memory_id, rec.version, rec))

                    scored.sort(key=lambda it: (-it[0], it[1], -it[2]))
                    top_matches = [
                        VectorMatch(
                            memory_id=it[3].memory_id,
                            score=float(it[0]),
                            project_id=it[3].project_id,
                            version=it[3].version,
                            status=it[3].status,
                            content=it[3].content,
                            model_id=it[3].model_id,
                            metadata=it[3].metadata,
                        )
                        for it in scored[:top_k]
                    ]
                    return VectorSearchResult(
                        query=query_text or f"<vector:{dim}d>",
                        matches=tuple(top_matches),
                        project_id=proj,
                        model_id=model_id or "",
                        total_candidates=len(candidates),
                    )

    def count(self, project_id: Optional[str] = None) -> int:
        """Count total vector records stored, optionally filtered by project_id."""
        self._ensure_open()
        candidate_projects = [project_id] if project_id else (
            [self.default_project_id] if self.default_project_id else list(self._known_projects)
        )
        total = 0
        for proj in candidate_projects:
            if not proj:
                continue
            with self.db.tenant_context(proj) as conn:
                with conn.cursor() as cur:
                    for tbl in DIMENSION_TABLE_MAP.values():
                        cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE project_id = %s;", (proj,))
                        row = cur.fetchone()
                        total += int(row[0]) if row else 0
        return total

    def close(self) -> None:
        """Release storage resources (idempotent)."""
        self._closed = True
