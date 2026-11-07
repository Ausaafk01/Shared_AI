"""Persistent PostgreSQL implementation of the AI Brain Knowledge Graph Storage."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import psycopg
from psycopg.errors import ForeignKeyViolation

from ai_brain.graph.exceptions import GraphValidationError, NodeNotFoundError, ProjectIsolationError
from ai_brain.graph.models import GraphNode, GraphRelationship, _freeze_container, _unfreeze_container
from ai_brain.graph.storage.base import GraphStore
from ai_brain.graph.types import normalize_node_type, normalize_relationship_type
from ai_brain.storage.database import DatabaseConnectionManager, get_database_manager

logger = logging.getLogger(__name__)


def ensure_project_exists(db: DatabaseConnectionManager, project_id: str) -> None:
    """Ensure parent project exists in projects table to satisfy FK constraints."""
    db.ensure_project_exists(project_id)


class PostgreSQLGraphStore(GraphStore):
    """PostgreSQL-backed Knowledge Graph storage engine.

    Guarantees:
    - ACID transaction safety and strict tenant isolation via RLS and SET LOCAL.
    - Foreign key referential integrity between nodes and directed relationship edges.
    - Cascading edge deletion when an attached node is removed.
    - Fast bidirectional adjacency lookups (incoming and outgoing relationships).
    - Preserves node and edge properties, provenance, weights, and timestamps.
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

    def _resolve_project(self, project_id: Optional[str]) -> str:
        proj = project_id or self.default_project_id
        if not proj:
            raise GraphValidationError(
                "project_id must be provided or default_project_id set on PostgreSQLGraphStore."
            )
        return proj

    def add_node(self, node: GraphNode) -> None:
        """Persist or update a GraphNode within its project boundary."""
        if self._closed:
            raise RuntimeError("PostgreSQLGraphStore is closed.")

        proj_id = node.project_id
        self._known_projects.add(proj_id)
        ensure_project_exists(self.db, proj_id)

        created_utc = (
            node.created_at.replace(tzinfo=timezone.utc)
            if node.created_at.tzinfo is None
            else node.created_at.astimezone(timezone.utc)
        )
        updated_utc = (
            node.updated_at.replace(tzinfo=timezone.utc)
            if node.updated_at.tzinfo is None
            else node.updated_at.astimezone(timezone.utc)
        )

        props: Dict[str, Any] = {
            "source_ref": node.source_ref,
            "metadata": _unfreeze_container(node.metadata),
        }

        with self.db.tenant_context(proj_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO graph_nodes (
                        project_id, node_id, node_type, name, properties, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s::jsonb, %s, %s
                    )
                    ON CONFLICT (project_id, node_id) DO UPDATE SET
                        node_type = EXCLUDED.node_type,
                        name = EXCLUDED.name,
                        properties = EXCLUDED.properties,
                        updated_at = EXCLUDED.updated_at;
                    """,
                    (
                        proj_id,
                        node.node_id,
                        str(node.node_type),
                        node.label,
                        json.dumps(props),
                        created_utc,
                        updated_utc,
                    ),
                )

    def _row_to_node(self, row: Sequence[Any]) -> GraphNode:
        project_id, node_id, node_type, name, props_raw, created_at, updated_at = row
        props = props_raw if isinstance(props_raw, dict) else json.loads(props_raw or "{}")

        source_ref = props.get("source_ref")
        meta = props.get("metadata", {})

        return GraphNode(
            node_id=node_id,
            node_type=node_type,
            project_id=project_id,
            label=name,
            source_ref=source_ref,
            metadata=meta,
            created_at=created_at,
            updated_at=updated_at,
        )

    def get_node(self, project_id: str, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by project_id and node_id."""
        if self._closed:
            raise RuntimeError("PostgreSQLGraphStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT project_id, node_id, node_type, name, properties, created_at, updated_at
                    FROM graph_nodes
                    WHERE project_id = %s AND node_id = %s;
                    """,
                    (proj, node_id),
                )
                row = cur.fetchone()
                return self._row_to_node(row) if row else None

    def get_nodes_by_project(self, project_id: str, node_type: Optional[str] = None) -> List[GraphNode]:
        """List all nodes in a project, optionally filtered by node_type."""
        if self._closed:
            raise RuntimeError("PostgreSQLGraphStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT project_id, node_id, node_type, name, properties, created_at, updated_at
                    FROM graph_nodes
                    WHERE project_id = %s
                """
                params: List[Any] = [proj]
                if node_type:
                    sql += " AND node_type = %s"
                    params.append(str(node_type).upper())
                sql += " ORDER BY created_at ASC;"

                cur.execute(sql, params)
                rows = cur.fetchall()
                return [self._row_to_node(r) for r in rows]

    def delete_node(self, project_id: str, node_id: str) -> bool:
        """Delete a node and its attached relationships from storage."""
        if self._closed:
            raise RuntimeError("PostgreSQLGraphStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM graph_nodes WHERE project_id = %s AND node_id = %s;",
                    (proj, node_id),
                )
                return cur.rowcount > 0

    def add_relationship(self, relationship: GraphRelationship) -> None:
        """Persist a directed relationship edge between two nodes in the same project."""
        if self._closed:
            raise RuntimeError("PostgreSQLGraphStore is closed.")

        proj_id = relationship.project_id
        self._known_projects.add(proj_id)
        ensure_project_exists(self.db, proj_id)

        created_utc = (
            relationship.created_at.replace(tzinfo=timezone.utc)
            if relationship.created_at.tzinfo is None
            else relationship.created_at.astimezone(timezone.utc)
        )

        props: Dict[str, Any] = {
            "provenance": _unfreeze_container(relationship.provenance),
            "metadata": _unfreeze_container(relationship.metadata),
        }

        with self.db.tenant_context(proj_id) as conn:
            with conn.cursor() as cur:
                # Enforce that both nodes exist in the project
                cur.execute(
                    "SELECT node_id FROM graph_nodes WHERE project_id = %s AND node_id IN (%s, %s);",
                    (proj_id, relationship.source_id, relationship.target_id),
                )
                found = {r[0] for r in cur.fetchall()}
                if relationship.source_id not in found:
                    raise NodeNotFoundError(
                        f"Source node '{relationship.source_id}' does not exist in project '{proj_id}'"
                    )
                if relationship.target_id not in found:
                    raise NodeNotFoundError(
                        f"Target node '{relationship.target_id}' does not exist in project '{proj_id}'"
                    )

                cur.execute(
                    """
                    INSERT INTO graph_relationships (
                        relationship_id, project_id, source_id, target_id,
                        relation_type, weight, properties, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s::jsonb, %s
                    )
                    ON CONFLICT (project_id, relationship_id) DO UPDATE SET
                        source_id = EXCLUDED.source_id,
                        target_id = EXCLUDED.target_id,
                        relation_type = EXCLUDED.relation_type,
                        weight = EXCLUDED.weight,
                        properties = EXCLUDED.properties;
                    """,
                    (
                        relationship.relationship_id,
                        proj_id,
                        relationship.source_id,
                        relationship.target_id,
                        str(relationship.relationship_type),
                        float(relationship.weight),
                        json.dumps(props),
                        created_utc,
                    ),
                )

    def _row_to_relationship(self, row: Sequence[Any]) -> GraphRelationship:
        (
            rel_id,
            project_id,
            source_id,
            target_id,
            relation_type,
            weight,
            props_raw,
            created_at,
        ) = row

        props = props_raw if isinstance(props_raw, dict) else json.loads(props_raw or "{}")
        prov = props.get("provenance", {})
        meta = props.get("metadata", {})

        return GraphRelationship(
            relationship_id=rel_id,
            source_id=source_id,
            target_id=target_id,
            relationship_type=relation_type,
            project_id=project_id,
            weight=float(weight),
            provenance=prov,
            metadata=meta,
            created_at=created_at,
        )

    def get_relationship(self, project_id: str, relationship_id: str) -> Optional[GraphRelationship]:
        """Retrieve a relationship by project_id and relationship_id."""
        if self._closed:
            raise RuntimeError("PostgreSQLGraphStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT relationship_id, project_id, source_id, target_id,
                           relation_type, weight, properties, created_at
                    FROM graph_relationships
                    WHERE project_id = %s AND relationship_id = %s;
                    """,
                    (proj, relationship_id),
                )
                row = cur.fetchone()
                return self._row_to_relationship(row) if row else None

    def get_relationships_by_project(
        self, project_id: str, relation_type: Optional[str] = None
    ) -> List[GraphRelationship]:
        """List all relationships in a project, optionally filtered by relation_type."""
        if self._closed:
            raise RuntimeError("PostgreSQLGraphStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT relationship_id, project_id, source_id, target_id,
                           relation_type, weight, properties, created_at
                    FROM graph_relationships
                    WHERE project_id = %s
                """
                params: List[Any] = [proj]
                if relation_type:
                    sql += " AND relation_type = %s"
                    params.append(str(relation_type).upper())
                sql += " ORDER BY created_at ASC;"

                cur.execute(sql, params)
                rows = cur.fetchall()
                return [self._row_to_relationship(r) for r in rows]

    def get_outgoing_relationships(self, project_id: str, node_id: str) -> List[GraphRelationship]:
        """List all outgoing relationship edges originating from node_id."""
        if self._closed:
            raise RuntimeError("PostgreSQLGraphStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT relationship_id, project_id, source_id, target_id,
                           relation_type, weight, properties, created_at
                    FROM graph_relationships
                    WHERE project_id = %s AND source_id = %s
                    ORDER BY created_at ASC;
                    """,
                    (proj, node_id),
                )
                rows = cur.fetchall()
                return [self._row_to_relationship(r) for r in rows]

    def get_incoming_relationships(self, project_id: str, node_id: str) -> List[GraphRelationship]:
        """List all incoming relationship edges terminating at node_id."""
        if self._closed:
            raise RuntimeError("PostgreSQLGraphStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT relationship_id, project_id, source_id, target_id,
                           relation_type, weight, properties, created_at
                    FROM graph_relationships
                    WHERE project_id = %s AND target_id = %s
                    ORDER BY created_at ASC;
                    """,
                    (proj, node_id),
                )
                rows = cur.fetchall()
                return [self._row_to_relationship(r) for r in rows]

    def delete_relationship(self, project_id: str, relationship_id: str) -> bool:
        """Delete a relationship edge."""
        if self._closed:
            raise RuntimeError("PostgreSQLGraphStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM graph_relationships WHERE project_id = %s AND relationship_id = %s;",
                    (proj, relationship_id),
                )
                return cur.rowcount > 0

    def count_nodes(self, project_id: Optional[str] = None) -> int:
        """Count graph nodes matching project filter."""
        if self._closed:
            raise RuntimeError("PostgreSQLGraphStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM graph_nodes WHERE project_id = %s;", (proj,))
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def count_relationships(self, project_id: Optional[str] = None) -> int:
        """Count graph relationships matching project filter."""
        if self._closed:
            raise RuntimeError("PostgreSQLGraphStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM graph_relationships WHERE project_id = %s;", (proj,))
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def close(self) -> None:
        """Close storage resources (idempotent)."""
        self._closed = True
