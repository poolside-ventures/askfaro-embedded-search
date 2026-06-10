"""Postgres backend — the server-side index (pgvector + tsvector).

Requires the `postgres` extra (sqlalchemy[asyncio], asyncpg) and a database
with the pgvector extension available. Lexical retrieval is weighted
tsvector + ts_rank_cd; semantic retrieval is pgvector cosine over an HNSW
index. Both are incremental: one upsert per object write, no rebuilds.

Embedding **spaces**: each space is its own `embedding_<space> vector(dim)`
column with its own HNSW index, queried independently. A single-space index
(the common case) uses one space named "default".

Delta sync uses a sequence-backed `updated_seq`. Under concurrent writers a
transaction can commit out of sequence order, so a cursor pulled mid-burst
may skip a row until the next sweep; pair cursor sync with a periodic
overlap sweep if writes are highly concurrent.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Sequence

from ..types import Filters, IndexDoc, RawHit, utcnow_iso
from .base import ChangeRow

DEFAULT_SPACE = "default"
_IDENT = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_space(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(
            f"invalid embedding space name {name!r}: must match [a-z][a-z0-9_]*"
        )
    return name


def _vector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.8g}" for x in vec) + "]"


def _ts(value: str | None) -> datetime | None:
    """asyncpg binds timestamptz params as datetime objects, not ISO strings."""
    return datetime.fromisoformat(value) if value else None


def _parse_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [float(x) for x in value.strip("[]").split(",") if x]
    return list(value)


class PostgresBackend:
    def __init__(
        self,
        dsn_or_engine,
        *,
        table: str = "faro_embedded_search_index",
        dim: int = 1536,
        spaces: dict[str, int] | None = None,
    ):
        from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

        if isinstance(dsn_or_engine, AsyncEngine):
            self._engine = dsn_or_engine
            self._owns_engine = False
        else:
            self._engine = create_async_engine(dsn_or_engine)
            self._owns_engine = True
        self.table = table
        self.seq = f"{table}_updated_seq"
        # spaces: {name: dim}. Back-compat: `dim=` configures one "default" space.
        self.space_dims: dict[str, int] = (
            {_validate_space(k): v for k, v in spaces.items()}
            if spaces
            else {DEFAULT_SPACE: dim}
        )
        self.spaces = list(self.space_dims)

    def _col(self, space: str) -> str:
        return f"embedding_{space}"

    async def create_schema(self) -> None:
        """Idempotent DDL — also adds new columns/indexes in place when the
        library (or the configured spaces) change. Apps with Alembic can
        transcribe this instead."""
        from sqlalchemy import text

        stmts = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"CREATE SEQUENCE IF NOT EXISTS {self.seq}",
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                partition_key TEXT,
                object_type TEXT NOT NULL,
                object_id TEXT NOT NULL,
                node_kind TEXT NOT NULL DEFAULT 'leaf',
                title TEXT,
                body TEXT,
                payload JSONB,
                attrs JSONB,
                search_vector tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(body, '')), 'B')
                ) STORED,
                source_updated_at timestamptz NOT NULL,
                embedding_indexed_at timestamptz,
                deleted_at timestamptz,
                updated_seq BIGINT NOT NULL,
                UNIQUE (object_type, object_id, node_kind)
            )""",
            f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS attrs JSONB",
            f"CREATE INDEX IF NOT EXISTS ix_{self.table}_fts ON {self.table} USING gin (search_vector)",
            f"CREATE INDEX IF NOT EXISTS ix_{self.table}_attrs ON {self.table} USING gin (attrs)",
            f"CREATE INDEX IF NOT EXISTS ix_{self.table}_partition ON {self.table} (partition_key, object_type) WHERE deleted_at IS NULL",
            f"CREATE INDEX IF NOT EXISTS ix_{self.table}_seq ON {self.table} (updated_seq)",
        ]
        for space, dim in self.space_dims.items():
            col = self._col(space)
            stmts.append(
                f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS {col} vector({dim})"
            )
            stmts.append(
                f"CREATE INDEX IF NOT EXISTS ix_{self.table}_{col}_hnsw ON {self.table} "
                f"USING hnsw ({col} vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
            )
        async with self._engine.begin() as conn:
            for stmt in stmts:
                await conn.execute(text(stmt))

    # -- write path -------------------------------------------------------

    def _insert_sql(self, *, use_nextval: bool) -> str:
        space_cols = [self._col(s) for s in self.spaces]
        cols = [
            "partition_key", "object_type", "object_id", "node_kind", "title",
            "body", "payload", "attrs", "source_updated_at",
            "embedding_indexed_at", "deleted_at", "updated_seq", *space_cols,
        ]
        seq_val = f"nextval('{self.seq}')" if use_nextval else ":updated_seq"
        vals = [
            ":partition", ":object_type", ":object_id", ":node_kind", ":title",
            ":body", "CAST(:payload AS jsonb)", "CAST(:attrs AS jsonb)",
            ":source_updated_at", ":embedding_indexed_at", ":deleted_at", seq_val,
            *(f"CAST(:vec_{s} AS vector)" for s in self.spaces),
        ]
        updatable = [c for c in cols if c not in ("object_type", "object_id", "node_kind")]
        sets = ", ".join(f"{c} = excluded.{c}" for c in updatable)
        return (
            f"INSERT INTO {self.table} ({', '.join(cols)}) VALUES ({', '.join(vals)}) "
            f"ON CONFLICT (object_type, object_id, node_kind) DO UPDATE SET {sets}"
        )

    def _space_params(self, embeddings: dict[str, list[float] | None]) -> dict:
        out = {}
        for s in self.spaces:
            vec = embeddings.get(s)
            out[f"vec_{s}"] = _vector_literal(vec) if vec else None
        return out

    async def upsert_row(
        self,
        doc: IndexDoc,
        embeddings: dict[str, list[float] | None],
        embedding_indexed_at: str | None,
    ) -> None:
        from sqlalchemy import text

        params = {
            "partition": doc.partition,
            "object_type": doc.object_type,
            "object_id": doc.object_id,
            "node_kind": doc.node_kind,
            "title": doc.title,
            "body": doc.body,
            "payload": json.dumps(doc.payload) if doc.payload is not None else None,
            "attrs": json.dumps(doc.attrs) if doc.attrs is not None else None,
            "source_updated_at": _ts(doc.source_updated_at),
            "embedding_indexed_at": _ts(embedding_indexed_at),
            "deleted_at": None,
            **self._space_params(embeddings),
        }
        async with self._engine.begin() as conn:
            await conn.execute(text(self._insert_sql(use_nextval=True)), params)

    async def delete_row(
        self, object_type: str, object_id: str, node_kind: str | None = None
    ) -> None:
        from sqlalchemy import text

        null_vecs = ", ".join(f"{self._col(s)} = NULL" for s in self.spaces)
        sql = f"""
            UPDATE {self.table}
            SET deleted_at = :now, title = NULL, body = NULL,
                {null_vecs + ',' if null_vecs else ''} updated_seq = nextval('{self.seq}')
            WHERE object_type = :object_type AND object_id = :object_id
              AND deleted_at IS NULL
        """
        params = {"now": _ts(utcnow_iso()), "object_type": object_type, "object_id": object_id}
        if node_kind is not None:
            sql += " AND node_kind = :node_kind"
            params["node_kind"] = node_kind
        async with self._engine.begin() as conn:
            await conn.execute(text(sql), params)

    # -- retrievers -------------------------------------------------------

    def _filter_sql(self, filters: Filters, params: dict) -> str:
        clauses = []
        if filters.partition is not None:
            clauses.append("partition_key = :f_partition")
            params["f_partition"] = filters.partition
        if filters.object_types:
            clauses.append("object_type = ANY(:f_types)")
            params["f_types"] = filters.object_types
        if filters.node_kinds:
            clauses.append("node_kind = ANY(:f_kinds)")
            params["f_kinds"] = filters.node_kinds
        if filters.attrs:
            clauses.append("attrs @> CAST(:f_attrs AS jsonb)")
            params["f_attrs"] = json.dumps(filters.attrs)
        return (" AND " + " AND ".join(clauses)) if clauses else ""

    @staticmethod
    def _hit(row, sim: float | None = None) -> RawHit:
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return RawHit(
            object_type=row["object_type"],
            object_id=row["object_id"],
            node_kind=row["node_kind"],
            partition=row["partition_key"],
            title=row["title"],
            payload=payload,
            sim=sim,
        )

    async def lexical_search(
        self, query: str, filters: Filters, limit: int
    ) -> list[RawHit]:
        from sqlalchemy import text

        params: dict = {"q": query, "limit": limit}
        where = self._filter_sql(filters, params)
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT partition_key, object_type, object_id, node_kind, title, payload,
                           ts_rank_cd(search_vector, plainto_tsquery('english', :q)) AS rank
                    FROM {self.table}
                    WHERE deleted_at IS NULL
                      AND search_vector @@ plainto_tsquery('english', :q){where}
                    ORDER BY rank DESC
                    LIMIT :limit
                    """
                ),
                params,
            )
            return [self._hit(r) for r in result.mappings().all()]

    async def semantic_search(
        self,
        query_vec: Sequence[float],
        space: str,
        filters: Filters,
        limit: int,
        min_score: float,
    ) -> list[RawHit]:
        from sqlalchemy import text

        if space not in self.space_dims:
            return []
        col = self._col(space)
        # The vector literal is inlined because asyncpg cannot infer the type
        # of one parameter used in both the SELECT and ORDER BY positions.
        # It is built from floats we control, never user input.
        vec = _vector_literal(query_vec)
        params: dict = {"limit": limit, "min_score": min_score}
        where = self._filter_sql(filters, params)
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT partition_key, object_type, object_id, node_kind, title, payload,
                           1 - ({col} <=> '{vec}'::vector) AS sim
                    FROM {self.table}
                    WHERE deleted_at IS NULL AND {col} IS NOT NULL{where}
                      AND 1 - ({col} <=> '{vec}'::vector) >= :min_score
                    ORDER BY {col} <=> '{vec}'::vector
                    LIMIT :limit
                    """
                ),
                params,
            )
            return [self._hit(r, sim=float(r["sim"])) for r in result.mappings().all()]

    # -- sync -------------------------------------------------------------

    async def changes_since(
        self, cursor: int | None, *, partition: str | None = None, limit: int = 500
    ) -> tuple[list[ChangeRow], int | None]:
        from sqlalchemy import text

        params: dict = {"cursor": cursor or 0, "limit": limit}
        where = ""
        if partition is not None:
            where = " AND partition_key = :partition"
            params["partition"] = partition
        vec_select = ", ".join(
            f"CAST({self._col(s)} AS text) AS vec_{s}" for s in self.spaces
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT partition_key, object_type, object_id, node_kind, title,
                           body, payload, attrs, {vec_select},
                           source_updated_at, embedding_indexed_at, deleted_at, updated_seq
                    FROM {self.table}
                    WHERE updated_seq > :cursor{where}
                    ORDER BY updated_seq ASC
                    LIMIT :limit
                    """
                ),
                params,
            )
            rows = result.mappings().all()
        changes: list[ChangeRow] = []
        for r in rows:
            payload = r["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            attrs = r["attrs"]
            if isinstance(attrs, str):
                attrs = json.loads(attrs)
            embeddings = {}
            for s in self.spaces:
                v = _parse_vector(r[f"vec_{s}"])
                if v is not None:
                    embeddings[s] = v
            changes.append(
                {
                    "object_type": r["object_type"],
                    "object_id": r["object_id"],
                    "node_kind": r["node_kind"],
                    "partition": r["partition_key"],
                    "title": r["title"],
                    "body": r["body"],
                    "payload": payload,
                    "attrs": attrs,
                    "embeddings": embeddings,
                    "source_updated_at": _iso(r["source_updated_at"]),
                    "embedding_indexed_at": _iso(r["embedding_indexed_at"]),
                    "deleted_at": _iso(r["deleted_at"]),
                    "updated_seq": int(r["updated_seq"]),
                }
            )
        new_cursor = changes[-1]["updated_seq"] if changes else cursor
        return changes, new_cursor

    async def apply_changes(self, rows: Sequence[ChangeRow]) -> None:
        from sqlalchemy import text

        if not rows:
            return
        sql = self._insert_sql(use_nextval=False)
        async with self._engine.begin() as conn:
            for row in rows:
                params = {
                    "partition": row.get("partition"),
                    "object_type": row["object_type"],
                    "object_id": row["object_id"],
                    "node_kind": row.get("node_kind", "leaf"),
                    "title": row.get("title"),
                    "body": row.get("body"),
                    "payload": json.dumps(row["payload"])
                    if row.get("payload") is not None
                    else None,
                    "attrs": json.dumps(row["attrs"])
                    if row.get("attrs") is not None
                    else None,
                    "source_updated_at": _ts(row["source_updated_at"]),
                    "embedding_indexed_at": _ts(row.get("embedding_indexed_at")),
                    "deleted_at": _ts(row.get("deleted_at")),
                    "updated_seq": row["updated_seq"],
                    **self._space_params(row.get("embeddings") or {}),
                }
                await conn.execute(text(sql), params)
            max_seq = max(int(r["updated_seq"]) for r in rows)
            await conn.execute(
                text(
                    f"SELECT setval('{self.seq}', GREATEST("
                    f"(SELECT last_value FROM {self.seq}), :max_seq))"
                ),
                {"max_seq": max_seq},
            )

    async def close(self) -> None:
        if self._owns_engine:
            await self._engine.dispose()


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()
