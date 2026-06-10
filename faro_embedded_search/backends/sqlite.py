"""SQLite backend — the on-device store and the shard interchange format.

Lexical retrieval uses FTS5 (BM25), kept in sync with the base table via
the canonical external-content triggers. Semantic retrieval is an exact
cosine scan over packed float32 blobs (numpy-accelerated when available);
exact scan is fast enough for per-user shards (tens of thousands of rows)
and trades zero index-maintenance for it. ANN acceleration via sqlite-vec
can be added without changing the file format.

The schema of this file IS the shard format: any runtime (e.g. a future
Swift reader) that understands this schema can retrieve against a shard
exported by a server backend.

Embedding **spaces**: a row can carry a vector from more than one model —
e.g. a `server` space (a hosted model) and a `local` space (an on-device
model). Each space is its own column, queried independently; vectors from
different models are never compared. A device shard typically carries only
the `local` space.
"""

from __future__ import annotations

import json
import re
import sqlite3
from array import array
from math import sqrt
from typing import Any, Sequence

from ..errors import ConfigurationError
from ..types import Filters, IndexDoc, RawHit, utcnow_iso
from .base import ChangeRow

try:
    import numpy as _np
except ImportError:  # pure-python cosine fallback
    _np = None

DEFAULT_SPACE = "default"

# Base table (no vector columns — those are added per space, see _ensure_columns).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_index (
    id INTEGER PRIMARY KEY,
    partition_key TEXT,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    node_kind TEXT NOT NULL DEFAULT 'leaf',
    title TEXT,
    body TEXT,
    payload TEXT,
    attrs TEXT,
    source_updated_at TEXT NOT NULL,
    embedding_indexed_at TEXT,
    deleted_at TEXT,
    updated_seq INTEGER NOT NULL,
    UNIQUE (object_type, object_id, node_kind)
);
CREATE INDEX IF NOT EXISTS ix_si_partition
    ON search_index (partition_key, object_type) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS ix_si_seq ON search_index (updated_seq);
CREATE TABLE IF NOT EXISTS search_meta (key TEXT PRIMARY KEY, value TEXT);
-- `porter` stemming so morphological variants match the way Postgres'
-- to_tsvector('english') stems them ("groceries" -> "grocery"). Without it,
-- SQLite (device) and Postgres (server) would rank the same corpus
-- differently, breaking the library's identical-semantics guarantee.
CREATE VIRTUAL TABLE IF NOT EXISTS search_fts
    USING fts5(title, body, content='search_index', content_rowid='id',
               tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS search_index_ai AFTER INSERT ON search_index BEGIN
    INSERT INTO search_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END;
CREATE TRIGGER IF NOT EXISTS search_index_ad AFTER DELETE ON search_index BEGIN
    INSERT INTO search_fts(search_fts, rowid, title, body)
        VALUES ('delete', old.id, old.title, old.body);
END;
CREATE TRIGGER IF NOT EXISTS search_index_au AFTER UPDATE ON search_index BEGIN
    INSERT INTO search_fts(search_fts, rowid, title, body)
        VALUES ('delete', old.id, old.title, old.body);
    INSERT INTO search_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END;
"""

_IDENT = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_space(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(
            f"invalid embedding space name {name!r}: must match [a-z][a-z0-9_]*"
        )
    return name


def _pack(vec: Sequence[float]) -> bytes:
    return array("f", vec).tobytes()


def _unpack(blob: bytes) -> array:
    a = array("f")
    a.frombytes(blob)
    return a


def _fts_query(query: str) -> str | None:
    """Quote each token so user input can't break FTS5 syntax (AND semantics,
    matching plainto_tsquery on the Postgres side)."""
    tokens = re.findall(r"\w+", query)
    if not tokens:
        return None
    return " ".join(f'"{t}"' for t in tokens)


class SQLiteBackend:
    def __init__(self, path: str = ":memory:", *, spaces: Sequence[str] = (DEFAULT_SPACE,)):
        self.spaces = [_validate_space(s) for s in spaces]
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        try:
            self._conn.executescript(_SCHEMA)
        except sqlite3.OperationalError as e:
            if "fts5" in str(e).lower():
                raise ConfigurationError(
                    "This SQLite build lacks the FTS5 extension, which the lexical "
                    "retriever needs. Most CPython builds include it; if yours doesn't, "
                    "use a Python whose bundled SQLite has FTS5 (or rebuild SQLite with "
                    "-DSQLITE_ENABLE_FTS5)."
                ) from e
            raise
        self._ensure_columns()
        self._conn.commit()

    @staticmethod
    def _col(space: str) -> str:
        return f"embedding_{space}"

    def _ensure_columns(self) -> None:
        """Add the attrs column and each space's vector columns if absent —
        so opening an index created by an older version (or with fewer spaces)
        just works."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(search_index)")}
        if "attrs" not in cols:
            self._conn.execute("ALTER TABLE search_index ADD COLUMN attrs TEXT")
        for space in self.spaces:
            col = self._col(space)
            if col not in cols:
                self._conn.execute(f"ALTER TABLE search_index ADD COLUMN {col} BLOB")
                self._conn.execute(
                    f"ALTER TABLE search_index ADD COLUMN {col}_dim INTEGER"
                )

    # -- write path -------------------------------------------------------

    def _next_seq(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM search_meta WHERE key = 'seq'"
        ).fetchone()
        seq = int(row["value"]) + 1 if row else 1
        self._conn.execute(
            "INSERT INTO search_meta (key, value) VALUES ('seq', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(seq),),
        )
        return seq

    def _bump_seq_floor(self, seq: int) -> None:
        """Keep the local counter monotonic past replicated sequence numbers."""
        row = self._conn.execute(
            "SELECT value FROM search_meta WHERE key = 'seq'"
        ).fetchone()
        if row is None or int(row["value"]) < seq:
            self._conn.execute(
                "INSERT INTO search_meta (key, value) VALUES ('seq', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(seq),),
            )

    def _upsert(self, row: ChangeRow, seq: int) -> None:
        embeddings = row.get("embeddings") or {}
        base_cols = [
            "partition_key", "object_type", "object_id", "node_kind", "title",
            "body", "payload", "attrs", "source_updated_at",
            "embedding_indexed_at", "deleted_at", "updated_seq",
        ]
        base_vals: list[Any] = [
            row.get("partition"),
            row["object_type"],
            row["object_id"],
            row.get("node_kind", "leaf"),
            row.get("title"),
            row.get("body"),
            json.dumps(row["payload"]) if row.get("payload") is not None else None,
            json.dumps(row["attrs"]) if row.get("attrs") is not None else None,
            row["source_updated_at"],
            row.get("embedding_indexed_at"),
            row.get("deleted_at"),
            seq,
        ]
        space_cols: list[str] = []
        space_vals: list[Any] = []
        for space in self.spaces:
            vec = embeddings.get(space)
            space_cols += [self._col(space), f"{self._col(space)}_dim"]
            space_vals += [_pack(vec) if vec else None, len(vec) if vec else None]

        cols = base_cols + space_cols
        vals = base_vals + space_vals
        placeholders = ", ".join("?" * len(cols))
        # Every non-key column updates from the incoming row on conflict.
        updatable = [c for c in cols if c not in ("object_type", "object_id", "node_kind")]
        set_clause = ", ".join(f"{c} = excluded.{c}" for c in updatable)
        self._conn.execute(
            f"INSERT INTO search_index ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT (object_type, object_id, node_kind) DO UPDATE SET {set_clause}",
            vals,
        )

    async def upsert_row(
        self,
        doc: IndexDoc,
        embeddings: dict[str, list[float] | None],
        embedding_indexed_at: str | None,
    ) -> None:
        self._upsert(
            {
                "partition": doc.partition,
                "object_type": doc.object_type,
                "object_id": doc.object_id,
                "node_kind": doc.node_kind,
                "title": doc.title,
                "body": doc.body,
                "payload": doc.payload,
                "attrs": doc.attrs,
                "embeddings": embeddings,
                "source_updated_at": doc.source_updated_at,
                "embedding_indexed_at": embedding_indexed_at,
                "deleted_at": None,
            },
            self._next_seq(),
        )
        self._conn.commit()

    async def delete_row(
        self, object_type: str, object_id: str, node_kind: str | None = None
    ) -> None:
        # Tombstone and strip content: the row must survive (so sync can
        # propagate the delete) but its text and vectors must not.
        null_vecs = ", ".join(
            f"{self._col(s)} = NULL, {self._col(s)}_dim = NULL" for s in self.spaces
        )
        sql = f"""
            UPDATE search_index
            SET deleted_at = ?, title = NULL, body = NULL,
                {null_vecs + ',' if null_vecs else ''} updated_seq = ?
            WHERE object_type = ? AND object_id = ? AND deleted_at IS NULL
        """
        params: list[Any] = [utcnow_iso(), self._next_seq(), object_type, object_id]
        if node_kind is not None:
            sql += " AND node_kind = ?"
            params.append(node_kind)
        self._conn.execute(sql, params)
        self._conn.commit()

    # -- retrievers -------------------------------------------------------

    def _filter_sql(self, filters: Filters) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if filters.partition is not None:
            clauses.append("si.partition_key = ?")
            params.append(filters.partition)
        if filters.object_types:
            clauses.append(
                f"si.object_type IN ({','.join('?' * len(filters.object_types))})"
            )
            params.extend(filters.object_types)
        if filters.node_kinds:
            clauses.append(
                f"si.node_kind IN ({','.join('?' * len(filters.node_kinds))})"
            )
            params.extend(filters.node_kinds)
        if filters.attrs:
            for key, value in filters.attrs.items():
                clauses.append(f"json_extract(si.attrs, '$.{key}') = ?")
                params.append(value)
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    @staticmethod
    def _hit(row: sqlite3.Row, sim: float | None = None) -> RawHit:
        return RawHit(
            object_type=row["object_type"],
            object_id=row["object_id"],
            node_kind=row["node_kind"],
            partition=row["partition_key"],
            title=row["title"],
            payload=json.loads(row["payload"]) if row["payload"] else None,
            sim=sim,
        )

    async def lexical_search(
        self, query: str, filters: Filters, limit: int
    ) -> list[RawHit]:
        match = _fts_query(query)
        if match is None:
            return []
        where, params = self._filter_sql(filters)
        rows = self._conn.execute(
            f"""
            SELECT si.*, bm25(search_fts) AS lex_score
            FROM search_fts
            JOIN search_index si ON si.id = search_fts.rowid
            WHERE search_fts MATCH ? AND si.deleted_at IS NULL{where}
            ORDER BY lex_score ASC
            LIMIT ?
            """,
            [match, *params, limit],
        ).fetchall()
        return [self._hit(r) for r in rows]

    async def semantic_search(
        self,
        query_vec: Sequence[float],
        space: str,
        filters: Filters,
        limit: int,
        min_score: float,
    ) -> list[RawHit]:
        if space not in self.spaces:
            return []
        col = self._col(space)
        where, params = self._filter_sql(filters)
        rows = self._conn.execute(
            f"""
            SELECT si.* FROM search_index si
            WHERE si.{col} IS NOT NULL AND si.deleted_at IS NULL{where}
            """,
            params,
        ).fetchall()
        scored = []
        for row in rows:
            sim = _cosine(query_vec, _unpack(row[col]))
            if sim >= min_score:
                scored.append((sim, row))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [self._hit(row, sim=sim) for sim, row in scored[:limit]]

    # -- sync -------------------------------------------------------------

    async def changes_since(
        self, cursor: int | None, *, partition: str | None = None, limit: int = 500
    ) -> tuple[list[ChangeRow], int | None]:
        sql = "SELECT * FROM search_index WHERE updated_seq > ?"
        params: list[Any] = [cursor or 0]
        if partition is not None:
            sql += " AND partition_key = ?"
            params.append(partition)
        sql += " ORDER BY updated_seq ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        keys = set(rows[0].keys()) if rows else set()
        changes = []
        for r in rows:
            embeddings = {}
            for space in self.spaces:
                col = self._col(space)
                if col in keys and r[col] is not None:
                    embeddings[space] = list(_unpack(r[col]))
            changes.append(
                {
                    "object_type": r["object_type"],
                    "object_id": r["object_id"],
                    "node_kind": r["node_kind"],
                    "partition": r["partition_key"],
                    "title": r["title"],
                    "body": r["body"],
                    "payload": json.loads(r["payload"]) if r["payload"] else None,
                    "attrs": json.loads(r["attrs"]) if r["attrs"] else None,
                    "embeddings": embeddings,
                    "source_updated_at": r["source_updated_at"],
                    "embedding_indexed_at": r["embedding_indexed_at"],
                    "deleted_at": r["deleted_at"],
                    "updated_seq": r["updated_seq"],
                }
            )
        new_cursor = changes[-1]["updated_seq"] if changes else cursor
        return changes, new_cursor

    async def apply_changes(self, rows: Sequence[ChangeRow]) -> None:
        for row in rows:
            self._upsert(row, row["updated_seq"])
            self._bump_seq_floor(row["updated_seq"])
        self._conn.commit()

    async def close(self) -> None:
        self._conn.close()


def _cosine(a: Sequence[float], b: array) -> float:
    if _np is not None:
        av = _np.asarray(a, dtype=_np.float32)
        bv = _np.frombuffer(b.tobytes(), dtype=_np.float32)
        if av.shape != bv.shape:
            return 0.0
        denom = float(_np.linalg.norm(av)) * float(_np.linalg.norm(bv))
        return float(av @ bv) / denom if denom else 0.0
    if len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = sqrt(na) * sqrt(nb)
    return dot / denom if denom else 0.0
