"""Backend protocol.

A backend owns storage and the two raw retrievers (lexical, semantic).
Fusion, collapsing, and diversity live in the core so every backend yields
identical search semantics.

Change rows — the sync contract between backends — are plain dicts:

    {
        "object_type": str,
        "object_id": str,
        "node_kind": str,
        "partition": str | None,
        "title": str | None,
        "body": str | None,
        "payload": dict | None,
        "attrs": dict | None,
        "embeddings": {space: list[float]},  # one entry per populated space
        "source_updated_at": str (ISO 8601),
        "embedding_indexed_at": str | None,
        "deleted_at": str | None,
        "updated_seq": int,
    }

`updated_seq` is a backend-local monotonic counter bumped on every write
(including tombstones), giving exact incremental delta sync with a single
integer cursor.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from ..types import Filters, IndexDoc, RawHit

ChangeRow = dict[str, Any]


@runtime_checkable
class Backend(Protocol):
    spaces: Sequence[str]  # configured embedding-space names

    async def upsert_row(
        self,
        doc: IndexDoc,
        embeddings: dict[str, list[float] | None],
        embedding_indexed_at: str | None,
    ) -> None: ...

    async def delete_row(
        self, object_type: str, object_id: str, node_kind: str | None = None
    ) -> None:
        """Tombstone (not hard-delete) so deletions propagate through sync."""
        ...

    async def lexical_search(
        self, query: str, filters: Filters, limit: int
    ) -> list[RawHit]: ...

    async def semantic_search(
        self,
        query_vec: Sequence[float],
        space: str,
        filters: Filters,
        limit: int,
        min_score: float,
    ) -> list[RawHit]: ...

    async def changes_since(
        self, cursor: int | None, *, partition: str | None = None, limit: int = 500
    ) -> tuple[list[ChangeRow], int | None]:
        """Rows with updated_seq > cursor, ordered; returns (rows, new_cursor)."""
        ...

    async def apply_changes(self, rows: Sequence[ChangeRow]) -> None:
        """Replicate change rows verbatim (embeddings included, no re-embed)."""
        ...

    async def close(self) -> None: ...
