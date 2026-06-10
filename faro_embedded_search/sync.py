"""Shard replication: copy index rows between backends without re-embedding.

The intended deployment is `index server-side, retrieve on-device`:
embeddings are computed once on the server (Postgres backend), and each
user's partition is replicated into a local SQLite shard the device
queries with identical semantics.
"""

from __future__ import annotations

from .backends.base import Backend
from .backends.sqlite import SQLiteBackend


async def replicate(
    source: Backend,
    dest: Backend,
    *,
    partition: str | None = None,
    cursor: int | None = None,
    batch: int = 500,
) -> int | None:
    """Pump changes from source to dest; returns the new cursor.

    Pass the previous return value as `cursor` for incremental delta sync
    (tombstones propagate deletes). Pass None for a full export.
    """
    while True:
        rows, cursor = await source.changes_since(
            cursor, partition=partition, limit=batch
        )
        if not rows:
            return cursor
        await dest.apply_changes(rows)


async def export_shard(
    source: Backend, dest_path: str, *, partition: str | None = None
) -> SQLiteBackend:
    """Export a partition into a fresh SQLite shard file and return it open."""
    dest = SQLiteBackend(dest_path)
    await replicate(source, dest, partition=partition)
    return dest
