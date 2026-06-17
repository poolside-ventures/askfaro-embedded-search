"""Postgres backend tests — run only when FARO_EMBEDDED_SEARCH_TEST_DSN is set, e.g.

    FARO_EMBEDDED_SEARCH_TEST_DSN=postgresql+asyncpg://postgres:postgres@localhost:5434/askfaro_embedded_search_test

The database must have the pgvector extension available.
"""

import os

import pytest

from askfaro_embedded_search import IndexDoc, SearchIndex, export_shard

DSN = os.environ.get("FARO_EMBEDDED_SEARCH_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="FARO_EMBEDDED_SEARCH_TEST_DSN not set")

from tests.conftest import DIM  # noqa: E402


@pytest.fixture
async def index(embedder):
    from sqlalchemy import text

    from askfaro_embedded_search.backends.postgres import PostgresBackend

    backend = PostgresBackend(DSN, table="fs_test_index", dim=DIM)
    await backend.create_schema()
    idx = SearchIndex(backend, embedder)
    yield idx
    async with backend._engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS fs_test_index"))
        await conn.execute(text("DROP SEQUENCE IF EXISTS fs_test_index_updated_seq"))
    await idx.close()


async def test_postgres_multi_space(embedder):
    from sqlalchemy import text

    from askfaro_embedded_search import IndexDoc, SearchIndex
    from askfaro_embedded_search.backends.postgres import PostgresBackend

    backend = PostgresBackend(DSN, table="fs_multi", spaces={"server": DIM, "local": DIM})
    async with backend._engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS fs_multi"))
        await conn.execute(text("DROP SEQUENCE IF EXISTS fs_multi_updated_seq"))
    await backend.create_schema()
    idx = SearchIndex(
        backend, embedders={"server": embedder, "local": embedder},
        default_space="server",
    )
    await idx.upsert_many([
        IndexDoc(object_type="note", object_id="n1", title="Quantum notes",
                 body="entanglement spooky action", partition="p1"),
        IndexDoc(object_type="email", object_id="e1", title="Invoice",
                 body="payment reminder overdue", partition="p1",
                 embed_spaces=["server"]),
    ])
    # Server space: email embedded -> semantic hit.
    assert any(r.object_id == "e1" and r.match_type in ("semantic", "hybrid")
               for r in await idx.search("overdue payment", space="server"))
    # Local space: note embedded -> semantic hit; email has no local vector,
    # so it can only surface lexically.
    assert any(r.object_id == "n1"
               for r in await idx.search("quantum entanglement", space="local"))
    for r in await idx.search("overdue payment", space="local"):
        if r.object_id == "e1":
            assert r.match_type == "keyword"

    async with backend._engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS fs_multi"))
        await conn.execute(text("DROP SEQUENCE IF EXISTS fs_multi_updated_seq"))
    await idx.close()


async def test_postgres_end_to_end(index, embedder, tmp_path):
    await index.upsert_many([
        IndexDoc(object_type="note", object_id="n1",
                 title="Quantum entanglement notes",
                 body="spooky action at a distance", partition="acct-1",
                 payload={"icon": "atom"}, attrs={"folder": "physics"}),
        IndexDoc(object_type="task", object_id="t1", title="Fix login bug",
                 body="users report oauth redirect loop", partition="acct-1"),
        IndexDoc(object_type="note", object_id="n2", title="Grocery list",
                 body="milk eggs bread", partition="acct-2",
                 attrs={"folder": "home"}),
    ])

    # Stemming parity: 'groceries' matches 'Grocery' via the english stemmer,
    # the same way SQLite porter stems it.
    assert (await index.search("groceries"))[0].object_id == "n2"

    # attrs filter (JSONB containment).
    folder_hits = await index.search("grocery quantum", attrs={"folder": "physics"})
    assert [r.object_id for r in folder_hits] == ["n1"]
    assert not await index.search("grocery quantum", attrs={"folder": "nope"})

    # Hybrid: lexical + semantic agree.
    results = await index.search("grocery milk")
    assert results[0].object_id == "n2"
    assert results[0].match_type == "hybrid"
    assert results[0].payload is None

    # Semantic-only: AND-lexical misses on partial overlap.
    results = await index.search("quantum physics")
    assert any(r.object_id == "n1" and r.match_type == "semantic" for r in results)

    # Payload round-trips.
    results = await index.search("entanglement")
    assert results[0].payload == {"icon": "atom"}

    # Update replaces old content.
    await index.upsert(IndexDoc(object_type="note", object_id="n2",
                                title="Hardware list", body="screws bolts",
                                partition="acct-2"))
    assert not await index.search("grocery milk")

    # Tombstone delete.
    await index.delete("task", "t1")
    assert not await index.search("oauth login")

    # Shard export: Postgres -> SQLite, identical retrieval semantics.
    shard_backend = await export_shard(index.backend, str(tmp_path / "s.db"),
                                       partition="acct-1")
    shard = SearchIndex(shard_backend, embedder)
    results = await shard.search("entanglement")
    assert [r.object_id for r in results] == ["n1"]
    assert results[0].payload == {"icon": "atom"}
    assert not await shard.search("hardware screws")  # other partition absent
    rows, _ = await shard.changes_since(None)
    tombstone = next(r for r in rows if r["object_id"] == "t1")
    assert tombstone["deleted_at"] is not None and tombstone["title"] is None
    await shard.close()
