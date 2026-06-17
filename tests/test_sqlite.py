import pytest

from askfaro_embedded_search import (
    CallableEmbedder,
    IndexDoc,
    SearchIndex,
    export_shard,
    replicate,
)
from askfaro_embedded_search.backends.sqlite import SQLiteBackend
from tests.conftest import bow_vector


@pytest.fixture
async def index(embedder):
    idx = SearchIndex(SQLiteBackend(), embedder)
    yield idx
    await idx.close()


def doc(oid: str, title: str, body: str = "", **kw) -> IndexDoc:
    return IndexDoc(object_type=kw.pop("object_type", "note"),
                    object_id=oid, title=title, body=body, **kw)


async def seed(index: SearchIndex):
    await index.upsert_many([
        doc("n1", "Quantum entanglement notes", "spooky action at a distance"),
        doc("n2", "Grocery list", "milk eggs bread"),
        doc("t1", "Fix login bug", "users report oauth redirect loop",
            object_type="task", partition="acct-1"),
        doc("t2", "Quarterly planning", "roadmap retrieval search milestones",
            object_type="task", partition="acct-2"),
    ])


async def test_hybrid_search_finds_exact_and_fuzzy(index):
    await seed(index)

    results = await index.search("grocery milk")
    assert results[0].object_id == "n2"
    assert results[0].match_type == "hybrid"

    # Partial token overlap: lexical AND-semantics misses, semantic still hits.
    results = await index.search("quantum physics")
    assert any(r.object_id == "n1" and r.match_type == "semantic" for r in results)


async def test_incremental_add_is_immediately_searchable(index):
    await seed(index)
    assert not await index.search("velociraptor")

    await index.upsert(doc("n3", "Velociraptor facts", "cretaceous predator"))
    results = await index.search("velociraptor")
    assert results and results[0].object_id == "n3"


async def test_update_replaces_old_content(index):
    await index.upsert(doc("n1", "Alpha topic"))
    await index.upsert(doc("n1", "Bravo topic"))

    assert not await index.search("alpha")
    results = await index.search("bravo")
    assert [r.object_id for r in results] == ["n1"]


async def test_delete_tombstones_and_strips_content(index):
    await seed(index)
    await index.delete("note", "n1")

    assert not await index.search("quantum entanglement")
    rows, _ = await index.changes_since(None)
    tombstone = next(r for r in rows if r["object_id"] == "n1")
    assert tombstone["deleted_at"] is not None
    assert tombstone["title"] is None and not tombstone["embeddings"]


async def test_filters(index):
    await seed(index)

    results = await index.search("planning roadmap", object_types=["task"])
    assert {r.object_type for r in results} == {"task"}

    results = await index.search("planning roadmap", partition="acct-1")
    assert all(r.partition == "acct-1" for r in results)
    assert not any(r.object_id == "t2" for r in results)


async def test_summary_nodes_collapse_to_one_object(index):
    await index.upsert_many([
        doc("n9", "Meeting notes 2026-06-02", "long transcript about hiring"),
        IndexDoc(object_type="note", object_id="n9", node_kind="summary",
                 title="Summary: hiring sync", body="decided to open two roles"),
    ])
    results = await index.search("hiring")
    assert len(results) == 1
    assert set(results[0].matched_node_kinds) == {"leaf", "summary"}


async def test_multi_space_dual_model_and_device_shard(tmp_path):
    # Two embedding spaces: a "server" model and an on-device "local" model.
    emb = CallableEmbedder(lambda ts: [bow_vector(t) for t in ts])
    backend = SQLiteBackend(spaces=("server", "local"))
    idx = SearchIndex(
        backend, embedders={"server": emb, "local": emb}, default_space="server"
    )
    await idx.upsert_many([
        # Note opts into all spaces (default) -> server + local vectors.
        IndexDoc(object_type="note", object_id="n1", title="Quantum notes",
                 body="entanglement spooky action", partition="p1"),
        # Email is server-only: no local vector will be produced.
        IndexDoc(object_type="email", object_id="e1", title="Invoice",
                 body="payment reminder client overdue", partition="p1",
                 embed_spaces=["server"]),
    ])
    # On the server space, the email IS semantically searchable.
    server = await idx.search("overdue payment", space="server")
    assert any(r.object_id == "e1" and r.match_type in ("semantic", "hybrid")
               for r in server)

    # Device shard carries ONLY the local space (smaller; no server vectors).
    shard = SearchIndex(
        await export_shard(backend, str(tmp_path / "device.db"),
                           partition="p1", spaces=("local",)),
        embedders={"local": emb}, default_space="local",
    )
    await idx.close()
    # Note has a local vector -> full semantic search offline.
    assert any(r.object_id == "n1"
               for r in await shard.search("quantum entanglement", space="local"))
    # Email has NO local vector -> never a semantic/hybrid hit on device...
    for r in await shard.search("overdue payment", space="local"):
        if r.object_id == "e1":
            assert r.match_type == "keyword"
    # ...but its text synced, so keyword search still finds it offline.
    assert any(r.object_id == "e1"
               for r in await shard.search("invoice", space="local"))
    await shard.close()


async def test_lexical_only_when_no_embedder():
    idx = SearchIndex(SQLiteBackend(), embedder=None)
    await idx.upsert(doc("n1", "Quantum entanglement notes"))
    results = await idx.search("entanglement")
    assert results and results[0].match_type == "keyword"
    await idx.close()


async def test_porter_stemming_matches_morphological_variants(index):
    # 'groceries' (query) must match 'grocery' (indexed) via FTS5 porter
    # stemming — the same way Postgres to_tsvector('english') stems them.
    # This is the cross-backend parity guarantee: server and device rank the
    # same corpus identically.
    await index.upsert(doc("n1", "Grocery list", "things to buy"))
    results = await index.search("groceries")
    assert [r.object_id for r in results] == ["n1"]


async def test_attrs_filter(index):
    await index.upsert_many([
        IndexDoc(object_type="tool", object_id="t1", title="Stripe payments",
                 body="charge customer cards", attrs={"category": "finance"}),
        IndexDoc(object_type="tool", object_id="t2", title="Twilio messaging",
                 body="charge ahead and send a text", attrs={"category": "comms"}),
    ])
    fin = await index.search("charge", attrs={"category": "finance"})
    assert [r.object_id for r in fin] == ["t1"]
    comms = await index.search("charge", attrs={"category": "comms"})
    assert [r.object_id for r in comms] == ["t2"]
    # Multi-key containment: both keys must match.
    none = await index.search("charge", attrs={"category": "finance", "tier": "pro"})
    assert none == []


async def test_attrs_round_trip_through_shard(index, embedder, tmp_path):
    await index.upsert(IndexDoc(object_type="tool", object_id="t1", title="Stripe",
                                body="payments", partition="p1",
                                attrs={"category": "finance"}))
    shard = SearchIndex(await export_shard(index.backend, str(tmp_path / "s.db"),
                                           partition="p1"), embedder)
    hit = (await shard.search("payments", attrs={"category": "finance"}))[0]
    assert hit.object_id == "t1"
    await shard.close()


async def test_shard_export_and_delta_sync(index, embedder, tmp_path):
    await seed(index)
    shard_path = str(tmp_path / "shard.db")
    shard = SearchIndex(await export_shard(index.backend, shard_path,
                                           partition="acct-1"), embedder)

    # Shard contains only acct-1 rows and answers identically for them.
    results = await shard.search("oauth login")
    assert [r.object_id for r in results] == ["t1"]
    assert not await shard.search("grocery milk")

    # Delta sync: new server row + a delete both propagate via cursor.
    _, cursor = await index.changes_since(None, partition="acct-1")
    await index.upsert(doc("t3", "Renew certificate", "tls cert expires soon",
                           object_type="task", partition="acct-1"))
    await index.delete("task", "t1")
    cursor = await replicate(index.backend, shard.backend,
                             partition="acct-1", cursor=cursor)

    assert (await shard.search("certificate"))[0].object_id == "t3"
    assert not await shard.search("oauth login")

    # Cursor is stable: nothing new means no rows moved.
    assert await replicate(index.backend, shard.backend,
                           partition="acct-1", cursor=cursor) == cursor
    await shard.close()
