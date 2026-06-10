"""Cross-backend equivalence: Postgres and SQLite must rank the same corpus
identically for the same queries.

This is the executable form of the library's core promise — "index
server-side, retrieve on-device, get the same results." It is what caught the
stemming divergence (SQLite wasn't stemming) and what will catch the next such
regression automatically, instead of by hand.

Runs only when FARO_EMBEDDED_SEARCH_TEST_DSN points at a pgvector database.
A deterministic embedder (the shared `embedder` fixture) gives both backends
identical vectors; the corpus is small enough that Postgres HNSW returns exact
nearest neighbours, so any difference is a real ranking divergence, not ANN
noise.
"""

import os

import pytest

from faro_embedded_search import IndexDoc, SearchIndex
from faro_embedded_search.backends.sqlite import SQLiteBackend

DSN = os.environ.get("FARO_EMBEDDED_SEARCH_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="FARO_EMBEDDED_SEARCH_TEST_DSN not set")

from tests.conftest import DIM  # noqa: E402

CORPUS = [
    IndexDoc(object_type="note", object_id="n1", title="Grocery list",
             body="milk eggs bread butter", attrs={"folder": "home"}),
    IndexDoc(object_type="note", object_id="n2", title="Quantum entanglement",
             body="spooky action at a distance physics", attrs={"folder": "science"}),
    IndexDoc(object_type="note", object_id="n3", title="Bread recipe",
             body="flour water yeast salt baking oven", attrs={"folder": "home"}),
    IndexDoc(object_type="task", object_id="t1", title="Fix login bug",
             body="oauth redirect loop authentication", attrs={"folder": "work"}),
    IndexDoc(object_type="task", object_id="t2", title="Quarterly planning",
             body="roadmap milestones objectives", attrs={"folder": "work"}),
    IndexDoc(object_type="contact", object_id="c1", title="Ada Lovelace",
             body="mathematician analytical engine", attrs={"folder": "people"}),
    IndexDoc(object_type="contact", object_id="c2", title="Grace Hopper",
             body="compiler nanosecond navy programming", attrs={"folder": "people"}),
    IndexDoc(object_type="email", object_id="e1", title="Invoice due",
             body="payment reminder client overdue", attrs={"folder": "mail"}),
    IndexDoc(object_type="weather", object_id="w1", title="Weather forecast",
             body="temperature rain city tomorrow", attrs={"folder": "misc"}),
    IndexDoc(object_type="tool", object_id="tl1", title="Stripe payments",
             body="charge card customer checkout", attrs={"folder": "finance"}),
]

# Each query is curated to have an unambiguous order so equality is meaningful.
QUERIES = [
    {"query": "groceries"},                                   # stemming: -> grocery
    {"query": "bread baking yeast"},                          # n3 then n1
    {"query": "quantum physics distance"},                    # n2
    {"query": "roadmap planning", "object_types": ["task"]},  # type filter -> t2
    {"query": "yeast", "attrs": {"folder": "home"}},          # attrs filter -> n3
    {"query": "oauth authentication login"},                  # t1
    {"query": "compiler programming"},                        # c2
    {"query": "card payment checkout"},                       # tl1
]


async def _run_all(index: SearchIndex):
    out = []
    for q in QUERIES:
        res = await index.search(**q, k=5)
        out.append([(r.object_id, r.match_type) for r in res])
    return out


async def test_postgres_sqlite_rank_identically(embedder, tmp_path):
    from sqlalchemy import text

    from faro_embedded_search.backends.postgres import PostgresBackend

    pg = PostgresBackend(DSN, table="fs_parity", dim=DIM)
    async with pg._engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS fs_parity"))
        await conn.execute(text("DROP SEQUENCE IF EXISTS fs_parity_updated_seq"))
    await pg.create_schema()
    pg_index = SearchIndex(pg, embedder)
    await pg_index.upsert_many(CORPUS)

    sq_index = SearchIndex(SQLiteBackend(str(tmp_path / "parity.db")), embedder)
    await sq_index.upsert_many(CORPUS)

    pg_results = await _run_all(pg_index)
    sq_results = await _run_all(sq_index)

    diffs = [
        f"{QUERIES[i]['query']!r}: postgres={p} sqlite={s}"
        for i, (p, s) in enumerate(zip(pg_results, sq_results))
        if p != s
    ]

    await pg_index.close()
    await sq_index.close()

    assert not diffs, "server/device ranking diverged:\n  " + "\n  ".join(diffs)
