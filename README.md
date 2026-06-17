# faro-embedded-search

**Incremental hybrid retrieval with identical semantics on the server and on-device.**

`faro-embedded-search` is a small, dependency-light library for searching a *continuously growing* collection of heterogeneous objects — notes, contacts, emails, tasks, tools, anything — without ever rebuilding an index. It is built for the pattern modern assistant apps need:

> **Index server-side, retrieve on-device.** Embeddings are computed once, centrally. Each user's slice of the index is replicated into a local SQLite shard, and the device answers queries locally — fast, offline, and private — with *exactly* the same ranking the server would produce.

Built and dogfooded by Faro; also the embedded retrieval engine of Scope.

## Why another retrieval library?

Most RAG tooling assumes a batch world: ingest a corpus, build a tree/graph/index, query it. Real applications have **object-level CRUD** — a contact edited, a task created, an email deleted — and small/on-device context windows that punish irrelevant results. `faro-embedded-search` makes two opinionated choices:

1. **Incremental-first.** One upsert per object write. Postgres (pgvector HNSW + tsvector GIN) and SQLite (FTS5 + vector blobs) both take per-row inserts natively, so there is no "rebuild" anywhere in the design. Hierarchical enrichment (summary and cluster nodes) lives in the *same flat pool* as extra rows — never on the write path.
2. **Rank-based fusion.** Lexical and semantic retrievers run in parallel and are fused with Reciprocal Rank Fusion (RRF, K=60). Because RRF consumes *ranks*, not raw scores, the incomparable scoring scales of `ts_rank_cd` (Postgres) and `bm25()` (SQLite FTS5) don't matter: the same corpus and query rank identically on both backends. That is what makes "same engine, server and device" honest rather than aspirational.

## Quick start

```python
from faro_embedded_search import IndexDoc, SearchIndex, OpenAICompatibleEmbedder
from faro_embedded_search.backends.sqlite import SQLiteBackend

index = SearchIndex(
    SQLiteBackend("search.db"),
    OpenAICompatibleEmbedder("https://api.openai.com/v1", api_key, "text-embedding-3-small"),
)

await index.upsert(IndexDoc(
    object_type="note", object_id="n1",
    title="Quantum entanglement notes",
    body="spooky action at a distance",
    partition="user-42",                      # isolation + shard key
))

results = await index.search("entanglement", partition="user-42", k=5)
# SearchResult(object_id='n1', match_type='hybrid', score=..., ...)
```

Deleting and updating are first-class — deletes are tombstones so they propagate through shard sync:

```python
await index.delete("note", "n1")
```

### Filtering

Filter by partition (the isolation/shard key), object type, node kind, or arbitrary structured attributes:

```python
await index.upsert(IndexDoc(
    object_type="tool", object_id="stripe/refund",
    title="Refund a charge", body="...",
    attrs={"category": "finance", "status": "active"},   # structured filter fields
))

# Only finance tools:
await index.search("refund a customer", object_types=["tool"], attrs={"category": "finance"})
```

`attrs` is matched by containment (all given keys must equal), backed by a JSONB GIN index on Postgres and `json_extract` on SQLite — so it filters identically on server and device.

#### Partitions

`partition` is the per-tenant isolation key (and the unit of on-device shard replication). It is **optional and defaults to `None`**, and a query's `partition` filter is an **exact match**: `search(..., partition="user-42")` returns only rows upserted with `partition="user-42"`. Two consequences worth knowing up front, because both fail quietly rather than with an error:

- A row indexed **without** a partition (`None`) will **not** appear in a query that filters by a partition, and vice-versa. Pick one convention per corpus — either always set a partition, or never — and don't mix.
- `search()` with no `partition` does **not** scope to a tenant; it searches across every partition. Pass the partition on every query in a multi-tenant index.

### Server-side: Postgres

```python
from faro_embedded_search.backends.postgres import PostgresBackend  # pip install faro-embedded-search[postgres]

backend = PostgresBackend("postgresql+asyncpg://...", table="faro_embedded_search_index", dim=1536)
await backend.create_schema()     # idempotent; or transcribe the DDL into your migrations
index = SearchIndex(backend, embedder)
```

### On-device: replicate a shard, query locally

```python
from faro_embedded_search import export_shard, replicate

# Full export of one user's partition into a SQLite file:
shard = await export_shard(server_backend, "user-42.db", partition="user-42")

# Later, incremental delta sync (inserts, updates, AND deletes):
cursor = await replicate(server_backend, shard, partition="user-42", cursor=cursor)
```

The shard is a plain SQLite file whose schema **is** the interchange format — any runtime that can read SQLite (a future Swift/Kotlin reader, for instance) can retrieve against it. Embeddings travel with the shard; the device never needs to re-embed the corpus. Query embedding on-device can use a local model, a cached vector, or one tiny server round-trip for the query alone.

## Tiering without batch trees

RAPTOR-style hierarchies buy small-context windows a lot — but a recursive tree can't be rebuilt on every insert. `faro-embedded-search` keeps the index flat and gets the benefit through **node kinds**:

- `leaf` — the object itself (default).
- `summary` — an optional one-line abstract of an object, indexed alongside it. O(1) per object, generated on your write path or a background pass.
- `cluster` — optional theme summaries produced by a periodic background sweep over existing embeddings.

All kinds live in the same pool and are retrieved by the same top-k ("collapsed" retrieval); by default multiple hits on one object collapse into its best row, with `matched_node_kinds` telling you which handles matched:

```python
await index.upsert_many([
    IndexDoc(object_type="note", object_id="n9", title="Meeting notes", body=transcript),
    IndexDoc(object_type="note", object_id="n9", node_kind="summary",
             title="Summary: hiring sync", body="decided to open two roles"),
])
```

## Multiple embedding spaces (dual-model / on-device)

A row can carry vectors from more than one model — typically a high-quality **server** model and a smaller **on-device** model — each as its own independently-queried space. Vectors from different models are never compared, so there's no mismatch.

```python
index = SearchIndex(
    backend,                                    # configured with both spaces
    embedders={"server": openai, "local": on_device_minilm},
    default_space="server",
)

# A note participates in both spaces (default); an email is server-only:
await index.upsert_many([
    IndexDoc(object_type="note",  object_id="n1", title="...", body="..."),
    IndexDoc(object_type="email", object_id="e1", title="...", body="...",
             embed_spaces=["server"]),         # no on-device vector
])

# On the server, query the server space; on-device, query the local space:
await index.search("quarterly plan", space="server")   # web/server
await index.search("quarterly plan", space="local")    # device shard
```

The device shard carries only the on-device space (smaller — no server vectors):

```python
shard = await export_shard(server_backend, "user-42.db",
                           partition="user-42", spaces=("local",))
```

Because a server-only object (the email above) has no local vector, on-device it's **keyword-searchable but not semantic** — its text still syncs, so FTS finds it offline; semantic search on it requires querying the server. This falls out of the per-space null-vector handling, no special-casing. Configure backends with matching spaces: `PostgresBackend(..., spaces={"server": 1536, "local": 384})`, `SQLiteBackend(..., spaces=("local",))`.

## Heterogeneous objects

Per-type behavior lives in code, not schema, via a tiny registry:

```python
from faro_embedded_search import register, docs_for

@register("contact")
def index_contact(c) -> IndexDoc:
    return IndexDoc(object_type="contact", object_id=str(c.id),
                    title=c.name, body=f"{c.role} at {c.company}",
                    payload={"avatar": c.avatar_url})

await index.upsert_many(docs_for("contact", some_contact))
```

`payload` is carried into results (and shards), so result lists render without joining back to your application database.

## Design notes

- **Embedding failure is non-fatal, but never silent.** A row written without a vector still serves lexical queries and gains semantic retrieval after a backfill — availability over completeness. Every embedding failure (at index time *and* query time) is logged at `WARNING` on the `faro_embedded_search` logger with the exception, so a misconfigured embedder (bad key, wrong endpoint) surfaces instead of silently degrading search to keyword-only. Wire that logger up in your app to catch it.
- **Exact semantic scan on SQLite.** Per-user shards are small (tens of thousands of rows); an exact cosine scan (numpy-accelerated when present) costs no index maintenance and returns exact results. ANN acceleration (e.g. sqlite-vec) can be added without changing the file format.
- **Diversity, not padding.** An optional per-group cap (`diversity_key`) drops near-duplicate siblings instead of deferring them.
- **Stemming parity.** Postgres uses the `english` text-search config; SQLite FTS5 uses the `porter` tokenizer. Both stem morphological variants ("groceries" → "grocery") so the server and a device shard rank the same corpus identically.
- **No framework.** Plain SQL on both backends, zero required dependencies in the core.

## Requirements

- **Python 3.11+.** The core (SQLite backend) needs only the standard library — SQLite's FTS5 and JSON1 extensions ship with CPython's bundled SQLite.
- **Postgres backend** (`[postgres]` extra) needs a database with the **pgvector** extension available. `create_schema()` runs `CREATE EXTENSION IF NOT EXISTS vector` and the rest of the DDL in one idempotent call (or transcribe it into your own migrations). It is safe to re-run; it also adds new columns in place when you upgrade the library.

## Troubleshooting

The library raises clear, actionable errors for the common setup mistakes (all subclasses of `FaroSearchError`):

| Symptom | Cause | Fix |
|---|---|---|
| `MissingDependencyError: PostgresBackend requires the 'postgres' extra` | Installed the core only | `pip install "faro-embedded-search[postgres]"` |
| `MissingDependencyError: OpenAICompatibleEmbedder requires the 'http' extra` | Missing httpx | `pip install "faro-embedded-search[http]"` |
| `ConfigurationError: pgvector extension isn't available…` | Postgres server has no `vector` extension | Enable/install [pgvector](https://github.com/pgvector/pgvector), or use `SQLiteBackend` (no extension needed) |
| `ConfigurationError: Embedding for space 'x' has dimension N, but the index is configured for M` | Embedder output dim ≠ the space's column dim | Match the dims: configure the backend with `spaces={'x': N}` (then re-create the schema), or use an embedder that emits M-dim vectors |
| `ConfigurationError: …embedders for space(s) [...] that the backend doesn't define` | `SearchIndex` has an embedder for a space the backend wasn't configured with | Give the backend matching spaces: `PostgresBackend(..., spaces={...})` / `SQLiteBackend(..., spaces=(...))` |
| `ConfigurationError: This SQLite build lacks the FTS5 extension` | Rare CPython build without FTS5 | Use a Python whose bundled SQLite has FTS5 |
| Postgres error: relation `…_index` does not exist | Forgot to set up the schema | Call `await backend.create_schema()` once (or run the equivalent migration) before indexing |

Other expected behaviours (not errors): searching a space with **no embedder** returns lexical-only results, and a doc indexed without a vector still serves keyword search — both by design (availability over completeness).

## Installation

```bash
pip install faro-embedded-search                # core (SQLite backend, stdlib only)
pip install faro-embedded-search[postgres]      # + Postgres/pgvector backend
pip install faro-embedded-search[http]          # + OpenAI-compatible embedder
pip install faro-embedded-search[numpy]         # + fast cosine on SQLite
```

## License

MIT
