# Changelog

All notable changes to `faro-embedded-search` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-06-10

### Added
- **Multiple named embedding spaces.** An index can hold a vector from more
  than one model per row — e.g. a `server` space (a hosted model) and a
  `local` space (an on-device model) — each its own column/index, queried
  independently. Vectors from different models are never compared.
  - `SearchIndex(backend, embedders={"server": e1, "local": e2}, default_space="server")`
  - Backends take a `spaces` config (`PostgresBackend(..., spaces={"server": 1536, "local": 384})`,
    `SQLiteBackend(..., spaces=("local",))`).
  - `IndexDoc.embed_spaces` lets an object type opt out of a space — e.g.
    emails embed server-side only, so on-device they're keyword-searchable but
    not semantic until you query the server.
  - `SearchIndex.search(..., space=...)` selects which space to query.
  - `export_shard(..., spaces=("local",))` ships a device shard carrying only
    the on-device space (smaller — no server vectors).

### Changed
- The internal sync change-row format renamed `embedding` →
  `embeddings: {space: vector}`. The single-embedder API is unchanged
  (`SearchIndex(backend, embedder)` → one `"default"` space).
- The default space's column is `embedding_default` (was `embedding` in 0.2.0).
  As this is pre-1.0 with no migration path yet, re-create/backfill a 0.2.0
  index rather than upgrading it in place.

[0.3.0]: https://github.com/poolside-ventures/faro-embedded-search/releases/tag/v0.3.0

## [0.2.0] - 2026-06-10

### Fixed
- **Cross-backend stemming parity.** SQLite FTS5 now uses the `porter`
  tokenizer, so morphological variants match the same way Postgres
  `to_tsvector('english')` stems them (e.g. "groceries" matches "grocery").
  Previously the SQLite (device) backend did not stem, so it could rank the
  same corpus differently from the Postgres (server) backend — a violation of
  the library's identical-semantics guarantee.

### Added
- **Generic attribute filter.** `IndexDoc.attrs` (a JSON dict) plus an
  `attrs=` argument to `SearchIndex.search` filters results by equality /
  containment (e.g. `attrs={"category": "finance"}`). Backed by a JSONB GIN
  index on Postgres and `json_extract` on SQLite. Removes the need to overload
  `partition` for non-isolation filters.
- `py.typed` marker so downstream type checkers see the library's annotations.
- Idempotent in-place schema upgrades: `create_schema()` (Postgres) and
  `SQLiteBackend(...)` add the new `attrs` column to indexes created by 0.1.0.

[0.2.0]: https://github.com/poolside-ventures/faro-embedded-search/releases/tag/v0.2.0

## [0.1.0] - 2026-06-10

Initial release.

### Added
- `SearchIndex` facade: incremental per-row `upsert` / `delete` (no rebuild
  step exists) and hybrid `search` with partition, object-type, and node-kind
  filters.
- Rank-based **Reciprocal Rank Fusion** (RRF, K=60) in the core, so every
  backend ranks identically for the same corpus and query regardless of its
  native lexical/vector score scales.
- **Postgres backend** (`[postgres]` extra): pgvector HNSW + weighted
  `tsvector` GIN, idempotent `create_schema()`.
- **SQLite backend** (stdlib only): FTS5 (BM25) + exact cosine scan
  (numpy-accelerated via the `[numpy]` extra). Its schema is the documented
  on-device shard interchange format.
- Node-kind tiering (`leaf` / `summary` / `cluster`) in one flat pool with
  per-object collapsing — an incremental-safe alternative to batch RAPTOR trees.
- Per-object-type indexer registry for heterogeneous objects.
- Pluggable embedders, including an `OpenAICompatibleEmbedder` (OpenAI,
  LiteLLM proxies, self-hosted). Embedding failure is non-fatal.
- Cursor-based shard replication (`replicate`, `export_shard`) with
  tombstone deletes, for the index-server-side / retrieve-on-device pattern.

[0.1.0]: https://github.com/poolside-ventures/faro-embedded-search/releases/tag/v0.1.0
