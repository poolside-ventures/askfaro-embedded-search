# Changelog

All notable changes to `faro-embedded-search` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow [Semantic Versioning](https://semver.org/).

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
