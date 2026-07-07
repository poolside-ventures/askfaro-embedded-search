# RFC: `core-search` — on-device embedded vector search in a single Rust core

**Status:** Draft · **Date:** 2026-06-20 · **Owner:** Faro
**Destined for:** the `faro-core` monorepo (this draft lives in `askfaro-embedded-search` because that is the engine it evolves; move on adoption).
**Consumers:** Scope (customer #0), and any app that wants local-first retrieval plug-and-play.

---

## 1. Summary

Make `askfaro-embedded-search` run **on-device** — fast, offline, private — with **vector search**, **incremental offline indexing**, and a **pluggable, per-device embedder**, as a **single Rust core** with thin language bindings (Python, Swift/Kotlin, WASM). The library is already designed for the *shape* of this (embedding spaces, SQLite shard contract, sync). What's missing is a native implementation and one embedder decision. This RFC consolidates the decisions and names the open questions.

The same engine powers **all** on-device search — agent tool selection (PCX), skills, notes, contacts, emails, tasks — not a per-feature retriever.

## 2. Goals / non-goals

**Goals**
- One retrieval engine, identical ranking on server and device (the existing promise, made native).
- Vector + lexical hybrid (RRF) on-device, over a replicated per-user SQLite shard.
- Incremental, offline indexing: new local items are embedded and searchable immediately, no rebuild.
- Pluggable embedder selected per device; **zero config** for the implementer.
- Drop-in coexistence with a customer's existing embedding (e.g. they use Pinecone) — we add our owned space, never replace theirs.

**Non-goals**
- Replacing a customer's cloud vector store. We are additive on-device infra.
- Running PCX *manifest construction* on-device (that stays server-side; see §8).
- Frontier embedding quality. We optimize quality-per-MB for small, fixed catalogs and per-user corpora.

## 3. Background — what already exists (verified in code)

- **Pure-Python library, zero deps.** It is the server + the reference implementation + the **SQLite interchange contract** (`backends/sqlite.py` header: the schema "IS the shard format … for a future Swift reader"). Match the schema → rank-compatible with the server.
- **Embedding spaces** (`index.py`): multiple embedders, each its own column; per-doc opt-in; shard carries only chosen spaces. Implemented + tested (`test_multi_space_dual_model_and_device_shard`).
- **Pluggable `Embedder`** (`embedder.py`): one async method `embed(texts) -> list[list[float] | None]`. No provider assumptions.
- **Hybrid retrieval:** FTS5 (`porter unicode61`, bm25) lexical + brute-force cosine semantic, fused by **RRF K=60** (ranks, not scores → identical across SQLite/Postgres). Parity is enforced by `test_parity.py`.
- **Sync:** `export_shard` / `replicate` / delta-sync (monotonic cursor + tombstones), tested. **Write-time and query-time embedding are independent** — the device can inherit server-precomputed corpus vectors and only embed the query (and any offline-created local writes).

## 4. Decision: single Rust core + thin bindings

Author the engine **once in Rust** and bind it everywhere, rather than maintaining parallel Python + Rust implementations.

- **Bindings:** PyO3 (Python), UniFFI (Swift/Kotlin), WASM (web). Precedent: HuggingFace `tokenizers` and `pydantic-core` are exactly Rust-core + PyO3 native wheels.
- **Why:** eliminates ranking drift entirely — there is nothing to keep in sync code-wise; only *data* (shards) sync, which the engine already handles.
- **Cost:** lose the pure-Python zero-dep property (ship native wheels per platform). Accepted.
- **The core owns both backends** behind one trait: **SQLite** (device) and **Postgres/pgvector** (server, via `sqlx`). See open question §11.1.

```
                 ┌──────────────────────── core-search (Rust) ────────────────────────┐
                 │  query orchestration · RRF fusion · the contract                    │
                 │  ┌─────────────┐   ┌──────────────────┐   ┌────────────────────┐    │
                 │  │ Backend     │   │ EmbedEngine      │   │ ModelManager       │    │
                 │  │  - SQLite   │   │  (pluggable,     │   │ (download/verify,  │    │
                 │  │  - Postgres │   │   per-device)    │   │  shared w/ core-stt)│   │
                 │  └─────────────┘   └──────────────────┘   └────────────────────┘    │
                 └───────┬───────────────────┬───────────────────────┬─────────────────┘
            PyO3 (server)│        UniFFI (iOS/Android)        WASM (web) / Tauri
```

Mirrors `core-stt`: opt-in heavy native dep, **host-downloads / crate-verifies** model files, thin API that hides the runtime.

## 5. Components

1. **Retriever** — reads the SQLite shard schema (float32 BLOB `embedding_<space>`, FTS5 `porter unicode61`, RRF K=60), rank-identical to the Python server. New deps: `rusqlite`; add an ANN (`sqlite-vec` / `usearch`) when shards outgrow the brute-force cosine scan (§11.2).
2. **`EmbedEngine`** (pluggable, per-device): default **EmbeddingGemma** ONNX via `ort` (reuses core-stt's `ort` + `tokenizers` + `ndarray` stack); optional Apple `NLContextualEmbedding` for a device-local-only index; OpenAI-compatible / server vectors when online.
3. **Backends** — SQLite + Postgres behind one trait (§4).
4. **Shared `ModelManager`** — promote `ModelSpec`/`verify` out of `core-stt` into a shared crate used by both STT and search.

## 6. The embedder

**Default: EmbeddingGemma-300M** (~200 MB, 768-dim Matryoshka → 256/128, multilingual 100+ langs). Chosen for **cross-lingual** retrieval (see §6.2). Benchmark backing in Appendix A.

### 6.1 The portable-output principle (why embedding ≠ STT / brain)

"Use the native model, skip the download" is correct **only when the output is portable**:
- **STT** → transcript (text). Portable. On Apple, use SpeechAnalyzer, skip the download. ✅
- **Brain (LLM)** → tool call / text. Portable. On Apple, use Foundation Models, skip the download. ✅
- **Embedding** → a vector in a **model-specific space**. **Not portable.** Every device + the server must agree on one model, or their vectors can't be compared. ❌

So embeddings invert the rule: **ship one shared cross-platform embedder everywhere, including Apple.** This is cheap — the embedder (~200 MB) is an order of magnitude smaller than the brain (~1 GB) or STT (~150 MB–1 GB). Apple's native embedder saves only ~200 MB and would break cross-device parity, so it is reserved for a device-*local-only* index, never the synced/shared space.

### 6.2 Cross-lingual

A multilingual embedder maps all languages into one shared meaning-space, so a **German query retrieves English-indexed content** (and vice-versa). Only the **semantic half** bridges languages — the lexical/FTS half never does (`Aufgabe` ≠ `task`). Cross-lingual retrieval therefore rides entirely on the embedding; this is the reason the multilingual model is chosen.

### 6.3 Future efficiency: language right-sizing
- **STT:** ship the user's language model (smaller/faster/often more accurate). Clean win.
- **Embedding:** do **not** mix monolingual embedders per content-language — that breaks cross-lingual and reintroduces incompatible spaces. Instead **right-size the single shared model** to the user's actual language footprint (an English-only user can get a tiny English embedder; a multilingual user gets EmbeddingGemma). Invariant: everyone in one index shares one model/space.

## 7. Embedding spaces & the hard rule

- **Shared "device" space:** the EmbeddingGemma vectors, computed identically on server and device, shipped in the shard. This is what makes cross-device search work (iPhone + Windows laptop + server all agree).
- **Bring-your-own coexistence:** a customer keeps their own embedding as a **second space** alongside ours; query ours, theirs, or fuse via RRF. We never replace their model.
- **The hard rule:** vectors compare **only within one space**. Model identity == space name. Changing the embedding model = a **new space** + coordinated cross-repo backfill; never an in-place re-embed. Pin `(model, version) → space` in the contract.

## 8. PCX integration (build server-side, retrieve on-device)

- **PCX manifest construction** (LLM-assisted summarization/cataloging) stays **server-side, online, build-time**. Not ported to device.
- **PCX retrieval** rides `core-search` **on-device** over a synced manifest shard.
- Offline-created tools/skills lack a manifest entry until they sync and the server processes them — acceptable.

## 9. Sync & offline indexing

- Server indexes (embeddings computed once, centrally); each user's slice replicates to a local SQLite shard; the device retrieves locally with identical ranking.
- **Offline incremental:** the device embeds new local items with the **same** shared model → same space → immediately searchable. On reconnect the server may re-embed authoritatively. Delta sync via cursor; tombstones propagate deletes.

## 10. Zero-config plug-and-play

The implementer calls `upsert` / `search` and nothing else. `core-search`:
- ships a **default embedder** and auto-downloads + verifies its model (shared `ModelManager`),
- **auto-selects** per device (bundled ONNX default; Apple local-only optional; server when online),
- **owns the device space end-to-end** so parity is automatic.

**The contract** (the one thing docs must make loud): SQLite schema · RRF K=60 · tokenizer `porter unicode61` · `(model, version) → space`. A shared **golden conformance suite** (extend `test_parity.py`) asserts every binding/backend ranks an identical corpus identically.

## 11. Open questions

1. **Postgres-in-Rust.** Port the pgvector/tsvector backend into the Rust core (via `sqlx`) so one core owns both backends, or keep a thin Python Postgres adapter feeding the shared Rust fusion? Full single-core argues for the former; it's the main lift.
2. **ANN vs brute-force.** SQLite semantic search is currently an exact cosine scan — fine for small per-user shards, but pick an ANN (`sqlite-vec` / `usearch`) threshold for larger ones. File format already tolerates adding it.
3. **EmbeddingGemma dims.** Validate recall at Matryoshka-truncated 256-dim (halves shard storage) before committing the stored dimensionality.
4. **Mobile inference runtime.** Confirm `ort` (ONNX Runtime) builds + performs for EmbeddingGemma on iOS/Android; else evaluate alternatives (Core ML export, MLC).

## 12. Phasing

1. **Spike:** EmbeddingGemma on `ort` in Rust → confirm §11.4 + recall parity with the Python reference on a shared corpus.
2. **`core-search` crate:** SQLite retriever + `EmbedEngine` + shared `ModelManager`; conformance suite green vs Python.
3. **Bindings:** PyO3 first (lets the server adopt the core), then UniFFI + WASM (device).
4. **Backends:** resolve §11.1 (Postgres-in-Rust).
5. **Scope integration:** register tools/skills/notes/contacts/emails/tasks as object types; wire PCX retrieval to `core-search`.

---

## Appendix A — empirical backing (Scope F-7, 2026-06-20, M1 Pro)

On-device tool-calling with progressive disclosure, full 34-tool Scope catalog:

| Approach | Tool acc | Context | Notes |
|---|---|---|---|
| Flat 34 schemas → caller | 72% | 13,298 tok | breaks Apple FM 4k window |
| PCX via caller-as-router (wrong) | 39% | 2.3k | a function-caller is a bad retriever |
| **PCX via embedded search → caller** | **89%** | **2.3k** | retrieval recall@5 = 100%, fits 4k |

Embedder quality on the 34-tool corpus (recall of gold tool, correct per-model prompt prefixes):

| Model | Size | Dims | r@1 | r@3 | r@5 | query |
|---|---|---|---|---|---|---|
| multilingual-e5-small | ~113 MB | 384 | 22% | 67% | 72% | 88 ms |
| nomic-embed-text (English-first) | ~80 MB | 768 | 50% | 83% | 100% | 29 ms |
| **EmbeddingGemma-300M** | ~200 MB | 768 | 67% | **100%** | **100%** | 99 ms |

Takeaways: progressive disclosure must select via **embedded search**, not the caller; **EmbeddingGemma** is the quality leader (and cross-lingual); E5-small is eliminated on quality. Caveat: 18 tool queries, tool-corpus only — confirm on a notes/contacts sample before final lock.
