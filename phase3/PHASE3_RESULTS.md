# Phase 3 — multi-backend integration & cross-DB relevance

Goal: prove the core thesis — **one primitive API, any vector DB** — by implementing
additional adapters and running the *same* HotpotQA dense retrieval on each backend,
then comparing relevance. See `docs/PHASE3.md` for the full backend plan.

## What was built
- **FAISS adapter** (`search_as_code/adapters/faiss_store.py`) — in-process, no server.
  Exact `IndexFlatIP` (cosine on normalized vectors). Composes `MemoryStore` for
  keyword/regex/hybrid so agent code behaves identically to a full backend.
- **SQLite adapter** (`search_as_code/adapters/sqlite_store.py`) — the "no vector DB
  needed" reference: float32 vectors as BLOBs in a plain SQL table, brute-force cosine
  in numpy. Persistent, stdlib-only, zero server.
- Both registered in the adapter registry; conformance-smoke-tested identical to the
  `memory` reference on dense / keyword / hybrid / metadata-filter.

## Cross-DB relevance — same HotpotQA vectors, same API, N=60 queries
Vectors were scrolled **out of the existing OpenSearch index** (100,978 docs, no
re-embedding) and loaded into each in-process backend, so every backend sees identical
vectors. Dense `query_vector(top_k=10)` via the one primitive API.

| backend | engine | recall@10 | all_found@10 | avg latency |
|---|---|---|---|---|
| OpenSearch (default) | HNSW m=16, ef_c=100 (under-built) | 0.792 | 0.617 | 2.8 ms |
| OpenSearch (tuned) | HNSW m=48, ef_c=512 | 0.900 | 0.800 | — |
| Milvus-lite | AUTOINDEX (embedded) | 0.875 | 0.767 | 140.7 ms |
| Chroma | HNSW (well-tuned default) | 0.908 | 0.817 | **0.9 ms** |
| FAISS | IndexFlatIP (exact) | **0.925** | **0.850** | 5.2 ms |
| SQLite | BLOB brute-force (exact) | **0.925** | **0.850** | 19.4 ms |
| memory | numpy brute-force (exact) | **0.925** | **0.850** | 60.5 ms |
| Qdrant | local mode (exact-like) | **0.925** | **0.850** | 185.8 ms |
| nmslib | HNSW (M=32, ef_c=200) | **0.925** | **0.850** | 354.7 ms |

**8 backends, one API** (`sac.connect(<backend>, ...)`; `--extra` adds Chroma/Qdrant/nmslib/Milvus).
Five hit *identical* exact relevance (0.925/0.850: FAISS, SQLite, memory, Qdrant, nmslib); Chroma's
HNSW is near-exact at the lowest latency; Milvus-lite slightly below; only OpenSearch's *default* HNSW
lags — and its tuned re-index (m=48) reaches 0.900 (see below). Qdrant needed an adapter fix (below).

## Findings
1. **Parity across exact backends.** FAISS, SQLite, and memory return *identical*
   relevance (0.925 / 0.850). Same vectors + same metric ⇒ same results, regardless of
   store. This is the "one API, any DB" thesis, demonstrated end-to-end.
2. **OpenSearch HNSW silently loses ~13 recall / ~23 all_found points** vs exact search
   on this corpus — a large gap.
3. **The gap is a tuning problem, not inherent to ANN.** Two independent proofs:
   (a) raising OpenSearch `ef_search` (100 → 512 → 2048) changed *nothing* (0.7917
   throughout) → the loss is baked into the graph at build time (`m` / `ef_construction`
   defaults, m=16), recoverable only by re-index; (b) **Chroma's HNSW on the identical
   vectors reaches 0.908** — 12 points above OpenSearch's 0.792 and near exact — at the
   *lowest* latency (0.8 ms). So a well-configured ANN nearly matches brute force; the
   OpenSearch default is simply under-built. Production gotcha: don't trust default HNSW
   build params; measure against exact.
4. **This reframes the HotpotQA headline.** The reported dense baseline (0.79) was
   depressed by ANN approximation — *exact* dense already reaches **0.925**, close to
   SAC's 0.96. So part of SAC's apparent multi-hop advantage over "dense" was actually
   **recovering ANN-approximation losses** (its fan-out + rerank re-surface true nearest
   neighbours the HNSW graph missed), on top of genuine multi-hop bridging. An honest,
   important nuance: agentic retrieval helps partly by *compensating for a lossy index*.

## Tuned-HNSW re-index — confirming the diagnosis
Rebuilt the HotpotQA index as `hotpotqa_tuned` with **m=48, ef_construction=512** (via
OpenSearch `_reindex`, reusing the same vectors — no re-embedding) and re-measured dense:

| HotpotQA dense (n=60) | recall@10 | all_found@10 |
|---|---|---|
| OpenSearch HNSW default (m=16, ef_c=100) | 0.792 | 0.617 |
| **OpenSearch HNSW tuned (m=48, ef_c=512)** | **0.900** | **0.800** |
| Exact (FAISS/SQLite/memory/Qdrant) | 0.925 | 0.850 |

Tuning recovers **+11 recall / +18 all_found points** — ~85% of the gap to exact — and
confirms the underperformance was a **build-parameter artifact**, not inherent to ANN or
the primitive API. (Note: `_reindex` completed server-side but opensearch-py raised
"got more than 100 headers" on the long-poll; the tuned index has all 100,978 docs.)

## Latency note
Exact brute force is fine at this scale (100k×768): FAISS 5 ms, SQLite 18 ms, pure numpy
60 ms per query. For ≤~1M vectors, exact search is a legitimate, higher-recall option;
ANN's value is throughput at much larger scale, at a measurable recall cost that must be
tuned (m/ef_construction), not assumed away.

## Status vs plan (`docs/PHASE3.md`)
| backend | status |
|---|---|
| memory, OpenSearch | ✅ shipped |
| **FAISS, SQLite** | ✅ new, measured (this report) |
| **Chroma** | ✅ measured (HNSW 0.908) |
| **Qdrant** | ✅ measured (0.925) after adapter fix: arbitrary string ids → uuid5 (orig in payload), and `search()`→`query_points()` for qdrant-client ≥1.10 |
| **nmslib** | ✅ measured (HNSW 0.925 — matches exact) |
| **Milvus-lite** | ✅ measured (embedded, 0.875) |
| Elasticsearch, MongoDB, Milvus-server | need a server (Docker socket denied here) |
| Pinecone | needs a cloud API key |

## Reproduce
```bash
python -m phase3.cross_db_relevance --n 60          # scroll OS vectors -> faiss/sqlite/memory, compare
```

_Follow-up: re-index HotpotQA with m=48 / ef_construction=512 and re-baseline dense
against exact; extend the matrix to Chroma/Qdrant/nmslib/Milvus-lite._
