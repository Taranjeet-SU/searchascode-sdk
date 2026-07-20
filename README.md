# search-as-code

**One `pip install`. One API. Any vector database.**

A *search-as-code* agentic harness: agents write portable Python against a single
primitive API — fan-out, fuse, rerank, dedup, extract — executed in a sandbox
with intermediate state kept **out of the model context**, no matter which vector
DB is underneath. No per-database SDK for the agent to learn.

```python
import search_as_code as sac

s = sac.Session("memory")                 # swap "memory" -> "qdrant" / "chroma" / "pgvector"
s.add([{"id": "1", "text": "vector databases power agentic retrieval"}])

hits = s.search("agent retrieval", top_k=5, mode="hybrid")
print(hits.to_evidence(fields=["title"]))   # compact, context-friendly
```

The agent code above is **identical** on every backend. Change the one string in
`Session(...)` and nothing else moves.

## Why

Five threads converge on the same idea (see `docs/CONCEPT.md` for the mapping):

- **Code-mode** (Cloudflare, Anthropic): LLMs write *code* far better than they emit
  tool-calls; run it in a sandbox and only final results return to the model.
- **Search-as-code** (Perplexity): expose the search stack as *atomic primitives*,
  not a monolithic `search()`; let the model orchestrate fan-out / fusion / verify.
- **Retrieval bottleneck** (Hornet, BrowseComp-Plus / arXiv:2508.06600): the
  retriever sets the accuracy ceiling — so the primitive layer must be good, and
  measurable.

This project is the missing piece: a **unified layer** so that harness works over
*every* vector DB behind one API.

## Architecture

```
agent-generated Python
        │  writes against
        ▼
┌─────────────────────────────────────────────┐
│ Session (harness handle)  ── out-of-context state store
│   search / search_many / rerank / fuse / extract
├─────────────────────────────────────────────┤
│ Primitives  (fan_out, fuse=RRF, dedup, rerank, freshness)   ← model-free, portable
├─────────────────────────────────────────────┤
│ VectorStore protocol  +  Capability emulation   ← DB differences hidden here
├──────────┬──────────┬──────────┬──────────────┤
│  memory  │  qdrant  │  chroma  │  pgvector ... │  ← adapters, same contract
└──────────┴──────────┴──────────┴──────────────┘
        ▲ runs inside
┌─────────────────────────────────────────────┐
│ Sandbox (LocalExecutor today; Docker/e2b/Pyodide pluggable) │
└─────────────────────────────────────────────┘
```

| Layer | Module | Role |
|---|---|---|
| Data model | `types.py` | `Document`, `Hit`, `ResultSet`, `Capabilities` — the lingua franca |
| Primitives | `primitives.py` | Portable atoms: fan-out, RRF fusion, dedup, rerank, freshness, extract |
| Adapters | `adapters/` | `VectorStore` contract; `memory` (reference), `qdrant`, `chroma`, `pgvector` |
| Harness | `session.py` | Binds backend+embedder, capability-aware search, out-of-context state |
| Sandbox | `sandbox.py` | Runs agent code; returns only `print`/`evidence` to the model |
| Filters | `filters.py` | One portable Mongo-ish filter dialect → translated per backend |

**Capability emulation** is what keeps agent code portable: if a backend lacks
keyword or hybrid search, the harness emulates it in-SDK (dense recall + lexical
rerank, RRF fusion) so `mode="hybrid"` behaves the same everywhere.

## Install

```bash
pip install -e .                 # core (numpy only) — memory backend works out of the box
pip install -e '.[qdrant]'       # + Qdrant
pip install -e '.[chroma]'       # + Chroma
pip install -e '.[pgvector]'     # + Postgres/pgvector
pip install -e '.[dev]'          # + pytest
```

The base install ships a dependency-free `HashEmbedder` and in-memory backend, so
the demo and tests run with no API key and no external services.

## Run it

```bash
python -m pytest -q          # 13 tests over the in-memory reference backend
PYTHONPATH=. python examples/demo.py
```

## The primitive API (what agent code writes)

```python
s = sac.Session("qdrant", collection="docs", embedder=my_embedder)

# fan out concurrently, RRF-fuse
cands = s.search_many(["q1", "q2", "q3"], top_k=8, mode="hybrid")

# keep the bulky set in the sandbox, out of context
s.remember("cands", cands)

# narrow to the few facts worth returning
best = s.rerank("original question", cands.where(lambda h: h.get("year") >= 2024), top_k=5)
evidence = best.to_evidence(fields=["title", "url"])   # only this goes back to the model
```

## Primitive-class × database support matrix

Rows are the 15 primitive classes from the [canonical taxonomy](docs/PRIMITIVES.md);
columns are vector databases. **Legend:** ✅ Supported · ⚪ Not eligible · ❌ Not
supported · 🕒 Planned.

*"Not eligible"* means the class is **not a database concern** — it belongs to the
SDK harness/host runtime (query rewriting, planning, sandboxed execution, evidence
verification, output rendering, evaluation). That split is the point of this
project: the DB owns retrieval, the harness owns everything around it.

| # · Primitive class | Layer | Mem | Qdr | Chr | pgv | Pine | Weav | Milv | ES | Vsp | Mgo | Rds |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 · Data contracts | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 1 · Source & corpus | **db** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2 · Query processing | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 3 · Search planning | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 4 · Candidate generation | **db** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 5 · Candidate manipulation | **db** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 6 · Content materialization | **db** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 7 · Scoring & ranking | **db** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 8 · Aggregation & analysis | **db** | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | 🕒 | ✅ | ✅ | ✅ | ✅ |
| 9 · Evidence & verification | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 10 · Context & output | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 11 · Runtime | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 12 · State & observability | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 13 · Evaluation & learning | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 14 · Composite macros | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |

At class granularity the database-layer classes are broadly ✅ (every listed store
does source/retrieval/filter/index/score); the real variation is in **class 8**
and inside the retrieval/ranking classes. The drill-down below opens those up.

### Capability drill-down (where the variation lives)

Each row is a database-relevant capability tagged with its parent class number.

| Capability (class) | Mem | Qdr | Chr | pgv | Pine | Weav | Milv | ES | Vsp | Mgo | Rds |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Source & schema introspection (1) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dense vector search (4) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Lexical / full-text BM25 (4) | ✅ | ✅ | 🕒 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Sparse-neural — SPLADE/ELSER/BM42 (4) | ❌ | ✅ | ❌ | 🕒 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Multi-vector / late-interaction — ColBERT (4) | ❌ | ✅ | ❌ | 🕒 | ✅ | 🕒 | ✅ | 🕒 | ✅ | ❌ | ❌ |
| Structured / scalar / SQL retrieval (4) | ❌ | ❌ | ❌ | ✅ | ❌ | 🕒 | 🕒 | ✅ | ✅ | ✅ | ✅ |
| Graph / traversal / community (4) | ❌ | ❌ | ❌ | 🕒 | ❌ | 🕒 | ❌ | ❌ | ❌ | ❌ | ❌ |
| Regex / exact-match (4) | ✅ | 🕒 | ✅ | ✅ | ❌ | ❌ | 🕒 | ✅ | 🕒 | ✅ | 🕒 |
| Geospatial / temporal retrieval (4·5) | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | 🕒 | ✅ | ✅ | ✅ | ✅ |
| Metadata filtering & governance (5) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Native indexing / upsert (6) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Server-side embedding (6) | ❌ | 🕒 | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🕒 | ❌ |
| Scoring — BM25 / vector similarity (7) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hybrid fusion — RRF / score (7) | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Native reranking — cross-encoder / multi-stage (7) | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Diversification / MMR (7) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🕒 | ❌ | ❌ |
| Aggregation & analytics (8) | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | 🕒 | ✅ | ✅ | ✅ | ✅ |

**A ❌ is "not native," not "unavailable to your agent."** Via capability
negotiation the SDK still exposes hybrid, keyword, regex, rerank (pluggable),
MMR, dedup, freshness, and compression on **every** backend by emulating them
client-side — so agent code is portable regardless of the marks above. The matrix
reflects native DB capability as of the 2025–2026 research base
([docs/RESEARCH.md](docs/RESEARCH.md)) and will drift as vendors ship features.
Columns: Mem = in-memory reference · Qdr Qdrant · Chr Chroma · pgv pgvector ·
Pine Pinecone · Weav Weaviate · Milv Milvus · ES Elasticsearch · Vsp Vespa ·
Mgo MongoDB Atlas · Rds Redis. **Bold "db" adapters shipped today: Mem, Qdr, Chr,
pgv;** the rest are native-capability references pending adapters.

## Adding a backend

Implement the `VectorStore` contract (`adapters/base.py`), return `Hit`s with
larger-is-better scores, translate the portable filter dialect, and declare
`capabilities()` honestly — the harness emulates whatever you report as `False`.
`adapters/memory.py` is the executable spec. Register with
`sac.register("mystore", MyStore)`.

## Status & roadmap

v0 (this repo): unified primitive API, adapter layer + capability emulation,
in-context-free state, local sandbox, reference + 3 real adapters, tests, demo.

Next: hardened sandbox backends (Docker/e2b/Pyodide) behind the `Sandbox`
interface · a code-mode agent loop (surface API → generate code → execute →
evidence) · an MCP server wrapper · a BrowseComp-Plus-style eval harness · more
adapters (Pinecone, Weaviate, Milvus, LanceDB).
