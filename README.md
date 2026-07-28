<h1 align="center">Search as Code</h1>

<p align="center"><b>One <code>pip install</code>. One API. Any vector database.</b></p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10+-blue">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green">
  <img alt="backends" src="https://img.shields.io/badge/backends-memory·opensearch·qdrant·chroma·pgvector·faiss·sqlite-orange">
  <img alt="tests" src="https://img.shields.io/badge/tests-95%20passing-brightgreen">
</p>

**Search as Code** is an agentic retrieval harness: instead of calling a fixed
`search()` endpoint, the LLM writes a short Python program against a unified
**primitive API** — search modes, fan-out, rerank, rephrase, fuse, dedup, MMR,
filter, aggregate — executed in a sandbox with intermediate state kept **out of
the model context**. The same agent code runs over **any vector database**
(OpenSearch, Qdrant, Chroma, pgvector, Pinecone, Weaviate, Milvus …) — no
per-database SDK, no per-database rewrite. It's *code-mode* retrieval (à la
Anthropic/Cloudflare) meets *search-as-code* primitives (à la Perplexity), made
database-agnostic.

```python
import search_as_code as sac

s = sac.Session("opensearch", index="docs", dim=768, embedder=my_embedder)
#   swap "opensearch" -> "qdrant" / "chroma" / "pgvector" / "memory" — nothing else changes

cands = s.search_many(["how do agents retrieve?", "agentic RAG"], top_k=40, mode="hybrid")
best  = s.rerank("how do agents retrieve?", cands, top_k=10)
print(best.to_evidence(fields=["title"]))     # compact, context-friendly
```

## 📦 Install

**Requires Python 3.10+.** Not yet on PyPI — install from source (editable):

```bash
git clone https://github.com/oro-jackson/searchascode-sdk.git
cd searchascode-sdk
pip install -e .                 # core: in-memory backend + dependency-free embedder (only needs numpy)
```

Then add the backend / extras you need:

| Command | Adds |
|---|---|
| `pip install -e '.[opensearch]'` | OpenSearch backend (also: `.[qdrant]` · `.[chroma]` · `.[pgvector]`) |
| `pip install -e '.[providers]'` | OpenAI embeddings + LLM |
| `pip install -e '.[all]'` | every backend + providers |
| `pip install -e '.[phase1]'` | everything to run the FiQA benchmark (torch, sentence-transformers, langchain) |
| `pip install -e '.[dev]'` | test + lint/type tooling (pytest, ruff, mypy) |

## 🚀 Quick start

```bash
python -c "import search_as_code as sac; print(sac.available())"   # ['chroma', 'faiss', 'memory', 'opensearch', ...]
python examples/demo.py                    # in-memory demo, zero setup, no API key
python examples/opensearch_quickstart.py   # needs .[opensearch] + a running OpenSearch
python -m pytest -q                        # 77 in-memory unit tests; +18 OpenSearch integration
```

The base install ships a dependency-free embedder + in-memory backend, so the
demo and unit tests run with **zero setup**.

## ⚡ Why it wins (measured, not claimed)

Benchmark on **BEIR FiQA** (57,638 docs in OpenSearch, **100 labeled queries**,
`gpt-4.1-mini`, gte-base embeddings + MS-MARCO reranker) — base hybrid search vs
MCP tool-calling vs Search-as-Code (totals over the 100 queries):

| mode | Recall@10 | nDCG@10 | latency | LLM calls | input tokens | cache hit | cost |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| base (hybrid) | 0.479 | 0.379 | 0.02 s | 0 | 0 | — | $0 |
| tool-calling (MCP) | 0.440 | 0.399 | 15.9 s | 6.2 | 416k | 27% | $0.23 |
| **Search as Code** | **0.549** | **0.408** | **7.7 s** | **2.6** | **335k** | **54%** | **$0.15** |

**SAC wins every axis that matters vs MCP tool-calling** — best Recall@10 (**+11
pts**) and nDCG@10, **~2.1× faster**, **~1.6× cheaper**, **2× the prompt-cache
hit** (54% vs 27%), and **<½ the LLM calls** — because intermediate results stay
in the sandbox instead of flowing back through context. Reproduce with
`python -m phase1.benchmark -n 100`; full run log in
[`benchmark_changelog.md`](benchmark_changelog.md), narrative in
[`phase1/RESULTS.md`](phase1/RESULTS.md).

## 🧰 Capabilities

| Group | What the agent can call |
|---|---|
| **Retrieval modes** | dense (ANN) · keyword (BM25) · hybrid (RRF) · regex · phrase/proximity · fuzzy · wildcard · prefix · fielded (field-boost) · more-like-this |
| **Query-side** | `rephrase` · `expand` · `decompose` · `hyde_search` · `prf_search` (Rocchio) · `smart_search` · `auto_filter` (self-query) |
| **Rank / fuse** | `rerank` (cross-encoder · lexical · Qwen) · RRF / weighted / relative-score fusion · `mmr` · `semantic_dedup` · `diversity_quota` |
| **Adaptive / gating** | `score_cutoff` · `adaptive_search` · `confidence` / `abstain` · `normalize_scores` |
| **Analysis** | `aggregate` · `facet` · `count_distinct` · `stats` |
| **Backends** | `memory` · `opensearch` · `qdrant` · `chroma` · `pgvector` · `faiss` · `sqlite` (+ `nmslib`, `milvus` refs) |
| **Harness** | sandboxed code-mode execution · out-of-context state store · capability emulation · retries + batched upserts · typed error codes |

See the full **[320-primitive taxonomy](docs/PRIMITIVES.md)** and the
**[primitive × database matrix](docs/DATABASES.md)** for exactly what each backend
supports natively vs. by emulation.

## 🧩 How it works

```
LLM writes Python  ─▶  Session (unified API, out-of-context state)
                        └─ Primitives: fan_out · fuse(RRF) · rerank · rephrase · dedup · mmr
                        └─ VectorStore protocol + capability emulation   ← DB differences hidden here
                        └─ adapters: memory · opensearch · qdrant · chroma · pgvector · faiss · sqlite
                   ─▶  Sandbox (only the final evidence returns to the model)
```

**Capability emulation** keeps agent code portable: if a backend lacks keyword or
hybrid search, the harness emulates it in-SDK, so `mode="hybrid"` behaves the same
everywhere. Add a backend by implementing one `VectorStore`
([`adapters/base.py`](search_as_code/adapters/base.py)) — `memory.py` is the
executable spec.

## 📚 Docs

| | |
|---|---|
| [`docs/CONCEPT.md`](docs/CONCEPT.md) | the idea + how 5 source articles map to the code |
| [`docs/PRIMITIVES.md`](docs/PRIMITIVES.md) | the 320-primitive canonical taxonomy |
| [`docs/DATABASES.md`](docs/DATABASES.md) | primitive × database support matrix |
| [`docs/CACHING.md`](docs/CACHING.md) | passing the SDK surface to the LLM efficiently |
| [`docs/SELECTION.md`](docs/SELECTION.md) | exposing the SDK in the prompt + how the LLM picks the right primitive (55 sources) |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | 150-source research base |
| [`benchmark_changelog.md`](benchmark_changelog.md) | scalability / throughput / latency / token benchmarks |
| [`phase1/`](phase1/) | OpenSearch benchmark: base vs tool-calling vs SAC + live UI |

## 🖥️ Live trace UI

```bash
streamlit run phase1/live_ui.py     # type a query → see all 3 modes' traces + a model-free primitives lab
streamlit run phase1/ui.py          # browse the static 100-query benchmark
```

## 🤝 Contributing

We develop in parallel using **git worktrees** (isolated checkouts per feature):

```bash
git worktree add ../sac-<feature> -b feat/<feature>   # isolated working copy
cd ../sac-<feature> && pip install -e '.[dev]'
# …make your change, then keep it green:
ruff check search_as_code && mypy search_as_code && pytest -q
```

- **Add a backend** by implementing one `VectorStore`
  ([`adapters/base.py`](search_as_code/adapters/base.py)); `memory.py` is the spec
  and the in-memory test suite is the contract every adapter must satisfy.
- **Add a primitive** in `primitives.py` (portable, model-free) or as a backend
  method (native); update [`docs/DATABASES.md`](docs/DATABASES.md).
- CI runs **ruff + mypy + pytest** on Python 3.10–3.12 plus a live-OpenSearch
  integration job ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## 🧭 Philosophy

The **database owns retrieval; the harness owns everything around it.** Expose
*real* primitives (not a monolithic `search()`), keep bulky intermediate state out
of the model context, and make the same agent program run on any vector DB.
Composite macros stay **bypassable** so generated code can always reach the atoms.
That split — DB-layer vs harness-layer — is exactly what the
[database matrix](docs/DATABASES.md) encodes.

## 🗺️ Roadmap

Shipped: unified primitive API · capability emulation · sandboxed code-mode
execution · 7 adapters (`memory`, `opensearch`, `qdrant`, `chroma`, `pgvector`,
`faiss`, `sqlite`) · LangChain SAC agent + tool-calling baseline · FiQA benchmark +
trace UI · typed errors · adapter resilience · CI.
Next: OpenSearch `sparse_neural_search` / native rerank / ColBERT (ml-commons) ·
hardened sandbox backends (Docker/e2b) · MCP server wrapper · learned
query→primitive router · more adapters (Pinecone, Weaviate, LanceDB).

## 📄 License

Apache-2.0 © 2026 search-as-code contributors.

<p align="center"><sub>search as code · agentic retrieval · code-mode · RAG · vector search · MCP · semantic search · LLM agents</sub></p>
