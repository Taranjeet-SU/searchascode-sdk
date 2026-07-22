<h1 align="center">Search as Code</h1>

<p align="center"><b>One <code>pip install</code>. One API. Any vector database.</b></p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10+-blue">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green">
  <img alt="backends" src="https://img.shields.io/badge/backends-memory·qdrant·chroma·pgvector·opensearch-orange">
  <img alt="tests" src="https://img.shields.io/badge/tests-85%20passing-brightgreen">
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

## ⚡ Why it wins (measured, not claimed)

Benchmark on **BEIR FiQA** (57k docs in OpenSearch, 100 labeled queries,
`gpt-4.1-mini`) — base hybrid search vs MCP tool-calling vs Search-as-Code:

| mode | Recall@10 | latency | LLM calls | input tokens | cache hit | cost |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| base (hybrid) | 0.479 | 0.02 s | 0 | 0 | — | $0 |
| tool-calling (MCP) | 0.348 | 5.9 s | 5.5 | 254k | 8% | $0.117 |
| **Search as Code** | **0.491** | 3.7 s | **2.0** | **142k** | **78%** | **$0.043** |

**SAC wins every axis that matters** — best recall, **2.7× cheaper** and **1.8×
fewer tokens** than tool-calling (intermediate results stay in the sandbox), 78%
prompt-cache hit, and 100% of generated programs executed cleanly. Full write-up:
[`phase1/RESULTS.md`](phase1/RESULTS.md).

## 🚀 Quickstart

```bash
pip install -e .                 # core: in-memory backend, no services, no API key
pip install -e '.[opensearch]'   # + OpenSearch     (also: qdrant / chroma / pgvector)
python examples/opensearch_quickstart.py
python -m pytest -q              # 77 unit tests (in-memory); +8 OpenSearch integration
```

The base install ships a dependency-free embedder + in-memory backend, so the
demo and unit tests run with zero setup.

## 🧩 How it works

```
LLM writes Python  ─▶  Session (unified API, out-of-context state)
                        └─ Primitives: fan_out · fuse(RRF) · rerank · rephrase · dedup · mmr
                        └─ VectorStore protocol + capability emulation   ← DB differences hidden here
                        └─ adapters: memory · opensearch · qdrant · chroma · pgvector
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
| [`phase1/`](phase1/) | OpenSearch benchmark: base vs tool-calling vs SAC + live UI |

## 🖥️ Live trace UI

```bash
streamlit run phase1/live_ui.py     # type a query → see all 3 modes' traces live
streamlit run phase1/ui.py          # browse the static 100-query benchmark
```

## Status

Shipped: unified primitive API · capability emulation · sandboxed code-mode
execution · 5 adapters (`memory`, `opensearch`, `qdrant`, `chroma`, `pgvector`) ·
LangChain SAC agent + tool-calling baseline · FiQA benchmark + trace UI.
Next: hardened sandbox backends (Docker/e2b) · MCP server wrapper · more adapters
(Pinecone, Weaviate, Milvus, LanceDB) · native rerankers.

<p align="center"><sub>search as code · agentic retrieval · code-mode · RAG · vector search · MCP · semantic search · LLM agents</sub></p>
