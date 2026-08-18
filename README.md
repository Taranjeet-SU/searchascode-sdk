<h1 align="center">Search as Code</h1>

<p align="center"><b>Your LLM writes retrieval code. Your corpus teaches it what works.<br>One <code>pip install</code>, any vector database.</b></p>

<p align="center">
  <a href="https://pypi.org/project/search-as-code/"><img alt="pypi" src="https://img.shields.io/pypi/v/search-as-code"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10+-blue">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green">
  <img alt="backends" src="https://img.shields.io/badge/backends-opensearch·qdrant·chroma·pgvector·faiss·sqlite·memory-orange">
</p>

---

**Search as Code (SaC)** replaces the fixed `search()` endpoint with a **primitive SDK the
LLM programs against**: one model turn writes a short Python strategy — search modes,
fan-out, RRF fusion, rerank, HyDE, MMR, dedup — executed in a sandbox where the bulky
intermediate results **never enter the model context**. A **continual harness** then makes
it corpus-specific: `explore` discovers what retrieval strategy your index rewards,
`forge` persists it as an inspectable, provenance-carrying primitive, and a **deep
LLM-as-judge** decides per query whether one cheap search was enough or the system should
go deeper — up to 10 authored hops.

Perplexity calls this architecture [the future of search for agents](https://research.perplexity.ai/articles/rethinking-search-as-code-generation).
**This is the open, bring-your-own-index version** — same agent code over OpenSearch,
Qdrant, Chroma, pgvector, FAISS, SQLite, or in-memory.

## 📊 The numbers (measured, with intervals — not vibes)

Tool-calling agents pay per hop: every result flows back through the context window.
Code-mode pays once. On **BrowseComp-Plus** (100k docs, Qwen3-Embedding-8B, n=100,
identical tools and search budget for both arms, gpt-4.1-mini):

| arm | model turns | input tokens/query | latency | recall@10 |
|---|:-:|:-:|:-:|:-:|
| tool-calling agent | 9.5 | 19,430 | 22.6 s | 0.088 |
| **Search as Code** | **1.0** | **629 (31× less)** | **13.0 s (1.7× faster)** | **0.124 (+41%)** |

<!-- GRAPH: docs/assets/cost_tokens.png — tokens per query, tool vs sac vs sac_explored -->
<!-- GRAPH: docs/assets/cost_hops.png — input tokens vs hop depth, tool vs sac -->
<!-- SLOT:SEEDED — sac_explored row + the explore-pays-for-itself paragraph -->
<!-- SLOT:HOTPOT_COST — per-hop-depth cost table (2/3/4-hop), tokens grow for tools, flat for SaC -->

And the guarantee that makes it safe to adopt: **SaC never ships a strategy that
underperforms your baseline.** Every forged primitive must beat `max(dense, hybrid)` on
held-out queries — measured with paired bootstrap confidence intervals — or the system
emits the baseline itself. That gate is in the SDK, with tests
([`HarnessForge.accept_code_primitive`](search_as_code/harness/forge.py)), and we publish
it firing: on strong retrievers it frequently selects the baseline. We think a retrieval
system that can prove it won't make things worse is worth more than one that claims magic.

## 📦 Install

```bash
pip install search-as-code                 # core: in-memory backend, zero further deps (numpy only)
pip install 'search-as-code[opensearch]'   # + your backend (also: qdrant / chroma / pgvector)
```

```python
import search_as_code as sac

s = sac.Session("opensearch", index="docs", dim=768, embedder=my_embedder)
#   swap "opensearch" -> "qdrant" / "chroma" / "pgvector" / "memory" — nothing else changes

cands = s.search_many(["how do agents retrieve?", "agentic RAG"], top_k=40, mode="hybrid")
best  = s.rerank("how do agents retrieve?", cands, top_k=10)
print(best.to_evidence(fields=["title"]))   # compact, context-friendly — this is ALL the model sees
```

Every example below runs offline with zero setup and no API key (a scripted generator
stands in for the LLM), and CI executes them so they cannot rot:

```bash
python examples/01_quickstart.py           # the primitive API + sandbox
python examples/02_opensearch.py           # a real backend
python examples/03_explore_first.py        # explore: discover the corpus strategy
python examples/04_harness_judge_forge.py  # the continual harness end to end
```

## 🔁 The continual harness: explore → forge → judge

This is the part no managed search API gives you: **the system learns your corpus and
keeps what it learned as code you can read.**

### 1. `explore` — discover what your index rewards

Point it at your corpus with a handful of labeled (or synthesized) queries. The LLM
authors a retrieval strategy per hop — including **raw OpenSearch DSL** probes
(`match_phrase` on a year, a part number, a proper name — the exact constraints dense
embeddings blur) — and the oracle records which strategies actually won. Structure is
**discovered, not assumed**: on entity-dense corpora it learns to keep queries whole; on
genuinely multi-document questions it learns to decompose and fuse.

### 2. `forge` — persist the win as a primitive, with provenance

The winning strategies are synthesized into a named, reusable primitive + skill +
few-shot exemplars, persisted to disk. Every artifact carries **provenance**: when it was
forged, what it was measured on, what baseline it beat and by how much (with the CI), and
what it superseded. Overwrites archive the old version. Contradictory learned rules are
retired, not accumulated.

```python
forge = sac.HarnessForge(store, registry)
report = forge.accept_code_primitive(          # the acceptance gate
    "my_corpus_strategy", "when to use it", authored_code,
    session=s, held=held_queries)              # beats max(dense, hybrid) or the baseline ships
print(report["accepted"], report["delta_vs_baseline"])
```

### 3. The deep LLM-as-judge — depth only when it pays

Most queries need one cheap search. Some need ten hops. The `DiagnosticJudge` reads
per-sub-fact cross-encoder coverage of the current candidates and decides: **PASS** (stop,
you have it) or **FAIL + a structured diagnosis** — which sub-fact is missing, why, which
technique to try next, what query to use. That diagnosis steers the next authored hop.

Validated leak-free (query-grouped splits, shipped renderer, grouped bootstrap):
**0.771 [0.666, 0.870]** balanced accuracy against the oracle, and judge-stopped runs
recover **95–103% of oracle-stopped recall** on HotpotQA and SearchUnify multi-hop
corpora. We also publish the no-LLM references it must beat (a tuned threshold gate
scores 0.738) — if a cheap gate ever matches it on your corpus, use the cheap gate.

```python
res = sac.agentic_solve(session, query, generator=llm)   # judge decides when to stop
res["ids"], res["hops"], res["codes"]                    # results + the authored strategies
```

## 🧰 The primitive surface

| group | primitives |
|---|---|
| retrieval modes | dense (ANN) · keyword (BM25) · hybrid (RRF) · phrase/proximity · fuzzy · wildcard · prefix · fielded · more-like-this · **raw OpenSearch DSL** |
| query-side | rephrase · expand · decompose · HyDE · PRF (Rocchio) · auto_filter (self-query) |
| rank / fuse | cross-encoder rerank · RRF / weighted / relative-score fusion · MMR · semantic dedup · diversity quota · reserve-per-subfact |
| gating | score_cutoff · confidence / abstain · score_cliff · consensus (with agreement signals that survive chaining) |
| introspection | describe_schema · seeded sample · content_type · describe(llm=True) |

Full catalog: [`docs/PRIMITIVES.md`](docs/PRIMITIVES.md) · per-backend support/emulation
matrix: [`docs/DATABASES.md`](docs/DATABASES.md). If a backend lacks a mode, the SDK
emulates it so agent code stays portable; where emulation is approximate, it says so
instead of pretending.

## 🧠 How it works

```
LLM writes Python  ─▶  Session (unified API, out-of-context state)
                        ├─ primitives: fan_out · fuse · rerank · hyde · mmr · dedup …
                        ├─ VectorStore protocol + capability emulation   ← DB differences end here
                        └─ adapters: opensearch · qdrant · chroma · pgvector · faiss · sqlite · memory
                   ─▶  Sandbox (timeouts, output caps; only final evidence returns to the model)
                   ─▶  Judge: PASS → done (1 turn) · FAIL → diagnosis → next authored hop (≤10)
                   ─▶  Forge: winning strategies → provenance-carrying primitives, reused next query
```

The database owns retrieval; the harness owns everything around it. Composite macros stay
bypassable, so generated code can always reach the atoms.

## 🔍 Why trust these numbers

This project maintains a standing, adversarial audit of its own claims — every headline
number carries a bootstrap confidence interval, negative results are published, and
retracted claims stay retracted. The summary lives in [`AUDIT.md`](AUDIT.md); the full
research trail (experiments, defect log, methodology corrections) lives in the companion
research repository. If you find a number this repo can't back, that's a bug — file it.

## 🗺️ Roadmap

- **StopGate** — the judge, a logistic gate, and a threshold gate behind one swappable interface, A/B-able on your corpus
- **Asymmetric query/passage embedding** — first-class support for bge/e5/Qwen instruction prefixes
- **`llm_batch` sub-model primitives** — batched extraction/verification inside the sandbox ([RLM](https://arxiv.org/abs/2512.24601)-style; the root model never sees raw tool output)
- **Single-turn deep mode** — one model turn authors a program that loops all 10 hops inside the sandbox
- **More adapters** — Pinecone, Weaviate, LanceDB · **MCP server wrapper** · hardened sandbox backends (Docker/e2b)

## 🤝 Contributing

```bash
git clone <this repo> && pip install -e '.[dev]'
make check          # lint + typecheck + tests + doc links — one target, same as CI
```

Add a backend by implementing one `VectorStore`
([`adapters/base.py`](search_as_code/adapters/base.py)); the parametrized conformance
suite ([`tests/test_conformance.py`](tests/test_conformance.py)) is the contract — it runs
against every installed backend in CI and has caught real bugs in three adapters.
See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 📄 License

Apache-2.0 © 2026 search-as-code contributors.

<p align="center"><sub>search as code · agentic retrieval · code-mode · continual harness · RAG · vector search · LLM agents</sub></p>
