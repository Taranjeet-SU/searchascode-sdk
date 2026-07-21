# Project status — Search as Code

_A living snapshot of what's done and what's pending._

## Progress by area

```
SDK core + primitives     ██████████ 100%   done
Database adapters (5)      ████████░░  80%   memory·opensearch·qdrant·chroma·pgvector (more pending)
Docs / taxonomy / research ██████████ 100%   320-primitive taxonomy, 150-source research, matrix
Phase 1: OpenSearch + FiQA ██████████ 100%   57k docs indexed, base recall validated
Agents (base/tool/SAC)     █████████░  90%   LangChain; judge loop + hops + reasoning
Benchmark + trace UI       █████████░  90%   100-query run + live UI (needs re-run w/ new agents)
Diagnosis + fixes          ██████████ 100%   dense-first, calibrated judge, accumulation, PRF
Phase 2: constraint probe  ██░░░░░░░░  20%   SAC 1.00 vs dense 0.42 proven; scale-up pending
Ontology + KG              ░░░░░░░░░░   0%   designed (PHASE2.md), not built — needs domain data
Learned components         ░░░░░░░░░░   0%   designed — online learning / query→primitive router
```

## ✅ Done

**SDK (`search_as_code/`)**
- Unified data model (`Document`/`Hit`/`ResultSet`/`Capabilities`) + typed errors
- Primitives: dense/keyword/hybrid/regex search, `search_many` (fan-out), `fuse`(RRF)/`rrf`,
  `dedup`, `rerank` (+`CrossEncoderReranker`), `mmr`, `freshness`, `compress`, `extract`,
  `expand`, `rephrase`, `decompose`, `hyde_search`, **`prf_search` (Rocchio)**
- 5 adapters: `memory`, `opensearch`, `qdrant`, `chroma`, `pgvector` (capability emulation)
- Sandbox (`LocalExecutor`) with out-of-context state; portable filter dialect
- 65 tests passing

**Docs**
- `PRIMITIVES.md` (320-primitive taxonomy), `DATABASES.md` (support matrix),
  `RESEARCH.md` (150 sources), `CACHING.md`, `CONCEPT.md`, `PHASE2.md` (architecture)

**Phase 1 (`phase1/`) — the benchmark**
- OpenSearch local (tarball, single node), BEIR **FiQA** ingested (57,638 docs, GPU embed)
- Base recall validated (dense R@10 0.45, matches published BEIR)
- Three query paths (LangChain): `run_base`, `run_tool_calling`, `run_sac`
- **100-query benchmark**: SAC best recall + 2.7× cheaper than tool-calling (pre-fix numbers)
- Live trace UI (Live + History tabs, prompt toggle, per-hop traces) + static UI

**Agents — refinements**
- LLM-as-judge retry loop (max 3 hops), **calibrated + semantic signals**
- **Confidence-aware hop accumulation** (keep best hop; fixes "perfect→destroyed")
- Reasoning capture; equal 4-formulation rephrasing for SAC & tool-calling
- Neighborhood-shift refinement (PRF / HyDE / decompose)

**Diagnosis (the important learning)**
- FiQA is single-strategy (dense ceiling) → SAC ties dense, tool-calling ≤ dense
- Fixed the keyword-fusion regression (dense-first weighted fusion)
- **Phase 2 probe: SAC 1.00 vs dense 0.42 on constraint/cross-doc queries** ← the real win

**Ops**
- Repo live: `github.com/oro-jackson/searchascode-sdk` (all work pushed)

## 🟡 In progress / partial
- **Benchmark re-run** with the fixed agents (judge calibration, dense-first, accumulation)
  so `phase1/RESULTS.md` reflects current behavior
- **Adapters**: 5 of ~12 targeted (Pinecone/Weaviate/Milvus/LanceDB/Vespa pending)

## ⬜ Pending (Phase 2 vision)
- **Ontology** (domain definitions only) — build + inject into rephraser/SAC/judge
- **Knowledge Graph** (relations, factual values, ranges/limits, tabular) — build + query primitives
- **KG/constraint primitives**: `graph_traverse`, `constraint_filter`, `max/min_version`,
  `entity_join`, `table_lookup` (probe whether new primitives are needed as complexity grows)
- **Result attribution** — tag each `Hit` with the primitive that found it; feed back to the agent
- **KG-consistency signals** for the judge (beyond cosine signals, now shipped)
- **Learned components** — online learning for judge calibration + SAC few-shot exemplars;
  query→primitive-combo router (the "Clf" idea)
- **Harder constraint eval** (ranges, multi-entity joins, tabular) + a **real domain dataset** (SU)
- **Stronger reranker** (hosted Cohere/Voyage — bge won't download here) for FiQA headroom
- **Discoverability**: set GitHub repo description + topics (needs a token); optional repo rename

## Services right now
- OpenSearch (:9200): started per session — dies on session teardown, restart from tarball
- Live UI (:8501): relaunch with `streamlit run phase1/live_ui.py`
