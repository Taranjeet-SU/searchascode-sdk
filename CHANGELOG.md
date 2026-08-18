# Changelog

## 0.1.0 — 2026-08-18

First release. Everything below is measured; see [`AUDIT.md`](AUDIT.md) for what the
numbers do and don't support.

### The product
- **Primitive SDK** over a unified `VectorStore` protocol: 7 adapters (opensearch,
  qdrant, chroma, pgvector, faiss, sqlite, memory) with capability emulation, typed
  errors, retries, and a parametrized conformance suite that runs against every
  installed backend in CI.
- **Code-mode execution**: the LLM authors one Python strategy per query against ~30
  primitives (dense/keyword/hybrid/raw-DSL search, fan-out, RRF/weighted fusion,
  cross-encoder rerank, HyDE, PRF, MMR, dedup, consensus, score gates), run in a
  sandbox with timeouts, output caps, and per-run namespace rebinding; only compact
  evidence returns to the model. Measured on BrowseComp-Plus (n=100, matched arms):
  **31× fewer input tokens, 1.7× lower latency, 1 model turn vs ~9.5** vs a
  tool-calling agent with the same tools and budget.
- **The continual harness**: `explore` (LLM-authored strategies + raw OpenSearch DSL
  probes, oracle-scored) → `forge` (persisted primitives/skills with full provenance:
  held-out metrics, CI, corpus fingerprint, supersession archive) → the
  **best-baseline acceptance gate** (`HarnessForge.accept_code_primitive`): nothing
  ships unless it beats max(dense, hybrid) on held queries.
- **DiagnosticJudge**: per-sub-fact cross-encoder coverage → PASS/FAIL + a structured
  diagnosis that steers the next hop. 0.771 [0.666, 0.870] held-out balanced accuracy
  (query-grouped, leak-free), 95–103% of oracle-stopped recall in-loop.
- `pip install search-as-code` ships the prompt surface (`sac.SAC_SYSTEM`), `py.typed`,
  and four zero-setup examples that CI executes.
