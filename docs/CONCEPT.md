# The concept, and how the five sources map onto this codebase

Search-as-code is the convergence of three ideas. This SDK is their union, made
**database-agnostic**.

## 1. Code-mode > tool-calls

> *"LLMs have seen a lot of code. They have not seen a lot of 'tool calls.'"*
> — Cloudflare, [Code Mode](https://blog.cloudflare.com/code-mode/)

Instead of the model emitting one tool-call per step, it writes a **program**
that orchestrates many calls, and only the final result returns to the context.
Anthropic's [Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
shows the payoff: filtering 10,000 rows in the sandbox instead of streaming them
through context is a ~98% token cut.

→ In this repo: **`sandbox.py`**. Agent code runs in `LocalExecutor`; bulky
intermediate results live in the namespace and the `Session` state store; only
`print(...)` and the `evidence` variable come back via `ExecResult.for_model()`.

## 2. Search as *primitives*, not a monolith

> *"We expose the components of the search stack as primitives within an SDK."*
> — Perplexity, [Rethinking Search as Code Generation](https://research.perplexity.ai/articles/rethinking-search-as-code-generation)

A fixed `search(query)` endpoint can't express fan-out, adaptive refinement, or
verification. Exposing atoms lets the model compose task-specific pipelines: fan
out over query variants → fuse → dedup → rerank → extract structured records.

→ In this repo: **`primitives.py`** (`fan_out`, `fuse` = Reciprocal Rank Fusion,
`dedup`, `rerank`, `freshness`, `extract`) and the `Session` methods that wrap
them. `examples/demo.py` mirrors Perplexity's CVE case study shape: fan-out,
adaptive filter, verify/extract.

## 3. The retriever sets the ceiling

> *"The retriever sets the ceiling; the interface decides how close the agent gets to it."*
> — Hornet, [Code Mode for Agentic Retrieval](https://hornet.dev/blog/code-mode-for-agentic-retrieval)

[BrowseComp-Plus (arXiv:2508.06600)](https://arxiv.org/abs/2508.06600) proves it
empirically: swapping BM25 for a strong embedder moves the *same* agent from
single digits to 70%+. So the primitive layer must expose real retrieval controls
(hybrid, filters, freshness, rerank) — and quality must be measurable.

→ In this repo: **`adapters/`** exposes those controls uniformly, and
`Capabilities` + emulation in `session.py` keep them available even on backends
that lack them natively. (A BrowseComp-Plus-style eval harness is on the roadmap.)

## The gap this fills

All three ideas assume you've already built the SDK for *your* search stack.
Perplexity built theirs; Hornet built theirs. If your retrieval lives in Qdrant
today and Pinecone tomorrow, you'd rewrite the harness each time.

**search-as-code** makes the harness itself portable: one primitive API, a thin
adapter per backend, capability emulation for the gaps. Write the agent's
retrieval program once; run it on any vector DB.
