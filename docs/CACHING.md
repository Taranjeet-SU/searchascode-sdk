# Passing the SAC SDK to the LLM efficiently (prompt caching)

**Question:** how do we hand the search-as-code SDK surface to the model on every
call without paying for those tokens each time? Three complementary methods,
grounded in the source articles and the OpenAI/Anthropic caching mechanics.

## 1. Code-mode framing (compact surface)

Present the SDK as a **typed code API in the system prompt** and have the model
*write Python* against it, rather than exposing hundreds of tool-call schemas
(Cloudflare "Code Mode", Anthropic "Code execution with MCP"). The whole
Phase-1 primitive surface — `search(mode=...)`, `search_many`, `rerank`,
`rephrase`, `fuse`, `dedup`, `mmr`, `filter`, `aggregate`, `remember/recall`,
`to_evidence` — fits in ~800–1,000 tokens. One `run_code` tool replaces N tool
definitions, so the per-call surface is small to begin with.

## 2. Stable prefix + automatic prompt caching (the core method)

**OpenAI prompt caching is automatic and prefix-based:** identical prompt
*prefixes* of ≥1,024 tokens are cached and billed at a large discount on
subsequent calls. For `gpt-4.1-mini` that is **$0.10 / 1M cached input vs
$0.40 / 1M uncached — a 75% reduction** on the repeated portion.

Design rule we follow: **everything static goes in the prefix, only the query
varies at the end.**

```
[ system message ]         ← SDK/primitive surface + rules + few-shot   (STATIC → cached)
[ user message ]           ← the one thing that changes: the query      (dynamic)
```

Because the system message is byte-for-byte identical across all 100 benchmark
queries, the first query pays full price to "warm" the cache and every
subsequent query pays the cached rate for that prefix. We **measure** this
directly — `phase1/llm.py` reads `usage.prompt_tokens_details.cached_tokens` and
prices cached vs uncached separately, so the benchmark reports real cache hit
rate and $ saved.

> Anthropic differs: caching is *explicit* via `cache_control` breakpoints
> (ephemeral, ~5-min TTL). If we swap providers, we add a cache breakpoint after
> the static surface instead of relying on automatic prefix matching. OpenAI
> needs no code change — just prefix stability.

Practical gotchas that break prefix caching (all avoided here):
- Any per-query token *before* the static block (timestamps, query text, a
  shuffled tool list) invalidates the prefix → keep them after it.
- Non-determinism in the surface (dict ordering, random few-shot) → the surface
  is a fixed string constant.

## 3. Progressive disclosure (keep the prefix small *and* relevant)

The canonical taxonomy has **320 primitives** (`docs/PRIMITIVES.md`); dumping all
of them would bloat the prefix and the model's decision space. Following
Anthropic's MCP guidance we expose only the **curated Phase-1 core** (~15
primitives) in the always-on prefix, plus a `help(primitive)` / `list_primitives`
affordance the generated code can call to pull a fuller signature on demand. The
common path stays cached; the long tail is loaded only when needed.

## What this means for the benchmark

- **Base search**: no LLM → no prompt, zero token cost (the cost floor).
- **Tool-calling**: pays for tool schemas + multi-turn intermediate results in
  context on every hop → highest token cost.
- **SAC (code-mode)**: static surface cached after query 1; intermediate
  candidate sets stay in the sandbox and never re-enter context → lowest LLM
  cost among the two LLM paths, which the token/cost columns quantify.

The benchmark's cost column is therefore an *apples-to-apples* measurement of
exactly this caching strategy, not an assumption.
