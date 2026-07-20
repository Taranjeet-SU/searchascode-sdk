# Phase 1 results — base vs MCP tool-calling vs Search-as-Code

**Corpus:** BEIR FiQA-2018 (57,638 docs) in OpenSearch · **Queries:** 100 test
queries with qrels · **Embeddings:** `thenlper/gte-base` · **LLM:** `gpt-4.1-mini`.

## Headline

| path | Recall@10 | nDCG@10 | MRR@10 | avg latency | LLM calls/q | input tokens | cache hit | total cost |
|---|---|---|---|---|---|---|---|---|
| base (hybrid) | 0.479 | 0.379 | 0.415 | **0.02 s** | 0 | 0 | — | **$0** |
| tool-calling (MCP) | 0.348 | 0.303 | 0.374 | 5.92 s | 5.48 | 254,359 | 8.3% | $0.117 |
| **SAC (code-mode)** | **0.491** | **0.398** | **0.457** | 3.70 s | **1.99** | **142,363** | **77.7%** | **$0.043** |

## What it says

1. **SAC has the best retrieval quality** — Recall@10 0.491 and nDCG@10 0.398,
   ahead of the strong hybrid `base` (0.479) and far ahead of tool-calling (0.348).
   The fan-out + RRF-fusion recipe the model writes genuinely helps recall.

2. **SAC is dramatically cheaper than tool-calling** — 1.8× fewer input tokens
   (142k vs 254k), **2.7× lower cost** ($0.043 vs $0.117), and fewer LLM calls
   (2.0 vs 5.5) — because intermediate candidate sets stay in the sandbox instead
   of re-entering the context on every hop.

3. **Prompt caching works as designed** — SAC's stable >1024-token code surface
   caches at **77.7%**, versus 8.3% for tool-calling's shifting context. This is
   the concrete payoff of the strategy in [docs/CACHING.md](../docs/CACHING.md).

4. **Tool-calling underperformed even the no-LLM baseline** on recall (0.348 <
   0.479). With `max_steps=6` the model sometimes exhausted its tool budget
   without calling `finish` (returning nothing) or submitted too few ids. Raising
   the step budget would recover some recall — but at *even higher* token cost,
   which only widens SAC's efficiency lead.

5. **SAC code was 100% executable** — all 100 generated programs ran without error
   in the sandbox.

## Example (query: "1 EIN doing business under multiple business names")

SAC wrote — and executed — this program (**Recall@10 = 1.0**):

```python
variants = [
    query,
    "EIN multiple business names",
    "Employer Identification Number doing business as multiple names",
    "business using one EIN for several business names",
]
pool = sac.search_many(variants, top_k=40, mode="hybrid")
reph = sac.rephrase_search(query, top_k=40, mode="dense")
fused = sac.fuse([pool, reph])
evidence = fused.dedup().top(10).ids()
```

On the same query, tool-calling issued `rephrase → search → rephrase → search →
search → search`, exhausted its 6-step budget without finishing, and returned
nothing (**Recall@10 = 0.0**).

## Caveats (honest)

- **Reranker:** MS-MARCO cross-encoders *hurt* FiQA (domain mismatch) and the
  FiQA-appropriate `bge-reranker` would not download in this environment, so the
  SAC recipe leads with hybrid + fan-out + fusion rather than cross-encoder rerank.
  A domain-matched reranker would lift all LLM paths further.
- **Tool-calling** could be tuned (more steps, better stop criteria); these numbers
  reflect a reasonable, not maximally-tuned, MCP baseline.
- Costs are tiny in absolute terms (100 queries), but the **ratios** — tokens,
  cache hit, calls, latency — are what scale.

## Reproduce

```bash
python -m phase1.benchmark -n 100 --reranker cross-encoder/ms-marco-MiniLM-L-12-v2
streamlit run phase1/ui.py     # per-query traces: generated code, steps, ids, cost
```
