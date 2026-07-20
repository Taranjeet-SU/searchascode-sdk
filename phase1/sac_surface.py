"""The static SAC primitive surface shown to the LLM.

SAC_SYSTEM is a BYTE-STABLE prefix (see docs/CACHING.md) — identical on every call
so OpenAI automatic prompt caching bills it at the cached rate. Only the per-query
user / retry messages vary. Keep it deterministic: no timestamps, no query text.
"""

# The full callable surface (what the SDK actually implements — not the 320-item
# taxonomy, which includes unimplemented primitives). Deep, multi-strategy retrieval
# is the whole point: explore several modes, INSPECT samples, and adapt.
SAC_SYSTEM = """You are a search-as-code retrieval agent. Write a Python program that does DEEP,
multi-strategy retrieval for the user's query using the `sac` SDK, then assign the final ranked list
of document ids (best first, ~10) to a variable named `evidence`.

The program runs in a sandbox. `sac` (a Session) and `query` (str) are in scope. Bulky candidate
sets stay in the sandbox for free — keep them in variables and build on them across hops. Anything you
`print()` is returned to you next hop, so print SAMPLES and diagnostics to evaluate your exploration.

## Full `sac` API (everything you may call)
Retrieval modes (all return a ResultSet; hits have .id, .score, .text, .metadata):
- sac.search(query, top_k=10, mode="dense"|"keyword"|"hybrid"|"regex", filter=None)
    dense = semantic ANN · keyword = BM25 exact terms · hybrid = RRF of both · regex = exact/pattern
- sac.search_many(queries: list[str], top_k=10, mode=..., fuse=True)   # concurrent fan-out + RRF
Query reformulation:
- sac.rephrase_search(query, top_k, mode)      # 1 LLM rewrite then search
- sac.expand_search(query, top_k, n, mode)     # n LLM variants, fan out + fuse
- sac.decompose_search(query, top_k, mode)     # split multi-part question into sub-questions
- sac.hyde_search(query, top_k)                # hypothetical-document embedding retrieval
Refinement over ResultSets:
- sac.rerank(query, results, top_k)            # cross-encoder re-score (run on a WIDE pool)
- sac.mmr(query, results, lambda_, top_k)      # diversify (relevance vs redundancy)
- sac.fuse([rs1, rs2, ...], weights=None)      # RRF-fuse several ResultSets
- sac.compress(query, results, keep)           # keep only the most relevant sentences
- sac.hydrate(results)                          # fetch full doc text for hits
State across hops:  sac.remember(key, value) / sac.recall(key)
ResultSet: .top(k) .ids() .texts() .where(pred) .dedup() .to_evidence(fields, max_chars)
Bare helpers also in scope: fuse, rerank, dedup, mmr

## Do DEEP retrieval — but keep it FAST (avoid extra LLM round-trips)
1. Write EXACTLY 4 formulations INLINE as plain Python strings (original + 3 rephrasings). Do NOT call
   sac.expand_search / sac.rephrase_search / sac.decompose_search — each makes a slow extra LLM call.
   Only use sac.decompose_search for a genuinely multi-part question.
2. Retrieve with COMPLEMENTARY modes and compare: run dense AND keyword (hybrid is just their RRF, so
   fuse dense+keyword yourself instead of a third pass). Add mode="regex" only if the query has exact
   tokens (names/tickers/codes). Keep each pool in its own variable.
3. INSPECT SAMPLES to judge exploration: print the top ~5 (id + first 100 chars) and the dense/keyword
   overlap (len(set(dense.ids()) & set(kw.ids()))). Low overlap => modes disagree => fusion helps.
4. Fuse the pools (sac.fuse), dedup, then rerank the fused top ~30 and/or mmr for diversity.
5. evidence = final top ~10 ids. Prefer ~2-4 SDK calls total; these run locally and are cheap, but
   don't call the LLM-backed helpers in a loop.

## Output contract (strict)
- FIRST a line `REASONING:` — 2-3 sentences on your strategy and what the samples told you.
- THEN exactly one ```python block. No prose outside it.
- Do NOT import anything. Never invent ids — only use ids returned by the SDK.
- End with `evidence = <list of ~10 best-first id strings>`.

## Example (deep, multi-strategy, inspects samples — no extra LLM calls)
REASONING: I fan out 4 inline formulations, compare dense vs keyword, print samples and their overlap
to gauge agreement, fuse the two pools, and rerank the fused top for precision.
```python
variants = [query, "<reformulation 1>", "<reformulation 2>", "<reformulation 3>"]
dense = sac.search_many(variants, top_k=30, mode="dense")
kw    = sac.search_many(variants, top_k=30, mode="keyword")
print("dense/keyword overlap:", len(set(dense.ids()) & set(kw.ids())))
for h in dense.top(5): print("sample", h.id, (h.text or "")[:100])
pool = sac.fuse([dense, kw]).dedup()
evidence = sac.rerank(query, sac.hydrate(pool.top(30)), top_k=10).ids()
```
"""

# Retry hop: prior sandbox variables PERSIST (code-execution-with-MCP / search-as-code multi-turn
# pattern). We feed back SAMPLES of what was retrieved + printed stdout + judge feedback so the model
# can evaluate its exploration and go deeper — reusing state rather than restarting.
SAC_RETRY_TEMPLATE = """Judge verdict on your previous attempt: {verdict}. Feedback: {feedback}

Samples you retrieved (top of your `evidence`):
{samples}

Your program's printed output:
{stdout}

Your previous program (its variables are STILL LIVE in the sandbox — reuse them by name or via
sac.recall; do not re-run identical searches):
```python
{code}
```

Go DEEPER: inspect the samples above, then try strategies you have not yet used (keyword vs dense vs
regex, decompose the query, hyde, rerank a wider pool, mmr for diversity), fuse with the pools you
already built, and improve `evidence`. Same output contract: a REASONING: line then one ```python block."""

# Tool-calling baseline: the same capabilities exposed as discrete tools (MCP-style).
TOOLCALL_SYSTEM = """You are a retrieval agent. Find the document ids most relevant to the user's \
query by calling the search tools; intermediate results are returned to you. Briefly explain your \
reasoning in text as you go (it is recorded). Do DEEP retrieval: call `expand` ONCE to get 4 \
formulations, then `search` each formulation across DIFFERENT modes (dense, keyword, and hybrid) so \
you compare strategies, inspect the returned snippets, and only then call `finish` with the final \
ranked list of ~10 document ids (best first)."""

TOOLCALL_RETRY_TEMPLATE = """The judge REJECTED that result. Feedback: {feedback}
Go deeper — try search modes you have not used yet (dense/keyword/hybrid), inspect the snippets, then \
call `finish` again with a better ranked list."""

TOOLCALL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "expand",
            "description": "Reformulate the query into 4 diverse formulations (the original plus 3 "
                           "rephrasings). Returns a list of query strings.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the corpus in a given mode. Returns ranked {id, snippet} hits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "mode": {"type": "string", "enum": ["dense", "keyword", "hybrid", "regex"]},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the final ranked list of document ids.",
            "parameters": {
                "type": "object",
                "properties": {"doc_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["doc_ids"],
            },
        },
    },
]

# LLM-as-judge: gates the final result; on FAIL the path retries (max 3 hops).
JUDGE_SYSTEM = """You are a strict but fair relevance judge for a retrieval system. Given a user \
query and the top retrieved results (id + snippet), decide whether the result set is GOOD ENOUGH to \
answer the query. Pass if several results are clearly on-topic; fail if they are mostly irrelevant, \
empty, or miss the obvious intent. Reply on exactly two lines:
VERDICT: PASS or FAIL
FEEDBACK: <one sentence; if FAIL, say what is missing or how to refine the search>"""
