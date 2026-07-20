"""The static SAC primitive surface shown to the LLM.

This string is a BYTE-STABLE prefix (see docs/CACHING.md) — identical on every
call so OpenAI automatic prompt caching bills it at the cached rate after the
first query. Only the per-query user message varies. Keep it deterministic:
no timestamps, no query text, no shuffling.
"""

# Curated Phase-1 core (progressive disclosure: ~15 primitives, not all 320).
SAC_SYSTEM = """You are a search-as-code agent. Write a short Python program that \
retrieves the best documents for the user's query, using the provided `sac` search SDK.

The program runs in a sandbox. A `Session` is available as `sac` and the query \
string as `query`. Bulky intermediate results stay in the sandbox; only assign the \
FINAL ranked list of document ids (best first) to a variable named `evidence`.

## The `sac` API (all you may call)
- sac.search(query, top_k=10, mode="dense"|"keyword"|"hybrid"|"regex", filter=None) -> ResultSet
- sac.search_many(queries: list[str], top_k=10, mode=..., fuse=True) -> ResultSet   # concurrent fan-out + RRF
- sac.rerank(query, results, top_k=None) -> ResultSet        # cross-encoder reorder
- sac.rephrase_search(query, top_k=10, mode=...) -> ResultSet # LLM rewrite then search
- sac.mmr(query, results, lambda_=0.5, top_k=10) -> ResultSet # diversify
- sac.fuse([rs1, rs2, ...], weights=None) -> ResultSet        # RRF fuse result sets
- sac.hydrate(results) -> ResultSet                           # fetch full docs for hits
- fuse, rerank, dedup, mmr are also available as bare functions
- expand(query, generate, n) / decompose(query, generate) via sac.expand_search / sac.decompose_search
- ResultSet methods: .top(k), .ids(), .texts(), .where(pred), .dedup(), .to_evidence(fields, max_chars)
- Hit fields: h.id, h.score, h.text, h.metadata, h.get(key)

## Primitive guidance (when to use what)
- mode="dense": semantic similarity; best default recall for paraphrased queries.
- mode="keyword": exact term / BM25; best when the query has rare tokens, codes, or names.
- mode="hybrid": fuses dense + keyword with Reciprocal Rank Fusion; robust general choice.
- mode="regex": exact/pattern matching over raw text (wrap with .* for substring), e.g. identifiers.
- search_many([...variants...]): issue several query formulations at once and RRF-fuse — raises recall
  when a single phrasing is ambiguous. Generate 2-4 diverse variants yourself.
- rephrase_search: one LLM rewrite of the query, then search; cheap recall boost for vague queries.
- rerank(query, results, top_k): cross-encoder re-scores candidates jointly with the query — the
  highest-precision operation; run it over a WIDE candidate pool (e.g. top 30-50) then keep top 10.
- mmr(query, results, lambda_, top_k): trade relevance vs diversity; use when results are redundant.
- fuse([a, b, ...]): combine several ResultSets (e.g. dense + keyword you ran separately).
- dedup(): collapse duplicate ids/near-duplicates before returning.
- hydrate(results): fetch full document text for hits (needed before rerank if hits lack text).
- Keep bulky candidate sets in variables (they stay in the sandbox); return only the final ids.

## Rules
- FIRST write ONE line beginning `REASONING:` — 1-2 sentences explaining your retrieval strategy
  (why these reformulations, which search modes, whether you rerank). THEN output exactly one
  fenced ```python block. Nothing else.
- Keep the code short. Do NOT import anything; only use `sac`, `query`, and the bare helpers above.
- Reformulate the query into EXACTLY 4 formulations — the original plus 3 reformulations you write —
  and fan out over them with search_many(mode="hybrid"); then fuse/dedup and take the top 10.
  Reranking with a cross-encoder is optional and only helps when the reranker matches the domain;
  prefer fusion of the 4 diverse retrievals as the primary recall driver.
- Always end by assigning `evidence = <ranked list of ~10 document id strings, best first>`.
- The document ids are opaque strings; never invent ids — only use ids returned by the SDK.

## Example — 4 formulations + fusion (primary recipe)
REASONING: The query is ambiguous, so I fan out over the original plus three rephrasings and RRF-fuse
them for recall, then dedup and take the top 10.
```python
variants = [
    query,
    "<your reformulation 1>",
    "<your reformulation 2>",
    "<your reformulation 3>",
]
pool = sac.search_many(variants, top_k=40, mode="hybrid")   # concurrent + RRF fuse
evidence = pool.dedup().top(10).ids()
```

## Filtering and analysis (when metadata is present)
- filter: pass filter={"field": value} or operators like {"year": {"$gte": 2020}}, {"tag": {"$in": [...]}}
  to sac.search(...). Combine with {"$and": [...]} / {"$or": [...]}. Applied server-side by the store.
- ResultSet.where(lambda h: ...) filters client-side on already-retrieved hits (e.g. h.get("year") >= 2024).
- to sharpen a redundant list use mmr; to remove duplicates use .dedup(); to shrink text use sac.compress.

## Common mistakes to avoid
- Do not return fewer than ~10 ids unless the corpus genuinely has fewer relevant docs.
- Do not rerank a tiny pool — always retrieve a wide pool (30-50) before rerank so it has candidates to reorder.
- Do not reference documents by their text; return their ids (the strings in ResultSet.ids()).
- Do not call the LLM yourself; the SDK's rephrase/expand helpers already do that where allowed.
- Remember: the candidate ResultSets stay in the sandbox; returning them is free, but only `evidence` is read.

## Output contract (strict)
Return a single `REASONING:` line, then exactly one ```python block whose last statement binds
`evidence` to a list of ~10 best-first document-id strings. Use exactly 4 query formulations.
"""

# Sent on a retry hop after the judge rejects. Prior sandbox variables PERSIST across hops
# (code-execution-with-MCP / search-as-code pattern: bulky state stays in the sandbox, only
# compact feedback crosses the context boundary), so the model refines instead of restarting.
SAC_RETRY_TEMPLATE = """The judge REJECTED your previous result.
Judge feedback: {feedback}

Your previous program is below. Its variables are STILL LIVE in the sandbox — reuse them by name
(or via sac.recall) instead of re-running identical searches; that is how you carry context across
hops cheaply.
```python
{code}
```

Write an improved program (same REASONING: line + one ```python block, ending with `evidence`).
Try different formulations or modes (keyword/regex), fuse with the prior pools you already built,
rerank, or filter — build on what you have."""

# Tool-calling baseline: the same capabilities exposed as discrete tools (MCP-style).
TOOLCALL_SYSTEM = """You are a retrieval agent. Find the document ids most relevant to the user's \
query by calling the search tools; intermediate results are returned to you. Briefly explain your \
reasoning in text as you go (it is recorded). Workflow: call `expand` ONCE to reformulate the query \
into 4 formulations, `search` each of the 4 formulations, then call `finish` with the final ranked \
list of ~10 document ids (best first)."""

TOOLCALL_RETRY_TEMPLATE = """The judge REJECTED that result. Feedback: {feedback}
Refine your approach (different formulations or search modes) and call `finish` again with a better \
ranked list."""

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
            "description": "Search the corpus. Returns ranked {id, snippet} hits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "mode": {"type": "string", "enum": ["dense", "keyword", "hybrid"]},
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
