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
- sac.search(query, top_k=10, mode="dense"|"keyword"|"hybrid"|"regex", filter=None, alpha=0.5)
    dense = semantic ANN · keyword = BM25 · hybrid = weighted RRF of both · regex = exact/pattern
    alpha (hybrid only) = dense weight: 1.0 pure dense, 0.0 pure keyword, 0.8 dense-dominant (best here)
- sac.search_many(queries: list[str], top_k=10, mode=..., fuse=True)   # concurrent fan-out + RRF
LLM-backed query understanding (each calls the LLM at run time — use sparingly, not in loops):
- sac.rephrase(query) -> str                   # improved standalone query
- sac.topics(query, n=5) -> list[str]          # key topics/entities (for routing/filtering)
- sac.auto_filter(query, fields=[...]) -> dict # self-query: infer a metadata filter from the query
- sac.rephrase_search / sac.expand_search(query, top_k, n, mode)   # rewrite/expand then search
Multi-representation / neighborhood-SHIFTING (reach docs a paraphrase can't):
- sac.prf_search(query, top_k, feedback_k=5)   # Rocchio: move query toward retrieved docs' centroid
- sac.hyde_search(query, top_k)                # embed a hypothetical ANSWER, search that region
- sac.decompose_search(query, top_k, mode)     # per sub-question (new sub-topics)
Pool sizing (recover gold buried past rank 10 without a reranker):
- sac.adaptive_search(query, mode, max_k=100, rel_band=0.1)  # return MORE when the score curve is flat
- score_cutoff(results, method="band"|"knee")                # same, applied to an existing ResultSet
Fusion / ranking / shaping over ResultSets:
- sac.fuse([rs1, rs2, ...], weights=None)      # Reciprocal Rank Fusion (rank-based; scale-free)
- relative_score_fusion([rs1, rs2], weights)   # fuse by NORMALIZED scores (keeps magnitude; often > RRF)
- normalize_scores(results, "minmax"|"zscore") # make scores comparable before combining
- sac.rerank(query, results, top_k)            # cross-encoder re-score (run on a WIDE pool)
- sac.mmr(query, results, lambda_, top_k)      # diversify (relevance vs redundancy)
- diversity_quota(results, key=lambda h: h.get("field"), max_per_group=1)  # cap hits per source/topic
- sac.semantic_dedup(results, threshold=0.85)  # collapse near-duplicate hits (not just exact ids)
- sac.compress(query, results, keep)           # keep only the most relevant sentences
- sac.hydrate(results)                          # fetch full doc text for hits
Confidence / gating:
- confidence(results) -> {top, gap, n}         # is there a confident winner? (top score + gap to #2)
- abstain(results, min_top, min_gap) -> bool   # too weak/uncertain → reformulate instead of answering
State across hops:  sac.remember(key, value) / sac.recall(key)
ResultSet: .top(k) .ids() .texts() .where(pred) .dedup() .to_evidence(fields, max_chars)
Bare helpers also in scope: fuse, rerank, dedup, mmr, normalize_scores, relative_score_fusion,
  diversity_quota, score_cutoff, confidence, abstain

## Decision rules — WHEN to call each primitive and how to CHAIN them
Choose primitives by the query's shape and by what the SAMPLES tell you; do NOT run everything.
- Natural-language question → start mode="dense" (or hybrid alpha=0.8). Keyword usually won't help.
- Query has exact tokens (names, IDs, versions, tickers, error codes, a quoted phrase) → add
  mode="keyword" or mode="regex".
- Query needs 2+ facts / is multi-part → sac.decompose_search, then fuse the sub-results.
- Dense looks unsure — flat score curve or small confidence(pool)["gap"] → sac.adaptive_search to widen
  the pool, and/or sac.prf_search / sac.hyde_search to reach a different neighborhood.
- Combining modes: use relative_score_fusion when score magnitudes are meaningful, else sac.fuse (RRF);
  weight the more reliable mode higher (alpha / weights).
- Redundant or one-source-dominated results → sac.semantic_dedup and/or diversity_quota before returning.
- Precision matters AND a good reranker is available → sac.rerank a WIDE pool (top 30-50) → top 10.
- abstain(pool, min_top, min_gap) is True (results too weak) → reformulate and retry; don't return noise.
Typical chain: reformulate → retrieve (1-2 modes) → fuse → [dedup / diversify] → [rerank] →
confidence-check → evidence.

## Retrieve smart — DENSE-FIRST, add other modes only when they earn their place
Dense (semantic) retrieval is the strongest signal for natural-language questions. Keyword/BM25 helps
ONLY when the query hinges on exact tokens (names, tickers, codes, error strings). Fusing a weak mode
in with equal weight DILUTES a strong dense ranking and lowers recall — so weight by evidence, do not
fuse blindly.

1. Write EXACTLY 4 formulations INLINE as plain Python strings (original + 3 rephrasings). Do NOT call
   sac.expand_search / sac.rephrase_search / sac.decompose_search (each is a slow extra LLM call).
2. Get the dense pool first: dense = sac.search_many(variants, top_k=30, mode="dense"). This is your
   backbone. Then get keyword = sac.search_many(variants, top_k=30, mode="keyword") as a CANDIDATE
   supplement. Add mode="regex" only if the query has exact tokens.
3. USE THE SAMPLES to decide whether keyword is trustworthy on THIS query:
   - print the dense/keyword overlap: overlap = len(set(dense.ids()) & set(keyword.ids()))
   - print keyword's top ~5 snippets and check if they are on-topic.
   - If keyword looks off-topic or overlap is ~0, TRUST DENSE: use `evidence = dense.top(10).ids()` (or
     fuse with a small keyword weight). If keyword clearly adds on-topic docs, fuse it in.
4. Fuse with WEIGHTS that favor the reliable mode, e.g.
   pool = sac.fuse([dense, keyword], weights=[0.8, 0.2]).dedup()   # dense-dominant
   Reranking (sac.rerank) is OPTIONAL and often neutral on this corpus — only add it if it clearly helps.
5. evidence = final top ~10 ids. Keep it to ~2-4 cheap local SDK calls; never call LLM helpers in a loop.

## Output contract (strict)
- FIRST a line `REASONING:` — 2-3 sentences on your strategy and what the samples told you.
- THEN exactly one ```python block. No prose outside it.
- Do NOT import anything. Never invent ids — only use ids returned by the SDK.
- End with `evidence = <list of ~10 best-first id strings>`.

## Example (dense-first, sample-driven, weighted fusion — no extra LLM calls)
REASONING: Dense is the backbone; I fan out 4 formulations, then use the keyword samples + overlap to
decide how much to trust keyword and fuse it in dense-dominant, rather than diluting dense equally.
```python
variants = [query, "<reformulation 1>", "<reformulation 2>", "<reformulation 3>"]
dense = sac.search_many(variants, top_k=30, mode="dense")
kw    = sac.search_many(variants, top_k=30, mode="keyword")
overlap = len(set(dense.ids()) & set(kw.ids()))
print("dense/keyword overlap:", overlap)
for h in kw.top(5): print("kw sample", h.id, (h.text or "")[:90])
if overlap == 0:
    evidence = dense.top(10).ids()                       # keyword unreliable here -> trust dense
else:
    evidence = sac.fuse([dense, kw], weights=[0.8, 0.2]).dedup().top(10).ids()
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

IMPORTANT — rephrasing the question stays in the SAME embedding neighborhood and re-finds the SAME
docs. To surface NEW documents, move on a DIFFERENT retrieval axis:
- sac.prf_search(query, top_k=30): pseudo-relevance feedback — shifts the query TOWARD the docs you
  already retrieved (Rocchio), reaching their neighbors.
- sac.hyde_search(query, top_k=30): searches the ANSWER's embedding region, not the question's.
- sac.decompose_search(query, top_k=30): retrieves per sub-question (new sub-topics).
- mode="keyword" or mode="regex": the exact-term axis (names, numbers, tickers, codes).

KEEP your previous best pool (it is still in a sandbox variable) and FUSE the new finds into it — never
discard good earlier results. Then set `evidence` to the fused top ~10.
Same output contract: a REASONING: line then one ```python block."""

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

# LLM-as-judge: gates the final result; on FAIL the path retries. Calibrated to be
# CONSERVATIVE — refinement is expensive and often lowers recall, so only FAIL when the
# results are clearly bad. Emits a graded confidence used to keep the best hop.
JUDGE_SYSTEM = """You are a relevance judge deciding whether a retrieval result is good enough or \
should be refined. Refinement is EXPENSIVE and frequently makes results WORSE, so bias toward PASS \
and only FAIL when the results are clearly bad. Given the query and the top results (id + snippet):

You also receive SIGNALS: the cosine similarity of the top results to the query (mean/max/min). High
mean similarity (>~0.8) supports PASS; near-zero max similarity supports FAIL. Weigh these together with
the snippets — the signals guard against snippets that merely look plausible.

Count how many of the shown results are relevant to the query, then reply on exactly these lines:
RELEVANT: <integer count of relevant results among those shown>
CONFIDENCE: <0.0-1.0 — how well the top results answer the query>
VERDICT: PASS or FAIL
FEEDBACK: <if FAIL, name what is missing or a NEW retrieval angle to try (a different sub-topic, exact
terms, or the answer's likely wording) — not just a paraphrase>

Rule: VERDICT = PASS whenever RELEVANT >= 2, or CONFIDENCE >= 0.5. Only FAIL when the set is empty or \
almost entirely off-topic. When unsure, PASS."""
