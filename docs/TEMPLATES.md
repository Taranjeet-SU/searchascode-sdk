# SAC retrieval templates — strategies, not primitive configs

A **template** is a named end-to-end retrieval *strategy* — a small procedure that composes
primitives at a chosen **effort tier**, some of them **adaptive** (retrieve, check whether the
top score "fell off", then escalate or stop). The learned router (`explore.fit`) picks a
template per query, which is what controls **both quality and latency**: cheap one-hop for easy
queries, deep only when a query needs it.

The registry lives in code (`search_as_code/explore/templates.py:TEMPLATE_DOCS`) so the router
and the LLM prompt can read it; this file is the human view.

## Tiers
- **light** — one hop, no LLM query-ops. Fast, for easy queries.
- **medium** — one extra signal (hyde / prf / mmr / multi-rephrase / part-number exact).
- **deep** — decompose fan-out and/or hyde, fused + reranked. For hard/multi-hop queries.
- **adaptive** — decide effort at run time from the score signal (escalate only when weak).

## The 16 templates

### light
| template | does | why it's a distinct choice |
|---|---|---|
| `light_dense` | single vector search, no rerank | cheapest path; baseline for easy semantic matches |
| `light_keyword` | BM25 term search only | exact term matching — wins on rare tokens/IDs where embeddings blur |
| `light_hybrid` | RRF(dense, keyword), one hop | balances semantics and terms without any LLM/rerank cost |
| `rephrase_rerank` | rephrase once → hybrid → rerank | the classic one-hop "clean the query, then precision-rerank" flow |

### medium
| template | does | why it's a distinct choice |
|---|---|---|
| `dense_rerank` | dense → cross-encoder rerank | precision on pure semantics; no lexical/LLM signal |
| `hyde_rerank` | hypothetical answer → dense → rerank | bridges vocab gap when query wording ≠ doc wording |
| `mmr_diverse` | dense → MMR diversify | kills near-duplicates; broad queries needing coverage not redundancy |
| `prf_rerank` | pseudo-relevance feedback → rerank | no LLM — uses the corpus to expand an under-specified query |
| `multi_rephrase` | N rephrasings → search each → fuse → rerank | query-variation ensemble; catches docs matching only one phrasing |
| `exact_partnum` | codes → exact+regex, fuse dense → rerank | the only strategy for identifiers/pins/codes where exact beats semantics |

### deep
| template | does | why it's a distinct choice |
|---|---|---|
| `decompose_rerank` | decompose → fan-out → fuse → rerank | multi-part/multi-hop questions no single query retrieves |
| `deep_hyde_decompose` | fuse(hyde, decompose, dense) → rerank | vocab-bridging + multi-hop; heavy generalist for hard prose |
| `deep_all` | fuse(dense, keyword, hyde, decompose) → rerank → MMR | max recall + precision + diversity; most expensive, last resort |

### adaptive (score-guarded)
| template | does | why it's a distinct choice |
|---|---|---|
| `score_guarded` | hybrid; if top score fell off → escalate to hyde+decompose, else return | spends deep effort ONLY when the cheap hop is uncertain — the latency saver |
| `escalating` | cascade dense → hybrid → deep, stop at first confident tier | finest-grained effort control; pays for exactly the depth each query needs |
| `confidence_gated_exact` | part# queries: try exact; escalate to dense+hyde only if weak | adaptive around identifiers — cheap exact when confident, semantic backup when not |

## Why this shape matters
The old set was primitive *configs* (which pools to fuse). These are *strategies* at different
costs, so the router's choice is an **effort/quality trade-off**, not just a fusion recipe. That
is what lets a deployment answer an easy query with `light_dense` (~one retrieval) and reserve
`deep_all`/`deep_hyde_decompose` for the few queries that need them — directly addressing the
high per-call latency a flat "always deep" recipe incurs on a real support corpus.

## Cost note (for `explore.fit`)
A per-query `StrategyContext` **memoizes** the shared sub-results (dense/keyword/hyde/decompose/
rephrase/exact/regex pools and rerank passes), so running all 16 templates for one query costs
about one of each primitive, not 16×. `label_llm`/`label_rerank` toggle whether the hyde/
decompose/rephrase pools and the cross-encoder run during labeling (needed for the medium/deep/
adaptive templates to be distinct).
