# Primitive/template selection — a learned retrieval-strategy router

**TL;DR (honest).** We built `sac.explore` — a pipeline that defines ~16 retrieval **strategy
templates**, labels synthetic queries by which template retrieves the gold doc, and trains a
classifier to **route each query to the cheapest template that solves it**. On standard single-hop
IR (BEIR), the learned router **ties "always dense" (+0.006 CV lift)** — dense recall@10 is
already high, so there is no routing headroom. The genuinely useful output was not the router but
the **per-corpus failure taxonomy** (which names *what to build* per dataset). This documents the
method, the numbers, and why it didn't beat the baseline — so the next iteration starts from truth.

---

## 1. The 16 strategy templates
Each template is a named recipe composed from SDK primitives, at an **effort tier**. (Full
registry: `search_as_code/explore/templates.py:TEMPLATE_DOCS`; docs: `docs/TEMPLATES.md`.)

| tier | templates |
|---|---|
| **light** | `light_dense`, `light_keyword`, `light_hybrid`, `rephrase_rerank` |
| **medium** | `dense_rerank`, `hyde_rerank`, `mmr_diverse`, `prf_rerank`, `multi_rephrase`, `exact_partnum` |
| **deep** | `decompose_rerank`, `deep_hyde_decompose`, `deep_all` |
| **adaptive** | `score_guarded`, `escalating`, `confidence_gated_exact` — escalate only when the top score "falls off" |

Cost is `tier + LLM penalty`, so non-LLM templates sort first (used by both the winner policy and
the cascade labeler).

### 1a. Primitives each template composes
Every template is built from the same SDK primitive set — the difference is *which* primitives it
chains and whether it gates on a score signal. `LLM?` marks templates that issue an LLM query-side
op (hyde/decompose/rephrase/expand), the real cost driver at label time. (Source of truth:
`templates.py`; primitive impls: `search_as_code/primitives.py` + `Session` search methods.)

| template | tier | primitives used (in order) | LLM? |
|---|---|---|---|
| `light_dense` | light | `dense` (vector search) | — |
| `light_keyword` | light | `keyword` (BM25) | — |
| `light_hybrid` | light | `hybrid` (RRF-fuse of dense+keyword) | — |
| `rephrase_rerank` | light | `rephrase_search` → `rerank` (cross-encoder) | ✓ |
| `dense_rerank` | medium | `dense` → `rerank` | — |
| `hyde_rerank` | medium | `hyde_search` → `rerank` | ✓ |
| `mmr_diverse` | medium | `dense` → `mmr` (diversify) | — |
| `prf_rerank` | medium | `prf_search` (pseudo-relevance feedback) → `rerank` | — |
| `multi_rephrase` | medium | `expand_search` (N rephrasings) → `fuse` → `rerank` | ✓ |
| `exact_partnum` | medium | `extract_codes` → `query_phrase`/`regex` + `dense` → `fuse` → `rerank` (else `hybrid`→`rerank`) | — |
| `decompose_rerank` | deep | `decompose_search` (sub-query fan-out) → `fuse` → `rerank` | ✓ |
| `deep_hyde_decompose` | deep | `fuse(hyde_search, decompose_search, dense)` → `rerank` | ✓ |
| `deep_all` | deep | `fuse(dense, keyword, hyde_search, decompose_search)` → `rerank` → `mmr` | ✓ |
| `score_guarded` | adaptive | `hybrid`; if `weak` (margin/thin gate) → `fuse(hyde_search, decompose_search, hybrid)` → `rerank` | ✓ |
| `escalating` | adaptive | `dense` → (gate) `hybrid` → (gate) `fuse(hyde_search, decompose_search, hybrid)` → `rerank` | ✓ |
| `confidence_gated_exact` | adaptive | `exact`+`regex`; if `weak` → `fuse(exact, dense, hyde_search)` → `rerank` (else `hyde_search`→`rerank`) | ✓ |

**Primitive glossary:** `dense` = single vector search · `keyword` = BM25 · `hybrid` = RRF fuse of
dense+keyword · `hyde_search` = generate hypothetical answer, then dense on it · `decompose_search`
= split into sub-queries, fan out, fuse · `rephrase_search` = rewrite the query once ·
`expand_search` = N query variants, fuse · `prf_search` = enrich from top hits (no LLM) ·
`query_phrase`/`regex` = exact identifier match · `fuse` = RRF combine of pools · `rerank` =
cross-encoder re-scoring · `mmr` = maximal-marginal-relevance diversification · `weak` = the
"top score fell off" gate (empty / <3 hits / low top-1↔top-2 margin) that drives the adaptive tier.

## 2. Labeling & training (`explore.dataset()` → `explore.train()`)
- **Grounded synthetic queries** generated from the corpus (`explore.synthesize`); the query's
  source doc is the gold (leakage-free).
- **Winner policy = recall@k, cheapest solver.** A template "solves" a query iff the gold doc is in
  its **top-k (recall@k)**; the label is the **cheapest** template that solves it. Labels are
  re-derived from stored per-template hits, so the policy is decoupled from the (expensive)
  labeling pass.
- **Cascade labeling** — evaluate templates cheapest-first, stop at the first cost group that
  solves, so expensive LLM strategies only run on the queries the cheap ones miss.
- **Features** = query embedding (gte-base) + lexical signals (length, #part-numbers, has-digit,
  question-ish…). **Model** = `HistGradientBoosting` (XGB-style), swappable via
  `explore.set_model(...)` (`hist_gb` / `logreg` / `random_forest` / `mlp`).
- **Outputs**: `router.pkl`, `labels.csv` (per-query recall@k for every template),
  `template_recall.csv`, `failures.json`.

## 3. Result — global router over 4 BEIR datasets (real qrels)
Pooled **3,024 real query–doc pairs** (nfcorpus + arguana + scidocs + scifact), all embedded with
gte-base so the feature space is shared.

| metric | value |
|---|---|
| oracle (any template solves@10) | 0.899 |
| **global CV accuracy** | **0.847 ± 0.004** |
| best single template (always `light_dense`) | 0.841 |
| **router lift over always-dense** | **+0.006** |

Per-dataset oracle: **arguana 0.964 · scifact 0.957 · nfcorpus 0.833 · scidocs 0.812.** Label
distribution is dominated by `light_dense` (~84% of solved queries) — i.e., dense recall@10 is
already high, so the other 15 templates are rarely the cheapest solver.

## 4. The actually-useful output — a per-corpus failure taxonomy
Unsolved queries (no template@10) are bucketed by cheap signals into four causes, and **each
corpus fails for a different reason → a different fix**:

| corpus | dominant failure | → what to build |
|---|---|---|
| nfcorpus (medical) | **synonym_metadata 70%** | query expansion / medical ontology |
| arguana (arguments) | **rank_collision 58%** | reranking / disambiguation |
| scidocs (citations) | **unexplained 80%** | relational / citation-graph signal |
| scifact (claims) | unexplained/rank (mostly solved) | — |

## 5. Why the router didn't beat dense (root causes)
1. **No complementarity.** On BEIR prose, dense dominates (~84%); there is little the router can
   route *to*. Routing only helps when templates are genuinely complementary.
2. **Winner is not predictable from cheap features.** For the ~16% dense misses, query features do
   not reliably predict which alternative wins → the classifier can't beat "always dense."
3. **Noisy 16-way label.** "Cheapest-that-solves" is noisy (ties/near-ties/synthetic artifacts), so
   model accuracy is upper-bounded by label noise.
4. **A heterogeneous *chunked* corpus we tried separately gave ~0 oracle** — the gold was one
   *exact chunk* among many near-duplicate chunks, so no strategy retrieved it at top-10 (a
   duplication artifact, not a retrieval gap). Fair measurement there needs document-level gold.

## 6. Honest conclusion & next
On single-hop prose IR, **a learned template router ≈ always-dense**; its value did not
materialize as routing. The **failure taxonomy** is the shipped value (a per-corpus "what to
build"). The path to a router that *does* beat dense (see `experiments/explore_improvement/`):
QPP-gated escalation (route only low-confidence queries), a **rule-based router as the
baseline-to-beat**, guaranteed-complementary templates (graph/ColBERT/conditional-HyDE) with
**3-class difficulty labels** instead of the noisy 16-way, and adaptive/iterative depth.

**Code:** `search_as_code/explore/{templates,router,training,fit}.py`;
`explore.dataset()/set_model()/train()`. **Related:** `experiments/multi_hop_synth_queries/`
(where code-mode SAC *does* win — multi-document retrieval).

## 7. Template routing on multi-hop synth data (does the router beat dense where templates are complementary?)

We re-ran the learned template router on the **newer multi-hop synthetic datasets**, where the gold is a **SET of N documents** (n_docs=2/3/4) that must **ALL** be retrieved. The success gate is therefore **all_golds@10** (a template solves iff `gold_set ⊆ top_k`), not the single-gold recall@10 used for single-hop BEIR (added as the purely-additive `all_golds=True` flag; shipped behavior untouched). Deep templates fire for real (gte-base dense, QwenReranker, gpt-4.1-mini generator for hyde/decompose/rephrase/expand).

### 7a. Headline — router CV vs always-dense vs best-single (the LIFT)

| corpus | n | oracle (any all-golds@10) | router CV (hist_gb) | best-single-template | always-dense | **lift vs always-dense** |
|---|---|---|---|---|---|---|
| hotpotqa_multihop | 600 | 0.900 | 0.617 ± 0.028 | light_dense 0.613 | 0.613 | **+0.004** |
| su_multihop | 450 | 0.838 | 0.684 ± 0.023 | light_dense 0.687 | 0.687 | **-0.003** |

### 7b. Winning-template distribution (cheapest all-golds solver)

Contrast with **BEIR single-hop, where ~84% of solved queries were won by `light_dense`** (no complementarity => nothing to route to).

- **hotpotqa_multihop** (solved=540/600): light_dense=61% of winners. Full: `light_dense` 331(61%), `light_keyword` 119(22%), `light_hybrid` 20(4%), `exact_partnum` 18(3%), `multi_rephrase` 12(2%), `rephrase_rerank` 9(2%), `dense_rerank` 8(1%), `decompose_rerank` 8(1%), `score_guarded` 7(1%), `hyde_rerank` 6(1%), `deep_all` 2(0%)
- **su_multihop** (solved=377/450): light_dense=69% of winners. Full: `light_dense` 259(69%), `light_keyword` 68(18%), `light_hybrid` 25(7%), `mmr_diverse` 7(2%), `dense_rerank` 6(2%), `score_guarded` 3(1%), `exact_partnum` 3(1%), `rephrase_rerank` 3(1%), `hyde_rerank` 3(1%)

### 7c. Per-template all_golds@10 recall (top strategies)

- **hotpotqa_multihop**: `light_hybrid` 0.695, `light_keyword` 0.670, `light_dense` 0.552, `exact_partnum` 0.035, `multi_rephrase` 0.020, `rephrase_rerank` 0.015 (caveat: cascade labeling under-measures dear templates on queries a cheaper one already solved).
- **su_multihop**: `light_hybrid` 0.696, `light_keyword` 0.591, `light_dense` 0.576, `mmr_diverse` 0.020, `dense_rerank` 0.013, `exact_partnum` 0.013 (caveat: cascade labeling under-measures dear templates on queries a cheaper one already solved).

### 7d. Failure taxonomy on unsolved (no template got all N golds@10)

- **hotpotqa_multihop** (60 unsolved checked): rank_collision 52%, unexplained 48%
- **su_multihop** (73 unsolved checked): unexplained 70%, rank_collision 30%

### 7e. Verdict

**Headline (the metric asked for): NO — the router still ties always-dense on multi-hop.**
hotpotqa +0.004 (CV 0.617 vs 0.613), su_multihop −0.003 (CV 0.684 vs 0.687). Both are inside
the CV std (±0.02–0.03) → statistical noise, the same non-result as single-hop BEIR (+0.006).
So switching to the harder **all_golds@10** gate did **not** hand routing a win.

**But the *reason* is different from BEIR, and this is the real finding.** On BEIR the router
failed because there was **no complementarity** (oracle 0.899 vs best-single 0.841 = a 6-pt gap —
nothing to route to). On multi-hop the complementarity is **genuinely large**: no single template
exceeds ~0.70 all_golds@10 recall, yet the **oracle (any template) = 0.90 (hotpot) / 0.84 (su)** —
a **~20-pt (hotpot) / ~14-pt (su) gap**, 2.5–3.5× BEIR's. Different queries really are solved by
different templates. The bottleneck therefore shifted from *"no complementarity"* (BEIR reason #1)
to *"the winning template is not predictable from cheap query features"* (BEIR reason #2): the
gte-base embedding + lexical signals don't tell the classifier which of dense / keyword / hybrid
will capture all N golds, so CV accuracy lands right on top of always-dense.

**Did the label finally spread? Partly, but not onto the deep templates we expected.** light_dense
fell from BEIR's ~84% of winners to **61% (hotpot) / 69% (su)**, with `light_keyword` emerging as a
real second winner (22% / 18%) and `light_hybrid` third. But the hypothesis that
`decompose_rerank` / `deep_*` / `escalating` would dominate on multi-hop was **not confirmed** —
they are almost never the *cheapest* all-golds solver (`deep_all` won 2/540 hotpot, 0/377 su).
Two compounding causes, stated honestly: (1) on these synthetic "concatenated-fact" queries, when
all N golds are retrievable at all, a single pooled dense/keyword/hybrid pass (top-25 → rerank)
usually already gets them into top-10, so the deep fan-out is rarely needed *and* never cheapest;
(2) **measurement caveat — cascade labeling only runs the deep templates on the ~10–40% residual
that the cheap tier missed**, so §7c's ~0.00–0.03 recall for the deep templates is measured on the
hardest subset only and must **not** be read as "deep templates don't work." What we *can* say: on
the residual hard set (4-hop oracle drops to 0.805 hotpot / 0.687 su), no template — deep included —
reliably closes the gap, and the unsolved are dominated by `rank_collision` + `unexplained`
(§7d), i.e. the golds are near-retrievable but buried, not semantically invisible.

**Bottom line.** Multi-hop gave us the complementarity BEIR lacked, but not a routing win: the
lift is still ~0. To convert the 14–20-pt oracle headroom into real recall you need either (a) a
label that is predictable from features (3-class difficulty instead of the noisy 16-way
cheapest-solver), or (b) to stop trying to *pick one* template and instead **union the cheap tier
+ escalate on a QPP confidence gate** — routing *depth*, not *identity*. Same next-step as §6.

## 8. Model bake-off + realized routed-recall — the router DOES beat dense (metric correction)

§7's "+0.004 tie" measured the **wrong thing**. CV accuracy asks "did the classifier name the
*exact cheapest winner*?" — and since `light_dense` is the cheapest winner ~61% of the time,
"always guess light_dense" already scores ~0.61, so the router only looks +0.004 better. But that
penalizes the router for picking a *different* template that **also retrieves all the golds** (just
wasn't cheapest). The metric that matters is **realized routed-recall**: take the router's predicted
template and check whether *it* actually got all golds@10. Measured that way, **routing beats
always-dense** (raw numbers: `experiments/primitive_selection/model_bakeoff.json`; charts:
`experiments/explore_learning/`).

### 8a. Realized routed-recall vs always-dense vs oracle
| corpus | n | always-dense recall | **router (grid-tuned hist_gb)** | oracle | **lift** |
|---|---|---|---|---|---|
| hotpotqa_multihop | 600 | 0.5517 | **0.6167** | 0.900 | **+0.065** |
| su_multihop | 450 | 0.5756 | **0.6022** | 0.838 | **+0.027** |

The router captures ~19% (HotpotQA) of the dense→oracle headroom — modest but real, and the opposite
sign of the "tie" §7 reported. (always-dense recall = `light_dense`'s realized all-golds@10 = §7c's
0.552 / 0.576 — *not* the 0.613 classification-accuracy baseline in §3.)

### 8b. Grid-CV bake-off — no head breaks the ceiling
5-fold CV **classification accuracy** clusters tightly; grid search barely moves it:

| model | hotpotqa CV acc | su CV acc | routed-recall (hotpot / su) |
|---|---|---|---|
| hist_gb | 0.630 | — | 0.612 / 0.602 |
| **hist_gb (grid)** | **0.641** | 0.668 | **0.617 / 0.602** |
| xgb | 0.622 | — | 0.598 / 0.596 |
| xgb (grid) | 0.637 | 0.695 | 0.607 / 0.596 |
| logreg | 0.620 | — | 0.580 / 0.582 |
| random_forest | 0.624 | — | 0.567 / 0.580 |
| mlp | 0.637 | — | 0.613 / 0.600 |

**Verdict:** grid-tuned **`hist_gb`** (`learning_rate 0.1, max_depth 8, max_iter 400`) is the default
head; **xgboost does not beat it**. All heads land within noise → the ceiling is *feature→winner
predictability*, not model capacity. The routing lift comes from the right **metric**, not a bigger
model. (See `experiments/explore_learning/README.md` for the full learning writeup + charts.)

