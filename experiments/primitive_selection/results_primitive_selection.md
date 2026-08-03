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
