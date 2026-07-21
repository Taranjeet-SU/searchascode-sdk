# Phase 2 — where code-as-search actually wins

## The reframe (from the Phase 1 diagnosis)

FiQA is **simple single-hop semantic** retrieval → dense is the ceiling → SAC can only
*match* it. Code-as-search wins **structurally** on **complex / multi-hop / constraint /
cross-document** queries, because the sandbox does **retrieval + computation** (set logic,
joins, max/min over versions, constraint satisfaction) that dense RAG and tool-calling
cannot.

Archetype query:
> *"latest release of SU where I can use embedding models **and** fine-tune them"*

Requires: find release v₁ (embedding support) + release v₂ (fine-tuning support) → return
**max(v₁, v₂)**. Dense returns passages, no computation. SAC computes it.

**Requirement:** the system must be good on **both** — simple queries (fall back to plain
dense) *and* multi-hop/constraint queries (compose primitives + compute). One agent,
query-adaptive.

## Architecture

```
                    ┌─────────────── Domain Ontology (definitions only) ───────────────┐
                    │  what terms mean in the domain: "embedding model", "fine-tune",   │
                    │  "release", "connector" … (no data, just canonical definitions)   │
                    └───────────────────────────────┬──────────────────────────────────┘
                    ┌───────────────────────────────┴──────────────────────────────────┐
                    │  Knowledge Graph (data + its limits)                               │
                    │  entities & relations · factual values · ranges/limits · tabular   │
                    │  info · "feature X introduced in release Y" · version bounds       │
                    └───────────────────────────────┬──────────────────────────────────┘
             injected as shared context into all three LLM components ↓
   ┌──────────────┐        ┌───────────────────────┐        ┌──────────────────────────┐
   │  Rephraser   │        │  SAC code generator   │        │  LLM-as-judge            │
   │ (ontology-   │        │ (ontology+KG aware:   │        │ (ontology+KG + SEMANTIC  │
   │  grounded    │        │  writes code that     │        │  SIGNALS: cosine sims,   │
   │  rewrites)   │        │  queries KG + docs +  │        │  KG-consistency, ranges) │
   └──────────────┘        │  computes answers)    │        └──────────────────────────┘
                           └───────────┬───────────┘
                                       ▼
             primitives: dense · keyword · regex · hybrid · rerank · mmr · fuse(RRF)
             · prf(Rocchio) · hyde · decompose · filter · aggregate · [KG: traverse,
               constraint, max/min-version, join, entity-lookup]  ← extend as needed
                                       ▼
                        [candidates] → samples + per-primitive ATTRIBUTION
                        (which code part found which good doc) → iterate
```

### Ontology vs KG (your distinction)
- **Ontology** — *domain-specific definitions only*. The vocabulary and what each term
  means. Grounds the rephraser (canonical terms) and the judge (is this on-topic *in this
  domain*). No instance data.
- **Knowledge Graph** — *data and its limits*: how topics relate, factual values, ranges
  and limits, tabular info, entity–entity relations, version bounds. This is what the SAC
  program **queries and computes over** for constraint/cross-doc answers.

### Judge with semantic signals (shipped)
The judge no longer reads snippets alone — it receives **SIGNALS** (cosine similarity of the
top results to the query: mean/max/min) and is calibrated to weigh them. Next: add
**KG-consistency signals** (do retrieved facts satisfy the query's constraints / ranges?).

### Learning (from data + online)
- **Ontology** learned/curated from the domain corpus (term extraction + definitions).
- **KG** built from technical docs, financial data, release notes, tables.
- **Judge prompts & SAC policy** improved by **online learning** from outcomes: log
  (query → generated code → primitive attribution → success), and use it to (a) tune the
  judge threshold against labels, (b) retrieve few-shot exemplars for the code generator,
  (c) train a query→primitive-combo router (the "Clf" idea).

## Primitive sufficiency (the empirical gate)
**Primitives are not assumed exhaustive.** Before scaling, take a **sample dataset** that
spans simple + multi-hop + constraint queries and test: *can the current primitives reach
the ceiling?* Where they can't, define new primitives (candidate gaps already visible: KG
traversal, constraint/`max-version`, entity-join, table-lookup). This probe drives the
primitive roadmap.

## Phased plan
1. **Primitive-sufficiency probe** — build a small synthetic *versioned-docs + constraint
   queries* eval (SU-release-note style) with known answers; run SAC vs dense vs tool-calling.
   Prove SAC wins on multi-hop/constraint and surface missing primitives. *(next)*
2. **Result attribution** — tag each `Hit` with the primitive that produced it; feed
   "which code part found which good doc" back to the agent.
3. **KG + constraint primitives** — build the KG from the corpus; add `graph_traverse`,
   `constraint_filter`, `max/min_version`, `entity_join`, `table_lookup`.
4. **Ontology** — domain definitions; inject into rephraser/SAC/judge as shared context.
5. **Learned components** — online-learning loop for judge calibration, SAC few-shot
   exemplars, and a query→primitive router from labelled data.

## Status
Shipped toward this: dense-first SAC, calibrated + **semantic-signal judge**, confidence-aware
hop accumulation, neighborhood-shift primitives (`prf_search`/`hyde`/`decompose`). Not yet:
ontology, KG, attribution, learned router, the constraint eval.
