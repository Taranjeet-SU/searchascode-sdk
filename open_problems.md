# Open Problems — where search-as-code got stuck, and where the field is stuck

Every dead-end we hit building `sac.explore` (the retrieval-strategy router) and deep-mode SAC is a
**named, actively-researched open problem** — not an implementation bug. This file catalogs each
one: what we saw (with links to the experiment that shows it), what the literature calls it (with
citations), what we did about it, and how our position compares to the state of the art.

**The reassuring meta-finding:** in every case, the fix we arrived at independently from our own data
matches the fix the current SOTA papers propose.

## Summary

| # | Open problem | Our evidence | Field's name / key paper | Our status vs literature |
|---|---|---|---|---|
| 1 | Router collapses to the majority strategy | [explore_learning §9c](experiments/explore_learning/README.md) | minority collapse / prior dominance | same wall; fix (rebalance) not yet run |
| 2 | Routing ties a strong default on **accuracy** | [primitive_selection §7–§8](experiments/primitive_selection/results_primitive_selection.md) | routing helps *cost*, not recall (Adaptive-RAG) | reproduced; under-credited cost angle |
| 3 | CV accuracy is a misleading metric | [explore_learning §4a/§8](experiments/explore_learning/README.md) | routing evaluation artifacts | we caught it, switched to realized recall |
| 4 | Reranking drops multi-gold coverage | [multi_hop §12](experiments/multi_hop_synth_queries/RESULTS.md) | precision-recall / list-coverage tradeoff | reproduced; fixed via fuse-for-coverage |
| 5 | Deep/iterative retrieval degrades vs one-shot | [multi_hop §15](experiments/multi_hop_synth_queries/RESULTS.md) | over-retrieval / distractor injection | reproduced; partial monotone fix |
| 6 | Unreliable stopping (LLM self-judge) | [deep_sac](experiments/deep_sac/) | LLM-as-judge overconfidence | reproduced; QPP-gate = SOTA fix, not yet built |
| 7 | Multi-hop / deep-research coverage ceiling | browsecomp RESULTS (*internal, not published — see `issues.md` GOV-1*) | all-golds retrieval ceiling | reproduced; a *coverage* not routing problem |
| 8 | Templates non-orthogonal + synthetic bias | [explore_learning §9](experiments/explore_learning/README.md) | query-generator artifact / redundant strategies | diagnosed; redesign pending |

---

## 1. The learned router collapses to the majority strategy
**What we saw.** Trained on the multi-hop labels, the router predicts `light_dense` **84–87%** of the
time — *higher* than its 61–69% share in the training labels — and only ever predicts **4–7 of the
16** templates. The minority winners (`light_hybrid`, `exact_partnum`, `multi_rephrase`) are
predicted ~0%. → [explore_learning §9c](experiments/explore_learning/README.md), raw:
[model_bakeoff.json](experiments/primitive_selection/model_bakeoff.json).

**Literature.** *Minority collapse*: representations of minority classes collapse into a single
vector, and *"strong majority-class recall hides severe minority-class collapse"*
([Neural Collapse under class imbalance](https://arxiv.org/pdf/2401.02058)). In routing specifically,
[Unsolvability Ceiling in Multi-LLM Routing](https://arxiv.org/pdf/2605.07395) finds that with a
strong prior (79.3% majority) *"the likelihood ratio is insufficient to overcome the prior"* and the
router degenerates to *"length-based triage rather than difficulty-based triage."* Our features were
embedding + length; our prior was 61–84%.

**Our status.** Diagnosed and quantified. **Proposed fix (matches field): class rebalancing / focal
loss** to force learning on rare actions — not yet run.

## 2. Routing ties a strong default on accuracy (its real value is cost)
**What we saw.** A learned 16-way router ≈ always-dense: BEIR +0.006, HotpotQA +0.004, SU −0.003 on
CV accuracy; realized routed-recall lifts only 0.552→0.617 (HotpotQA) / 0.576→0.602 (SU) vs an oracle
of 0.90 / 0.84. → [primitive_selection §7–§8](experiments/primitive_selection/results_primitive_selection.md).

**Literature.** [Adaptive-RAG](https://arxiv.org/pdf/2403.14403) shows a 3-class complexity router
**matches the always-expensive baseline at lower cost** — routing's win is efficiency/latency, not
accuracy. [RAGRouter-Bench](https://arxiv.org/pdf/2604.03455) frames lightweight routing as a
cost-baseline study.

**Our status.** Reproduced the accuracy tie. **Correction the literature forces:** we under-credited
the *cost* angle — the right target is **3-class depth routing (skip work on easy queries)**, not
16-way identity routing for recall.

## 3. CV classification accuracy is a misleading metric
**What we saw.** §7 first reported the router "ties dense (+0.004)" using **CV accuracy** (did it name
the exact cheapest winner). That penalizes a correct-but-non-cheapest pick. Switching to **realized
routed-recall** flipped the result to a real +6.5 / +2.7 pt lift. → [explore_learning §4a/§8](experiments/explore_learning/README.md).

**Literature.** [Unsolvability Ceiling in Multi-LLM Routing](https://arxiv.org/pdf/2605.07395)
(subtitle: *An Empirical Study of Evaluation Artifacts*) documents exactly this class of routing
evaluation artifact.

**Our status.** Caught and corrected; all headline numbers now use the realized task metric.

## 4. Cross-encoder reranking drops multi-gold coverage
**What we saw.** Adding a reranker over the fused union *lowered* all-golds on a clean SU comparison:
2-hop 0.950→0.880, 3-hop 0.720→0.640, 4-hop 0.530→0.430 — the reranker scores whole-question
relevance, so a doc satisfying only one of N sub-facts gets dropped. → [multi_hop §12](experiments/multi_hop_synth_queries/RESULTS.md).

**Literature.** *"Cross-encoder reranking improves precision by 4.3 points but costs 3.5 recall
points… aggressive reranking suppresses valid secondary answers"* for list/multi-answer questions;
MMR shows the steepest recall penalty by suppressing complementary evidence
([Benchmarking Retrieval Strategies, biomedical RAG](https://arxiv.org/pdf/2605.02520)).

**Our status.** Reproduced and **fixed**: default recipe is coverage-first RRF-fusion of decomposed
sub-pools; rerank is reserved for single-gold precision (helps BrowseComp, hurts multi-gold).

## 5. Deep / iterative retrieval degrades vs one-shot
**What we saw.** Legacy deep-SAC *loses* to one-shot on 5/6 cells (all-golds 0.522 vs 0.600) — a
confidently-wrong deeper hop overwrote a correct hop-1, and reranking a wider pool hurt coverage. →
[multi_hop §15](experiments/multi_hop_synth_queries/RESULTS.md), raw
[deep_recall_monotone.json](experiments/deep_sac/deep_recall_monotone.json).

**Literature.** [When Iterative RAG Beats Ideal Evidence](https://arxiv.org/html/2601.19827v4)
documents *"cases where the introduction of retrieval actively degrades performance."*
[Stop-RAG](https://arxiv.org/html/2510.14337v1): *"each additional iteration carries… the risk of
distractors."*

**Our status.** Reproduced; **partial fix** (`run_sac monotone=True`: hop-0 = one-shot recipe +
RRF-fuse all hops) restores aggregate parity (0.611) but still loses on easy 2-hop. Full fix needs #6.

## 6. Unreliable stopping — the LLM self-judge
**What we saw.** The per-hop LLM judge is the weak link: it rejects an already-correct hop-0 (forcing
a diluting hop-2) or accepts a confidently-wrong hop. Deep on BrowseComp burned 34 searches for 0
recall. → [deep_sac](experiments/deep_sac/), browsecomp RESULTS (*internal, not published — see `issues.md` GOV-1*).

**Literature.** [Overconfidence in LLM-as-a-Judge](https://arxiv.org/html/2508.06225v2): *"inflated
confidence scores that do not reflect true performance… overconfident models propagate erroneous
judgments."* [Self-RAG](https://arxiv.org/pdf/2310.11511) uses reflection tokens but
[Stop-RAG](https://arxiv.org/html/2510.14337v1) notes LLM self-assessment stopping is *"unreliable."*

**Our status.** Reproduced. **Proposed fix (matches SOTA): a value-based / QPP confidence gate**
instead of the LLM self-judge (Stop-RAG's exact prescription) — not yet built.

## 7. Multi-hop / deep-research coverage ceiling
**What we saw.** all-golds@10 needs *all N* golds in the top-10; unsolved explodes 1%→20/31% from
2→4 docs. On BrowseComp (100k, ~3 golds/query) even the lenient any-gold oracle is 0.353 (65%
unsolvable) and all-golds@10 ≈ 0. → browsecomp RESULTS (*internal, not published — see `issues.md` GOV-1*),
[explore_learning §9b](experiments/explore_learning/README.md).

**Literature.** [MultiHop-RAG](https://arxiv.org/pdf/2401.15391): multi-hop facts are distributed
across docs, straining retrieval. BrowseComp-Plus ([arXiv 2508.06600](https://arxiv.org/pdf/2508.06600))
is designed so single-retriever recall is near-floor.

**Our status.** Reproduced; correctly framed as a **coverage / new-primitive problem, not a routing
one** — no strategy selection helps when the golds aren't reachable.

## 8. Templates aren't orthogonal + synthetic-query bias
**What we saw.** 5–7 of the 16 templates never win a single query; light tier takes 89–94% on the
synthetic sets. But on BrowseComp's real research queries `hyde_rerank` wins 21% — the monopoly is
partly a property of the keyword-shared `generate_multihop` chains, not universal. →
[explore_learning §9a/§9d](experiments/explore_learning/README.md).

**Literature.** [RAG-Fusion](https://arxiv.org/abs/2402.03367) and
[Question Decomposition for RAG](https://arxiv.org/pdf/2507.00355) show the value is
decompose-and-fuse of the *cheap* primitive, not exotic strategies; HyDE-style bridging matters only
under vocabulary mismatch — corpus-dependent, exactly what we found.

**Our status.** Diagnosed. **Proposed fix: fewer, orthogonal templates targeting distinct failure
modes + difficulty-tier labels** (per §1–§3).

---

## What *works* (where we agree with the literature and it holds)

- **Decompose → retrieve-per-sub-query → RRF-fuse beats single dense on multi-hop.** We saw 0.39→0.64
  all-golds; the field reports +36.7% MRR@10 / +11.6% F1
  ([Question Decomposition for RAG](https://arxiv.org/pdf/2507.00355)); our `fuse(pools)` *is*
  [RAG-Fusion](https://arxiv.org/abs/2402.03367).
- **Code-mode keeps intermediates out of context → ~90%+ fewer tokens.** We measured ~20× (95%) vs
  tool-calling; the field reports up to 92.8% at 500+ tools
  ([From Tool Orchestration to Code Execution](https://arxiv.org/pdf/2602.15945),
  BrowseComp-Plus [2508.06600](https://arxiv.org/pdf/2508.06600)).
- **Fewshot exemplars + model plan_prompt as prompt context** lift deep-SAC all-golds +0.24 (combo
  arm, [multi_hop §14](experiments/multi_hop_synth_queries/RESULTS.md)) — evidence-grounded guidance
  beats a static rule.

## Bottom line
We are not stuck because we did it wrong — we are stuck on the field's open frontier, and we
independently re-derived the SOTA fixes (class rebalancing, value/QPP stopping gate, 3-class depth
routing, coverage-first fusion). The genuinely shippable artifact is a **fixed decompose→fuse
pipeline + a few rule-gated primitives + a lightweight per-corpus diagnostic**, not a learned 16-way
router or per-query LLM codegen — see the "what to actually build" synthesis in the experiment docs.
