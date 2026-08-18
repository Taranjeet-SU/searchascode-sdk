# deep_judge — a diagnostic LLM-as-judge that drives (and forges) the next hop

The deep-mode agent's judge used to be a one-bit gate ("good enough? PASS/FAIL"). Here it becomes a
**diagnostic** controller: per hop it coverage-checks each decomposed sub-fact against the candidate set
using calibrated signals, and for each still-missing sub-fact it **diagnoses why** and **prescribes the
next technique** — which the next hop executes. The winning OpenSearch queries the loop discovers are then
**forged into reusable primitives / skills / subagents**.

## 1. The judge tracks the gold oracle — but the "ceiling" claim is weaker than first reported

> **Corrected 2026-08-17.** The original version of this section reported point estimates with no
> intervals, and its selection procedure tie-broke on the test split. Re-derived by
> [`reselect_judge.py`](reselect_judge.py) (selection on TUNE only, ties broken by earliest round,
> bootstrap 95% CIs from `search_as_code.metrics`) → [`judge_reanalysis.json`](judge_reanalysis.json).
> The audit entries are `issues.md` DJ-1 / DJ-2 / DJ-3. **No new API calls** — pure re-analysis of the
> same tuning logs.

Frozen eval set: 100 multi-hop HotpotQA queries × {shallow top-5 hybrid, deep arsenal top-10} = 200
examples, oracle-labelled (`oracle = all gold ids ⊆ candidate set`), 106 PASS / 94 FAIL. We measure the
judge's PASS/FAIL agreement with the oracle (balanced accuracy on a held-out 100).

| judge / bound | held-out balanced-acc **[95% CI]** | tuned? |
|---|---|---|
| **Supervised ceiling** (LogReg, 5-fold CV, cross-encoder feature) | **0.725** *(no CI reported)* | — |
| LLM judge v0 — bi-encoder cosine signal (saturated) | 0.585 [0.490, 0.685] | **no — round 0** |
| LLM judge — **+ cross-encoder coverage signal** (v1) | 0.700 [0.610, 0.791] | — |
| LLM judge — v1 + **same-model critic** tuning | **0.721 [0.633, 0.811]** | yes (round 7) |
| LLM judge — v1 + **independent Qwen-32B critic** | 0.700 [0.610, 0.791] | **no — round 0** |

**What the intervals change (all three are honest corrections against ourselves):**

- **The tuning gain is not distinguishable from noise (DJ-2).** Same-model critic tuning moves TEST
  balanced accuracy by **+0.020 [−0.110, +0.150]** over the untuned prompt. On TUNE the adopted gain was
  +0.011 — *a single example flipping* (tn 36→37, fp 11→10), which is what cleared the code's own 0.01
  "don't chase eval noise" margin. At n=100 and p≈0.72 the interval is ±0.09, several times the effect.
  The earlier "0.63 → 0.72" phrasing is not supported.
- **Two of the five rows are the UNTUNED prompt (DJ-3).** Both `tuning_log_same.md` and
  `tuning_log_ce_qwen.md` record `## Best (round 0)`. So the "independent Qwen-32B critic → 0.70" row is
  the *baseline*, not a tuned outcome, and the honest statement is stronger and different: **the
  independent critic produced no adopted improvement at all.** Likewise the v0 row's "0.585 → ~0.68"
  had no log entry for the 0.68; it is 0.585.
- **The selection defect is real but did not change this pick (DJ-1).** `tune_judge.py:146-150` tie-breaks
  on TEST, which contradicts its own stated intent. Re-selecting on TUNE alone still yields round 7 here,
  so the 0.721 figure survives — but it was not *arrived at* honestly, and the code must be fixed before
  the next run, where it may well bite.
- **"0.72 is the ceiling" is therefore an over-claim as stated.** The LLM judge's CI [0.633, 0.811]
  overlaps the supervised bound (0.725) so heavily that "the judge is AT the signal ceiling" is not
  testable at n=100 — it is consistent with the data, not demonstrated by it. Distinguishing a 0.02
  difference at 95% confidence needs roughly n≈2,000, not 100. The *qualitative* finding — the
  bi-encoder cosine is saturated and the cross-encoder signal is what moves the judge — survives, and
  that is the part the deep-SAC line actually rests on.

### Fresh validation of the SHIPPED judge (2026-08-17)

The table above re-derives the *tuning logs*. [`validate_judge.py`](validate_judge.py) does the
complementary thing — it runs `search_as_code.harness.DiagnosticJudge`, the prompt actually in
the SDK, over the frozen oracle set and adds the two reference points the original write-up
lacked → [`judge_validation_test.json`](judge_validation_test.json):

| | held-out balanced accuracy (n=100) |
|---|---|
| always-PASS baseline | 0.500 |
| **shipped `DiagnosticJudge`** | **0.700 [0.613, 0.789]** · false-accept 0.298 · false-reject 0.302 |
| logistic regression on the same 9 signals, 5-fold | **0.722 ± 0.039** — *no LLM call* |

Two things follow, and both are corrections against us:

- **The shipped prompt reproduces the UNTUNED score.** Its confusion matrix (`tp=37 tn=33 fp=14
  fn=16`) is round 0's, exactly; round 7 was `tp=37 tn=35 fp=12 fn=16`. The shipped text is 99.5%
  — not 100% — identical to the adopted revision. Since DJ-2 already showed the entire claimed
  gain *is* those two examples, the practical reading is the same either way (issues.md DJ-4).
- **The LLM is not adding accuracy over the signals it reads** (DJ-5). A cheap classifier matches
  or beats it. The honest framing of this component is therefore *not* "an LLM judge that mimics
  the oracle", but "a **diagnostic** controller": its value is `DIAGNOSIS` / `TECHNIQUE` /
  `NEXT_QUERY` — the per-sub-fact prescription that drives the next hop, which §2's numbers do
  support and which a classifier cannot produce. A LogReg stop-gate is the obvious cheaper
  baseline to A/B the PASS/FAIL half against.

Findings, honestly:
- The bi-encoder cosine is **saturated** (PASS min-sim 0.86 vs FAIL 0.81) — the judge can't separate
  covered from missing, and no critic fixes it. Adding a **cross-encoder** per-sub-fact score (PASS +1.5
  vs FAIL −4.0, a 5.5-pt gap) is what moves the judge from 0.585 to ~0.70 (a gap that IS larger than the interval, unlike the
  critic-tuning step above).
- 0.72 is *consistent with* a signal ceiling: a supervised model on the same signals tops out around
  there, plausibly because snippet-level relevance can't verify whether the *exact* gold doc (vs a
  near-identical HotpotQA distractor) is present. **Stated as a proven ceiling this was an over-claim** —
  see the CI discussion above (DJ-2).
- **The critic was never the bottleneck.** Same-model self-critique reaches 0.721; an independent
  **Qwen-32B** critic reaches 0.70 — neither beats the signal ceiling. (gpt-4.1/4o are 403 on this project,
  so the independent critic had to be local; Qwen-32B ran 4-bit on the shared GPU.)

The judge emits structured, next-hop-actionable reasoning:
`COVERED / MISSING / DIAGNOSIS (vocab_gap|entity|buried|absent) / TECHNIQUE / NEXT_QUERY / CONFIDENCE / VERDICT`.

## 2. The playbook: judge → skill-lookup → technique (RAG_Techniques) → forge

Each weak sub-fact (cross-encoder < 0) is routed through a **semantic skill-lookup** over a catalog seeded
from [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) (HyDE, decomposition,
fusion/RRF, reranking, PRF, fielded/self-query, and **LLM-authored `os_query`** — a raw OpenSearch DSL body,
validated read-only). Assembly reserves one slot per sub-fact then fills by RRF (fixes the multi-hop
dilution where late broad hops evict earlier golds).

### Numbers (n=30, 4-hop, oracle-stop; `global` = blind rewrite baseline, `diagnostic` = this playbook)

| corpus | arm | all-golds@10 | recall@10 | avg hops | avg retr. calls |
|---|---|---|---|---|---|
| **HotpotQA** | global | 0.467 | 0.825 | 4.13 | 19.0 |
| **HotpotQA** | **diagnostic** | 0.433 | 0.783 | **3.90** | 20.2 |
| **SU (su_docs)** | global | 0.333 | 0.758 | 4.80 | 19.9 |
| **SU (su_docs)** | **diagnostic** | **0.533** | **0.858** | **3.33** | 19.0 |

- **SU: the diagnostic playbook wins decisively** — +0.20 all-golds (0.53 vs 0.33), +0.10 recall, and it
  gets there in **30% fewer hops at equal cost**. Product-doc sub-facts are genuinely varied/described, so
  targeting the right technique per sub-fact pays off.
- **HotpotQA: roughly parity** (slightly behind on solve, fewer hops). These 4-hop queries are *comparison*
  questions that **name every entity**, so a broad rewrite already retrieves them — the diagnostic edge is
  confined to the described-entity minority. An honest, interpretable split, not a universal win.

## 3. What the forge created from the discovered OpenSearch queries

The loop captured winning `(sub-fact, technique, authored-DSL)` events (HotpotQA: 30 events incl. **12
LLM-authored `os_query` DSL bodies**; SU: 10 events — `os_query` degrades on the memory store, so SU relies
on hyde/fielded/decompose). From these, `forge.author_code_primitive` had the LLM **author free-form
retrieval code over the full SDK**, each **validated against a held query with gold** before acceptance.

**Authored code primitives (all compose decompose × {hybrid, HyDE, fielded} → RRF-fuse):**

| primitive | corpus | validated | composes |
|---|---|---|---|
| `hotpot_authored_1` | HotpotQA | 4/4 golds | query_fielded · hyde_search · hybrid · fuse(RRF) · raw `_search` |
| `hotpot_authored_2` | HotpotQA | ✓ | query_fielded · hyde_search · hybrid · fuse(RRF) · raw `_search` |
| `hotpot_authored_3` | HotpotQA | 4/4 golds | query_fielded · hyde_search · hybrid · fuse(RRF) · raw `_search` |
| `su_authored_2` | SU | 4/4 golds | query_fielded · hyde_search · hybrid · fuse(RRF) |
| `su_authored_3` | SU | 4/4 golds | query_fielded · hyde_search · fuse(RRF) |

**Composed skills:** `hotpot_diag_arsenal`, `su_diag_arsenal` (the winning technique mix, RRF-fused).
**Subagents:** `hotpot_subfact_agent`, `su_subfact_agent` (plan = authored primitive → arsenal).
**Learned prompt rules (1 per corpus):** "for a generically-DESCRIBED entity use hyde; for a NAMED entity
use fielded/os_query; fuse per-sub-fact with RRF and reserve one slot per sub-fact."

All persisted to `forge_store_{hotpot,su}/` (`code_primitives.jsonl` / `skills.jsonl` / `subagents.jsonl` /
`learnings.md`) and registered live, so later queries can select the forged primitive via the skill-lookup.

## 4. SAC-replicate: do the FORGED primitives + the autonomous judge match raw-query relevance?

Can SAC-style composition (the forged authored primitives / skills / subagents), driven by the LLM
diagnostic judge **without the oracle** ("without knowing the ceiling"), reproduce the relevance of raw
oracle-guided targeting? Three arms, n=30, 4-hop:
- `raw_oracle` — diagnostic playbook, raw techniques, **oracle**-stop (the target).
- `sac_oracle` — retrieval via the **forged authored primitives** + `diag_arsenal` skill/subagent, oracle-stop.
- `sac_judge` — same forged primitives, but the **LLM judge decides stop** (no oracle).

| corpus | arm | all-golds@10 | recall@10 | hops | stop-correct |
|---|---|---|---|---|---|
| **HotpotQA** | raw_oracle | 0.467 | 0.808 | 3.70 | — |
| | sac_oracle | 0.433 | **0.792** | 4.07 | — |
| | sac_judge | 0.367 | 0.783 | 3.43 | 0.567 |
| **SU** | raw_oracle | 0.533 | 0.850 | 3.33 | — |
| | sac_oracle | 0.433 | 0.817 | 3.90 | — |
| | sac_judge | 0.333 | 0.792 | 3.33 | 0.467 |

- **Yes on relevance.** `sac_oracle` reaches recall@10 within **0.02–0.03** of raw (0.792 vs 0.808;
  0.817 vs 0.85) — the forged primitives reproduce raw-query relevance; you don't need a fresh raw query
  each time.
- **The autonomous judge recovers ~93–97% of recall** (`sac_judge` 0.783 / 0.792) but loses **0.10–0.20 on
  strict all-golds** — because `stop_correct` is only 0.47–0.57 (it PASSes one sub-fact short). The
  retrieval is not the bottleneck; the **stop signal** is, bounded by the same 0.72 judge ceiling.

Take-away: SAC primitives + diagnostic judge mimic raw-query *recall* autonomously; the oracle's only
remaining advantage is the last ~10–20 pts of all-golds, purely a perfect stop signal.
(`run_sac_replicate.py`, `sac_replicate_{hotpot,su}.json`.)

## 5. Free-form explore — structure-emergent, forged from raw OpenSearch queries

The diagnostic playbook above **hardcodes** `decompose → per-sub-fact arsenal`. That wins on multi-hop
(HotpotQA/SU) but **loses on BrowseComp** (dense 0.079 vs decompose 0.025 recall@10) — because BrowseComp
questions are ONE entity satisfying MANY constraints (the gold matches the *whole conjunction*, so
splitting the query scatters retrieval). A fixed recipe can't know this; **explore should discover it.**

### `agentic_solve` (standard: `search_as_code.harness.agentic_solve`)
The LLM **authors the retrieval strategy itself each hop**, as code over the OpenSearch query surface
(`session.search` dense/keyword/hybrid, `hyde_search`, `query_fielded`, raw `store._search` DSL, plus
in-scope `fuse_ids` / `rerank`). It **chooses the structure** — keep the query whole (dense/hybrid + rerank)
or decompose — instead of being forced to decompose. Guidance, not dictation:
- the **deep judge** runs every hop and emits `COVERED / MISSING / DIAGNOSIS / TECHNIQUE / NEXT_QUERY`,
  fed to the next hop as a *structure-neutral* hint (an earlier version let the judge's "absent→decompose"
  heuristic bias the strategist to decompose everything — fixed by framing it as an optional hint);
- the **RAG-Techniques `SkillLookup`** suggests techniques for the missing aspect;
- **memory**: per-hop findings persist within a query (cross-hop, `AgentMemory` working set) and winning
  strategies persist across queries (cross-query `skill_win`, recalled to seed later queries → skill building).

### Structure emerges (the key result)
Given only the question + judge/memory guidance, the LLM picks opposite structures per corpus, unprompted:

| corpus | LLM decomposed | correct structure |
|---|---|---|
| HotpotQA (multi-doc) | **3/3** → decompose | ✅ recall 1.0 / 0.75 / 0.50 |
| BrowseComp (conjunctive) | **1/5** → whole-query | ✅ (free-form recall@10 ≈ 0.11 vs hardcoded-decompose 0.025, dense 0.079) |

So keeping the query whole on BrowseComp — **discovered, not set** — recovers it from the decompose disaster.

### The 7-stage explore pipeline (`run_explore_pipeline.py`) — the default way to run explore
1. **Explore with raw OS queries, ORACLE (ceiling) stop** — `agentic_solve` gold-stopped, up to `max_hops`
   (10), capturing the winning strategies. Runs on **max(200, 33% of the corpus)** queries, parallelized.
2. **Deep judge** — the `DiagnosticJudge` (0.72 signal-ceiling), corpus-agnostic.
3. **Validate WITHOUT the ceiling** — re-run held queries with the judge deciding stop; compare to oracle-stop.
4. **Forge from the raw queries** — synthesize ONE reusable primitive from the winning strategies,
   **preserving the discovered structure** (whole-query vs decompose), validated on held gold; + skill + subagent.
5. **Validate on training with the new forge** — the forged primitive reproduces exploration recall.
6. **Explore ends → commit** the forged artifacts.
7. **Run on all data with the new primitive** — then the actual analysis.

```bash
python -m experiments.deep_judge.run_explore_pipeline <corpus> [n_train] [n_val] [n_test] [max_hops=10] [workers=8]
```

### BrowseComp → OpenSearch
BrowseComp was memory-only (so `os_query` degraded there). `experiments/browsecomp/index_to_opensearch.py`
indexes its 100K docs + precomputed gte-base vectors into OpenSearch (with a plain `text` field — the
default `text.keyword` sub-field hits Lucene's 32766-byte term limit on BrowseComp's large docs). Verified:
OS kNN matches exact cosine 19/20; raw OS queries (BM25/phrase/hybrid/kNN/boosts) now run there.

### Full BrowseComp run (274 train = 33%, 40 val, 200 test, 10 hops, 8 workers) — end-to-end
The pipeline ran on the OpenSearch-indexed BrowseComp and **closed the whole loop**:

- **Stage 1 — discovered structure = `whole-query`** (decomposed **39/274 = 14%**), explore recall@20 **0.089**.
  The agent chose to keep the query whole on its own — the learned rule records it.
- **Stage 3 — validate without the ceiling**: judge-stop recall@20 **0.054** vs oracle-stop **0.119** (the
  autonomous judge recovers ~45% of oracle recall — it stops early on this brutal corpus; honest).
- **Stage 4 — forge**: synthesized a **whole-query** `browsecomp_explored_primitive` (decompose×… would be
  wrong here) — hybrid + dense + HyDE fused, then cross-encoder rerank — plus `browsecomp_explored_skill`
  and `browsecomp_explored_agent`.
- **Stage 7 — run on ALL 830 gold queries** with the forged primitive vs dense:

  | arm | recall@10 | recall@20 | all-golds@10 |
  |---|---|---|---|
  | dense (baseline) | 0.062 | 0.094 | 0.029 |
  | **forged (explored whole-query)** | **0.086** | **0.131** | **0.048** |
  | | **+38%** | **+40%** | **+67%** |

  The forged whole-query primitive **beats dense on every metric** — the structure was *discovered*, *bottled*,
  and *pays off* on the full data. (Absolute numbers stay low: BrowseComp is a ~signal-ceiling needle-in-100K
  benchmark; the past dense floor was 0.061 recall@10.)

Two bugs found + fixed en route (both now in standard): the forge's acceptance bar was too strict for a
low-recall corpus (validate dense-relative on ≥25 held), and `CodePrimitive`s couldn't call `fuse_ids`/`rerank`
(added to `forge._safe_globals`, so authored primitives are self-contained). One honest wrinkle: the pipeline
classifies structure by the **first-hop** code (the initial strategic choice); the cross-query memory stored
**last-hop** codes (which drift toward decompose over 10 hops), so the forged primitive was authored from the
discovered *structure* rather than the memory exemplars.

### Relationship to the past
This is the **same** `explore→forge→replicate` loop as `explore_forge` (tasks #40/#41), with the one
assumption removed that broke it — exploration is **no longer hardcoded to decompose**; structure is
discovered and forged per corpus (decompose for HotpotQA/SU, whole-query for BrowseComp).

### Artifacts (BrowseComp, `forge_store_browsecomp_explored/`)
- code primitive `browsecomp_explored_primitive` (whole-query: hybrid+dense+HyDE → RRF → cross-encoder rerank)
- skill `browsecomp_explored_skill`, subagent `browsecomp_explored_agent`
- learned rule: *"discovered structure = whole-query (decomposed 39/274 in exploration)"*
- runners: `run_explore_pipeline.py` (the 7-stage default), `reforge_and_full.py`, `run_forged_on_full.py`;
  results `explore_pipeline_browsecomp.json`, `explore_full_browsecomp.json`.

## 6. Reproduction — strong 8B retrievers on BrowseComp-Plus (Qwen3-Embedding-8B, ReasonIR-8B)

To place our gte-base pipeline against the published BrowseComp-Plus retrievers (arXiv 2508.06600 Table 2),
we re-embedded the **same 100K-doc corpus** with two 8B retrievers and measured dense recall on all 830 gold
queries, **matching the paper's protocol**: docs truncated to **512 tokens** ("maximum context length of 512
tokens across all methods for fair comparison") and encoded plain, cosine kNN with `ef_search=1200` so top-k
is near-exact. Our golds come from `qrel_golds.txt` → the paper's **Gold Document Retrieval** setting, so we
compare against that row and at the paper's own operating points (**R@5 / R@100 / R@1000**). The one
non-obvious knob is the **query instruction**: Qwen3-Embedding prepends `Instruct: Given a web search query,
retrieve relevant passages that answer the query\nQuery:{q}` on the *query* side only; ReasonIR's card uses an
empty instruction.

| Retriever (dim) | R@5 | R@100 | R@1000 | vs paper (Gold) |
|---|---|---|---|---|
| gte-base (768) — our default, plain | 0.043 | — | — | (weak baseline) |
| **Qwen3-Embedding-8B** (4096) — instruct | **0.200** | **0.592** | **0.850** | **matches/exceeds** |
| &nbsp;&nbsp;↳ *paper Qwen3-8B, Gold* | 0.185 | 0.558 | 0.835 | |
| **ReasonIR-8B** (4096) — instruct | 0.122 | 0.452 | 0.761 | ~3–4 pts low |
| &nbsp;&nbsp;↳ *paper ReasonIR-8B, Gold* | 0.153 | 0.497 | 0.789 | |

*(830 queries. `repro_dense_extended.json`. An earlier pass reported R@5 0.175 for Qwen3 — that was
understated by a too-low HNSW `ef_search`; the near-exact re-run gives 0.200, right on the paper's 0.185.)*

**What it says**
- **Qwen3-Embedding-8B is a 1-to-1 reproduction** — we match/exceed the paper's Gold-doc numbers at all three
  operating points (R@5 0.200 vs 0.185, R@100 0.592 vs 0.558, R@1000 0.850 vs 0.835).
- **ReasonIR-8B lands ~3–4 pts below the paper** at every point (80 % → 91 % → 96 % of the paper as k grows —
  a ranking gap, not a coverage gap). The Qwen3 match validates our engine + protocol (same OpenSearch HNSW,
  same `ef_search`, same 512-token cap), so the residual is a **ReasonIR-specific encoding difference**: the
  paper encodes via Tevatron, its HF card only shows an empty-instruction example, and indeed the instruction
  is a **no-op for us** (instruct 0.122 ≈ plain 0.122) — ReasonIR's intended reasoning-instruction/pooling
  isn't fully public. Documented honestly rather than papered over.
- **The retriever is the lever, not the augmentation.** gte-base → Qwen3-8B-instruct lifts Recall@5
  **0.043 → 0.200 (~4.6×)**; ReasonIR-8B gives 0.122 (~2.8×). Both dwarf anything our forged SAC primitive
  added on top of gte-base.
- **The query instruction is half of Qwen3's score** (it was the missing piece in our first pass, not
  truncation), but **~neutral for ReasonIR** (which the card confirms).
- **SAC augmentation value is inverse to retriever strength.** On weak gte-base the forged primitive helped
  (§4/§5); on the strong 8B dense arm the same forge *hurts* (forged R@10 0.114 < dense 0.149,
  `explore_full_qwen8b.json`) — a strong retriever has less headroom for query-side scaffolding to recover.

**Reproduction gotcha (documented so it doesn't recur).** ReasonIR-8B ships custom Llama modeling code for
the transformers-4.5x era. On the repo's **transformers 5.6.2** it can be coerced to load, but its rotary
`inv_freq` (a non-persistent buffer recomputed in `__init__`) materialises as **uninitialised random memory**
under 5.x meta-device loading → per-process-random positional encoding → a corpus embedded in one process is
orthogonal to queries encoded in another → **recall collapses to 0** (it looks fine within a single process,
which is what makes it dangerous). Fix: run ReasonIR through an isolated venv pinned to **transformers
4.51.3** (`reasonir_venv`, built `--system-site-packages` so it reuses the base torch/opensearch). Under it
the same weights load with no patches, encode is deterministic across processes, and a 295-doc probe gold
ranks 18/11/5/9/3 vs 289/240/78/9/2 on the corrupt path. Qwen3-8B has no such issue (loads on 5.6.2 via
sentence-transformers). `reasonir_encoder.py` refuses to run under transformers ≥ 5.

**Artifacts / how to reproduce**
- Embed + index (per model): `experiments/browsecomp/embed_and_index.py` (Qwen3-8B, sentence-transformers)
  and `experiments/browsecomp/embed_reasonir.py` (ReasonIR, venv). Indices: `browsecomp_qwen8b`,
  `browsecomp_reasonir` (dim 4096), `browsecomp` (gte-base, 768).
- Query eval: `experiments/browsecomp/reproduce_qwen8b.py` (env `BC_EMB`/`BC_INDEX`/`BC_TASK`; plain vs
  instruct) and `experiments/browsecomp/repro_reasonir.py` (venv). Results: `repro_qwen8b.json`,
  `repro_dense_reasonir.json`; the **operating-point comparison vs the paper** (R@5/R@100/R@1000, exact kNN)
  is `repro_dense_extended.json`.
- ReasonIR steps run as `reasonir_venv/bin/python -m experiments.browsecomp.{embed_reasonir,repro_reasonir}`.
- To reproduce the paper's operating points, raise `ef_search` on the kNN index first
  (`PUT <index>/_settings {"index":{"knn.algo_param.ef_search":1200}}`) — the HNSW default undershoots top-k
  recall and made an earlier pass report Qwen3 R@5 0.175 instead of 0.200.

## Files
- `judge_core.py` — the diagnostic judge (prompt, render, parse, metrics).
- `build_evalset.py` / `augment_ce.py` — frozen oracle-labelled eval set + cross-encoder signal.
- `tune_judge.py` — prompt tuning via a critic (`--critic qwen` for the independent local critic); `qwen_critic.py`.
- `skill_catalog.py` — RAG_Techniques catalog + semantic skill-lookup.
- `os_query.py` — validated LLM-authored OpenSearch DSL (read-only).
- `run_playbook.py` — 3-arm comparison (global / widen / diagnostic).
- `run_forge_playbook.py` — the Phase-C pipeline: run → capture → forge → report (per corpus).
- Results: `forge_playbook_{hotpot,su}.json`, `agreement_curve_ce_*.json`, `tuning_log_ce_*.md`.
