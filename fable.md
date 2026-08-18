# fable.md — from audited research repo to a shippable Search-as-Code product

*Written 2026-08-18, after a four-track review (SDK primitives, deep judge, explore→forge,
open-problems staleness) whose findings are logged in `issues.md` §17. This file is the plan:
the target architecture, every fix and algorithm change needed, the validation protocol, the
repo cleanup, the competitive positioning, and the RLM question. It supersedes the scattered
"proposed cleanup" notes in `STRUCTURE.md` and the task board in `CHANGELOG.md`.*

---

## 0. The thesis (what the product is)

**One `pip install`. The LLM writes retrieval code against a primitive SDK, over any vector DB.
The system starts at dense, and only goes deeper when a validated judge says dense failed —
escalating up to 10 hops of authored strategies (including raw OpenSearch DSL). Everything it
learns per corpus is forged into named, provenance-carrying skills/primitives/subagents that
future queries reuse. Cost is the headline: 1 model turn and ~25× fewer input tokens than
tool-calling, structurally.**

That last sentence is the only headline claim that currently survives audit
(`experiments/multi_hop_synth_queries/RESULTS.md` §4b: 1 turn vs ~8–9, ~610 vs ~15,000–18,800
input tokens, widening with hop depth). Everything else in this plan exists to make the other
claims — quality parity with dense, learned per-corpus skills, a trustworthy judge — true **by
construction**, not by narrative.

### The five commitments (from the review, agreed as the immediate work)

1. **Restore the dense-default gate to HEAD** (FRG-1) — ✅ committed `a8b74a8` on
   `fix/dense-default-gate` (also stops forging from the forced first hop — part of FRG-2);
   needs merge to `main`.
2. **Re-validate the judge properly** (DJ-6..9) — group-split, tune-only calibration,
   production-identical rendering, beat a tuned `min_ce` threshold or drop the LLM from the gate.
3. **Fix the pseudo-score feed** (DJ-10) — `agentic.py:214`, `playbook.py:153`.
4. **Unit-test `primitives.py`** (TEST-5) — the layer the product is named after has ~5 tests.
5. **Forge artifacts get provenance + a real acceptance bar** (FRG-3/4).

---

## 1. Where we actually are (honest state)

**Survives audit:** the cost/token win of code-mode; explore's corpus-knowledge lift (+0.06
recall, benefits *both* arms); the adapter conformance suite (caught 3 real bugs on day one);
coverage-first RRF assembly (`playbook._reserve`); the `issues.md` audit culture itself.

**Does not survive (as of §17):**
- The judge's PASS/FAIL is behaviorally a one-feature threshold (`min_ce > 0.1`) with a mis-set
  cut point; its validation split leaks 52/76 queries; its thresholds were calibrated on the test
  half; in production 3 of its 9 signals are constant (DJ-6..10).
- "Structure-emergent" exploration is largely dictated by the author prompt and misattributed to
  hop 1 (FRG-2). The forged skill/subagent artifacts are vestigial (FRG-3).
- The dense-default gate — the README's "SAC never underperforms dense" guarantee — is absent
  from HEAD (FRG-1).
- 21/26 primitives untested; MMR/`.info`/normalize_scores/emulation correctness bugs
  (SDK-C15..18); sandbox has no timeout and a misleading security test (SDK-C19).
- Quality-vs-tool-calling claims retracted; only cost holds (EXP-2, P1-1/7, CB-1).

**Read first:** `issues.md` §13 (fix order), §17 (this review). The cross-cutting diagnosis is
unchanged: *documented properties with no enforcing test, failing silently.* Every workstream
below therefore lands with its test in the same commit.

---

## 2. Competitive landscape — what we must match or beat

### Perplexity — "Search as Code" (the thesis validator)
Perplexity rearchitected their search into an **Agentic Search SDK**: models are the control
plane, generate Python in a compute sandbox, and orchestrate atomic primitives (retrieval,
ranking, filtering, fanouts, dedup, rendering — e.g. `sdk.search.web_many()`,
`sdk.llm.extract_many()`). Claims: beats OpenAI/Anthropic/Exa on 4/5 benchmarks incl.
**BrowseComp 0.805**, 85% token reduction on a CVE case study, "new cost-performance frontier."
Two design choices worth copying: (a) the SDK is *deliberately incomplete* — agents build missing
components on the fly in code; (b) state persists on a **filesystem, not an in-memory REPL**
("better reliability on long trajectories").
Source: https://research.perplexity.ai/articles/rethinking-search-as-code-generation

### Hornet — "retrieval engine for agents" (the verification validator)
Hornet's bet is a **verifiable API surface**: schema-first design, and three verification levels —
syntactic (OpenAPI-valid configs), semantic (cross-config conflict errors agents can self-correct
from), **behavioral (relevance/ranking/latency metrics observable and comparable)** — so agents
learn/configure/optimize retrieval autonomously, with safe deploy/rollback.
Source: https://hornet.dev/blog/how-we-build-a-retrieval-engine-for-agents

### Glean — enterprise agentic search (the learned-retriever validator)
Glean's **Waldo** is an RL-trained agentic search model that runs *before* the frontier model —
doing only search/retrieval — claiming ~50% lower latency and ~25% fewer tokens; platform is a
permissions-aware knowledge graph with purpose-built per-task indexes.
Source: https://www.glean.com/blog/waldo-launch

### Our wedge (say this in the README, and make it true)
All three are **closed services over their own index**. We are the only one of the four that is:
1. **Database-agnostic** — same authored code over OpenSearch/Qdrant/Chroma/pgvector/FAISS/…,
   with a *published conformance suite* as the contract (Hornet's "behavioral verification," but
   open and self-hostable).
2. **Corpus-learning in the open** — explore→forge persists *inspectable* Python primitives,
   skills, and few-shot exemplars per corpus (Glean's Waldo idea, but as readable code instead of
   model weights; Perplexity's "build components on the fly," but *kept* and versioned).
3. **Honest by construction** — every claim ships with a CI, an interval, and a standing audit.
   Nobody else's benchmark page links to its own defect log. That is a feature; keep it.

**Beat/match targets:** match Perplexity's *shape* (primitives + sandbox + code-authoring) with
credible open numbers on BrowseComp-Plus (retrieval flavor, arXiv 2508.06600 — not their
web-scale 0.805, which is a different task); beat Hornet on *verifiability of learned artifacts*
(they verify configs; we verify forged code against dense with CIs); match Glean's "search-first,
cheaper tokens" story with our measured 25–31×.

---

## 3. Target architecture — the SAC loop

```
                       ┌─────────────────────────────────────────────────┐
                       │              CONTINUAL HARNESS (per corpus)     │
                       │  skills · code-primitives · subagents ·         │
                       │  few-shot exemplars · learnings — all with      │
                       │  provenance {source hops, held metric+CI,       │
                       │  corpus fingerprint, date, supersedes}          │
                       └───────────────▲─────────────────┬───────────────┘
                                       │ forge (gated)   │ inject (recipes + fewshot)
 query ──► TRIAGE ──► hop 0: DENSE (or forged primitive if gate-approved)
                │                       │
                │                 DEEP JUDGE (calibrated stop)
                │                 PASS ──► return evidence (1 LLM turn total, cheap path)
                │                 FAIL + diagnosis ─► ESCALATE: hops 1..10
                │                       │  LLM authors code per hop:
                │                       │   · raw OS DSL (bool/match_phrase) to pin constraints
                │                       │   · decompose→fan_out→RRF when diagnosis says multi-doc
                │                       │   · hyde/prf/fielded/fuzzy per diagnosis
                │                       │  pools ACCUMULATE (monotone), reserve-per-subfact
                │                       └─► judge re-scores each hop; budget-capped
                └── every solved query: capture (winning hop, code, structure, cost)
```

Four properties this architecture must guarantee, each currently violated:

**P1 — Never below dense.** Hop 0 *is* dense (or a forged primitive that beat dense on held
queries with a CI — the dense-default gate, FRG-1). Escalation only *adds* pools (monotone
accumulation already in `agentic.py:196-198`), so recall is non-decreasing in hops by
construction. The gate must live in the SDK (`HarnessForge.accept()`), not in an experiment
script that a merge can drop.

**P2 — Depth is earned, not default.** The judge is the escalation controller. Today it can't be
trusted (DJ-6..10). Until WS2 lands, ship the **calibrated `min_ce` threshold gate** as the
default stop (it ties the LLM judge at zero cost — DJ-8/9) and keep the LLM judge for its
*diagnosis* only. This also answers open-problem #6 with the repo's own data.

**P3 — Not every query gets the same code.** Today the author prompt hardcodes "keep the query
WHOLE / do NOT decompose" (FRG-2), so authored code is near-uniform. Fix: the prompt becomes
structure-neutral; per-query variation comes from (a) the judge's diagnosis, (b) triage class,
(c) **few-shot exemplars retrieved by query-type from the forged store** (the mechanism
`explore_learning` showed works: "evidence, not prediction"). Measure diversity explicitly
(distinct strategy signatures per 100 queries) — it's a product claim, so it needs a number.

**P4 — Raw-OS queries produce dataset-specific artifacts.** The raw-DSL-first hop is the *probe*
that discovers what the corpus rewards (exact years, part numbers, field boosts). Winning DSL
patterns are what forge generalizes into skills. Keep `os_first=True` for **explore mode** but
classify structure from the *winning* hop, not `codes[0]` (FRG-2), and let `author_os_query`
introspect the schema first (SDK-A7).

---

## 4. Workstreams

### WS1 — Foundations: make the primitives layer true (1–2 weeks)
*The product is named after this layer; it must be boringly correct.*

- Write `tests/test_primitives.py` covering all 26 primitives (TEST-5). Property-style where
  possible: fusion permutation-invariance, mmr order preservation through `.top()`, info
  propagation through every ResultSet-constructing call, normalize_scores singleton behavior.
- Fix SDK-C15 (mmr writes MMR scores back), SDK-C16 (route `.info` through `rerank`/`fuse`/
  `normalize_scores` — or make `_derive` the only constructor), SDK-C17 (singleton → 1.0).
- Emulation honesty (SDK-C18): either make `_regex` exhaustive (scan via scroll/batch on backends
  that support it; on pure-ANN backends raise `EmulationQualityWarning` and document the pool
  bound) or mark regex `supported=False` and let callers fall back explicitly. Give `_keyword` a
  real BM25 (port `FastMemoryStore`'s df/length-norm index from `bc_common.py` — BC-3, it's
  already written). Collapse the four `hybrid` implementations into one shared helper with one
  pool multiplier.
- Sandbox (SDK-C19): add wall-clock timeout + output cap + per-run namespace refresh (rebind
  primitives each `run()`, keep only the user state dict); delete
  `test_sandbox_blocks_open_and_import`, replace with timeout/cap tests; inject `query`; inject
  all 26 primitives with sandbox-callable signatures (bind `generate` so `expand(query)` works).
  Adopt Perplexity's conclusion: keep **filesystem-persisted state** for long trajectories rather
  than growing in-memory namespaces.
- Wire or delete: `mark_degraded` (wire into the 61 bare-except sites — LEG-5's fix, finally),
  `Capabilities.max_top_k` (set on OpenSearch = 10,000), `native_rerank`/`multi_vector`
  (delete until read), `quality_filter` (delete), export `surface.SAC_SYSTEM` from `__init__`.
- Ship `py.typed`. Extend conformance to qdrant/pgvector (and nmslib/milvus or delete those
  adapters — an adapter without conformance coverage does not ship).

**Exit criteria:** `make check` green with the new tests; conformance parametrized over ≥7
backends; zero known behavior-divergence between native and emulated modes without a warning.

### WS2 — The judge: earn the stop signal (1–2 weeks, parallel with WS1)
*Goal: a stop/escalate gate that provably matches the oracle within CI on held-out queries, plus
a diagnosis whose marginal value is measured, so it can steer explore, forge, and live runs.*

Protocol (fixes DJ-6..15 in one redesign):
1. **Rebuild the eval set** with query-level splits (`GroupShuffleSplit` by query id; shallow+deep
   examples of one query never straddle the split). Grow it: 100 → 500+ queries per corpus,
   stratified across BrowseComp-Plus / SU multihop / HotpotQA multihop, so the judge is validated
   on the corpora it will run on (it's currently HotpotQA-synthetic only). Bootstrap **by query**.
2. **Fix the inputs before the prompt.** Real scores end-to-end (DJ-10): normalized CE scores in
   `agentic.py`/`playbook.py`, not `1/(rank+1)`. One renderer — `DiagnosticJudge.render` — used
   by tuning, validation, and production (DJ-14). Align the premise: judge coverage of the
   query's *gold structure* (subclaims capped at the corpus's docs-per-query), not always-6
   sub-facts (the DJ premise-mismatch).
3. **Baselines first.** Tuned `min_ce` threshold (fit on tune only) and query-grouped LogReg are
   the floor. The LLM judge ships only if it beats them on PASS/FAIL *or* its diagnosis shows
   measured lift (step 5). Recalibrate all prompt thresholds on tune only (DJ-7).
4. **Prompt + few-shot tuning, done right:** critic-revision rounds (existing `tune_judge.py`
   machinery, with the DJ-1 selection fix already landed) **plus few-shot exemplars drawn from the
   tune split only** — k=4–8 worked examples per verdict class, retrieved per-corpus from the
   forged store so the judge sees the corpus's own failure shapes. Select on TUNE, report TEST
   once, with grouped bootstrap CIs. Adopt only if ΔTUNE clears the CI, not 1–2 examples (DJ-2).
5. **Measure the diagnosis' marginal value** (DJ-12) with the controls that already exist: run
   `diagnostic` vs `widen` (same reserve assembly, untargeted) vs `global` at n≥100/corpus. If
   targeted-next-hop doesn't beat untargeted widening, simplify: threshold gate + generic
   escalation, and the LLM call disappears from the hot path (pure cost win).
6. **Kill stale claims** in the same PR: `README.md:113/114`, `run_explore_pipeline.py:6/103`
   (DJ-13); the unsourced "supervised ceiling 0.725" row.

**Exit criteria:** judge-stop recall ≥ 90% of oracle-stop recall on all three corpora (currently
45% on BrowseComp), reported with CIs; a `StopGate` protocol in the SDK with three
implementations (threshold / logreg / llm-judge) that are A/B-able by one flag.

### WS3 — Forge + explore: a real continual harness (2–3 weeks)
*Goal: prime-agent's loop — durable state, small evidence-backed updates, rollback — for
retrieval artifacts.*

- **Provenance record** on every artifact (FRG-4): `{created, corpus_fingerprint, source_query_ids,
  winning_hops, held_metric+CI, baseline_metric (dense), accepted_by, supersedes}`. `save()`
  appends versions; `create_skill` never silently overwrites. `learnings.md` rules get the same
  treatment: superseding a rule retires the old one from the injected block (contradictory-rules
  bug, FRG-4).
- **Acceptance gate in the SDK** (FRG-1/3): `HarnessForge.accept(primitive)` runs it on ≥30 held
  queries against the dense baseline and adopts only on CI-clearing improvement; otherwise emits
  the dense-default primitive. `min_recall=0.0` and the `0.9×dense` bar die. The gate result is
  part of provenance. One integration test forges→rejects→falls-back so no merge can drop it
  silently again.
- **Attribution + structure honesty** (FRG-2): capture per-hop `{code, ids, marginal_recall}`
  (the `capture` hook exists, unused); credit the hop whose pool contained the golds; classify
  structure by AST (does the winning code fan out?) not regex; author prompt becomes
  structure-neutral (present whole-query and decompose as equal options with the corpus's own
  exemplars). Re-run the structure-emergence claim at n≥50/corpus.
- **Skill synthesis from the winning code**, not a hardcoded ternary (FRG-3): forge prompts the
  LLM with the top-k winning programs + their per-hop marginal recall and asks for a
  *parameterized* primitive (query-slots, field names from `describe_schema`). Execute
  `LearnedSubagent` in the loop or delete the class.
- **Diversity mechanism (P3):** the forged store keys exemplars by triage class × diagnosis;
  `agentic_solve` retrieves 2–3 nearest exemplars into the author prompt. Metric: strategy-
  signature entropy over the eval set, reported next to recall.
- **Wire the two explores together or rename.** `sac.explore` (ProfilePack/templates/router) and
  the agentic explore share zero code today. Decision: the agentic pipeline is the product path;
  `sac.explore` survives as the *corpus-profiling* stage that feeds it (`describe`, synth-query
  generation, template-based *labeling* for exemplar mining) and its stub Router/Codegen stages
  are deleted (SDK-R7). `docs/EXPLORE.md` rewritten to point at one pipeline (OPM-1 residue:
  it still headlines `cv_accuracy`).
- Security/robustness: forge-time `exec` moves into the WS1 sandbox (timeout, caps, no
  `__import__` shim); `to_skill` stops swallowing exceptions — degraded runs are counted (FRG-5).

**Exit criteria:** `run_explore_pipeline` end-to-end on a fresh corpus produces a store whose
every artifact answers "where did you come from, what did you beat, by how much, ±what" — and a
rejected forge produces a working dense-default run, tested in CI.

### WS4 — The escalation controller (1 week, after WS2)
- `agentic_solve` becomes the single entry point with an explicit budget object:
  `Budget(max_hops=10, max_llm_calls, max_searches, max_tokens)` — accounted, returned in the
  result, and enforced (no more hardcoded `steps: 1` — P1-8).
- Hop schedule: hop 0 dense/forged → on FAIL, hop 1 raw-OS DSL (schema-introspected, SDK-A7;
  validated read-only — fix landed for SDK-C1) → hops 2+: diagnosis-directed authored code.
- Session state cleared per query (P1-4); reranker/embedder `_ensure` behind a lock (SDK-C7);
  memory writes batched (SDK-C8).
- Cost telemetry per hop (input/cached/output tokens, searches) so the "goes deeper only when
  needed" claim is a measured curve: recall-vs-hops and cost-vs-hops per corpus.

### WS5 — Primitive surface completeness (1 week, parallel)
The catalog must cover "what makes search work" and be *portable or honestly marked*:

| tier | primitives | status |
|---|---|---|
| retrieval modes | dense · bm25/keyword · hybrid · phrase/proximity · fuzzy · wildcard · prefix · fielded · regex · more-like-this | exist; keyword/regex emulation fixed in WS1; README table scoped to reality (OpenSearch-only ops marked) |
| query-side | rephrase · expand · decompose · HyDE · PRF (Rocchio) · auto_filter · **step-back prompting** · **query2doc** | mostly exist; add the two bolded |
| rank/fuse | cross-encoder rerank · RRF · weighted/relative fusion · MMR · semantic_dedup · diversity_quota · **reserve-per-subfact (promote `_reserve` to a public primitive — it's the multi-hop workhorse)** | exist; ONE fusion implementation (SDK-R2/LEG-6: 8 copies → 1) |
| gating | score_cutoff · adaptive_search · confidence/abstain · **StopGate (threshold/logreg/judge — WS2)** · **QPP signals as a public `score_signals`** | consolidate |
| introspection | describe_schema · sample(seeded — SDK-C4/P4-4) · content_type · describe(llm) | exists; seed the sample |
| asymmetric embedding | query-vs-passage prefixes (port `phase2/embed_models.py` — P2-5; it was worth more than every augmentation measured) | **missing from SDK, high value** |
| sub-LM | `llm_map` / `llm_extract_many` over result sets (Perplexity's `sdk.llm.extract_many`, RLM's `llm_batch`) — batched sub-model calls whose outputs stay in the sandbox | **new; the RLM bridge (§6)** |

Rule (Perplexity's, adopted): the SDK stays deliberately incomplete — the sandbox + forge exist
so agents build the rest; but everything *in* the catalog is tested and portable-or-marked.

### WS6 — Validation: one harness, three corpora, matched arms (continuous)
- **One `Arm` abstraction** (LEG-1: four incompatible harnesses today): shared toolset/prompt
  budget/token accounting; arms = `dense`, `dense+rerank` (the missing control, P1-10),
  `bm25`, `hybrid`, `tool-calling` (the *good* one — `chatbot/toolcalling.py` design, CB-1),
  `sac` (unseeded), `sac_explored`, `tool_explored` (explore seeds both — EXP-3's lesson).
- **Corpora:** BrowseComp-Plus (830q, gold-complete subset, seeded shuffle — keep, it's the
  cleanest sampling in the repo); SU 2/3/4-doc multihop; HotpotQA-derived 2/3/4-doc multihop.
  Fix BC-1 (qrels out of /tmp) and document BC-2 (KW_CHARS=2000 lexical caveat) wherever a
  lexical number appears — or re-index full text with the WS1 BM25.
- **Reporting rules (already policy, now enforced):** paired bootstrap CIs on every delta
  (`sac.bootstrap_ci`); a `caveats` field in every results JSON printed by report generators
  (§13's recommendation); forge-disjoint eval slices; worker exceptions re-raised (EXP-1);
  headline metrics: recall@10, all-golds@10, nDCG@10 × input-tokens, LLM calls, searches, hops.
- **The scoreboard the README shows:** per corpus — SAC ≥ dense (gate property, must hold by
  construction), SAC vs dense+rerank, SAC vs matched tool-calling, cost curves. Negative cells
  stay in the table. That's the differentiator no competitor will copy.

### WS7 — Repo cleanup: ship a product, not a lab notebook (1 week)

**Status of the §14 structural improvements (verified against the tree, 2026-08-18).** More has
landed than the issue log's FIXED markers show — the fix sweeps did not annotate the STR entries:

| item | status | evidence / what remains |
|---|---|---|
| STR-1 wheel never tested | **half done** | CI has a "wheel installs + README quickstart" job (`ci.yml:38`) — the cheap fix (a). Remaining: `src/` layout (b) so untested imports are impossible by construction |
| STR-2 no conformance suite | **largely done** | `tests/test_conformance.py` runs in CI over every installed backend. Remaining: qdrant/pgvector/nmslib/milvus coverage (→ §17 TEST-5, WS1) |
| STR-3 research/customer in-tree | **partial** | `phase2/3` → `internal/legacy/`; `phase4/` (51 Altera files) still at root; no companion repo/submodule; GOV-1 history decision still open |
| STR-4 no lockfile / pins | **partial** | `requirements/{dev,experiments}.txt` exist; `transformers>=4.40,<5` upper bound landed (the BC-4 lesson, `pyproject.toml:44,:55`). Remaining: a compiled lock, torch bound |
| STR-5 no pre-commit | **done** | `.pre-commit-config.yaml` with `detect-private-key` + `check-added-large-files` + ruff |
| STR-6 docs unbuildable, no link check | **partial** | `check_doc_links.py --staged` in CI + `make check`. Remaining: mkdocs build + nav |
| STR-7 version duplicated, no tags, stale dist | **partial** | version single-sourced (`dynamic = ["version"]`, `pyproject.toml:7,:74`). Remaining: **0 git tags**, stale `dist/` 0.0.1 wheel still present — delete it, tag v0.1.0 at M5 |
| STR-8 four top-level phases | **partial** | phase2/3 archived under `internal/legacy/`. Remaining: `phase4/` (customer risk) and `phase1/` (load-bearing but unshippable — move its `llm/metrics/agents` into `search_as_code/bench/`) |
| STR-9 missing governance files | **done** | `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff` all present at root |
| STR-10 no task runner | **done** | `Makefile` `make check` = the CI set; CLAUDE.md points at it |
| STR-11 examples not CI-run | **done** | `examples/01–04` exist, zero-setup, executed by CI (also closes EX-1, EX-4) |
| STR-12 two changelogs + STATUS | **partial** | `benchmark_changelog.md` folded into `CHANGELOG.md`. Remaining: **`docs/STATUS.md` still exists** (DOC-4 — delete it), CHANGELOG still mixes task board + log |
| STR-13 stale import count | moot | superseded by the phase2/3 move (MRG-1/5 cleaned the stragglers) |

So WS7's *remaining* work is: `src/` layout · `phase4` + `experiments/` internal split (+ GOV-1
history decision) · `phase1` promotion into the package · mkdocs · tag a release + delete `dist/`
· delete `docs/STATUS.md` · compiled lockfile · docs diet below.

Current top level: `phase1/`, `phase4/`, `chatbot/`, `benchmarks/`, 2-in-1 changelog, ~14 root md
files, a customer CSV, an SSH `.pub` key (unstageable now via pre-commit, but still on disk). Target:

```
searchascode/                      # the public repo (this one, renamed focus)
├── src/search_as_code/            # STR-1: src layout — wheel is what's tested
├── tests/                         # unit + conformance (the published contract, STR-2)
├── docs/                          # mkdocs site: concept, primitives, databases, explore,
│                                  # harness, judge — with lychee link-check in CI (STR-6)
├── examples/                      # 01 quickstart · 02 backend · 03 explore-first ·
│                                  # 04 harness/judge/forge — all CI-executed (already true)
├── benchmarks/                    # the WS6 harness + published results JSON (public corpora only)
├── README.md · CHANGELOG.md · CONTRIBUTING.md · SECURITY.md · CITATION.cff · LICENSE
├── fable.md                       # this plan, until done, then folded into docs/
├── issues.md · open_problems.md · learnings_standard.md   # the audit trail — keep, it's the brand
└── (soul.md + STRUCTURE.md fold into CLAUDE.md + docs/; one constitution, not three)

searchascode-internal/             # separate repo or submodule (STR-3, mem0's model)
└── phase1..4, chatbot, SU/Altera/BrowseComp data & runs, experiments/* that touch customer data
```

- `phase1`'s load-bearing pieces (`sac_surface` → already `surface.py`; `llm.py`, `metrics.py`,
  agents) move **into the package** (`search_as_code/bench/`) so experiments are reproducible
  from a release (STR-8); phases 2–4 and `chatbot/` go internal.
- Governance mechanized: `.pre-commit-config.yaml` with `detect-private-key` +
  `check-added-large-files` (STR-5, kills the GOV-2 class); the customer-artifact guard stays in
  CI; **decision needed from repo owner:** history rewrite for the GOV-1/MRG-3 files already in
  `origin/main` history (a `.gitignore` cannot fix it; documented, still open).
- Versioning: single-source version, tag releases, delete stale `dist/` (STR-7); lockfile +
  upper bounds on `transformers`/`torch` (STR-4 — the ReasonIR lesson, BC-4).
- Docs diet: README (pitch + quickstart + scoreboard + audit-status), one CHANGELOG, mkdocs for
  the rest. `docs/STATUS.md` deleted (DOC-4). `learnings_standard.md` finally created (LEG-2) —
  it's the promotion path six findings died waiting for.
- GEN-1 class closed for good: `Session` gets a `complete()`-style single-string generator slot;
  the six `out[0]`-then-resplit consumers are already fixed — add the regression test that locks
  the contract.

### WS8 — issues.md triage: what is already fixed vs. what this plan must still do

172 logged entries as of 2026-08-18. Verified against the tree (not just the FIXED markers —
the fix sweeps under-annotated).

**Already fixed — do not redo** (verified in code):
- *Marked FIXED in the log:* ADP-1/2/3 · CI-3 · DJ-4 · DOC-8/10 · EX-3/4 · EXP-5/6 · GOV-4 ·
  MRG-1/2/5.
- *Fixed but unannotated:* **GEN-1/2/3** (shared `_genutil.py` helper; `session.py:141` cites it) ·
  **SDK-C1** (allowlist condition fixed, documented at `os_query.py:83`) · **SDK-C2** ($and/$or/$not
  translated, `opensearch.py:203-216`) · **SDK-C4** (seeded `sample_seed`, `opensearch.py:49,:67` —
  the P4-4 port) · **SDK-C5** (re-embeds gold text, `training.py:421-425`) · **SDK-C7** (locks in
  `rerankers.py:24,:68`) · **SDK-A1** (availability gating), **SDK-A2** (realized_recall evaluator),
  **SDK-A4** (triage hints) · **PKG-1** (extras: `learn`, opensearch in `all`, `<5` transformers
  bound) · **TEST-3** (conformance suite) · **DOC-1/DOC-5/DOC-6** (surface.py in-package; badge;
  links) · **FRG-1** (gate, `a8b74a8`, pending merge) · STR-2/5/9/10/11 per the WS7 table.
  *Action item: add dated FIXED annotations to these entries so the log matches the tree.*

**Still open, mapped to the workstream that owns it:**

| owner | open entries |
|---|---|
| WS1 (primitives/sandbox) | SDK-C15 (mmr scores) · SDK-C16 (.info dies on rerank/fuse) · SDK-C17 (normalize_scores singleton) · SDK-C18 (keyword/regex emulation, 4× hybrid) · SDK-C19 (sandbox timeout/caps/poisoning, misleading test) · SDK-C3 ($eq on string metadata → `.keyword` sub-field) · SDK-C10 (MemoryStore swallows kwargs, dim check off) · SDK-C11/BC-3 (port FastMemoryStore BM25) · SDK-C12 (`all_rerank` KeyError) · SDK-C13/C14 leftovers · SDK-A8 (freshness unreachable) · SDK-R8 (`mark_degraded` 0 callers — wire or delete) · TEST-5 (21/26 primitives untested; 4 adapters uncovered; no `py.typed`; playbook test hangs) |
| WS2 (judge) | DJ-6 (query-level split leak) · DJ-7 (thresholds fit on test half) · DJ-8 (LogReg row-order artifact) · DJ-9 (one-feature-threshold equivalence, mis-set cut) · DJ-10 (pseudo-scores → constant signals) · DJ-11 (stage-3 gold leak + shared memory) · DJ-12 (diagnosis value untested) · DJ-13 (stale 0.72 claims in README/pipeline) · DJ-14 (renderer mismatch, inverted critic verdict) · P1-5/P1-6 (legacy judge stop conditions) · P2-1's unaudited sibling `align_prompts.calibrate_judge` |
| WS3 (forge/explore) | FRG-2 remainder (structure-neutral prompt, winning-hop attribution, `decomposed=` detector) · FRG-3 (acceptance bars, skill-from-winning-code, dead subagent) · FRG-4 (provenance, contradictory learnings.md) · FRG-5 (unseeded-control contamination, exec-at-load, swallowed to_skill) · SDK-A3 (vacuous `default_verify` reward) · SDK-A5 (validate-before-keep unimplemented) · SDK-A6/A7 (corpus-specific prompts; schema introspection in os_query) · SDK-R7 (dead Router/Codegen stubs) · DS-5 (forge-store provenance of acceptance rule) |
| WS4 (controller) | P1-4 (session state leaks across queries) · SDK-C8 (AgentMemory O(n²), unbounded) · P1-8 (hardcoded turns / uncounted in-code LLM calls) · P4-6 (unlocked Usage accounting) |
| WS5 (surface) | SDK-R2..R6 remainders (one RRF ×8 → LEG-6, one decompose ×4, one sandbox namespace ×5 → P4-7, dup rerank/profile helpers) · P2-5 (asymmetric embedding promotion) · DOC-2 (`help`/`list_primitives` affordance — implement or de-document) · DOC-7 (mark unimplemented taxonomy entries) |
| WS6 (validation) | LEG-1 (one `Arm` abstraction over the 4 harnesses) · P1-1/P1-7/P2-2/CB-1 (matched-arm FiQA/multi-hop re-run with the good tool baseline, or restate) · P1-2 (mirror hop-fusion into baselines) · P1-10 (dense+rerank control arm) · P1-13 (one token-accounting definition) · P1-14 (README repro command's reranker default) · P1-15 (seeded query sampling convention) · EXP-1 (re-raise worker exceptions) · EXP-4 (run multi-hop arms through the gate) · BC-1 (qrels out of /tmp) · BC-2 (KW_CHARS caveat or full-text BM25) · BC-5 (cite the paper row) · BC-6 (EF.K import-time mutation) · DS-1 (re-run the deep-explore arm with the fixed profile) · DS-3 (archive `eval_recall.py`) · P2-1-RESULT (strike the learned-profile claim from CHANGELOG/MULTI_DATASET_REPORT) · P2-3 (testzip in `ensure()`) · P2-4/P4-1/P4-9 (rename mislabeled arms) · BM-1 (realistic bench vectors) |
| WS7 (structure/governance) | STR-1b/3/4/6/7/8/12 remainders (WS7 table) · **GOV-1/MRG-3 history-rewrite decision (owner)** · GOV-2 (key still on disk — move it out) · GOV-3/P4-3/LEG-3 (phase4 exit) · DOC-4 (delete STATUS.md) · DOC-3/DOC-9 (doc corrections) · LEG-2 (create `learnings_standard.md`) · LEG-4 (chatbot: document or move) · LEG-7 (delete stale dist/) · LEG-8 (fold explore_improvement) · CI-1 (lint/type the whole tree) · CI-2 (coverage denominator) · EX-2 (un-ignore \*.csv for fixtures) · P3-1/P2-6..10, P4-2/P4-5/P4-8 (internal-repo hygiene, port `bootstrap_ci`/judge-calibration ideas before archiving) |
| annotations | OPM-1 (date-stamp open_problems.md statuses) · LEG-5 (degradation counting — lands via SDK-R8 wiring in WS1) |

Reading of the balance: **~35 entries are genuinely fixed, ~15 are moot/superseded, and ~120
remain open** — but they concentrate: WS1+WS2+WS3 close every 🟥 that touches correctness of the
product path; WS6 closes every 🟥 that touches published claims; WS7 closes the governance tail.
Nothing in the log falls outside a workstream.

---

## 5. Algorithm changes (the delta from today's design, numbered)

1. **Dense-first, judge-gated escalation** replaces "always run the pipeline": hop 0 = dense or
   gate-approved forged primitive; hops 1–10 only on FAIL. (Answers open-problems #2 and #5 with
   the cost framing the literature says is routing's real value.)
2. **StopGate abstraction**: calibrated `min_ce` threshold as default (it ties the LLM judge —
   DJ-8/9); LLM judge retained only if WS2 step 5 shows diagnosis lift. QPP `score_signals`
   become a public primitive either way. (Closes open-problem #6.)
3. **Monotone-by-construction depth**: accumulate + reserve-per-subfact + final RRF; rerank only
   inside single-gold sub-pools (open-problem #4's fix, kept and promoted to a public primitive).
4. **Structure-neutral authoring with retrieved exemplars**: per-query strategy variation comes
   from forged few-shot exemplars keyed by triage class × diagnosis, not from a prompt-imposed
   default. Diversity is a reported metric.
5. **Raw-OS DSL as schema-introspected probe** in explore; winning DSL patterns are the raw
   material forge parameterizes into skills. Structure attributed to the winning hop by AST.
6. **Evidence-gated forging**: accept only on CI-clearing win vs dense on ≥30 held queries;
   dense-default otherwise; full provenance; rollback = pointing at the previous version.
7. **Asymmetric query/passage embedding** as a first-class Session capability (worth more than
   every augmentation measured — P2-5/qwen8b finding).
8. **Judge trained per corpus with few-shot from its own tune split**, validated group-split,
   rendered identically in tuning/validation/production.
9. **Sub-LM primitives** (`llm_map`/`llm_extract_many`): batched small-model calls over result
   sets, outputs staying in the sandbox — extraction/verification hops stop costing main-model
   context (the RLM pattern, and Perplexity's `extract_many`).
10. **Class-rebalanced exemplar mining** (open-problem #1's unrun fix) applied where it now
    matters: not the 16-way router, but ensuring minority strategy classes survive into the
    few-shot store.

---

## 6. RLM (prime-agent) — can we use it? Yes, incrementally; don't rebuild on it.

What RLM is (Zhang, Kraska, Khattab — arXiv 2512.24601; Prime Intellect's blog): treat the
context as a **variable in a persistent Python REPL**; the root LM writes code to inspect/
transform it and spawns **recursive sub-LM calls** (`llm_batch`) whose outputs return as
variables, not context tokens. Tools are restricted to sub-LMs; the root model never sees raw
tool output. Result: ~unbounded effective context, big token savings on long-horizon research
tasks.

**We are already 70% of an RLM substrate.** SAC's sandbox *is* the REPL with retrieval bound in;
"intermediate state out of the model context" *is* context-as-variable; the 25× token win is the
same mechanism RLM exploits. What we're missing, in adoption order:

1. **`llm_batch` as a primitive (WS5)** — cheap, immediate: sub-LM extraction/verification over
   pooled candidates inside the sandbox. This is the 80/20 of RLM for retrieval and directly
   attacks BrowseComp's verify-heavy queries.
2. **Persistent per-session REPL state on the filesystem** (WS1 sandbox rework) — align with both
   RLM and Perplexity's finding that filesystem beats in-memory for long trajectories. Enables
   multi-query research sessions where hop-N of query 2 reuses pools from query 1.
3. **Root-LM orchestration mode** — `agentic_solve(mode="rlm")`: one root turn writes a program
   that loops hops *inside the sandbox*, calling the StopGate between iterations, sub-LMs for
   judging/extraction. This collapses our "1 turn per hop" to "1 turn per query at depth 10" —
   the natural endpoint of the cost thesis. Do this after WS2/WS4, because it makes the
   stop-gate load-bearing.
4. **Not now:** prime-agent's RL training of the harness ("learned context folding"). We have no
   RL infra; our equivalent lever is the forged store + few-shot mining, which is the
   prompt-space version of the same idea. Revisit if/when the product has usage data.

Positioning bonus: "SAC is the retrieval SDK an RLM harness plugs into" (batch-friendly,
summaries under output caps, sandbox-native) is exactly the integration surface prime-agent's
design asks for — worth a doc page and an example.

---

## 7. Milestones

| # | milestone | contents | acceptance |
|---|---|---|---|
| M0 (done/underway) | the five commitments | gate restored · judge re-validation started · pseudo-scores fixed · primitives tests · forge provenance | `make check` green; §17 Tier-1 entries marked FIXED |
| M1 | Foundations | WS1 + WS5 consolidation | conformance ≥7 backends; 0 untested primitives; one RRF/decompose/sandbox |
| M2 | Trustworthy stop | WS2 | judge/gate ≥90% of oracle recall, 3 corpora, CIs; StopGate A/B-able |
| M3 | Continual harness | WS3 + WS4 | fresh-corpus pipeline → provenance-complete store; SAC ≥ dense by construction; diversity metric reported |
| M4 | The scoreboard | WS6 full run | matched-arm results on BC-Plus/SU/HotpotQA with CIs, negative cells included, README audit table regenerated |
| M5 | Ship | WS7 + RLM items 1–2 | `pip install search-as-code` runs the whole story; internal split done; v0.1.0 tagged; docs site live |

Sequencing: M1 ∥ M2 → M3 → M4 → M5. M4 re-runs cheaply whenever anything above changes — that's
the point of one harness.

## 8. Risks / open questions

- **The judge may not earn its LLM call** (DJ-8/9/12). Acceptable outcome: threshold gate +
  generic widening ships, LLM diagnosis becomes an optional explain-mode. The architecture
  doesn't depend on which wins — that's why StopGate is an abstraction.
- **Strong retrievers erase SAC's retrieval lift** (qwen8b: forged ≤ dense). The product story
  must lead with cost + portability + learned skills, with quality as "never worse, sometimes
  better on weak/lexical corpora" — the gate makes that claimable.
- **BrowseComp all-golds may stay ≈0** (open-problem #7 — a coverage ceiling). Report R@100/1000
  reachability alongside; sub-LM verification (RLM item 1) is the most plausible mover.
- **GOV-1 history rewrite** needs an owner decision; until then the public repo carries
  SU-derived data in history.
- **Shared checkout, concurrent agents** (MRG-4/5 class): worktrees per workstream, merge to main
  frequently, guards read the index (`--staged`), CI widened beyond `search_as_code/` (CI-1).

## 9. Definition of "a successful GitHub repo showcasing a product"

A stranger lands on the README and within 10 minutes: installs from PyPI, runs an offline
example, sees the scoreboard (with intervals and its own caveats), runs explore on *their* corpus
and gets a provenance-carrying forged primitive that provably didn't underperform dense — and can
read `issues.md` to see exactly how the numbers were earned. No competitor offers the last part.

## Sources
- Perplexity, *Rethinking Search as Code Generation* — https://research.perplexity.ai/articles/rethinking-search-as-code-generation
- Hornet, *How we build a retrieval engine for agents* — https://hornet.dev/blog/how-we-build-a-retrieval-engine-for-agents · https://hornet.dev/blog/the-case-for-a-new-retrieval-engine-for-agents
- Glean, *Waldo launch* — https://www.glean.com/blog/waldo-launch · https://www.glean.com/platform/api
- Zhang, Kraska, Khattab, *Recursive Language Models* — https://arxiv.org/abs/2512.24601 · https://alexzhang13.github.io/blog/2025/rlm/
- Prime Intellect, *RLM: the paradigm of 2026* — https://www.primeintellect.ai/blog/rlm · prime-agent — https://github.com/PrimeIntellect-ai/prime-agent
- BrowseComp-Plus — https://arxiv.org/abs/2508.06600
