# Changelog

## 2026-08-17 — audit-fix sweep (`fix/audit-sweep`)

Worked `issues.md` end to end. 10 commits, each one issue-family. Highlights:

**Correctness (the silent-failure class the audit called dominant)**
- GEN-1/2/3: nine consumers took `out[0]` of a line-splitting generator, so the "validated
  multi-hop recipe" ran on ONE sub-fact, HyDE embedded a preamble line, and pretty-printed JSON
  was dropped. One shared helper (`_genutil`) + 11 regression tests.
- SDK-C1..C14: dead `os_query` allowlist, `$or` filters running unfiltered, `$eq` on strings
  matching nothing, unseeded `sample()` defeating resume, failure taxonomy reading a stripped
  `d.vector`, `regexp` never matching, thread-unsafe reranker loading, O(n²) memory writes,
  `MemoryStore` ignoring `dim`, per-query corpus re-tokenization.
- SDK-A1..A5: the 16-template space collapsing to ~4 under shipped defaults; the "self-improving"
  harness learning from a reward that was just "returned ≥1 id"; triage decomposing every
  conjunctive question; the `validate()` gate no stage implemented.
- BC-4: `_fix_meta_buffers` never rebuilt `inv_freq` — the buffer that actually corrupted ReasonIR.

**Measurement honesty**
- Promoted `phase4/metrics.py` → `search_as_code.metrics` (bootstrap CIs, `compare`, recall@k /
  all-golds@k / nDCG). It was the only significance testing in the repo.
- DJ-1/2/3: re-derived the judge headline. 0.721 **[0.633, 0.811]**; the tuning gain is
  +0.020 [−0.110, +0.150] (not distinguishable from noise); two of five table rows were the
  *untuned* prompt. "0.72 is the signal ceiling" softened to what the data supports.
- P1-7 / EXP-2: re-ran multi-hop with matched prompts and explore-first applied to BOTH harnesses.
  The quality margin does not survive (`sac_explored − tool_explored` = −0.10/−0.07/−0.08, ns);
  the cost margin does, and is large (1 turn vs ~9; ~600 vs ~15–19k input tokens). Explore is a
  real ingredient but a corpus-knowledge win, not a code-mode win. `RESULTS.md` §4b.

**Controls, so the next defect of each class is loud**
- `tests/test_conformance.py`: the adapter contract the README claimed, run against every
  installed backend. Enforcing it immediately found 3 real bugs (ADP-1/2/3).
- CI was RED before this branch (ruff 37, mypy 52 errors on committed HEAD). Now 0/0, plus new
  jobs for the wheel smoke test, conformance, doc links and the customer-artifact guard.
- `make check` as the single definition of "keep it green"; `.pre-commit-config.yaml`;
  pinned `requirements/`; `scripts/check_no_customer_artifacts.py`.
- Governance: 48 tracked customer files untracked, `*.pub` ignored (the SSH key), version
  single-sourced, `learnings_standard.md` created (soul.md named it 3× and it never existed).

Tests: 199 unit + 23 OpenSearch integration. `issues.md`: 126 → 136 entries (10 found *by* the
new controls), 7 marked FIXED.

---


Running log of everything built/changed/learned. Newest first.

## Phase 4 — answer-generation benchmark (standard)
- Generic RAG answer-gen eval: `phase4/metrics.py` (SQuAD EM/token-F1 + bootstrap 95% CI),
  `phase4/answer_gen.py` (arms: closed-book / vanilla-RAG / tool-RAG / SAC; shared generator+corpus;
  closed-book = contamination control).
- **HotpotQA (n=200, gpt-4.1-mini):** SAC EM **0.520**/F1 **0.673** > tool 0.500/0.659 > vanilla
  0.470/0.626 > closed-book 0.310/0.423. SAC tops answer quality; retrieval adds +0.25 F1 over closed-book.
- **Generalizable learning:** on a domain the generator already knows well, retrieval lifts
  *citation/source-grounding* far more than final-answer text — measure BOTH answer and citation.
- **Gotcha (custom GTE-v1.5 / "new"-arch embedders via transformers):** meta-device init leaves
  non-persistent buffers (position_ids, rotary cos/sin) uninitialized -> GPU device-assert / NaN. Fix:
  `low_cpu_mem_usage=False`, disable unpad/memory-efficient-attention, and re-materialize those buffers
  on the model device.


---

## 2026-08-11 — Diagnostic LLM-as-judge + forged-primitive playbook (SDK)
- `search_as_code.harness.DiagnosticJudge`: STOP/CONTINUE controller; per-sub-fact CROSS-ENCODER coverage
  signal (primary) + bi-encoder/lexical/cliff. Oracle-agreement 0.63->0.72 balanced-acc = the SIGNAL
  ceiling (same-model self-critique 0.721; independent Qwen-32B 0.70; neither beats it).
- `harness.diagnostic_solve`: decompose -> per-sub-fact arsenal -> reserve-slot assembly -> judge +
  RAG-Techniques `SkillLookup` (NirDiamant/RAG_Techniques) route weak sub-facts to hyde/fielded/rerank/
  decompose/prf/authored `os_query`. `forged=` runs through forged primitives (SAC-replicate).
- Results (n=30, 4-hop): SU diagnostic 0.53 vs 0.33 all-golds (+0.10 recall, 30%% fewer hops); HotpotQA
  ~parity. Forge authored 5 free-form code primitives + skills/subagents from discovered OpenSearch queries.
- SAC-replicate: forged primitives reproduce raw-query recall (sac_oracle~raw_oracle); autonomous judge
  keeps recall within ~0.02-0.06 but loses ~0.10-0.20 strict all-golds to an imperfect stop.
- pip test: `tests/test_diagnostic_playbook.py` (raw 0.875 / forged-SAC 0.750 recall@10). Write-up:
  `experiments/deep_judge/README.md`.

---


# STATUS & TASK BOARD (updated 2026-07-22)

Granular status so another agent can pick up the work. Read this first, then the dated log below.

## How to run (environment)
- **Repo:** `/home/taranjeet.bakshi/code_search_harness`. Python venv: `source .venv/bin/activate`
  (fallback `source phase1/.venv/bin/activate`).
- **Secrets:** `export $(grep -v '^#' ~/taxonomy/.env | xargs)` — provides `OPENAI_API_KEY` (gpt-4.1-mini).
  Loaded automatically by `phase1/common.py:load_env()`.
- **OpenSearch:** local tarball on `localhost:9200` (no auth). Restart from tarball if a session kills it;
  data persists. Check: `curl -s localhost:9200/_cat/indices?h=index,docs.count`.
  Live indices: fiqa(57638), fiqa_openai_large(57638), hotpotqa(100978), scifact(5183), nfcorpus(3633),
  arguana(8674), sac_learned(1). scidocs/trec-covid pending stage-2.
- **GPU is SHARED** with another user (`aditti.ramsisaria`). ONLY kill your own PIDs. Run GPU jobs
  serially (Qwen reranker OOMs with >2 workers). gte-base embedder + Qwen3-Reranker-0.6B (max_length 512).
- **Datasets:** BEIR zips in `phase2/data/<name>/`. Small ones download fast from the UKP mirror;
  the 2GB HotpotQA is cached under `~/.cache/huggingface/datasets/BeIR___hotpotqa` (load offline).
  GOTCHA: validate zips with `zipfile.testzip()` before use — parallel downloads truncate silently.
- **Key files:** SDK `search_as_code/`; benchmark harness `phase1/`; experiments `phase2/`
  (`beir.py` loader/registry, `beir_run.py` ingest+eval, `run_campaign.sh`/`run_stage2.sh` drivers,
  `make_report.py` report, `learned.py`/`learn_rules.py`/`align_prompts.py` learning pipeline).
- **Reports:** `phase2/MULTI_DATASET_REPORT.md` (auto-gen), `docs/PHASE3.md` (multi-backend plan),
  `phase2/runs/*.json` (per-dataset results, gitignored).

## Task board
### ✅ Done
- **SDK + 5 adapters + sandbox + 320-primitive taxonomy + DB matrix** (pushed: github.com/oro-jackson/searchascode-sdk).
- **Phase-1 FiQA benchmark** — base vs MCP-tool vs SAC, trace UIs (`phase1/live_ui.py`), 100-query run.
- **FiQA diagnostics** — ceiling (best fixed ~0.47, oracle 0.57, recall@100 0.73), miss analysis (case-by-case),
  embedder swap gte→OpenAI-large (nDCG 0.39→0.53), QwenReranker (flat-query 0.31→0.44).
- **New/updated primitives** — normalize_query, rare_terms, quality_filter, smart_search, retrieve_rerank,
  score_cutoff/adaptive_search, normalize_scores, relative_score_fusion, diversity_quota, semantic_dedup,
  confidence/abstain, topics, auto_filter, prf_search, hyde; synonym-aware expand.
- **Learning pipeline** — `learn_rules.py` (mine aliases/glossary/synonyms/routes), `align_prompts.py`
  (exemplars + judge calibration), `learned.py` (runtime pull from `sac_learned` OpenSearch index).
- **HotpotQA base** — SAC 0.96/0.92 vs dense 0.79/0.62 (flagship multi-hop win). `phase2/runs/hotpot.json`.
- **Multi-dataset base numbers (5/5 + HotpotQA) — CAMPAIGN COMPLETE.** SciFact, ArguAna, NFCorpus, SciDocs,
  TREC-COVID + HotpotQA all done; final scoreboard in dated log. `phase2/runs/*.json`, `MULTI_DATASET_REPORT.md`.

- **5-dataset campaign (#19)** — DONE; final scoreboard in dated log.
- **HotpotQA learning→final (#17)** — DONE; +2.7pts all_found from learned synonyms.
- **Conditional learning pass (#22)** — DONE; benefit scales with multi-hop structure, self-limits on ArguAna.
- **Phase 3 multi-backend (#21)** — DONE (in-process): FAISS+SQLite adapters + Qdrant fix; 6-backend HotpotQA
  cross-DB relevance; exact-parity proven + ANN-tuning finding. `phase3/`, `phase3/PHASE3_RESULTS.md`.

- **Selective GitHub push (#18)** — DONE. Pushed 54 standard files (SDK adapters incl. faiss_store/
  sqlite_store + qdrant fix, phase2 benchmark+learning code, phase3 cross-DB code, all reports+CHANGELOG)
  to `oro-jackson/searchascode-sdk` main (`c350d23..af1cc91`). Verified NO artifacts/secrets staged
  (run data, dataset zips, logs, learned_*/impact_* all gitignored).

### 🔄 In progress
- **All tracked tasks complete.** Optional next: Phase 3 extensions (nmslib/Milvus-lite adapters; server
  backends need infra; re-index HotpotQA with tuned HNSW to re-baseline dense against exact).

### ⬜ Planned (granular, ordered)
1. **Selective GitHub push** [#18, #20] — push ONLY standard code: SDK incl. new `adapters/faiss_store.py`,
   `sqlite_store.py`, fixed `qdrant.py`, registry; `phase2/beir*.py`, `make_report.py`, `learn_rules.py`,
   `align_prompts.py`, `impact_eval.py`, `run_*.sh`; `phase3/*.py`; reports `MULTI_DATASET_REPORT.md`,
   `phase3/PHASE3_RESULTS.md`, `docs/PHASE3.md`, `CHANGELOG.md`. KEEP gitignored: `sac_learned` index,
   `phase2/runs/`, `phase2/data/`, `phase3/*.log`, `phase3/*.json`, learned_*/impact_* artifacts.
2. **Phase 3 extensions** (optional): nmslib + Milvus-lite adapters; server backends (ES/Mongo/Milvus/
   Pinecone) need infra; re-index HotpotQA with tuned HNSW (m=48) to re-baseline dense against exact.
3. **Ongoing:** keep CHANGELOG + reports updated each step; ground each step in papers/articles.

### ⚠️ Gotchas / lessons for future agents
- Validate downloaded zips (`testzip()`); the UKP mirror truncates on parallel/timeout downloads.
- `pkill -f "pattern"` can match its own bash wrapper (exit 143/144) — kill by explicit PID.
- Shared GPU: only your PIDs; serialize Qwen jobs; max_length 512 to avoid OOM.
- Agentic search is NOT universally better — it *hurt* on ArguAna. Apply primitives conditionally.
- FiQA has no learnable routing structure (single-strategy) → learning is net-neutral there. Learning pays
  off on heterogeneous data.

---

## 2026-08-11 — Diagnostic LLM-as-judge + forged-primitive playbook (promoted to the SDK)
- **Diagnostic judge** (`search_as_code.harness.DiagnosticJudge`): a STOP/CONTINUE controller that
  coverage-checks each sub-fact with calibrated signals (per-sub-fact CROSS-ENCODER score is the primary
  signal; bi-encoder cosine is saturated) and emits `MISSING/DIAGNOSIS/TECHNIQUE/NEXT_QUERY/VERDICT` the
  next hop consumes. Oracle-agreement **0.63→0.72 balanced-acc** once the cross-encoder signal is added —
  and 0.72 **is the signal ceiling**: a supervised model tops out there, and neither same-model
  self-critique (0.721) nor an independent **Qwen-32B** critic (0.70) beats it. The residual is
  snippet-level (can't verify the exact gold vs a distractor), not a reasoning/critic limit.
- **Playbook** (`harness.diagnostic_solve`): decompose → per-sub-fact arsenal (hybrid+HyDE+fielded RRF) →
  reserve-one-slot-per-sub-fact assembly (fixes multi-hop dilution) → judge + **RAG-Techniques skill
  lookup** (`harness.SkillLookup`, seeded from NirDiamant/RAG_Techniques) routes each weak sub-fact to
  HyDE/fielded/rerank/decompose/PRF/**authored os_query** (`harness.author_os_query`, validated read-only DSL).
- **Forge from discovered OpenSearch queries** (`experiments/deep_judge/run_forge_playbook.py`): on
  HotpotQA+SU (n=30, 4-hop) the loop captured winning queries and the LLM **authored 5 free-form code
  primitives over the full SDK** (all validated on held queries; compose hybrid+HyDE+fielded+RRF), plus
  composed skills + subagents + a learned rule per corpus, persisted to `forge_store_{hotpot,su}/`.
  Numbers: **SU diagnostic 0.53 vs 0.33 all-golds (+0.10 recall, 30% fewer hops)**; HotpotQA ~parity
  (its comparison queries name their entities, so broad retrieval already suffices).
- **SAC-replicate** (`run_sac_replicate.py`): the forged SAC primitives reproduce raw-query relevance —
  `sac_oracle` recall within ~0.02–0.03 of `raw_oracle`; the autonomous judge (`sac_judge`, no oracle)
  keeps recall within ~0.02–0.06 but loses ~0.10–0.20 on *strict* all-golds because its stop decision is
  right only ~47–57% of the time (the same 0.72 ceiling). **Retrieval is not the limiter — the stop
  signal is.**
- **Pip test**: `tests/test_diagnostic_playbook.py` reproduces raw≈SAC recall via the installed package API.
- Full write-up: `experiments/deep_judge/README.md`.

---

## Multi-dataset campaign (in progress)
- **Datasets added** — SciFact, NFCorpus, ArguAna, SciDocs, TREC-COVID (BEIR, real qrels,
  diverse query types) on top of FiQA + HotpotQA. Generic ingest/eval harness (`phase2/beir.py`,
  `phase2/beir_run.py`, `phase2/run_campaign.sh` — serial, shared GPU).
- **Goal** — per-dataset base numbers (dense vs SAC vs tool), then learning → final, in one report.
- **Research grounding (this step)** — BEIR (Thakur et al. 2104.08663) shows the best dense model
  (TAS-B) beat BM25 on only 8/18 datasets zero-shot; BM25/hybrid stays strong out-of-domain. Motivates
  the *heterogeneous* dataset spread here — dense doesn't universally win, so the value is a system that
  *routes* per query type (hybrid alpha, keyword, rerank, decompose) rather than one fixed retriever.
- **NFCorpus smoke** — dense recall@10 0.171 / hybrid 0.180 over 323 queries (hard many-gold medical set).
- **SciFact DONE** — full 300q: dense 0.843 / hybrid 0.860 r@10. Subset 40q: dense 0.858, hybrid 0.876,
  **SAC 0.876**, tool 0.866; all_found@10 0.825 uniform. Term-heavy → hybrid gives a small lift, SAC
  matches hybrid (routes to it, no hops needed). Near-saturated: easy datasets don't need agentic search.
- **ArguAna DONE** — full 1406q: dense 0.752 / hybrid 0.730 r@10. Subset 40q: dense **0.850**, hybrid
  0.750, SAC 0.725, tool 0.500. **Anti-result**: dense is best; SAC/tool *hurt*. ArguAna = "given a long
  argument, find its counter-argument" — the raw argument IS the ideal query, so rephrase/decompose
  degrades it. Lesson: agentic manipulation must be *conditional*; on some tasks the plain query wins.
- **NFCorpus DONE (n=40)** — full 323q: dense 0.189 / hybrid 0.170 r@10. Subset 40q: dense 0.189, hybrid
  0.170, SAC 0.179, tool 0.177; all_found@10 0.025–0.05. Many-gold medical set → recall@10 structurally
  capped; SAC ≈ dense (no hop/route lever helps much). Honest low-ceiling dataset.
- **SciDocs DONE → CAMPAIGN COMPLETE (5/5).** Corpus 25,657, ~4.93 gold/q. Full 1000q: dense 0.245 /
  hybrid 0.218 r@10. Subset 40q: **SAC 0.241 > dense 0.222 ≈ hybrid 0.221 > tool 0.153**. Modest SAC win
  (citation retrieval benefits from rerank/pool-expansion; tool-calling clearly worse). all_found ~0
  (multi-gold). Final scoreboard below.
- **FINAL SCOREBOARD (recall@10, SAC-subset N=40, + HotpotQA):**
  | dataset | type | dense | hybrid | SAC | tool | verdict |
  |---|---|---|---|---|---|---|
  | HotpotQA | multi-hop | 0.79 | 0.95 | **0.96** | 0.75 | SAC wins big (all_found 0.92 vs 0.62) |
  | SciDocs | citation | 0.22 | 0.22 | **0.24** | 0.15 | SAC edges out |
  | SciFact | claim/term-heavy | 0.86 | 0.88 | **0.88** | 0.87 | SAC ties hybrid |
  | NFCorpus | medical many-gold | **0.19** | 0.17 | 0.18 | 0.18 | ~tie (r@10 capped) |
  | ArguAna | long-argument | **0.85** | 0.75 | 0.73 | 0.50 | dense wins; SAC HURTS |
  | TREC-COVID | many-gold(~493) | 0.016 | 0.017 | 0.017 | 0.014 | uninformative metric |
  **Thesis confirmed:** agentic code-search helps in proportion to query complexity — decisive on multi-hop,
  small edge on citation, neutral on term-heavy/many-gold, and it can *hurt* when the raw query is already
  optimal (ArguAna). No single retriever wins everywhere → the value is per-query routing.
- **TREC-COVID DONE** — corpus 171,332, **avg 493 gold/query**. Full 50q: dense 0.019 / hybrid 0.020 r@10.
  Subset 40q: dense 0.016, hybrid 0.017, SAC 0.017, tool 0.014; all_found 0. **CAVEAT: recall@10 is
  saturated-low here** (max ≈ 10/493 ≈ 0.02) — all methods hit the ceiling, so recall@10 is *uninformative*
  for this dataset. Proper metric = nDCG@10 or recall@100. Reported for completeness; excluded from the
  "who wins" claim. Confirms the many-gold caveat also affecting NFCorpus/SciDocs.
- **SciDocs + TREC-COVID zips truncated** (14M/19M partials from a timed-out parallel download → BadZipFile) —
  stage-1 campaign skipped both. Re-download: TREC-COVID valid at 71MB; SciDocs larger than expected (~120MB+).
  First stage-2 attempt raced the downloads (checked zips at 03:05 before complete) → skipped again. Fixed
  `run_stage2.sh` to *poll testzip() until valid* (cap ~20 min) then run. Re-launched. Lesson: gate on
  data-readiness, not just GPU-readiness.
- **SciFact grounding** — 5,183 abstracts, 300 test claims; BM25 nDCG@10 ≈ 0.662, dense improves Recall@100
  but BM25 wins early-rank on term-heavy queries → SciFact favors keyword/hybrid, a good contrast to
  HotpotQA (multi-hop) and FiQA (semantic). Related agentic work found: "Think Before You Retrieve:
  Test-Time Adaptive Search with Small LMs" (2511.07581), "Claim-Aware Scientific RAG: Evidence-First
  Retrieval and Abstention" — parallel our adaptive-routing + confidence/abstain primitives.
- **Planned Phase 3** — multi-backend adapters (FAISS/nmslib/Elasticsearch/Milvus/SQLite/Mongo/Pinecone)
  + HotpotQA cross-DB relevance (`docs/PHASE3.md`).

## Phase 4 — answer-generation benchmark (global RAG authenticity, task #24)
- **HotpotQA answer-gen DONE (n=200, EM/F1 + bootstrap 95% CIs, gen=gpt-4.1-mini, k=5).** All arms share
  generator+corpus+prompt; only retrieval differs; closed-book = contamination control.
  | arm | EM [95% CI] | F1 [95% CI] |
  |---|---|---|
  | **SAC** | **0.520 [0.450,0.585]** | **0.673 [0.614,0.728]** |
  | tool-RAG | 0.500 [0.430,0.565] | 0.659 [0.600,0.713] |
  | vanilla-RAG | 0.470 [0.405,0.535] | 0.626 [0.568,0.683] |
  | closed-book | 0.310 [0.250,0.375] | 0.423 [0.362,0.485] |
  - **SAC tops answer quality**, clearly beats vanilla RAG (+0.05 EM/+0.05 F1); edges tool-calling (CIs
    overlap → the two agentic methods tie, both > vanilla). Retrieval lift over closed-book +0.20–0.25 F1
    (real value, not memorization). Harness: `phase4/{metrics.py,answer_gen.py}`, `runs/answergen_hotpotqa.json`.
  - Method note: authentic protocol (deterministic EM/F1 = leaderboard metric; contamination control;
    equal budget). 2WikiMultiHopQA/MuSiQue queued (need their own corpora built).

## Phase 3 extensions (task #23)
- **Tuned-HNSW re-index CONFIRMS the ANN finding.** Rebuilt HotpotQA as `hotpotqa_tuned` (m=48,
  ef_construction=512) via OpenSearch `_reindex` (same vectors, no re-embedding). Dense recall@10
  **0.792 → 0.900**, all_found@10 **0.617 → 0.800** — recovers ~85% of the gap to exact (0.925/0.850).
  Definitively a *build-parameter artifact*, not inherent to ANN. (opensearch-py raised "got more than
  100 headers" on the reindex long-poll, but the copy finished: tuned index has all 100,978 docs; measured
  separately.) `phase3/reindex_tuned.py`.
- **nmslib + Milvus-lite adapters** — new in-process backends (`adapters/nmslib_store.py`,
  `milvus_store.py`), registered. nmslib = HNSW (lazy one-shot build); Milvus-lite = embedded MilvusClient
  local-file (COSINE). Both compose MemoryStore for keyword/regex/hybrid; smoke-tested OK. Milvus fix: uri
  must be a non-existent path (milvus-lite makedirs it) — use `mkdtemp()/milvus.db`, not a pre-created file.
- **8-backend cross-DB matrix DONE** — one API, 8 stores: OpenSearch-default 0.792 (tuned 0.900),
  Milvus-lite 0.875, Chroma 0.908, and **FAISS/SQLite/memory/Qdrant/nmslib all exact 0.925/0.850**.
  nmslib HNSW matches exact (good defaults); Chroma near-exact at lowest latency (0.9ms). Thesis
  ("one primitive API, any vector DB") demonstrated across 8 backends. `phase3/PHASE3_RESULTS.md`.

## Phase 3 — multi-backend (in progress, task #21)
- **FAISS + SQLite adapters** — new in-process backends (`adapters/faiss_store.py`, `sqlite_store.py`),
  registered in registry. FAISS = exact IndexFlatIP (cosine); SQLite = float32-BLOB brute-force (the
  "no vector DB needed" reference). Both compose `MemoryStore` for keyword/regex/hybrid → fully capable,
  identical to the reference on dense/keyword/hybrid/metadata-filter (conformance smoke passed).
- **Cross-DB HotpotQA relevance (`phase3/cross_db_relevance.py`)** — scrolled the 100,978 existing vectors
  OUT of OpenSearch (no re-embedding), loaded into FAISS/SQLite/memory, ran the SAME dense retrieval via
  the one primitive API. Result (n=60):
  | backend | recall@10 | all_found@10 | latency |
  |---|---|---|---|
  | OpenSearch (HNSW ANN) | 0.792 | 0.617 | 3.0ms |
  | FAISS (exact) | 0.925 | 0.850 | 5.2ms |
  | SQLite (exact) | 0.925 | 0.850 | 17.6ms |
  | memory (exact) | 0.925 | 0.850 | 60.0ms |
  - **Parity proven:** the 3 exact backends are *identical* (0.925/0.850) — one API, any DB, same relevance.
  - **IMPORTANT reframing:** OpenSearch's HNSW ANN silently loses ~13 recall / ~23 all_found points vs
    exact. So the HotpotQA "dense baseline" (0.79) was depressed by ANN approximation — **exact dense already
    hits 0.925**, close to SAC's 0.96. Part of SAC's apparent multi-hop win over dense was actually
    *recovering ANN-approximation losses* (fan-out + rerank re-surface true neighbors HNSW missed), not
    purely reasoning. Honest nuance; SAC still adds the true-multi-hop bridging on top.
  - **ef_search does NOT recover it** — tested ef_search=100/512/2048, all give 0.7917 (no change). So the
    loss is *build-time* (HNSW `m`/`ef_construction` defaults, m=16), not query-time; only a **re-index** with
    higher m/ef_construction closes it.
  - **Chroma (`--extra`) proves it's tuning, not ANN:** Chroma's HNSW on the SAME vectors hits **0.908/0.817
    at 0.8ms** — 12 pts above OpenSearch's 0.792 and near exact (0.925). A well-built ANN nearly matches brute
    force; OpenSearch's default is simply under-built.
  - **Qdrant adapter FIXED + measured** — two bugs: (1) point ids must be uuid/int → map arbitrary strings to
    uuid5 (original in `payload._sac_id`); (2) qdrant-client ≥1.10 dropped `.search()` → use `.query_points()`.
    Qdrant local mode then matches **exact 0.925/0.850** (168ms; local mode warns >20k points). Also hardened
    the eval loop to guard per-backend query failures.
  - **FINAL 6-backend matrix:** OpenSearch 0.792 (under-built HNSW), Chroma 0.908 (tuned HNSW, 0.8ms),
    FAISS/SQLite/memory/Qdrant all **0.925/0.850 (exact)**. **"One primitive API, any vector DB" demonstrated
    across 6 backends** — identical relevance from exact stores; only the ANN build config varies. Phase 3 core done.
  - **Follow-up:** re-index HotpotQA with m=48/ef_construction=512 (or use exact/FAISS for ≤1M corpora);
    re-baseline HotpotQA dense against exact (0.925) — narrows SAC's headline margin.
- **Remaining backends** (`docs/PHASE3.md`): Chroma/Qdrant adapters exist (pip-install to test live); nmslib
  + milvus-lite pip-installable in-process; Elasticsearch/MongoDB/Milvus-server need a server (no Docker
  here); Pinecone needs a cloud key. In-process subset (FAISS/SQLite/memory) done + measured.

## Learning / router / primitives (Phase 2)
- **Conditional-learning sweep DONE (task #22)** — mined rules + measured learned-synonym Δ on 4 datasets:
  | dataset | synonyms mined | queries expanded | base r@10 | learned r@10 | Δ |
  |---|---|---|---|---|---|
  | HotpotQA | 15 (+8 aliases) | 24 | 0.803 | **0.817** | **+0.014 r@10 / +2.7pts all_found** |
  | SciDocs | 16 | 17 | 0.230 | 0.232 | +0.003 |
  | SciFact | 5 | 5 | 0.863 | 0.863 | 0.000 |
  | NFCorpus | 4 | 2 | 0.167 | 0.167 | 0.000 |
  | ArguAna | **0** | 0 | 0.793 | 0.793 | 0.000 |
  **Elegant conditional result:** ArguAna mined **zero** rules — its dense-misses aren't lexical
  (argumentative stance, not vocabulary), so the learner produces nothing to apply → the earlier
  "manipulation hurts ArguAna" risk is *self-limited*: no rules = no harm. HotpotQA (most rules, most
  expansions) is the only real gain. **Learned-synonym benefit scales with multi-hop/entity structure;
  elsewhere it's a safe no-op.** Answers "where does learning pay off": on lexical-mismatch multi-hop.
  Artifacts: `phase2/runs/learned_<ds>.json`, `impact_<ds>.json`, `run_learn_sweep.sh`.

- **HotpotQA end-to-end (base→learning→final) DONE (task #17):**
  - *Base:* SAC **0.96/0.92** vs dense 0.79/0.62, hybrid 0.95/0.90, tool 0.75/0.68 (recall@10 / all_found@10).
  - *Learning:* generalized `learn_rules.py`/`align_prompts.py` to any dataset via `beir.eval_data()`; mined
    **8 aliases + 15 synonyms** (entity/attribute: heritage→{origin,descent,ethnicity}, head coach→{coach,
    manager}, starred→{cast,featuring,starring}, …) + judge threshold 0.05 (F1 0.99); stored in `sac_learned`
    id=hotpotqa. 0 exemplars (no exploration data for hotpot — gracefully skipped).
  - *Final (learned profile on dense path, deterministic, n=150):* dense raw 0.803/0.633 →
    learned-normalized 0.807/0.647 → **synonym-expand+fuse 0.817/0.660 (+1.3 / +2.7 pts)**.
  - **Finding:** learned synonyms HELP on HotpotQA (+2.7 all_found) but were net-neutral on FiQA — multi-hop
    entity vocabulary benefits from expansion to bridge hops. Learning transfers where routing structure /
    vocab-bridge exists; it does so *without an LLM at query time* (cheap dense clawing toward the SAC ceiling).
- **HotpotQA base** — SAC **recall@10 0.96 / all_found@10 0.92** vs dense 0.79/0.62, hybrid 0.95/0.90,
  tool 0.75/0.68. First dataset where SAC decisively beats dense (multi-hop fan-out pays off).
- **Learned profile + runtime pull** (`learned.py`) — aliases/glossary/synonyms/exemplars/judge-threshold
  mined offline (`learn_rules.py`, `align_prompts.py`), stored in OpenSearch `sac_learned`, injected at
  runtime into normalize_query/expand/prompt. "standard code + custom learned code".
- **FiQA learned impact** — net-neutral (normalize) / negative (blanket synonym-expand): learning needs
  heterogeneous data; rules must be applied conditionally, not blanket.
- **Judge calibration** — threshold tuned vs qrels (FiQA ~trivial: threshold 0).
- **QwenReranker** (Qwen3-Reranker-0.6B) — lifts FiQA flat-query recall@10 0.31→0.44 (ms-marco *hurt*: 0.28).
- **New primitives** — normalize_query, rare_terms, quality_filter, smart_search, retrieve_rerank,
  score_cutoff/adaptive_search, normalize_scores, relative_score_fusion, diversity_quota, semantic_dedup,
  confidence/abstain, topics, auto_filter, prf_search, multi-representation (hyde); synonym-aware expand.
- **Miss analysis** — 27% of FiQA gold missed@100; 58% at rank 100–500 (near-miss, flat score curve);
  case-by-case remedies (rerank / keyword-boost / rephrase / HyDE / contextual chunking / label artifacts).
- **Ceiling** — FiQA best fixed ~0.47 (dense-weighted hybrid); oracle 0.57; recall@100 0.73.
  **Embedder swap** gte-base→OpenAI text-embedding-3-large: nDCG@10 0.39→0.53 (matches SOTA).
- **Decision rules in the prompt** (`docs/SELECTION.md`) — when to call/chain each primitive.

## Phase 1 (foundation)
- SDK: unified primitive API over any vector DB; 5 adapters (memory/opensearch/qdrant/chroma/pgvector);
  sandboxed code-mode execution; 320-primitive taxonomy; 150-source research; DB support matrix.
- OpenSearch + FiQA (57k) benchmark; base vs MCP tool-calling vs SAC (LangChain); 100-query run; trace UIs.
- Repo: github.com/oro-jackson/searchascode-sdk


---

# Benchmark changelog (folded from benchmark_changelog.md)

# Benchmark changelog & plan

Living log of all benchmarking activity for the **search-as-code** harness. Every
benchmark has an ID, a metric, a method, a **status**, and a **results** block
that is filled in when it runs. Newest results are appended under each item.

**Status legend:** ⬜ planned · 🟡 running · ✅ done · ⚠️ partial/blocked · ❌ failed

## Environment (captured at run time)
- Host: Linux, Python 3.13 (`.venv-dummy`), CPU + **NVIDIA RTX 5090** GPU.
- OpenSearch **2.17.1**, single node on `:9200`, index `fiqa` = **57,638 docs**.
- Embedder: `thenlper/gte-base` (768-d, GPU). Reranker: `cross-encoder/ms-marco-MiniLM-L-12-v2`.
- Agent LLM: `gpt-4.1-mini` (OpenAI), key from `~/taxonomy/.env`.
- Harness: `benchmarks/bench.py` (subcommands below) + `phase1/benchmark.py` (agent paths).
- Each run writes raw JSON to `benchmarks/results/` and a summary here.

## How to reproduce
```bash
pip install -e '.[phase1]'                       # deps
python -m benchmarks.bench scalability           # Section A
python -m benchmarks.bench throughput            # Section B
python -m benchmarks.bench micro                 # Section E2
python -m benchmarks.bench resilience            # Section E3
python -m benchmarks.bench embedding             # Section E4
python -m phase1.benchmark -n 10 \
  --reranker cross-encoder/ms-marco-MiniLM-L-12-v2   # Sections C, D, E1
```

---

## Section A — Scalability
How the system behaves as corpus size and ingest volume grow.

| ID | Benchmark | Metric | Backend | Status |
|----|-----------|--------|---------|:--:|
| A1 | Ingest throughput vs batch size | docs/sec | OpenSearch | ✅ |
| A2 | Query latency vs corpus size (1k/10k/50k) | ms/query (p50/p95) | memory | ✅ |
| A3 | Index build time + memory footprint vs corpus size | s, MB | memory | ✅ |
| A4 | Fan-out (`search_many`) scaling vs #queries | total s, speedup | memory | ✅ |

**Results**
- A1 ✅ OpenSearch bulk ingest (5,000 docs, 64-d): batch=100 → **7,474 docs/s** · batch=500 → 8,096 · batch=1000 → **8,584 docs/s**. Larger batches help modestly (+15% from 100→1000); the batched-upsert change pays off.
- A2 ✅ in-memory dense query latency (brute-force cosine, p50/p95): 1k → **0.21 / 0.21 ms** (5,377 qps) · 10k → 2.58 / 2.92 ms (384 qps) · 50k → **14.9 / 16.7 ms** (65 qps). Latency scales ~linearly with corpus (brute-force) — fine for dev/small corpora; use OpenSearch HNSW for large ones.
- A3 ✅ in-memory build + footprint: 1k → 3 ms / 0.5 MB · 10k → 20 ms / 5.1 MB · 50k → **105 ms / 25.6 MB** (≈512 B/doc at 128-d float32). Linear, cheap.
- A4 ✅ `search_many` fan-out (10k corpus): 1 q → 37.4 ms/q · 4 q → 3.79 · 8 q → 3.73 · **16 q → 3.51 ms/q** — thread fan-out amortizes per-query cost ~10× vs serial (first call includes matrix build).

---

## Section B — Throughput (APIs per second)
Sustained query rate the retrieval layer can serve.

| ID | Benchmark | Metric | Backend | Status |
|----|-----------|--------|---------|:--:|
| B1 | Single-thread QPS per mode (dense/keyword/hybrid/regex) | queries/sec, ms/query | OpenSearch (fiqa) | ✅ |
| B2 | Concurrent QPS vs worker count (1/2/4/8/16) | queries/sec, p95 ms | OpenSearch (fiqa) | ✅ |

**Results** (live `fiqa`, 57,638 docs; 300 queries/mode)
- B1 ✅ single-thread QPS (p50): **keyword 558 qps** (1.73 ms) · **dense 377 qps** (2.64 ms) · **hybrid 84 qps** (11.9 ms — runs dense+keyword then RRF) · **regex 8.4 qps** (119 ms — scans the `.keyword` subfield, inherently costly). Guidance: reach for regex only on genuinely exact-token needs.
- B2 ✅ concurrent dense QPS: 1 → 402 · 2 → 770 · **4 → 969 (peak)** · 8 → 918 · 16 → 867. Scales ~2.4× to 4 workers, then plateaus/declines (single OpenSearch node + client GIL). Sweet spot ≈ 4–8 concurrent.

---

## Section C — AI-agent latency
End-to-end latency of the three retrieval paths (base / MCP tool-calling / SAC code-mode).

| ID | Benchmark | Metric | Status |
|----|-----------|--------|:--:|
| C1 | Per-path end-to-end latency over N queries | mean, p50, p95 s | ✅ |
| C2 | Per-hop latency + hop-count distribution (LLM paths) | s/hop, #hops | ✅ |

**Results** (`phase1.benchmark -n 100`, gpt-4.1-mini, FiQA — stable sample)
- C1 ✅ mean end-to-end latency: **base 0.021 s** (no LLM) · **SAC 7.69 s** · **tool-calling 15.89 s**. **SAC is ~2.1× faster than MCP tool-calling** — one code program vs many serial tool round-trips.
- C2 ✅ LLM calls/query: **SAC 2.57** vs **tool-calling 6.15** — SAC makes ~58% fewer model round-trips because intermediate results stay in the sandbox.

---

## Section D — Token consumption & cost
LLM economics per path (the code-mode efficiency thesis).

| ID | Benchmark | Metric | Status |
|----|-----------|--------|:--:|
| D1 | Input / output / cached tokens per query, per path | tokens, $/query | ✅ |
| D2 | Prompt-cache hit rate (SAC stable prefix) | % cached input | ✅ |

**Results** (`-n 100`, totals over 100 queries)
- D1 ✅ **SAC:** 335,146 input / 40,693 output tokens, **$0.1453 total (~$0.00145/query)**. **tool-calling:** 415,665 input / 60,016 output, **$0.2288 total (~$0.00229/query)**. base: $0. **SAC is ~36% cheaper per query** — it sends ~19% fewer input and ~32% fewer output tokens (intermediate hits stay in the sandbox) *and* more of its input is cache-billed.
- D2 ✅ prompt-cache hit rate: **SAC 53.5%** (179,456 of 335,146 input tokens cache-billed — the stable `SAC_SYSTEM` prefix) vs **tool-calling 26.9%**. Measured from `usage.prompt_tokens_details.cached_tokens`.

---

## Section E — Quality, primitives & reliability
Retrieval quality (the ceiling), primitive micro-throughput, and resilience overhead.

| ID | Benchmark | Metric | Status |
|----|-----------|--------|:--:|
| E1 | Retrieval quality per path | Recall@10 / nDCG@10 / MRR@10 | ✅ |
| E2 | Primitive micro-throughput (fuse/mmr/semantic_dedup/rerank/score_cutoff) | ops/sec, ms/call | ✅ |
| E3 | Resilience overhead (retry wrapper, batched vs single upsert) | µs overhead, docs/sec | ✅ |
| E4 | Embedding throughput (gte-base, GPU) | texts/sec | ✅ |

**Results**
- E1 ✅ retrieval quality (`-n 100`, stable): **Recall@10** — **SAC 0.5487**, base 0.4788, tool-calling 0.4397. **nDCG@10** — **SAC 0.4076**, tool-calling 0.3988, base 0.3792. **MRR@10** — tool-calling **0.4754**, SAC 0.4459, base 0.4153. **SAC wins Recall@10 (+7 pts over base, +11 over tool-calling) and nDCG@10**; tool-calling edges MRR (it front-loads one strong hit). SAC also beats the README's older 0.491 R@10 — the newer primitives/prompt help.
- E2 ✅ (pool=200 hits, mean over 300 calls): `confidence` 166k ops/s · `score_cutoff` 125k · `dedup`/`diversity_quota` 71k · `fuse` 9.4k · `relative_score_fusion` 6.7k · `rerank(lexical)` 1.3k (0.75 ms) · **`mmr` 161 ops/s (6.2 ms)** · **`semantic_dedup` 73 ops/s (13.6 ms, embeds each hit)**. Takeaway: pure-rank/score primitives are effectively free; the vector/embedding primitives (mmr, semantic_dedup) dominate cost — apply them only to a trimmed pool.
- E3 ✅ `with_retry` overhead **0.14 µs/call** (direct 0.02 → wrapped 0.16); `chunked()` **45.1M items/sec**. Resilience wrappers are negligible on the hot path.
- E4 ✅ gte-base embedding throughput (RTX 5090, 2,000 texts): batch 32 → 8,333 texts/s · batch 128 → 11,979 · **batch 256 → 12,343 texts/s**. At ~12k texts/s, embedding is not the ingest bottleneck (OpenSearch bulk at ~8.6k docs/s is).

---

## Final summary (run 2026-07-22, all 16 benchmarks ✅)

**Scalability** — OpenSearch bulk ingest **8.6k docs/s** (batch 1000); in-memory brute-force dense scales linearly (50k docs → p95 16.7 ms, 25.6 MB); fan-out amortizes to 3.5 ms/query at 16-wide. For large corpora use OpenSearch HNSW, not the in-memory backend.

**Throughput (APIs/sec)** — live FiQA (57k docs): keyword **558 qps**, dense **377 qps**, hybrid **84 qps**, regex **8.4 qps** single-thread; concurrent dense peaks at **~970 qps @ 4 workers**.

**AI-agent (N=100, stable)** — SAC wins on nearly every axis vs MCP tool-calling:

| path | Recall@10 | nDCG@10 | MRR@10 | latency | LLM calls | input tok | cache hit | cost/100q |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| base (hybrid) | 0.479 | 0.379 | 0.415 | 0.02 s | 0 | 0 | — | $0 |
| tool-calling | 0.440 | 0.399 | **0.475** | 15.9 s | 6.15 | 415,665 | 26.9% | $0.229 |
| **SAC** | **0.549** | **0.408** | 0.446 | **7.7 s** | **2.57** | 335,146 | **53.5%** | **$0.145** |

**SAC: best Recall@10 (+11 pts vs tool-calling) & nDCG@10, ~2.1× faster, ~36% cheaper, 2× the cache-hit rate, <½ the LLM calls.** Tool-calling only edges MRR@10.

**Reliability/primitives** — resilience wrappers are free (`with_retry` 0.14 µs; `chunked` 45M items/s); embedding on the RTX 5090 hits **12.3k texts/s**; only `mmr` (161 ops/s) and `semantic_dedup` (73 ops/s) are costly primitives — apply them to trimmed pools.

**Provenance:** all raw JSON in `benchmarks/results/`; agent summary in `phase1/runs/bench_summary.json`. Agent metrics are the full 100-query run.

## Heartbeat log
3-minute progress heartbeats are appended to `benchmarks/HEARTBEAT.md` and posted
in-session while benchmarks run.
