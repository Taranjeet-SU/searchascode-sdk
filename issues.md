# issues.md — repo-wide audit

A full read-through of every folder and file in this repo, logging **approach**, **code**, and
**redundancy** issues. Newest sections appended at the bottom; within a section the highest-impact
issues come first.

- **Severity:** 🟥 blocker / wrong results · 🟧 friction / risk · 🟨 minor / cosmetic
- **Category:** `[A]` approach (the design or the claim is wrong) · `[C]` code (bug, fragility,
  efficiency) · `[R]` redundancy (duplicate implementations, dead code)
- Started 2026-08-13. Per-experiment friction logs live alongside their experiment
  (e.g. [`experiments/qwen8b_sac/issues.md`](experiments/qwen8b_sac/issues.md)); this file is the
  repo-wide view.

> Audit note: a run was in flight during this audit (`run_explore_pipeline browsecomp` with
> Qwen3-Embedding-8B, writing to `experiments/qwen8b_sac/`). Nothing was modified.

---

## 1. `search_as_code/` — the shippable SDK

### Approach

#### 🟥 SDK-A1 `[A]` The 16-template label space collapses to ~4 templates under the default labeling config
`explore/templates.py:87-105` — `StrategyContext.hyde/decompose/rephrased/expanded` **silently
return `self.dense()` / `self.hybrid()`** when `use_llm` is false, and `rerank()` returns its input
unchanged when no reranker is set. `Explorer.dataset()` defaults to `label_llm=False,
label_rerank=False` (`explore/engine.py:169`). Under those defaults:

| template | what it actually runs with the defaults |
|---|---|
| `hyde_rerank`, `dense_rerank`, `mmr_diverse`(≈), `decompose_rerank` | `light_dense` |
| `rephrase_rerank`, `multi_rephrase`, `score_guarded`, `escalating` | `light_hybrid` |
| `deep_hyde_decompose`, `deep_all` | fuse of dense+dense(+keyword) |

`best_from_hits` then breaks the resulting ties by **cheapest cost** (`explore/router.py:44-47`), and
`light_dense` has cost 0 — so it wins essentially every tie *by construction*.
`open_problems.md` #1 and #8 attribute this to "minority collapse" and "non-orthogonal templates";
the mechanical cause is that with the shipped defaults **most templates are literally the same
function**. Either make LLM labeling the default, or refuse to label templates whose dependencies
are unavailable (mark them `unavailable`, not `miss`).

**Scope — the published numbers are NOT affected.** Every experiment that trained a router passes
`label_llm=True, label_rerank=True` explicitly (`phase2/beir_train.py:50`, `phase2/beir_qrels.py:66`,
`experiments/browsecomp/explore_router.py:47`, `experiments/explore_forge/run_forge.py:139`), so
`explore_learning` §4b and `primitive_selection` measured the real templates. This is a **footgun
default in the shipped SDK**: `docs/EXPLORE.md`'s own quickstart is `explorer.fit(n=5000)` with no
flags, which silently labels a degenerate template space. The two failure modes look identical in
the output (`light_dense` dominates), so a user cannot tell which one they hit.
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — availability gating: unavailable templates are recorded as unavailable, not labeled (`explore/templates.py:275-296`).

#### 🟥 SDK-A2 `[A]` The router still reports the metric the repo says it abandoned
`open_problems.md` #3 states CV classification accuracy is a misleading routing metric and that "all
headline numbers now use the realized task metric". But the shipped training subsystem still returns
`cv_accuracy` / `router_lift_over_fixed` (= CV accuracy − majority-class share) as its headline
(`explore/training.py:476-483`), and `Explorer.train()` records `cv_acc` / `vs_fixed` into the pack
manifest (`explore/engine.py:200-205`). There is **no realized-recall evaluator anywhere in the
SDK** — the correction lives only in the experiment write-ups.
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — `realized_recall` evaluator + `primary_metric` note (`explore/training.py:544-595`). Residue: `engine.py:207-212` manifest still records `cv_acc`; `docs/EXPLORE.md` still headlines it (see §17 OPM-1).

#### 🟥 SDK-A3 `[A]` The "self-improving harness" has no real reward, so it learns from noise
`default_verify` accepts any non-empty id list with score `1.0` (`harness/loop.py:19-21`). Three
consequences with the shipped defaults:
1. `plan_execute_verify` breaks on the first skill (`score >= accept=0.75`), so the bounded
   Plan–Execute–Verify loop is **always single-step** — `max_steps` never binds (`harness/loop.py:29-38`).
2. `post_write_memory` writes "the skill `X` worked" to long-term memory on **every** non-empty run
   (`harness/hooks.py:72-76`), so `recall()` later biases the plan toward whatever ran first.
3. `reflect()` gates forging on `result.score < threshold` (0.5) — never true — so every non-empty
   multi-hop run forges a skill + subagent (`harness/forge.py:298`, and
   `harness/harness.py:137` fabricates `score=1.0 if fused else 0.0` for the subagent path).
The docstrings call this "evidence-backed" online learning; the evidence is "returned ≥1 id".

#### 🟧 SDK-A4 `[A]` `triage` pushes almost every long query into `decompose_arsenal`
`_MULTI_HINT` matches a bare `\band\b` (plus `each`, `both`, `as well as`), and `n_clauses >= 3`
counts comma/and splits (`harness/triage.py:30-32`, `:74`). Any conjunctive-constraint question — the
BrowseComp shape — is classified `multi_hop` → `decompose_arsenal` → `depth="multi"` → the subagent
path. That is exactly the structure `experiments/deep_judge/README.md` §5 shows is *wrong* for
conjunctive corpora (decompose 0.025 vs dense 0.079 recall@10). `agentic_solve` fixed this for the
authored-strategy path; `Harness`/`triage` still hardcodes it.
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — strong/weak multi-hints + entity counting (`harness/triage.py:31-47,:96-108`).

#### 🟧 SDK-A5 `[A]` "Validate-before-keep" is documented as an invariant but never implemented
`explore/engine.py:61-64` defines `Stage.validate()` and the module docstring makes rejecting
non-improving stages "the honesty rule". **No stage in `default_pipeline()` overrides it** — every
stage keeps its output unconditionally, so `status="rejected"` is unreachable via the engine, and
`docs/EXPLORE.md`'s robustness table over-claims.

#### 🟧 SDK-A6 `[A]` Corpus-specific content baked into the standard SDK (violates `soul.md` rule 1)
- `primitives.expand` hardcodes finance examples in its prompt (`'CD'->'certificate of deposit'`)
  and `DEFAULT_ALIASES` is FiQA/UK-English specific (`primitives.py:348-352`, `:416-419`).
- `harness/os_query.py:24-29` hardcodes the field names `text`/`title` **and** a HotpotQA example
  (`"The Cardboard Crown"`) in the system prompt.
- `explore/stages.py:226-231` prompts for questions "From the technical document below" — an
  Altera/SU-era assumption for a general SDK.

#### 🟧 SDK-A7 `[A]` `os_query` never introspects the schema it writes queries against
`author_os_query` asserts a `title` field exists (SDK-A6) instead of calling
`store.describe_schema()`, which the SDK already provides. On BrowseComp (no `title`) the authored
body silently matches nothing, burning the retry budget. The introspect-first pattern is documented
in `docs/INTROSPECTION.md` and unused here.

#### 🟨 SDK-A8 `[A]` `freshness` is unreachable from the sandbox it was designed for
`primitives.freshness` requires caller-supplied `now`/`half_life` because "the sandbox forbids
wall-clock nondeterminism" (`primitives.py:288-292`), but `LocalExecutor` binds no clock and no
`__import__`, so sandboxed code cannot obtain `now`. The primitive is referenced only by
`__init__`, `sandbox`, and one unit test — it is used by no experiment.

### Code

#### 🟥 SDK-C1 `[C]` The `os_query` read-only allowlist is dead code
`harness/os_query.py:38`:
```python
if not k.startswith("_") and k not in _ALLOWED and not isinstance(k, str):
```
JSON object keys are always `str`, so `not isinstance(k, str)` is always `False` and the whole
condition can never fire. Only the `_BANNED` set is actually enforced — the docstring's promise
("only read-only query clauses are allowed") is not met. Fix: drop the `isinstance` clause.
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — allowlist condition corrected; documented at `os_query.py:83`.

#### 🟥 SDK-C2 `[C]` The OpenSearch adapter silently drops `$and` / `$or` / `$not` filters
`adapters/opensearch.py:143-144` skips any key starting with `$` (`# nested and/or omitted in
reference adapter`), while `filters.validate()` **accepts** those operators and its docstring
promises boundary validation so that "server-side adapters that would otherwise silently drop bad
operators fail fast". Net effect: `search(filter={"$or": [...]})` runs **unfiltered** and returns
more results than requested, with no error. Either translate them or raise.
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — `$and`/`$or`/`$not` translated to bool clauses (`adapters/opensearch.py:203-229`); unsupported operators raise.

#### 🟥 SDK-C3 `[C]` `$eq` filters on string metadata match nothing on OpenSearch
`_to_filter` emits `{"term": {field: val}}` (`adapters/opensearch.py:147`). Dynamically-mapped
string metadata is indexed as `text` + a `.keyword` sub-field, so a `term` query on the bare field
does not match. Equality filters therefore fail *closed* (zero hits) on the backend the whole repo
runs on.
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — `_term_field` routes string equality to the `.keyword` sub-field (`adapters/opensearch.py:171`).

#### 🟥 SDK-C4 `[C]` Random `sample()` makes the corpus fingerprint change every run, defeating resume/drift
`corpus_fingerprint` hashes `store.sample(12)` and calls it "deterministic-ish"
(`explore/engine.py:67-81`), but `OpenSearchStore.sample` uses `function_score` +
`random_score` with **no seed** (`adapters/opensearch.py:340-346`). So the fingerprint differs on
every invocation → `fingerprint_changed()` is always True → `run_pipeline` re-runs **every stage
every time**, and resumability only works on `MemoryStore` (whose `sample` is deterministic
first-n — see SDK-C10). Fix: seed the random_score, or fingerprint on `count` + a sorted-id sample.
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — seeded `random_score` sample (`adapters/opensearch.py:49,:67`, the P4-4 port).

#### 🟥 SDK-C5 `[C]` The 4-way failure taxonomy is structurally broken on OpenSearch
`classify_failure` computes semantic similarity from `d.vector` of docs returned by `store.get()`
(`explore/training.py:374-378`), but every network adapter **strips the vector field** from
`_source` before building the `Document` (`adapters/opensearch.py:325-327`, `:162-163`). So
`best_sem` is always `0.0`, and the classifier can only ever emit `low_similarity` (if lexical is
also low) or `synonym_metadata`. `rank_collision` requires `best_lex >= lex_lo`, so the taxonomy
that commit 953daf9 shipped as a headline is mislabelling every OpenSearch query. Fix: re-embed the
gold text (as `duplication_scan` already does) instead of reading `d.vector`.
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — gold text re-embedded instead of reading stripped `d.vector` (`explore/training.py:421-425`).

#### 🟧 SDK-C6 `[C]` `templates.regex()` can never hit on OpenSearch
`explore/templates.py:121-127` passes `re.escape(code)` to `mode="regex"`. `query_regex` runs a
`regexp` query, which OpenSearch **anchors to the whole field value** — the adapter's own docstring
says callers must wrap with `.*` (`adapters/opensearch.py:197-198`). So the regex pool is always
empty, silently weakening `exact_partnum` and `confidence_gated_exact` to their non-regex halves.

#### 🟧 SDK-C7 `[C]` Reranker/embedder lazy loading is not thread-safe → duplicate GPU loads
`CrossEncoderReranker._ensure` and `QwenReranker._ensure` do a plain `if self._model is None`
check-then-load (`rerankers.py:24-31`, `:61-75`). `agentic_solve`/`run_explore_pipeline` run 8
worker threads sharing one reranker, so N threads can enter `_ensure` concurrently and each load a
copy of the model. This is a plausible root cause of the CHANGELOG's standing gotcha ("Qwen
reranker OOMs with >2 workers"). Fix: a `threading.Lock` around `_ensure`.
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — `threading.Lock` around `_ensure` in both rerankers (`rerankers.py:24,:68`).

#### 🟧 SDK-C8 `[C]` `AgentMemory.remember()` rewrites the whole file per call and is not thread-safe
`harness/memory.py:81-84` appends then calls `save()`, which truncates and rewrites every item →
O(n²) I/O over a run, plus an embedding call per write. There is no lock, yet
`run_explore_pipeline.py:111` comments the shared instance as "(thread-safe)" and only guards *some*
call sites with an external `mlock`. Long-term memory also grows unbounded (no cap/dedup).

#### 🟧 SDK-C9 `[C]` Dataset "atomicity" is per-file, so a crash can silently misalign X and y
`explore/training.py:173-175` writes `feat_{bi}.npy`, then `lab_{bi}.jsonl`, then the checkpoint —
three separate atomic writes. A crash between the first two leaves a feature shard with no label
shard; `load_dataset` globs the two patterns **independently** and `np.concatenate`s
(`:195-206`), so features and labels shift relative to each other with no error. The resume path
also ignores a changed `batch_size` even though the checkpoint records it.

#### 🟧 SDK-C10 `[C]` `MemoryStore` swallows every constructor kwarg, so dimension checking is off by default
`adapters/memory.py:26` is `def __init__(self, **_)`. `Session._check_dims` looks for
`store.dim`/`store._dim` (`session.py:149`) and finds neither, so on the **default backend used by
tests, demos and the SU experiments** only intra-batch consistency is checked, never the intended
dimension. This is what forced the `SAC_DIM` env hack in `experiments/qwen8b_sac/issues.md` #2.

#### 🟧 SDK-C11 `[C]` `MemoryStore.query_keyword` recomputes corpus DF on every query
`adapters/memory.py:80-105` tokenizes **every document** and rebuilds the document-frequency table
per call — O(N · tokens) per keyword search, no index, no caching. `explore/multihop.py` issues
`n_docs - 1` keyword searches per candidate chain (`:45-72`), so generating 1,000 4-hop queries is
~3,000 full-corpus scans. Cache `df`/`toks_by_id` and invalidate on `upsert`.

#### 🟨 SDK-C12 `[C]` `TemplateRouter` falls back to a template name that does not exist
`predict()` and `route_plan()` return `"all_rerank"` when `self.model is None`
(`explore/router.py:101`, `:111`). `TEMPLATE_NAMES` contains no such entry, so the value KeyErrors
in `run_template` — the unfitted-router path is untested and broken. (Nearest real name: `deep_all`.)
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — unfitted fallback is the real template `deep_all` (`explore/router.py:45`).

#### 🟨 SDK-C13 `[C]` `consensus()`'s extra signals are lost on the next chained call
`primitives.consensus` attaches `.agreement`, `.votes`, `.n_lists` to the returned `ResultSet`
(`primitives.py:198-200`), but `ResultSet.top/dedup/where` all construct a **new** `ResultSet`
(`types.py:65-85`), so the documented gating signals vanish the moment agent code chains anything.
Return them as a tuple or a dataclass.

#### 🟨 SDK-C14 `[C]` Smaller code issues
- `_TransformersEmbedder.embed` does one forward pass **per text** (`embeddings.py:130-141`) — no
  batching, while `explore/fit._batch_embed` was added specifically to fix this elsewhere.
- `QwenReranker.dev` / `.tok` are only assigned inside `_ensure` (`rerankers.py:61-75`), so any
  attribute access before the first call raises `AttributeError`.
- `_resilience` docstring promises "exponential backoff with jitter"; the implementation is
  deterministic (`_resilience.py:6`, `:56`) — the second docstring contradicts the first.
- `describe_schema()` returns `fields` on OpenSearch but `metadata_keys` on memory
  (`adapters/opensearch.py:373` vs `adapters/memory.py:149`); consumers must know both
  (`session.py:124` already special-cases it).
- `adapters.available()` lists all nine backends regardless of whether their client libraries are
  installed (`adapters/registry.py:87`) — the README presents it as an availability check.
- `Session.hydrate` → `OpenSearchStore.get` issues **one unbatched `mget`** for however many ids it
  is handed (`adapters/opensearch.py:316-319`); `chunked()` is used for upserts only.
- `OpenSearchStore.delete` swallows every exception per id (`:330-335`).
- `ensure_index` no-ops when the index exists, without checking that the existing mapping's
  dimension matches `dim` — a silent mismatch on re-use.
- `normalize_query`'s `"cheque's"` alias is unreachable: the tokenizer splits on `\w+|\W+`
  (`primitives.py:418`, `:426`).
- `EmbeddingError` is exported and tested but never raised by any SDK code path.

### Redundancy

#### 🟥 SDK-R1 `[R]` `explore/fit.py` is a dead second copy of the whole labeling+training pipeline
`fit_router()` (`explore/fit.py:86-181`) duplicates `training.build_dataset` +
`training.train_router_model`: it collects queries, embeds, labels via `label_via_templates`,
trains, writes `router.pkl`/`router_labels.jsonl`/`router_meta.json` and records the `router`
stage. It is **referenced nowhere** — not by `Explorer.fit()` (which calls `dataset()` + `train()`),
not by `explore/__init__.py`, not by any test or experiment. `router.train_router`
(`explore/router.py:147-169`) exists *only* to serve it, and uses **different hyperparameters**
(`max_iter=300, lr=0.08`) than the live path (`400 / 0.07`, `training.py:41-45`). Two divergent
trainers, one of them unreachable. Keep `_collect_queries`/`_batch_embed`, delete the rest.

#### 🟧 SDK-R2 `[R]` Five implementations of reciprocal-rank fusion
`primitives.fuse` (on `ResultSet`, `primitives.py:33`), `harness/loop.fuse_ids` (`:41`),
`harness/playbook._rrf` (`:26`), `harness/forge._safe_globals._fuse_ids` (`:87`), and
`harness/agentic._exec` rebinding `fuse_ids` to `playbook._rrf` (`:74`). All use `k=60`; four
operate on id lists and could be one function.

#### 🟧 SDK-R3 `[R]` Four decompose implementations behind three different prompts
`primitives.decompose` ("minimal set of simpler sub-questions"), `harness/loop.decompose_query`
("distinct factual sub-questions… 4 max" + a lexical regex fallback), and **inline copies** in
`harness/skills._decompose_fielded:95-101` and `harness/skills._decompose_arsenal:145-151`
(byte-identical prompt + parsing, 2-6 subs). `playbook`/`agentic` call `P.decompose` and cap at 6.
So the number of sub-facts a query gets depends on which entry point you came through.

#### 🟧 SDK-R4 `[R]` `_decompose_fielded` and `_decompose_arsenal` are near-identical
`harness/skills.py:85-114` vs `:136-164` — same decomposition block, same fielded+dense pooling;
`_decompose_arsenal` only adds a HyDE pass. One should call the other with a flag.

#### 🟧 SDK-R5 `[R]` Two incompatible sandbox namespaces
`sandbox.LocalExecutor._build_namespace` binds `sac` → **the Session instance** plus ~20 bare
primitives (`sandbox.py:62-90`). `harness/forge._safe_globals` binds `sac` → **the module**, plus
`P`, `fuse`, `fuse_ids`, `rerank`, and a restricted `__import__` (`forge.py:68-104`).
`agentic._exec` then overrides two of those (`agentic.py:72-84`). LLM-authored code that works in
one executor breaks in the other, and only `forge`'s variant restricts imports.

#### 🟨 SDK-R6 `[R]` Duplicated helpers and metrics
- `_llm_profile` exists twice with the same prompt: `session.py:121-141` and `explore/stages.py:149-166`.
- `extract_codes` exists twice with **different regexes**: `harness/triage.py:47` and
  `explore/templates.py:33` (the triage one also false-positives on e.g. `COVID 19`).
- The cross-encoder rerank helper is duplicated verbatim: `forge._safe_globals._rerank`
  (`forge.py:94-101`) and `agentic._rerank_helper` (`agentic.py:62-69`).
- `primitives.result_diversity` vs `Session.diversity` compute the same metric with **different
  redundancy thresholds** (0.9 vs 0.92) and different vector sources (stored vs re-embedded);
  `primitives.max_similarity` vs `Session.answerability` likewise overlap.
- Skills `definition_lookup` and `hybrid_search` are the **same function** `_hybrid`
  (`harness/skills.py:200-201`), so semantic skill lookup can return either at random.

#### 🟨 SDK-R7 `[R]` Dead / unreachable subsystems
- `Harness.child()` / `spawn()` / `max_depth`: `_run_subagents` uses `arsenal_single` whenever the
  registry has it — which is always, since it is a builtin — so the recursive subagent path never
  executes (`harness/harness.py:117-135`).
- `RouterStage` is a `NotImplementedError` stub that records the `router` stage as `planned`, while
  `Explorer.train()` writes the *same* stage key as `ok`/`rejected` (`explore/stages.py:252-258` vs
  `explore/engine.py:200`). `CodegenStage` requires `router`, so it is permanently `skipped`.
- `primitives.quality_filter` and `primitives.freshness` are used by no experiment (see SDK-A8).
- `consensus`, `content_type`, `max_similarity`, `result_diversity`, `score_cliff` are **not
  exported** from `search_as_code/__init__.py` although they are injected into the sandbox and
  advertised in the prompt surface and docs.

---

## 2. `tests/`, `examples/`, packaging, CI, and repo governance

#### 🟥 GOV-1 `[A]` Files marked "INTERNAL, do not push" are already on the public `main`
`.gitignore` was just extended with:
```
experiments/su_multihop/
# BrowseComp-Plus benchmark — INTERNAL, do not push
experiments/browsecomp/
```
but **`.gitignore` does not untrack already-committed files**. `origin/main` (the public
GitHub repo per `soul.md` rule 1) currently contains **19 `experiments/browsecomp/` files**
(`RESULTS.md`, `bc_perquery*.jsonl`, `bc_recall*.json`, figures, all the runners) and **6
`experiments/su_multihop/` files including `data/su_multihop_{2,3,4}docs.jsonl`** — SearchUnify-derived
query/answer data. Verify with:
```bash
git ls-tree -r --name-only origin/main | grep -E '^experiments/(browsecomp|su_multihop)'
```
Fix requires `git rm --cached` on those paths + a push (and a decision about history).

#### 🟧 GOV-2 `[C]` A customer SSH key sits in the repo root and is **not** gitignored
`production-aa032620s.searchunify.com-gaganjot.singh@grazitti.com.pub` — a filename carrying a
customer hostname and a colleague's email — is untracked but matches **no** `.gitignore` rule, so a
single `git add -A` stages it. (`SearchUnify_Evaluation_*.csv` in the same directory *is* covered by
`*.csv` / `SearchUnify_Evaluation*`.) Add an explicit ignore and move the key out of the repo.

#### 🟧 GOV-3 `[C]` 51 tracked `phase4/altera*` files are gitignored-but-tracked
`.gitignore` lists `phase4/altera_*.py`, `phase4/ALTERA_RESULTS.md`, `phase4/models/` — yet
`git ls-files phase4` returns 51 files including `altera.py`, `altera_agent.py`,
`ALTERA_RESULTS.md`. They are **not** on `origin/main` today (the selective-push discipline held),
but they are tracked on this branch, so any merge of `feat/deep-sac` → `main` publishes customer
work. The gitignore entries give a false sense of protection.

#### 🟥 TEST-1 `[A]` The newest, most-promoted code paths have zero tests
No test exercises `harness.agentic_solve` (the README's "recommended entry point"),
`DiagnosticJudge.judge`, `os_query.author_os_query` / `_validate`, `rag_techniques.SkillLookup`,
`forge.author_code_primitive`, `CodePrimitive.to_skill`, or `explore.write_dataset_csv` /
`analyze_failures` / `duplication_scan`. SDK-C1 (the dead allowlist branch) is exactly the class of
bug one unit test would have caught. `tests/test_agent_harness.py` covers the *older* `Harness`
path; `agentic.py` is not even committed yet.

#### 🟧 TEST-2 `[C]` The only playbook test cannot run in CI and asserts a hardcoded LLM-dependent number
`tests/test_diagnostic_playbook.py` needs a live OpenSearch `hotpotqa` index, an `OPENAI_API_KEY`,
`phase1` (excluded from the wheel), and **two gitignored artifact paths**
(`experiments/deep_judge/forge_store_hotpot`, `experiments/multi_hop_synth_queries/data/*.jsonl`,
`:19-20`). It then asserts `raw_m >= 0.55` on an LLM-driven pipeline (`:91`) — non-deterministic and
guaranteed-skip in CI. `_forged_primitives()` returns `[]` when the store is absent (`:62-63`), so
the SAC half of the test silently disappears rather than failing loudly.

#### 🟧 TEST-3 `[A]` The "adapter contract" is claimed but never enforced across adapters
`README.md` says "the in-memory test suite is the contract every adapter must satisfy" and
`adapters/base.py` says the primitive layer emulates whatever an adapter reports `False`. There is
**no parametrized conformance suite**: `faiss_store`, `sqlite_store`, `nmslib_store`,
`milvus_store` have **zero tests** (and `qdrant`/`chroma`/`pgvector` are `omit`ted from coverage in
`pyproject.toml:70-71`). `test_units.py:test_emulates_missing_modes_on_dense_only_backend` uses a
hand-rolled fake instead. Bugs like SDK-C2/C3 (OpenSearch filter translation) are invisible because
`test_opensearch.py:77` only checks simple equality on a numeric field.
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — `tests/test_conformance.py` parametrized over installed backends, in CI. Coverage gap for qdrant/pgvector/nmslib/milvus tracked in §17 TEST-5.

#### 🟧 TEST-4 `[C]` The suite structurally cannot see the resume/drift bug
`tests/test_explore.py:272 test_fingerprint_detects_drift` runs on `MemoryStore`, whose `sample()`
is deterministic first-n. On the OpenSearch adapter the same fingerprint is random per call
(SDK-C4), so the property the test asserts is false in production and green in CI.

#### 🟧 PKG-1 `[C]` Declared dependencies don't cover the shipped code paths
- `explore/training.py` imports **sklearn** (`:42-59`) and optionally **xgboost** (`:63`), but
  neither appears in `pyproject.toml` — not in `dependencies`, not as an extra. `Explorer.fit()` on
  a base install dies with a bare `ModuleNotFoundError`, not the SDK's own
  `MissingDependencyError`. Same for `torch`/`sentence-transformers`, which
  `rerankers.py`/`embeddings.py` lazy-import with no extra other than `[phase1]`.
- `all = [qdrant, chroma, pgvector, psycopg, openai]` **omits `opensearch-py`**
  (`pyproject.toml:52-58`) even though the README documents `[all]` as "every backend + providers"
  and OpenSearch is the backend every experiment uses.
- No extras exist for `faiss`, `sqlite`, `nmslib`, `milvus`, yet `adapters.available()` and the
  README badge advertise them as backends.
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — `learn`/`learn-xgb` extras added; `opensearch-py` in `all`; `transformers>=4.40,<5` bound.

#### 🟧 CI-1 `[A]` CI lints and type-checks only `search_as_code/`
`.github/workflows/ci.yml` runs `ruff check search_as_code` and `mypy search_as_code`. Everything
else — `phase1/` (the harness every experiment imports), all of `experiments/`, `phase2-4`,
`chatbot/`, `benchmarks/` (~19k of the repo's ~25k Python lines) — is **never linted or
type-checked**, and `[tool.mypy] exclude` explicitly drops `phase1/`, `examples/`, `tests/`.

#### 🟨 CI-2 `[C]` `pytest --cov` reports coverage against a source set that includes untestable adapters
`[tool.coverage.run] omit` skips qdrant/chroma/pgvector but keeps `faiss_store`, `sqlite_store`,
`nmslib_store`, `milvus_store` — four untested modules — in the denominator, so the coverage number
is not comparable across commits that add adapters.

#### 🟨 EX-1 `[R]` `examples/` covers only the oldest API
Two examples: `demo.py` (memory + `LocalExecutor`) and `opensearch_quickstart.py`. Nothing
demonstrates `agentic_solve`, `explore`, the forge, or the diagnostic judge — the things the README
now leads with. The one worked example of the current flow
(`experiments/deep_judge/example_connect_explore_deploy.py`) is untracked and lives under
`experiments/`.

#### 🟨 EX-2 `[C]` `*.csv` is globally gitignored
`.gitignore:27` ignores **every** `.csv` in the repo. `explore.write_dataset_csv` writes
`labels.csv` / `template_recall.csv` as documented, reusable artifacts, and any small fixture CSV a
test might want would be silently untracked.

---

## 3. `docs/` and root markdown

#### 🟥 DOC-1 `[A]` The documented LLM-facing surface is not in the shipped package
`docs/SELECTION.md:3` states that "the LLM never sees Python signatures alone — it gets
`phase1/sac_surface.py::SAC_SYSTEM`", and `docs/CACHING.md` builds its whole prompt-cache argument on
that prefix. But `pyproject.toml:59-60` ships only `search_as_code*`, so **`sac_surface.py` is not in
the wheel**. A `pip install search-as-code` user gets the primitives with no prompt surface, no
decision rules, and no chaining recipes — the part the docs call the mechanism. `SAC_SYSTEM` is
7,667 chars of exactly the guidance the SDK needs to be usable as documented and it lives in the
internal harness (`grep -rn SAC_SYSTEM` → `phase1/` only).
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — the surface moved into the package as `search_as_code/surface.py`; 2026-08-18: also exported as `sac.SAC_SYSTEM` and verified present in the wheel.

#### 🟧 DOC-2 `[C]` `docs/CACHING.md` documents an affordance that does not exist
It claims progressive disclosure works via "a `help(primitive)` / `list_primitives` affordance the
generated code can call to pull a fuller signature on demand" (`docs/CACHING.md`, §3). Neither
`help` nor `list_primitives` is bound in `sandbox.LocalExecutor._build_namespace`
(`sandbox.py:62-90`) — and `help` is not in the builtins allowlist either — nor do they exist in the
package. Generated code calling them raises `NameError`.

#### 🟧 DOC-3 `[A]` `docs/INTROSPECTION.md` advertises the behaviour that breaks resumability
"`sample()` draws a *real random* slice from the store (OpenSearch uses `random_score`, not
first-n)" is presented as a feature. It is the direct cause of SDK-C4: `corpus_fingerprint` hashes
that sample, so the explore pipeline sees corpus drift on every run. The doc also records the
`fields` vs `metadata_keys` split (SDK-C14) as intended cross-backend behaviour rather than a
contract inconsistency.

#### 🟧 DOC-4 `[C]` `docs/STATUS.md` is stale enough to mislead
It reports "Database adapters (5) 80%", "65 tests passing", "Learned components 0% — designed",
"Phase 2: constraint probe 20%", and a live "Services right now" section. Reality: 9 registered
backends, **106 unit tests** (`pytest -q --ignore=tests/test_opensearch.py
--ignore=tests/test_diagnostic_playbook.py` → 106 passed), the learned router is implemented
(`explore/training.py`), and phases 2–4 have completed reports. A status file that tracks
session-scoped service state cannot stay true; either generate it or delete it.

#### 🟧 DOC-5 `[C]` Test-count claims disagree with the suite in three places
`README.md:10` badge says **95 passing**; `README.md:65` says "77 in-memory unit tests; +18
OpenSearch integration"; `docs/STATUS.md` says 65. The suite collects **106 unit + 18 OpenSearch +
1 skipped integration**. Pick one source and generate the badge.

#### 🟧 DOC-6 `[C]` Public README links to a file that is not published
`README.md:128` and `:185` link to `benchmark_changelog.md`, which is **not on `origin/main`**
(`git ls-tree -r --name-only origin/main | grep -c benchmark_changelog` → 0). It is the only broken
relative link across the 46 public `.md` files, and it is cited as the evidence for the headline
benchmark. `open_problems.md` (public) similarly links to `experiments/browsecomp/RESULTS.md`, which
is *currently* published but is the file GOV-1 says must be removed — removing it breaks three links.

#### 🟨 DOC-7 `[A]` "320-primitive taxonomy" reads as capability, not roadmap
`docs/PRIMITIVES.md` legitimately declares itself a catalog of implemented **+ planned** primitives
("Total catalog entries: **320**", `:14`), but `README.md:143` sells it as "the full 320-primitive
taxonomy ... for exactly what each backend supports natively vs. by emulation". The doc contains
~310 distinct backticked primitive names; the package exports 75 symbols total (~30 of them
primitives). Nothing marks the ~280 unimplemented entries as unavailable at a glance.

#### 🟨 DOC-8 `[C]` `soul.md` calls itself always-loaded but nothing loaded it
`soul.md:7` — "This is our `CLAUDE.md`-style always-loaded file" — while no `CLAUDE.md` and no
`.claude/settings.json` existed in the repo until this audit. Every rule in it (capture-as-you-go,
standard-vs-internal, cite sources) depended on an agent choosing to read it.
**FIXED** 2026-08-13: added `CLAUDE.md` + a `SessionStart` hook that states the issue-logging
routine and points at `soul.md`/`STRUCTURE.md`.

#### 🟨 DOC-9 `[C]` Smaller staleness
- `docs/CONCEPT.md` ends with "(A BrowseComp-Plus-style eval harness is on the roadmap.)" — it
  exists (`experiments/browsecomp/`, 830-query runs).
- `docs/DATABASES.md` claims the SDK "exposes hybrid, keyword, regex, rerank, MMR, dedup,
  freshness, and compression on **every** backend by emulating them client-side" — true for those,
  but `query_fielded`/`query_phrase` are **not** emulated (the matrix marks memory 🕒) even though
  `harness/skills.py` and `explore/templates.py` depend on them and `hasattr`-guard instead.
- `docs/EXPLORE.md`'s robustness table lists validate-before-keep as active (see SDK-A5) and
  documents `fit()`'s labeling knobs without noting that the defaults collapse the template space
  (SDK-A1).

---

## 4. `phase1/` — the benchmark harness behind the headline numbers

#### 🟥 P1-1 `[A]` The FiQA benchmark's tool-calling arm gets 3 tools; the SAC arm gets ~20
`phase1/sac_surface.py`: `TOOLCALL_TOOLS` is **`['expand', 'search', 'finish']`** with a **511-char**
system prompt, while the SAC arm receives a **7,667-char** `SAC_SYSTEM` and a sandbox namespace
holding `hyde_search`, `prf_search`, `rerank`, `fuse`, `mmr`, `consensus`, `semantic_dedup`,
`adaptive_search`, `score_cutoff`, `diversity_quota`, `compress`, `answerability` … Verify:
```bash
python -c "from phase1 import sac_surface as s; print([t['function']['name'] for t in s.TOOLCALL_TOOLS]); print({k: len(getattr(s,k)) for k in dir(s) if k.isupper() and isinstance(getattr(s,k),str)})"
```
The tool arm **cannot rerank, decompose, fuse, diversify, or HyDE at all.** So the README's headline
table ("tool-calling (MCP) 0.440 vs Search as Code 0.549 … SAC wins every axis that matters vs MCP
tool-calling", `README.md:111-129`) compares a 3-tool agent against a 20-primitive agent with 15×
the prompt guidance. Whatever code-mode's real advantage is, this run does not isolate it.

#### 🟥 P1-2 `[A]` `run_sac` fuses every hop; `run_tool_calling` returns only its last answer
`phase1/agents.py:218-221` — with `monotone=True` (the default) the SAC arm's final ids are
`_rrf_ids(pooled)`, an RRF union of **all** hops. The tool arm returns the ids from its final
`finish` call (`:261`, `:290`) with no cross-hop union. The SAC arm therefore gets a structural
recall advantage on the same budget, and `phase1/agents.py:137-142` documents this as a deliberate
fix for deep-SAC losing to one-shot — it was never mirrored into the baseline.

#### 🟧 P1-3 `[A]` Hop-2 switches the SAC system prompt, which breaks the cache claim it is credited for
`_sys_for_hop` returns `SAC_SYSTEM` on hop 0 and `SAC_DEEP_SYSTEM` on hop ≥ 1
(`phase1/agents.py:151-157`). `docs/CACHING.md` attributes SAC's "2× the prompt-cache hit (54% vs
27%)" to a byte-stable prefix; a second, different 2,259-char prefix cannot hit the cache warmed by
the first. The measured advantage is partly "SAC makes fewer LLM calls", which is a real but
different claim.

#### 🟧 P1-4 `[C]` Session state leaks across queries in the benchmark
`run_sac` clears exactly three keys — `agreement`, `lists`, `answerable` (`phase1/agents.py:166`) —
but the `Session` is reused for the whole 100-query run and generated code is encouraged to
`sac.remember(...)` arbitrary keys. Anything else a program stashes stays visible to later queries
via `sac.recall(...)`. Either use a fresh Session per query or clear `session._state` wholesale.

#### 🟧 P1-5 `[C]` The judge can stop the loop on a confident FAIL
`phase1/agents.py:215`: `if (accept or conf >= 0.75 or hop == max_retries) and not low_agree: break`.
`JUDGE_SYSTEM` defines `CONFIDENCE` as "how well the top results answer the query", so a FAIL with
conf ≥ 0.75 is *mostly* unreachable given its own `PASS whenever CONFIDENCE >= 0.5` rule — but the
condition is still wrong in intent: it treats the judge's quality score as a stop signal
independent of its verdict.

#### 🟨 P1-6 `[A]` The judge is explicitly biased toward PASS
`JUDGE_SYSTEM`: "Refinement is EXPENSIVE and frequently makes results WORSE, so bias toward PASS…
PASS whenever RELEVANT >= 2, or CONFIDENCE >= 0.5… When unsure, PASS." This is defensible (it
encodes open-problem #5) and honestly documented, but it means hop counts, latency and cost for
**both** LLM arms are set by a deliberately lenient gate, and judge-accuracy figures measured with
this prompt are not independent of it.

#### 🟥 P1-7 `[A]` The "fair" multi-hop study gives the two arms materially different prompts
`experiments/multi_hop_synth_queries/eval_fair.py` is a genuinely shared toolset (one `Tools`
instance, same 8 tools, same budget) — a real improvement over P1-1. But the prompts are not
matched:
- `CODE_SYS` (`:355-372`) hands the code arm **the winning strategy as a worked program**
  (`subs = decompose(question)` → search each → `fuse(pools)[:10]`) *and* the key insight from
  §12 of the results: "`rerank(question, ...)` scores whole-question relevance, so it can DROP docs
  that satisfy only ONE sub-fact — use it to sharpen a SINGLE sub-pool, not over the fused union."
- `TOOL_SYS` (`:176-181`) gets five lines of generic advice ending "…`rerank` to pull the best
  candidates forward" — i.e. it is nudged toward precisely the operation the other arm is warned
  costs multi-gold recall.

The file's own docstring says "Only the harness differs". It does not: the guidance differs, in the
direction that favours the arm under test. Fix: give both arms the same strategy text and the same
rerank caveat.

#### 🟧 P1-8 `[C]` `avg_model_turns` for the code arm is hardcoded, and its extra LLM calls are uncounted
`code_harness` returns `{"steps": 1}` literally (`eval_fair.py:261`). But `decompose()` and
`rephrase()` are `Tools` methods that call `gen.complete(...)` (`:96-104`) *from inside* the
generated program, so a program that decomposes and rephrases makes **three** LLM round-trips while
reporting one turn. Tokens are captured (`sgen.usage`) but the headline "1 model turn vs ~5-7" in
`RESULTS.md` is a definition, not a measurement.

#### 🟧 P1-9 `[C]` Invalid output scores better than partially-correct output
`Tools.final` (`eval_fair.py:141-143`): `return (cand or _rrf(self.pool))[:K]` where `cand` keeps
only ids seen in retrieval. An arm that returns 3 valid ids is scored on 3; an arm that returns
hallucinated ids (or crashes — `code_harness` swallows every exception at `:256-257`) falls back to
the RRF of the **entire 50-doc pool** and is scored on 10. The fallback can outscore a genuine
partial answer, and it fires more often for the arm that only ever sees 8 ids per observation.

#### 🟨 P1-10 `[A]` The dense baseline is not given the wide-pool-then-rerank treatment
The `dense` arm is one `search(top_k=10)` (`eval_fair.py:299`), while tool/sac pool 50 candidates
and may cross-encoder rerank them for free. A `dense→rerank` control (same 50-pool, no LLM) is the
obvious missing arm; without it the study shows "harness beats one dense search", not "harness beats
a tuned single-pass pipeline". §12's rerank ablation covers part of this but not as a baseline arm.

#### 🟥 P1-12 `[C]` `LLM.as_generator()` line-splits the completion — see §5, this is the trigger
`phase1/llm.py:72-78` adapts the LLM to the SDK's `generate(prompt) -> list[str]` contract by
splitting the completion on newlines and stripping leading list markers/digits. That is correct for
`expand`/`decompose`/`topics`, which want a list — but six SDK consumers take `out[0]` and then
re-split it, so they see **only the first line** of the model's answer. Full analysis in §5.

#### 🟨 P1-13 `[C]` "input tokens" means different things in the two benchmark harnesses
`phase1/llm.py:21-25` stores `input_tokens = prompt_toks - cached` (uncached only, with
`cached_input_tokens` separate), while `eval_fair.py:187-188` sums LangChain's
`usage_metadata["input_tokens"]`, which is the **total** including cached. The README's FiQA token
column and the multi-hop token column are therefore not comparable.

#### 🟨 P1-11 `[R]` A fifth RRF copy, and a duplicated line
`phase1/agents.py:123 _rrf_ids` and `eval_fair.py:36 _rrf` are two more copies of SDK-R2's fusion
(now seven in total). `eval_fair.py:177-179` also executes `(rc, al), meta = m[a]` twice inside the
aggregation loop — harmless, but it signals the block was pasted.

---

## 5. Systemic — the `generate(prompt) -> list[str]` contract is consumed wrongly in six places

This is one root cause with repo-wide measurement consequences, so it gets its own section.

#### 🟥 GEN-1 `[C]` Six consumers take `out[0]` of a line-list generator and then re-split it
The SDK's generator contract is `generate(prompt) -> list[str]`, and the only adapter used in this
repo — `phase1/llm.py:72-78 LLM.as_generator()` — implements it by **splitting the completion into
lines**. `primitives.decompose` consumes that correctly (iterates the list). These six do not: they
take `out[0]` (the *first line*) and then call `.splitlines()` on it, which can only ever yield one
element:

| site | intent | actual result with a line-splitting generator |
|---|---|---|
| `harness/skills.py:97-98` (`_decompose_fielded`) | 2–6 sub-questions | **1 sub-question** |
| `harness/skills.py:147-148` (`_decompose_arsenal`) | 2–6 sub-questions | **1 sub-question** |
| `harness/loop.py:62-65` (`decompose_query`) | 2–4 sub-questions | `len(subs) >= 2` fails → **silently falls back to the lexical regex split**, LLM output discarded |
| `explore/fit.py:25-29` (`_rephrase`) | `n` paraphrases | **1 paraphrase**, regardless of `n` |
| `explore/multihop.py:81-84` (`_gen`) | parse `{...}` JSON | regex runs on line 1 only → **pretty-printed JSON never matches → chain silently skipped** |
| `session.py:138-139` / `explore/stages.py:163-164` (`_llm_profile`) | "4-6 short lines" | **1 of 4–6 lines** |

Reproduced (no API needed — emulating `as_generator()` with a 3-line completion):
```python
def gen(prompt):                     # exactly what LLM.as_generator() does
    text = "What year was Film A released?\nWho directed Film B?\nWhich studio produced Film C?"
    return [ln.strip("-*0123456789. \t") for ln in text.splitlines() if ln.strip()]

P.decompose("q", gen)                          # -> 3 subs        ✅ correct consumer
harness.loop.decompose_query(q, gen)           # -> ['Which two films share a director',
                                               #     'what year did each release']  ← lexical fallback
skills._decompose_arsenal parsing              # -> ['What year was Film A released?']  ← 1 sub
explore.fit._rephrase(session, "q", n=3)       # -> ['What year was Film A released?']  ← 1 paraphrase
```

**Why it matters beyond tidiness:**
- `decompose_arsenal` is the skill `triage` recommends for **every** `multi_hop` query
  (`harness/triage.py:75`) and is documented as "MULTI-HOP (best), validated"
  (`harness/skills.py:207`, `docs/HARNESS.md`). With the line-splitting adapter it retrieves for
  **one** sub-fact plus the whole query — so the "validated multi-hop recipe" has been running at a
  fraction of its intended fan-out.
- `Harness`'s subagent path never uses LLM decomposition at all; it uses the `\band\b|,|;|vs`
  regex. `docs/HARNESS.md` presents subagents as "one child harness per decomposed sub-question".
- The router dataset's `rephrases=2` knob is a no-op beyond the first paraphrase, which shrinks and
  de-diversifies the training set behind `open_problems.md` #1/#8.
- `Session.describe(llm=True)` — the flagship introspection feature in `docs/INTROSPECTION.md` —
  returns one line of a 4–6 line profile.
- 14 experiment scripts construct `Session(..., generator=gen.as_generator())`
  (`grep -rn "as_generator()"`), so every one of them inherits whichever of these paths it touches.

**Fix:** make the consumers accept both shapes — `txt = "\n".join(out) if isinstance(out, list) else str(out)`
(which `explore/stages.py:235` already does correctly for `_gen_queries`) — or give `Session` a
`complete()`-style single-string generator slot alongside the list one.
**FIXED** (pre-2026-08-18, annotated 2026-08-18) — shared `_genutil.py` helper; consumers accept both generator shapes (see `session.py:141`).

#### 🟧 GEN-2 `[C]` The same promotion introduced the bug the experiment version doesn't have
`experiments/multi_hop_synth_queries/generate.py:89-96` calls `llm.complete()` (raw string) and
regexes the **whole** response — correct. The standard port
`search_as_code/explore/multihop.py:80-84`, promoted to the SDK as
`sac.explore.generate_multihop`, changed that to `out[0]`. `tests/test_explore.py:241
test_generate_multihop` passes because its stub generator returns single-line JSON, so the
regression is invisible to CI. The published datasets (`data/multihop_*docs_queries.jsonl`) came
from the experiment version and are unaffected — but anyone using the *documented SDK* entry point
silently loses every chain whose JSON is pretty-printed.

#### 🟧 GEN-3 `[C]` `hyde_search` / `answerability` embed only the first line of the hypothetical answer
`session.py:350` and `:381` both do `doc = (gen(prompt) or [query])[0]`. A single-paragraph
completion survives (no newlines), but a preamble line ("Here's a passage:"), a title line, or a
multi-paragraph answer means the embedded "hypothetical document" is that first fragment instead of
the passage. `as_generator()`'s `strip("-*0123456789. \t")` also mangles any passage that starts
with a year or figure. HyDE is used by `sf_arsenal`, `skills._hyde`/`_arsenal_single`/
`_decompose_arsenal`, `templates.hyde`, `Tools.hyde`, and the authored primitives — so every HyDE
number in the repo was measured through this path.

---

## 6. `phase2/`, `phase3/`, `phase4/`, `chatbot/`, `benchmarks/`, remaining `experiments/`

#### 🟧 LEG-1 `[R]` Four independent SAC-vs-tool-calling comparison harnesses
`phase1/agents.py` (+`sac_surface.TOOLCALL_TOOLS`), `experiments/multi_hop_synth_queries/eval_fair.py`
(`TOOL_SCHEMAS`/`tool_harness`), `chatbot/toolcalling.py`, and `phase2/hotpot_eval.py` /
`phase2/beir_run.py` each define their own tool schema, their own agent loop, and their own metric
plumbing. They disagree on toolset (3 vs 8 tools — P1-1), on prompt content (P1-7), on what
"model turn" counts (P1-8) and on what "input tokens" means (P1-13). Any cross-experiment comparison
of these numbers is invalid, and a fix to one arm does not propagate. This is the single largest
duplication in the repo; one shared `Arm` abstraction (tool schema + budget + accounting) would
subsume all four.

#### 🟧 LEG-2 `[C]` `learnings_standard.md` is mandated by the constitution and does not exist
`soul.md` names it in rule 2, in the docs table (`:31`), and in the "after a finding" workflow
(`:53`); `STRUCTURE.md:41` lists it as a deliverable; `research.md:16` points at it. The file is
**absent from the repo**. So the documented mechanism for promoting learnings out of custom work into
the SDK has never had a destination — which is consistent with the SDK-side gaps in §1 (fixes landed
in experiment code and were never generalized). Either create it or repoint the workflow at
`CHANGELOG.md`/`issues.md`.

#### 🟧 LEG-3 `[A]` `phase4/` is 51 tracked files of customer-specific work inside the shippable repo
`phase4/altera_*.py` (31 py files, ~19 with swallow-all `except` blocks) plus
`phase4/ALTERA_RESULTS.md` and `phase4/models/gte-alt-v1/` (a 1,418-line vendored `modeling.py`).
`.gitignore` lists these paths but they were committed before the rule (GOV-3).
`STRUCTURE.md:51` calls `phase2/3/4` "earlier eval phases — **not** imported by experiments", i.e.
they are dead weight in the tree but still carry the governance risk. See P4-1/P4-2 in §9 for the
specifics (what is actually hardcoded, and the accurate version of the "fork vs SDK" question —
**correction:** only one phase4 file defines its own fusion helper, so the fork claim as first
written here was too strong).

#### 🟧 LEG-4 `[C]` `chatbot/` is undocumented in the repo map and untested
9 tracked files, absent from `STRUCTURE.md`'s map (verified by diffing the directory list against the
doc). It contains a fourth tool-calling implementation (LEG-1) and an "arena" UI, and has no tests.
Only 3 of its 8 modules import the SDK directly (the rest go through `phase1`).
**Correction (deep pass):** an earlier version of this entry said the directory "ships committed
logs" — it does not. `git ls-files chatbot` returns only the 9 `.md`/`.py` files;
`arena_8502.log` / `arena_screen.log` exist on disk but are correctly ignored by `chatbot/*.log`.
See CB-1 in §12 for what the deep pass did find here.

#### 🟧 LEG-5 `[C]` Silent-failure density is highest exactly where correctness matters
Bare `except Exception:` / `except:` blocks that swallow and continue: **`search_as_code/` 49**,
`experiments/` 32, `phase4/` 19, `phase2/` 5. In the SDK these are concentrated in
`explore/templates.py:_memo` (a failing template is recorded as "did not solve"),
`harness/skills.py` (every skill silently degrades to `_dense`), and `harness/playbook.apply_technique`
(a failing technique silently returns a plain hybrid search). The consequence is that **measurement
cannot distinguish "this strategy lost" from "this strategy crashed"** — which is how SDK-C5, C6 and
GEN-1 all stayed invisible. Recommendation: keep the fallbacks, but count them (a
`degraded: {reason: n}` field on every result) and fail the run if the rate is non-trivial.

#### 🟨 LEG-6 `[R]` Two more RRF copies, bringing the total to eight
Adding to SDK-R2/P1-11: `experiments/deep_judge/run_playbook.py`,
`experiments/explore_forge/run_transparent.py`, `experiments/multi_hop_synth_queries/eval_recall.py`.
Eight files now implement `1.0 / (k + rank + 1)`.

#### 🟨 LEG-7 `[C]` Stale build artifacts and an undocumented `dist/`
`dist/search_as_code-0.0.1-py3-none-any.whl` + `.tar.gz` from 2026-07-28 sit in the tree (gitignored,
so harmless to the remote) while `pyproject.toml` still says `version = "0.0.1"` and `CHANGELOG.md`
records many changes since. Any `pip install dist/*.whl` gets July code under the current version
number. `dist/` and `search_as_code.egg-info/` are also absent from `STRUCTURE.md`.

#### 🟨 LEG-8 `[C]` `experiments/explore_improvement/` contains only a `RESEARCH.md`
No runner, no results — an experiment directory that is really a research note. Either fold it into
`research.md` or add the study it describes. (`experiments/su_multihop/` is the mirror case: it is
gitignored locally yet published on `origin/main` — see GOV-1.)

---

## 7. Fix order (recommended)

Ranked by "wrong conclusions or wrong results per hour of work", not by severity alone.

| # | issue | why first | est. |
|---|---|---|---|
| 1 | **GEN-1** | one root cause, six call sites, silently degrades the *documented best* multi-hop recipe; a 1-line fix per site | ~1h |
| 2 | **SDK-C1** | the "validated read-only" DSL gate does not run at all; 1-line fix + 1 test | ~15m |
| 3 | **SDK-C4** | every explore run re-does every stage; seed the sample or change the fingerprint | ~30m |
| 4 | **SDK-C5** | the 4-way failure taxonomy mislabels 100% of OpenSearch queries; re-embed gold text | ~30m |
| 5 | **P1-1 / P1-7** | the two headline comparisons are not apples-to-apples; either match the arms or restate the claims in `README.md` / `RESULTS.md` | ~3h |
| 6 | **SDK-C2 / C3** | filters fail open (`$or` ignored) and fail closed (`$eq` on strings) on the primary backend | ~1h |
| 7 | **GOV-1** | "INTERNAL, do not push" data is on the public remote right now | ~30m + a decision |
| 8 | **SDK-C7 / C8** | thread-unsafe model loading and O(n²) memory writes under the 8-worker pipelines | ~1h |
| 9 | **SDK-A1** | make the labeling default honest (or refuse to label unavailable templates) | ~1h |
| 10 | **LEG-5** | count the silent fallbacks, so the next bug of this class is visible instead of inferred | ~2h |

**The cross-cutting theme.** Nine of the seventeen blockers share one shape: *a documented property
that no test covers and whose failure mode is silent.* The allowlist that never rejects, the filter
clause that is dropped, the template that degrades to `light_dense`, the decomposition that returns
one sub-fact, the fingerprint that always differs, the `validate()` gate no stage implements. The
codebase is unusually well documented — which is why the gaps between doc and behaviour are the
dominant defect class rather than crashes. Two structural changes would prevent most recurrences:
**(a)** make degradation loud (count fallbacks, return `degraded` reasons — LEG-5), and **(b)** test
the documented invariant, not the happy path (an adapter conformance suite — TEST-3 — plus one test
per documented property in §1).

---

## 8. `phase2/` — BEIR campaign, learning pipeline, routers (deep pass, 30 files)

#### 🟥 P2-1 `[A]` The learned-profile result is measured on the queries the rules were mined from
`learn_rules.py:mine()` iterates the **test** qrels (`qids = [x for x in qr if …][:n]`, n=120), finds
gold docs dense missed, and asks the LLM for an alias/glossary/synonym rule **derived from that gold
document's text**. `impact_eval.py:main()` then evaluates the resulting profile over
`[x for x in qr if …][:n]` with n=150. Both iterate the same dict in insertion order, so **the first
120 of the 150 evaluation queries are exactly the mining set** — 80 % contamination. `run_learn_sweep.sh`
runs precisely this pair (`--n 150` mine → `--n 150` impact) for four datasets. The CHANGELOG's
"+2.7 pts all_found from learned synonyms" (HotpotQA) and the learning-pipeline numbers in
`MULTI_DATASET_REPORT.md` inherit the leak. `align_prompts.calibrate_judge` has the same shape (it
tunes the judge threshold on `qr[:n]`, which is then used at eval time). Fix: mine on a train split,
evaluate on a disjoint one, and report both.

#### 🟥 P2-2 `[A]` `MULTI_DATASET_REPORT.md` states the two arms have the same primitives
Its methodology section reads: "**SAC** — gpt-4.1-mini writes Python over primitives
(search/fan-out/fuse/rerank/decompose/expand/rephrase/mmr/…) … **tool-calling (MCP-style)** — same
LLM, same budget, but each primitive is a discrete tool call". The tool arm is
`sac_surface.TOOLCALL_TOOLS` = `['expand', 'search', 'finish']` (P1-1) — it has no `fuse`, `rerank`,
`decompose`, `mmr` or fan-out tool at all. This is the same defect as P1-1 but stated more strongly,
in a report that reads as the paper-ready summary. Every "SAC vs tool" row in that report
(`beir_run.py:main`, `hotpot_eval.py:main`, both with `max_retries=1`) also carries the P1-2 hop-fusion
asymmetry.

#### 🟧 P2-3 `[C]` The documented zip-corruption gotcha was fixed in a shell script, not in the loader
`CHANGELOG.md` lists it twice as a standing trap ("Validate downloaded zips (`testzip()`); the UKP
mirror truncates on parallel/timeout downloads"). `run_stage2.sh` does implement the check
(`python -c "import zipfile;zipfile.ZipFile(...).testzip()"` in a wait loop) — but the library path
`beir.py:ensure()` still does `requests.get(...)` → `zipfile.ZipFile(zp).extractall(DATA)` with **no
`testzip()` and no size check**, so any other caller re-hits the same trap. The fix belongs in
`ensure()`.

#### 🟧 P2-4 `[A]` `synth_eval.py`'s "SAC 1.00 vs dense 0.42" compares unequal access to the answer
This probe is cited in `docs/STATUS.md` as "the real win". Two asymmetries:
1. `SAC_CONSTRAINT_SYSTEM` embeds **the full worked solution** in the prompt — retrieve each feature,
   read `hit.get("version")`, `max(...)` by version tuple — i.e. the arm is told the algorithm.
2. The SAC arm reads the answer from **`metadata["version"]`**, while `dense_answer()` must regex
   `r"version (\d+\.\d+)"` out of the passage text and returns the *first* match from the top-3.
The qualitative conclusion (retrieval alone cannot compute `max`, code can) is sound and worth
keeping; the 1.00-vs-0.42 number is a property of the harness, not of code-mode. No tool-calling arm
is present for comparison.

#### 🟧 P2-5 `[A]` Asymmetric query/passage encoding exists in `phase2/` but was never promoted to the SDK
`embed_models.py:9-20` handles it correctly and explicitly ("bge/e5 need DIFFERENT prefixes for
queries vs passages — getting this wrong tanks recall"), yielding `(query_embed, passage_embed)`
pairs. `Session` still has **one symmetric `embedder`**, which is exactly the blocker recorded in
`experiments/qwen8b_sac/issues.md` #4 (Qwen3-8B dense R@10 0.149 plain → 0.277 instructed, i.e. the
instruction is worth more than every augmentation measured in the repo). A capability the repo
already had, in the directory `soul.md` rule 2 says to promote *from*, sat unpromoted for months —
the concrete cost of LEG-2's missing `learnings_standard.md` workflow.

#### 🟧 P2-6 `[C]` Two of the three router formulations default to a worker count the GPU can't hold
`router_explore.py` (default `--workers 4`) and `ceiling_model.py` (default `--workers 4`) both use
`mp.get_context("spawn").Pool(workers)`, and **each worker process loads its own**
`SentenceTransformer` + reranker (`QwenReranker` in `router_explore`, a MiniLM CE in `ceiling_model`).
`CHANGELOG.md`'s gotcha list says "Qwen reranker OOMs with >2 workers" and "run GPU jobs serially" —
so the shipped default of the exploration script is the configuration the CHANGELOG warns against.

#### 🟧 P2-7 `[R]` Three incompatible router formulations
`phase2/router_model.py` + `router_explore.py` learn over **8 arms**
(`dense/keyword/hybrid_.8/prf/dense+rerank/hybrid+rerank/expand_fuse/expand_fuse+rerank`);
`search_as_code/explore/templates.py` learns over **16 templates**; `phase4/altera_router_train*.py`
has its own. They share no feature extractor, no label policy (best-arm vs cheapest-solver) and no
metric (`router_model.py` reports *realized routed recall* — which is exactly the correction
`open_problems.md` #3 demanded and SDK-A2 says never reached the SDK). The best-designed evaluator in
the repo is in the deprecated phase.

#### 🟨 P2-8 `[C]` `learned.py` details
- `alias_map()` opens with `m = dict(sac.__dict__.get("_", {}))` and then immediately `m = {}` —
  a vestigial line that does nothing (`learned.py:73-77`).
- `expand_seeds()` compares a lower-cased synonym key against the raw query
  (`query.replace(term, s) if term in query else f"{query} {s}"`), so any capitalised occurrence
  takes the append branch instead of substituting — the two behaviours have different retrieval
  effects and the choice is accidental.
- `LearnedProfile.load()` connects with `index="_meta"` purely to obtain a client handle.

#### 🟨 P2-9 `[C]` `device="cuda"` hardcoded in five scripts
`learn_rules.py`, `align_prompts.py`, `impact_eval.py`, `miss_analysis.py`, `miss_cases.py`,
`qwen_ab.py` construct `SentenceTransformer(..., device="cuda")` with no
`torch.cuda.is_available()` fallback, while `beir_run.py`, `beir_train.py`, `hotpot_eval.py`,
`ceiling_model.py` and `router_explore.py` guard it. The unguarded ones cannot run on a CPU box.

#### 🟨 P2-10 `[C]` `parse_args()` called two or three times per entry point
`router_explore.py:__main__` (`ap.parse_args().n, ap.parse_args().workers`) and
`ceiling_model.py:__main__` (three calls) re-parse `sys.argv` per argument. Harmless today; breaks
the moment an argument has a side effect or a default is computed.

---

## 9. `phase3/` + `phase4/` (deep pass, 55 files)

*Findings are described without reproducing customer identifiers; see the files themselves for values.*

#### 🟥 P4-1 `[A]` "SAC" names three different systems, and the customer-facing report uses the weakest one
- `phase1/agents.run_sac` / `eval_fair.code_harness`: **an LLM writes a Python program** in a sandbox.
- `phase4/altera_eval.py:retrieve("sac")`: a **hand-written** `decompose → dense+bm25 fan-out → RRF →
  Qwen rerank` pipeline. No LLM authors code, no sandbox. It is reported as the `sac` arm in
  `runs/altera_eval.json` and `ALTERA_RESULTS.md`.
- `harness/playbook.diagnostic_solve(forged=…)`: replay of previously forged primitives.

The authors were aware — `altera_codesac.py`'s docstring opens "**TRUE** code-mode SAC over the
Altera KB with the FULL primitive surface", which only makes sense as a contrast with the arm already
called `sac`. Any reader comparing the Altera "SAC" pass-rate to the FiQA or multi-hop "SAC" numbers
is comparing different systems. Rename the fixed pipeline (e.g. `fanout_rrf`) and keep `sac` for the
code-authoring agent.

#### 🟥 P4-2 `[C]` A tunnel blip is recorded as a retrieval miss
`phase4/altera.py:_search()` retries 4× and then **returns `[]`** — "degrade to [] rather than
crashing the whole run on a tunnel blip". Every arm in `altera_eval.py` / `altera_latency.py` /
`altera_claude_code.py` retrieves through it, so a transient SSH-tunnel failure is scored as
"retrieved nothing" and folded into the pass-rate, indistinguishable from a genuine miss. On a
customer-facing benchmark this is the highest-stakes instance of LEG-5. Minimum fix: count the
degradations and print them beside the result; better, mark the question `errored` and exclude it.

#### 🟧 P4-3 `[C]` Customer identifiers are hardcoded in tracked files
`phase4/altera.py` hardcodes the internal OpenSearch host:port, three real index names, and a
vector-field UUID; ~23 `altera_*.py` files carry at least one such identifier; `altera_eval.py`,
`altera_codesac.py` and `altera_latency.py` hardcode an absolute path to a customer evaluation
spreadsheet whose **filename names both the customer and the benchmarked vendor**. All are gitignored
*and tracked* (GOV-3), and **19 of the 21 `phase4/run_*.sh` wrappers are tracked** — `.gitignore`
only covers `phase4/run_altera*.sh`, so `run_call.sh`, `run_claude_code.sh`, `run_dump.sh`,
`run_eval_explore.sh`, `run_explore.sh`, `run_fit.sh`, `run_hyde.sh` are tracked and each `source
phase4/.secrets` (which exists on disk, 97 bytes, correctly ignored).

#### 🟧 P4-4 `[A]` The seeded-random sample that fixes SDK-C4 already exists here
`phase4/altera_synth.py:sample_cards()` uses
`{"function_score": {"query": …, "random_score": {"seed": seed, "field": "_seq_no"}}}` — a
**reproducible** random sample. `adapters/opensearch.py:sample()` uses `random_score` with **no
seed**, which is what breaks `corpus_fingerprint` and the whole resume/drift design (SDK-C4). The fix
is a two-line port from a file in the deprecated phase. Same shape as P2-5: the capability existed and
was never promoted.

#### 🟧 P4-5 `[A]` A fourth router formulation, still scored on CV accuracy
`altera_router_train.py` learns over 5 arms (`dense/keyword/kb/kb_expanded/hybrid`) and
`altera_router_train2.py` over its own set — so the repo has **four** incompatible router
formulations (8-arm phase2, 16-template SDK, 5-arm phase4, phase4-v2; P2-7 undercounted). It reports
`cv_acc` vs a majority-arm baseline and prints a verdict string from that comparison — the metric
`open_problems.md` #3 disowns (SDK-A2). Its `disagree_rate` ("arms disagree on X% of queries ←
routing headroom") is a *better* signal and appears nowhere else.

#### 🟧 P4-6 `[C]` Shared `LLM()` mutated from 8 worker threads → cost accounting races
`altera_eval.py:main` passes one `LLM()` into `ThreadPoolExecutor(max_workers=8)`, and each
`process_one` makes up to 15 calls through it. `phase1/llm.py:Usage.add` does
`self.input_tokens += …` on shared attributes with no lock — `+=` on an attribute is not atomic in
CPython, so the reported `llm_cost_usd` under-counts by an unknown amount. Same pattern in
`altera_claude_code.py` (4 workers) and `altera_synth.py` (12 workers).

#### 🟧 P4-7 `[R]` A fifth sandbox namespace
`altera_codesac.py` builds its own `_SAFE` builtins dict + `agent_namespace(question)` and execs the
authored program there, importing `SAC_SYSTEM` from `phase1`. With `sandbox.LocalExecutor`,
`forge._safe_globals`, `agentic._exec` and `eval_fair.code_harness`'s inline `ns`, that is **five**
mutually incompatible execution namespaces for LLM-authored retrieval code (SDK-R5 said two).

#### 🟨 P4-8 `[A]` Two genuinely promotable assets are stranded in the customer phase
- `phase4/metrics.py`: SQuAD-standard EM / token-F1 **plus `bootstrap_ci` (2000-sample 95 % CI)** —
  the only significance testing anywhere in the repo. Every headline number elsewhere is a bare mean.
- `phase4/altera_eval.py`'s **judge calibration**: it first scores the *vendor's own* answers and
  checks the judge reproduces the spreadsheet's published pass-rate before trusting it on the new
  arms. `harness/DiagnosticJudge` has no equivalent hook.
- Positive note for contrast: `altera_learn.py` mines its profile "from the KB ONLY (no sheet access
  → no test leakage)", i.e. phase4 **avoided** the leak that phase2's `learn_rules.py` has (P2-1).

#### 🟨 P4-9 `[C]` `altera_claude_code.py`'s arm name overstates what runs
The `claude_code` arm is a **static** Python function ("Claude's retrieval strategy, encoded from
driving real queries") — a hand-written recipe, not a live agent. The docstring is honest about it;
the arm label in the results JSON is not.

#### 🟨 P3-1 `[C]` `phase3/cross_db_relevance.py` is clean; two small notes
The cross-backend parity study is well constructed (same vectors scrolled out of OpenSearch, one
query embedding, per-backend latency) and its expected-result framing is honest. Notes:
`qids = list(queries)[:n]` is first-n rather than sampled, and `scroll_vectors` checks
`got >= max_docs` only after finishing a 1000-doc batch, so it can overshoot the cap by up to 999
docs — harmless here, misleading if someone caps it low.

---

## 10. `experiments/browsecomp/` + `experiments/deep_judge/` (deep pass, 34 files)

#### 🟥 DJ-1 `[C]` The judge's "held-out" 0.721 is tie-broken on the test split
`tune_judge.py:146-150` selects the best round as:
```python
# select on TUNE but require a real margin (>0.01) so we don't chase eval noise; tie-break on TEST
if best is None or cm["balanced_acc"] > best["tune"]["balanced_acc"] + 0.01 or (
        abs(cm["balanced_acc"] - best["tune"]["balanced_acc"]) <= 0.01
        and tm["balanced_acc"] > best["test"]["balanced_acc"]):
```
The second branch chooses between near-tied rounds **by TEST balanced accuracy** — so the reported
number is no longer a clean held-out estimate, contradicting the module's own stated intent ("Track
TEST (held-out) each round for honest generalization"). That number propagates to
`search_as_code/harness/diagnostic_judge.py:18` ("Tuned default (round-7 critic revision; held-out
balanced-acc 0.721)"), to `deep_judge/README.md` §1, to `open_problems.md` #6, and to the
"0.72 **is** the signal ceiling" conclusion the whole deep-SAC line rests on. Fix: select on TUNE only,
break ties by round index, and report TEST once.

#### 🟥 DJ-2 `[A]` The tuning gain is one example wide, and no interval is reported
From `tuning_log_ce_same.md`: round 0 → TUNE 0.760 / TEST 0.700; adopted round 7 → TUNE 0.771 /
TEST 0.721. The TUNE delta is **+0.011** — the confusion matrices differ by `tn 36→37, fp 11→10`,
i.e. **a single example flipped** — which is what cleared the code's own 0.01 "don't chase eval noise"
margin. On TEST it is two examples. At n=100 and p≈0.72 the 95 % interval is **±0.088**, four times
the claimed improvement, and none of the judge tables anywhere in the repo report an interval —
although `phase4/metrics.py:bootstrap_ci` exists (P4-8). Every "0.63 → 0.72" style delta in
`deep_judge/README.md` §1 needs a CI before it can support the ceiling claim.

#### 🟥 DJ-3 `[A]` The "independent Qwen-32B critic" row is the untuned prompt
`tuning_log_ce_qwen.md` records `## Best (round 0)` with TUNE 0.760 / TEST 0.700 — round 0 is the
**unmodified `INITIAL_PROMPT`**, so the Qwen-32B critic never produced a revision that was adopted.
`deep_judge/README.md` §1 lists "LLM judge — v1 + **independent Qwen-32B critic** | 0.70" alongside
the same-model critic's 0.721 as if both were tuned outcomes, and concludes "neither beats the signal
ceiling". The honest statement is stronger and different: *the independent critic produced no adopted
improvement at all* — its 0.70 is the baseline. Same for `tuning_log_same.md` (bi-encoder variant,
best = round 0, TEST 0.585), whose README row reads "0.585 → ~0.68" with no log entry for the 0.68.

#### 🟧 BC-1 `[C]` A published file reads its gold labels from a `/tmp` scratchpad path
`bc_common.py:34-36` hardcodes
`BC_REPO = Path("/tmp/claude-1001/-home-taranjeet-bakshi-code-search-harness/<session-uuid>/scratchpad/BrowseComp-Plus")`
and `load_golds()` reads `qrel_golds.txt` from under it. The directory happens to still exist on this
machine, so the benchmark runs today — but it is a session-scoped temp path from a *different*
session, it embeds a local username and session UUID, and **`bc_common.py` is on public `origin/main`**
(GOV-1). For anyone else the file is unrunnable, and one `/tmp` cleanup makes the 830-query
BrowseComp line unreproducible here too. Move the qrels into the experiment directory (or an env var
with a clear error).
**FIXED** 2026-08-18 — qrels re-fetched from texttron/BrowseComp-Plus into `experiments/browsecomp/data/qrel_golds.txt` (2,407 lines) and `bc_common.py` repointed; sanity: 830 queries, 2.9 golds/query, dense gte-base R@10 0.0705 (~published 0.0624, harness-subset difference).

#### 🟧 BC-2 `[A]` The BrowseComp keyword/hybrid arms index 6 % of each document
`bc_common.py:19-22`: `KW_CHARS = 2000` — "BrowseComp-Plus docs are long (~33KB avg). A pure-Python
BM25 over the full 3.3GB of text is intractable, so the keyword index uses each doc's first
KW_CHARS characters." Honestly documented in the file, but it means **every BM25, hybrid and
`fuse(dense, keyword)` number on BrowseComp** — including the arms compared against dense in
`RESULTS.md` and the `explore`/forge numbers in `deep_judge/README.md` §5 — was measured with ~6 % of
each document indexed lexically. The dense arm is unaffected (full precomputed vectors), so the
comparison is systematically tilted against every lexical strategy. This caveat does not appear in
either results write-up.

#### 🟧 BC-3 `[R]` `FastMemoryStore` is a private fix for SDK-C11 that was never promoted
`bc_common.py:66-115` subclasses `MemoryStore` purely to precompute the keyword index, with the
diagnosis in its docstring: "the stock MemoryStore re-tokenizes every doc on every keyword query,
which is unusable at ~100K docs". It is complete (df + per-doc counts + doc lengths, `threading.Lock`
around the build) and battle-tested at 100K docs. `soul.md` rule 2 says a generalizable fix found in
custom work must land in `search_as_code/` — this one stayed local, so `phase2/beir_qrels.py` and
`beir_train.py` (which load up to 200K docs into the *stock* `MemoryStore` and then run 16 templates
× N queries through `query_keyword`) still pay the O(N·tokens)-per-query cost. Porting it is a
copy-paste.

#### 🟧 BC-4 `[C]` The SDK's meta-buffer fix does not cover the buffer that actually broke ReasonIR
`reasonir_encoder.py`'s docstring is an exemplary write-up of the failure: under transformers 5.x the
rotary `inv_freq` non-persistent buffer materialises as uninitialised memory → per-process-random
positional encoding → cross-process recall collapse. But
`search_as_code/embeddings.py:_fix_meta_buffers` only re-registers `position_ids` and — gated on
`hasattr(m, "cos_cached") and hasattr(m, "inv_freq")` — `cos_cached`/`sin_cached`. Modern Llama-style
RoPE modules expose **`inv_freq` without `cos_cached`**, so the guard is false and **nothing is
fixed**; `inv_freq` itself is never re-materialised. The SDK therefore has a partial fix for exactly
this bug family and would silently hit the same corruption on the next such model. The workaround
(a pinned venv) is documented but lives outside the code.

#### 🟨 BC-5 `[C]` Two different "paper" reference numbers for the same model
`reproduce_qwen8b.py`'s docstring targets "the paper (Recall@5 14.5%, nDCG@10 20.3%)";
`deep_judge/README.md` §6 compares against "paper Qwen3-8B, **Gold**: R@5 0.185". Both are presumably
real rows of arXiv 2508.06600 (different retrieval settings), but nothing in the script says which
setting it targets, so a reader cannot tell whether 0.200 beat the intended reference or a different
one. State the table/row in both places.

#### 🟨 BC-6 `[C]` `eval.py` retunes a shared module constant at import time
`experiments/browsecomp/eval.py:23-25` does `EF.K = 20` after importing `eval_fair`, then imports
`Tools`/`tool_harness`/`code_harness` from it. It works (the functions read the module global at call
time) and it is commented — but it silently changes the depth of a harness another experiment reports
`@10` numbers from, and the two experiments' results are only comparable if you notice this line.
Pass `k` as a parameter instead.

**Positive notes worth keeping** (not defects): `browsecomp/eval.py` restricts the sample to queries
whose **golds are all present in the corpus** before measuring `all_golds` and shuffles with a fixed
seed — the cleanest sampling in the repo; `build_evalset.py` freezes the candidate states once so
every judge-prompt variant is scored on identical inputs, and weights the hop mix toward the regime
where a stop signal matters; `tune_judge.py`'s adopt-only-if-TUNE-improves guard is the right shape
(DJ-1 is one line inside it).

---

## 11. Remaining `experiments/` (deep pass: deep_sac, explore_forge, explore_learning, primitive_selection, su_multihop, multi_hop remainder)

#### 🟥 DS-1 `[C]` The "explore-seeded deep SAC" arm was fed a one-line profile — the negative result is confounded
`deep_sac/run_deep_sac.py:103` builds the arm-2 hint with
`describe_session.describe(n_samples=8, llm=True)`, and `Session._llm_profile` prompts for
"**In 4-6 short lines** describe: (1) what kind of data this is, (2) the key entities/fields,
(3) **which retrieval primitives fit best**" — then returns `out[0]`, the **first line only**
(GEN-1). With `LLM.as_generator()` the arm therefore received item (1) and **never received item (3),
the recommended primitives** — the only actionable part.

That arm is the basis of a published negative conclusion. From the public
`deep_sac/deep_recall.json` (HotpotQA 2-hop): `sac_deep` recall@10 **0.92** / all_golds 0.86 with 2.98
searches, vs `sac_deep_explore` **0.79** / 0.74 with **7.6** searches; and
`make_deep_report.py:189` writes it up as "Seeding the deep agent with the `describe(llm=True)`
corpus profile **dropped** HotpotQA …". The finding may well survive a fixed profile — but as measured
it cannot distinguish "corpus profiling hurts" from "a truncated one-line hint hurts". This is the
clearest downstream cost of GEN-1 and it should be re-run first.

#### 🟧 DS-2 `[A]` `su_multihop` / `deep_sac` read customer docs from outside the repo, and their results are public
`su_multihop/run_su_multihop.py:39` and `deep_sac/run_deep_sac.py:44` both load
`~/scripts/data/su_docs_2.csv` — customer documents outside the repository, with no checked-in
fixture or documented provenance, so neither study is reproducible by anyone else. Meanwhile
`run_deep_sac.py`'s own docstring says "**INTERNAL — never pushed. SU uses internal customer docs**",
yet 10 `experiments/deep_sac/` files *are* on public `origin/main` (`deep_recall*.json`,
`deep_recall*_perquery.jsonl`, `make_deep_report.py`, `_launch.sh`). I checked the published
contents: they carry **only aggregate metrics and per-query numbers — no query text, no document
text** (`{"corpus": "hotpotqa", "hop": 2, "arm": …, "recall": 1.0, …}`), so this is a policy
inconsistency rather than a content leak. Decide which it is and make the docstring and `.gitignore`
agree.

#### 🟧 DS-3 `[A]` Two multi-hop harnesses with different arm definitions, both reported as "SAC vs tool"
`multi_hop_synth_queries/eval_recall.py` and `eval_fair.py` both measure dense/tool/sac on the same
datasets but define the arms differently: in `eval_recall` the tool arm is "ONE search per turn …
**RRF-accumulate**" and the sac arm "plans all sub-queries in one shot, batch-search, fuse in code";
in `eval_fair` both arms share the 8-tool `Tools` object and a budget. `RESULTS.md` presents the
`eval_fair` run as the headline and calls the earlier one "an earlier, non-tool-matched recall run",
which is honest — but both files remain, both write results, and only the docstrings distinguish them.
Consider deleting or clearly archiving `eval_recall.py` so no one re-reports its numbers.

#### 🟨 DS-4 `[A]` `primitive_selection` and `explore_learning` are the methodologically strongest studies here
Recorded as a positive so a future cleanup does not flatten them: `run_multihop_router.py` passes
`label_llm=True, label_rerank=True` (so the templates genuinely differ — the SDK-A1 trap is avoided),
states the all-golds gate up front, and pools 2/3/4-hop with an `n_docs` tag;
`explore_learning/make_charts.py` renders from `model_bakeoff.json` rather than hardcoded numbers;
and `explore_learning/README.md` §4a documents its own metric correction (CV accuracy → realized
recall) before reporting the headline. That correction is the one the SDK never adopted (SDK-A2).
Only caveat: the HyDE-based templates in those runs still went through GEN-3, so `hyde_rerank` /
`deep_hyde_decompose` were labelled with a first-line-only hypothetical document.

#### 🟨 DS-5 `[C]` `explore_forge` validates a forged primitive on the queries it was forged from
`explore_forge/run_forge.py` mines winning patterns from the exploration set and then accepts the
authored primitive using held queries drawn from the same pool (the later
`deep_judge/run_explore_pipeline.py:171` improved this to `held_list = test[:5] or train[:5]`, i.e. a
test slice with a train fallback). The forge-acceptance bar is therefore weaker in the older study
than in the newer one, while both write `forge_store_*` artifacts that later runs load and reuse. Note
which store came from which acceptance rule.

---

## 12. `phase1/` remainder, `chatbot/`, `benchmarks/` (deep pass, 24 files)

#### 🟥 CB-1 `[A]` A properly-designed tool-calling baseline exists in `chatbot/` — the headline benchmark uses the weak one
`chatbot/toolcalling.py` is what P1-1's arm should have been. Its tool design explicitly follows
Anthropic's "writing tools for agents" guidance (its docstring says so), and crucially
`_search_docs()` internally does **hybrid retrieval → `hydrate` → cross-encoder rerank → top_k**
(`toolcalling.py:88-96`), so the tool arm gets reranking *inside* the tool. It also has `read_docs`
for full-text inspection, a structured `finish`, and it **counts a hop only for retrieval rounds, not
the finish round** (`:117-119`) — a more careful turn definition than `eval_fair`'s hardcoded
`steps: 1` (P1-8).

Meanwhile `phase1/sac_surface.TOOLCALL_TOOLS`'s `search` is a bare `session.search(...)` with no
rerank, no hydrate and no read tool — and *that* is the arm in the README's headline table and in
`MULTI_DATASET_REPORT.md`. The fair comparison was achievable with code already in this repository.
Point the benchmark at this tool design (or port its `search_docs`) and re-run before quoting
"SAC wins every axis that matters vs MCP tool-calling" again.

#### 🟧 P1-14 `[C]` The README's reproduce command uses a reranker that the repo says will not download
`README.md:126-128` says "Reproduce with `python -m phase1.benchmark -n 100`". That entry point
defaults to `--reranker BAAI/bge-reranker-base` (`phase1/benchmark.py:44`, `:126`), while
`phase1/RESULTS.md:62-63` states "FiQA-appropriate `bge-reranker` **would not download in this
environment**" and gives the actual command with
`--reranker cross-encoder/ms-marco-MiniLM-L-12-v2` (`:73`) — which is also the reranker the README's
own results table credits ("gte-base embeddings + MS-MARCO reranker", `:114`). So the documented
one-liner is not the command that produced the documented numbers, and on a fresh machine it fails or
silently degrades at the first `rerank` call. Change the default to the ms-marco model.

#### 🟨 P1-15 `[C]` First-N query selection in the FiQA benchmark
`phase1/benchmark.py:47` (`qids = [...][:n]`), `eval_base.py`, `answer_gen.py:main` and
`hotpot_eval.py` all take the first N qrels-bearing queries in file order rather than a seeded
sample. `phase1/ceiling.py` and `phase2/router_explore.py` *do* sample with a seed, and
`browsecomp/eval.py` shuffles with `random.seed(0)` — so the convention is inconsistent across
studies that get compared to each other. `MULTI_DATASET_REPORT.md` discloses the first-N choice for
the SAC/tool subset and pairs the baselines on the same subset, which is the right mitigation; the
others do not mention it.

#### 🟨 BM-1 `[C]` `benchmarks/bench.py` measures ANN latency with uniform-random vectors
`bench_scalability`/`bench_throughput` build queries as `[rng.random() for _ in range(dim)]`
(`:87`, `:93`, `:163`) with the justification "latency doesn't depend on vector *meaning*". That
holds for brute-force backends, but for HNSW the number of distance computations depends on the
neighbourhood structure a query lands in, and uniform-[0,1) vectors all sit in the positive orthant —
an unrealistic, mutually-similar distribution. The absolute OpenSearch QPS/latency figures in
`benchmark_changelog.md` may therefore not transfer to real embeddings. Sampling stored corpus
vectors (with jitter) as queries would cost nothing and remove the caveat.

#### 🟨 BM-2 `[A]` `benchmarks/` is the one harness with a clean design and no results in the repo
`bench.py` isolates each component (ingest throughput, corpus-size latency scaling, fan-out, QPS,
primitive micro-ops, retry/chunking overhead, embedding throughput), writes raw JSON per run and has
a `--json` mode. Its outputs go to `benchmarks/results/`, which is gitignored except the logs — so
`benchmark_changelog.md` is the only surviving record, and it is the file the public README links to
and that is **not published** (DOC-6). Publish the JSON or stop linking the changelog.

**Positive notes:** `phase1/metrics.py` is a correct, standard BEIR-style implementation (recall/nDCG
with graded gains, MRR, and `evaluate()` skipping unlabelled queries); `phase1/ceiling.py` and
`phase2/ceiling_model.py` both report a per-query **oracle** alongside the best fixed combo, which is
the right way to frame routing headroom; `chatbot/evaluate.py` opens by stating its own limitation
("FiQA ships relevant-doc labels, not gold answers, so answer quality is bounded by whether we
surface the right passages") before reporting anything.

---

## 13. Revised fix order after the deep pass (supersedes §7)

§7 was written from the first pass (74 issues). The deep pass added 39 more, including six blockers
that outrank most of the original list because they change *published conclusions* rather than code
behaviour.

### Tier 1 — a published conclusion is currently unsupported
| # | issue | what changes if fixed |
|---|---|---|
| 1 | **DJ-1 + DJ-2 + DJ-3** | The "0.72 is the signal ceiling" result: the held-out number is tie-broken on test, the tuning gain is 1–2 examples with a ±0.088 interval, and the "independent Qwen-32B critic" row is the *untuned* prompt. Re-select on TUNE only, add bootstrap CIs (`phase4/metrics.py` already has them), relabel the Qwen row. |
| 2 | **DS-1** | "Seeding the deep agent with `describe(llm=True)` **dropped** recall" was measured with a **one-line** profile (GEN-1) that omitted the recommended-primitives item. Fix GEN-1, re-run the arm. |
| 3 | **P1-1 + P1-7 + P2-2 + CB-1** | "SAC beats MCP tool-calling on every axis": the tool arm has 3 tools vs ~20 primitives, the multi-hop study gives the code arm the winning recipe *and* the rerank caveat, and a properly-designed tool arm already exists in `chatbot/toolcalling.py`. Either match the arms or restate the claim. |
| 4 | **P2-1** | The learned-profile lift is measured on the queries the rules were mined from (80 % overlap). Split, re-run. |
| 5 | **P4-1** | "SAC" names three different systems; the customer-facing report uses the fixed pipeline, not the code-mode agent. Rename. |

### Tier 2 — silent wrongness in code
| # | issue | why |
|---|---|---|
| 6 | **GEN-1** | one root cause, six call sites, degrades the documented best multi-hop recipe; also the trigger for DS-1 |
| 7 | **SDK-C1** | the "validated read-only" DSL gate never rejects anything (1-line fix) |
| 8 | **P4-2** | a tunnel blip is scored as a retrieval miss on the customer benchmark |
| 9 | **SDK-C4** (fix in **P4-4**) | unseeded `sample()` defeats resume/drift; the seeded version already exists in `phase4/altera_synth.py` |
| 10 | **SDK-C5** | the 4-way failure taxonomy mislabels every OpenSearch query (adapters strip `d.vector`) |
| 11 | **SDK-C2 / C3** | `$or` filters silently ignored; `$eq` on string metadata silently matches nothing |
| 12 | **BC-4** | `_fix_meta_buffers` never re-materialises `inv_freq`, the buffer that actually corrupted ReasonIR |

### Tier 3 — governance, then the rest
13. **GOV-1** (internal data on public `main`), **GOV-2** (unignored customer SSH key), **P4-3**
    (tracked customer identifiers + 19 tracked `run_*.sh`), **BC-1** (published file reads gold
    labels from a `/tmp` scratchpad path).
14. **BC-3 / BC-2** — port `FastMemoryStore` into the SDK (it fixes SDK-C11 and unblocks
    `phase2/beir_*`); document the `KW_CHARS = 2000` caveat wherever a BrowseComp lexical number
    appears.
15. Then §7's tiers 8–10 (thread-safety, labeling defaults, fallback counting).

### What the deep pass changed about the diagnosis

The first pass concluded the dominant defect class was "a documented property that no test covers,
failing silently". The deep pass sharpens that in two ways:

1. **The best methodology in the repo is in the directories marked deprecated or internal.**
   Realized-recall routing evaluation (`phase2/router_model.py`), bootstrap CIs and SQuAD metrics
   (`phase4/metrics.py`), judge calibration against a published baseline (`phase4/altera_eval.py`),
   leakage-free profile mining (`phase4/altera_learn.py`), a correct tool-calling baseline
   (`chatbot/toolcalling.py`), seeded random sampling (`phase4/altera_synth.py`), and a working
   `MemoryStore` keyword index (`browsecomp/bc_common.py`). The shipped SDK and the headline
   benchmarks use the weaker version of each. `soul.md` rule 2 ("improve the SDK, don't fork it") has
   a matching companion gap: **`learnings_standard.md`, the file the constitution names as the
   promotion path, does not exist** (LEG-2). That is the mechanism behind at least six findings
   (P2-5, P4-4, P4-8, BC-3, CB-1, SDK-A2).

2. **Where the numbers are honest, they are honest by narrative, not by construction.** Many
   write-ups disclose their own weaknesses in prose (`KW_CHARS`, first-N subsets, "an earlier
   non-tool-matched run", "would not download in this environment"). But the disclosure lives in a
   docstring or a README paragraph while the *number* travels alone into `README.md`,
   `CHANGELOG.md` and `open_problems.md`. Attaching the caveat to the artifact — a `caveats` field in
   every results JSON, printed by the report generators — would prevent the whole class.

---

## Coverage of this audit

**Read in full:** all of `search_as_code/` (46 files), `tests/` (9), `docs/` (13), `examples/` (2),
root markdown + config, `phase1/` (16), `phase2/` (30), `phase3/` (3), `phase4/` (52),
`chatbot/` (8), `benchmarks/` (2), and every `experiments/` subdirectory (browsecomp 17,
deep_judge 17, multi_hop_synth_queries 8, deep_sac 4, explore_forge 4, explore_learning 1,
primitive_selection 1, su_multihop 1, qwen8b_sac 1) — ~205 code files, ~25k lines, plus 35 markdown
files and the tuning logs.

**Not covered:** `phase4/models/gte-alt-v1/` vendored model code (1,563 lines of third-party
`modeling.py`/`configuration.py` — read only far enough to confirm it is vendored HF code, not
project logic); binary/data artifacts (`*.npy`, corpus JSONL, figures); `.venv-dummy/`; and the
generated `dist/` wheel.

---

## 14. Structural comparison with Flask, mem0 and LangChain — repo-layout issues

Added 2026-08-17. The audit above judges this repo against itself (does the code do what the docs say?).
This section judges its **structure** against three reference projects, each chosen because it already
solved a problem we have:

- **Flask** ([pallets/flask](https://github.com/pallets/flask)) — the canonical single-package Python
  library layout.
- **mem0** ([mem0ai/mem0](https://github.com/mem0ai/mem0)) — the closest peer: an AI-memory SDK with
  provider-plugin adapters *and* a benchmark suite, i.e. the same SDK-plus-research shape as this repo.
- **LangChain** ([langchain-ai/langchain](https://github.com/langchain-ai/langchain)) — the largest
  many-backends-behind-one-API project in this space.

All three layouts were read on 2026-08-17 from the GitHub contents API and are reproducible:

```bash
for r in pallets/flask mem0ai/mem0 langchain-ai/langchain; do
  echo "== $r"; curl -s "https://api.github.com/repos/$r/contents/" | grep '"name"'; done
curl -s https://api.github.com/repos/langchain-ai/langchain/contents/libs        # 7 libs
curl -s https://api.github.com/repos/langchain-ai/langchain/contents/libs/partners  # 15 provider pkgs
curl -s https://raw.githubusercontent.com/mem0ai/mem0/main/.gitmodules           # evaluation/ is a submodule
```

### 14.0 The layouts, side by side

| aspect | Flask | mem0 | LangChain | **this repo** |
|---|---|---|---|---|
| importable package | `src/flask/` | `mem0/` | `libs/<pkg>/<module>/` | `search_as_code/` at repo root |
| tests | `tests/` (root) | `tests/` (root) | `libs/<pkg>/tests/{unit,integration}_tests/` | `tests/` (8 files) |
| cross-backend contract | n/a | n/a | **`libs/standard-tests/langchain_tests/`** — shipped conformance suite (`integration_tests/vectorstores.py`, `embeddings.py`, `retrievers.py`, `cache.py`, `base_store.py`) each integration subclasses | none (README claims one) |
| provider/backend units | n/a | `mem0/{vector_stores,llms,embeddings,reranker}/` + `mem0/configs/` | **15 separate installable packages** under `libs/partners/` (chroma, qdrant, openai, …) | 9 adapters in one dist, behind extras |
| research / benchmarks | absent from repo | **`evaluation/` is a git submodule** → `github.com/mem0ai/memory-benchmarks` | absent from repo root | `experiments/` (107 files) + `phase1-4` (102) + `chatbot/` (9) + `benchmarks/` (8), all in-tree |
| examples | `examples/{tutorial,celery,javascript}`, each a standalone project (`tutorial/` has its own `pyproject.toml`, `LICENSE.txt`, `README.rst`, `tests/`) | `examples/`, `cli/`, `server/` | `libs/*/scripts` + external docs repo | 2 loose scripts |
| dependency pinning | `uv.lock` | `poetry.lock` | per-lib `uv.lock` (e.g. `libs/core/uv.lock`) | **none** |
| pre-commit | `.pre-commit-config.yaml` | `.pre-commit-config.yaml` | `.pre-commit-config.yaml` | **none** |
| task runner | — (pre-commit + uv) | `Makefile` | per-lib `Makefile` (e.g. `libs/core/Makefile`) | **none** (commands live in prose) |
| changelog | `CHANGES.rst` (one) | release notes in docs | per-package | **two** (`CHANGELOG.md`, `benchmark_changelog.md`) + `docs/STATUS.md` |
| docs | `docs/` + `.readthedocs.yaml` (Sphinx build) | `docs/` (Mintlify site) | separate docs build | `docs/` = 13 loose `.md`, **no build config** |
| governance files | `LICENSE.txt`, `CHANGES.rst`, `.editorconfig` | `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` | `CITATION.cff`, `AGENTS.md` | `LICENSE` only |
| agent instructions | — | `CLAUDE.md`, `AGENTS.md`, `LLM.md`, **plus a per-package `mem0/CLAUDE.md`** | `CLAUDE.md`, `AGENTS.md` | `CLAUDE.md`, `soul.md`, `STRUCTURE.md` ✅ |

Two things this repo already does as well as or better than the comparators, recorded so a cleanup does
not remove them: the **agent-instruction layer** (`CLAUDE.md` + `soul.md` + `STRUCTURE.md` + a
`SessionStart` hook) is more developed than mem0's, and **`issues.md` itself** has no analog in any of
the three.

### Issues

#### 🟥 STR-1 `[A]` Root-level package layout means the published wheel is never the thing that gets tested
`pyproject.toml:59-60` (`[tool.setuptools.packages.find] include = ["search_as_code*"]`) selects a
subset of a tree in which `search_as_code/` sits at the repo root beside `phase1/`, `experiments/`,
`chatbot/`. Because the root is on `sys.path`, `pytest` (`pyproject.toml:63`, `testpaths = ["tests"]`)
and every experiment import the **working copy**, never the built artifact — so nothing in CI or local
dev exercises what `pip install search-as-code` actually delivers. **This is not hypothetical: it is the
mechanism behind DOC-1** — `phase1/sac_surface.py::SAC_SYSTEM`, which `docs/SELECTION.md:3` calls the
LLM-facing surface, is excluded from the wheel and no test noticed. Flask's `src/flask/` (verified: `src/`
contains only `flask/`) makes this failure impossible by construction — you cannot `import flask` from a
Flask checkout without installing it. Fix, cheapest first: **(a)** add a CI job that builds the wheel,
installs it into a clean venv, and runs the README quickstart + `import search_as_code; sac.available()`;
**(b)** longer term, move to `src/search_as_code/`.

#### 🟥 STR-2 `[A]` Nine backends ship in one distribution with no conformance suite — the field's answer is a shared test package
`README.md:225-227` states "the in-memory test suite is the contract every adapter must satisfy". No such
contract exists (TEST-3), and the distribution model makes it worse: 9 backends in one wheel behind
extras that do not match the adapters (`pyproject.toml:36-58` — no extra for `faiss`, `sqlite`, `nmslib`,
`milvus`; `all` omits `opensearch-py`, PKG-1), while `adapters.available()` reports all nine regardless of
what is installed (`adapters/registry.py:87`, SDK-C14). LangChain solves exactly this with two structures
we have neither of: **15 separate installable packages** under `libs/partners/`, and
**`libs/standard-tests/langchain_tests/`**, a *published* conformance suite whose
`integration_tests/vectorstores.py` / `embeddings.py` / `retrievers.py` every integration subclasses — so
a provider that silently drops a filter clause fails someone else's test. mem0 takes the lighter version
of the same idea: `mem0/vector_stores/` + `mem0/configs/` with a per-provider config class beside each
adapter. **Why it matters here:** SDK-C2 (`$or` filters silently dropped), SDK-C3 (`$eq` on string
metadata matches nothing) and SDK-C6 (`regex` never hits on OpenSearch) are all *contract* violations that
a parametrized suite would have caught on day one — `tests/test_opensearch.py:77` only checks equality on
a numeric field. Fix without splitting the distribution: a `tests/conformance.py` exposing a
`VectorStoreConformance` base that is parametrized over every registered adapter, plus one extra per
backend. This entry does not restate TEST-3; it records the *distribution model* as the reason the gap
persists, and names the prior art to copy.

#### 🟥 STR-3 `[A]` Research and customer work live inside the shipped repo; both peers keep them structurally out
`experiments/` (107 tracked files), `phase1-4` (102), `chatbot/` (9) and `benchmarks/` (8) sit in the same
tree and the same git history as the published package — 226 tracked files against the SDK's 46.
The comparators do the opposite, and one of them does it with the exact mechanism that would fix
GOV-1/GOV-3 here: **mem0's `evaluation/` is a git submodule** pointing at a separate repository
(`.gitmodules` → `https://github.com/mem0ai/memory-benchmarks`), and LangChain's root holds nothing but
`libs/` and config. Three consequences already logged separately, whose *common cause* is this layout:
1. **Governance is enforced by `.gitignore` discipline rather than by structure — and it has already
   failed twice** (GOV-1: 19 `experiments/browsecomp/` + 6 `experiments/su_multihop/` files on the public
   `main`; GOV-3: 51 tracked `phase4/altera*`). A submodule or a second repo makes "internal never gets
   pushed" a property of the topology, not of a rule an agent has to remember.
2. **~19k of ~25k Python lines are never linted or type-checked** (CI-1) because CI only covers
   `search_as_code/` — a scope that is only defensible *because* the other 19k are in the same tree.
3. **No experiment is reproducible from a release.** 34 files under `experiments/` import `phase1`
   (`grep -rl 'from phase1\|import phase1' experiments --include=*.py | wc -l` → 34), and `phase1` is
   excluded from the wheel (STR-1) and from mypy (`pyproject.toml:88`).
Fix: move `experiments/` + `phase2-4` + `chatbot/` to a companion repo (submodule if they must stay
visible), keeping `search_as_code/`, `tests/`, `docs/`, `examples/` as the distribution.

#### 🟧 STR-4 `[C]` No lockfile and no pinned dev/ML requirements, in a repo already burned by an unpinned dependency
Flask ships `uv.lock`, mem0 `poetry.lock`, LangChain a `uv.lock` per lib. This repo has none, and every
dependency is an open-ended floor (`pyproject.toml:24 numpy>=1.24`, `:42-47 sentence-transformers>=2.2`,
`torch>=2.0`, `langchain-openai>=0.1`). The cost is already recorded: **BC-4** — transformers 5.x
materialising ReasonIR's rotary `inv_freq` as uninitialised memory, collapsing recall — with the
workaround living in a hand-maintained venv *outside* the repo (`~/reasonir_venv`), documented only in
prose. `pyproject.toml:81-84` likewise carries a comment explaining a mypy 3.12-vs-runtime-3.10 split that
a lock would express directly. Fix: `requirements/{dev,docs,tests}.txt` compiled with `uv pip compile`
(Flask's model) or a lockfile, and an upper bound on `transformers`/`torch` in the `[phase1]` extra.

#### 🟧 STR-5 `[C]` No `.pre-commit-config.yaml` — all three comparators have one, and it is the standard guard against GOV-2
Flask, mem0 and LangChain each ship `.pre-commit-config.yaml` at root. Here, ruff and mypy run **only** in
CI and **only** over `search_as_code/` (`.github/workflows/ci.yml`), so nothing checks a commit before it
is made. Sitting in the working tree right now: an SSH public key naming a customer host and a colleague's
email (GOV-2, matched by no `.gitignore` rule), an 868KB customer evaluation CSV, `dist/`, `.coverage`,
`.venv-dummy/`. `CLAUDE.md:50` mitigates this with a *rule* ("Never `git add -A`"). The upstream
convention is a *mechanism*: `check-added-large-files`, `detect-private-key`, `end-of-file-fixer`, plus
ruff/ruff-format as hooks. A pre-commit config would make GOV-2 unstageable rather than merely forbidden.

#### 🟧 STR-6 `[C]` `docs/` is a folder of loose markdown with no build and no link check
Flask has `.readthedocs.yaml` + a Sphinx `docs/`; mem0 builds a Mintlify site; LangChain builds its docs in
CI. Here `docs/` is 13 `.md` files with no `conf.py`, no `mkdocs.yml`, no nav, no build step and no link
checker (verified: nothing matching `conf.py|mkdocs|mint|docusaurus|package.json` in `docs/`). That is
precisely why three documentation defects survived to this audit: **DOC-6** (the public README links to
`benchmark_changelog.md`, which is not on `origin/main` — a link checker is a 5-line CI job), **DOC-5**
(three different test counts across `README.md:10`, `README.md:65`, `docs/STATUS.md`), and **DOC-4** (a
`STATUS.md` reporting "65 tests" and "5 adapters" against a reality of 106 and 9). Fix: mkdocs + `lychee`
in CI; generate the test-count badge instead of hand-writing it.

#### 🟧 STR-7 `[C]` Version is duplicated, there are zero tags, and the publish workflow fires on a release that has never been cut
`pyproject.toml:7` and `search_as_code/__init__.py:72` both hardcode `0.0.1`; `git tag` returns **0 tags**;
`.github/workflows/publish.yml` triggers `on: release: [published]` — an event this repo has never
produced. Meanwhile `dist/` holds a wheel built 2026-07-28 carrying that same `0.0.1` while `CHANGELOG.md`
records months of subsequent change (LEG-7), so `pip install dist/*.whl` silently yields July code under
the current version string, and no PyPI artifact can be mapped back to a commit. Flask single-sources the
version and every change lands in `CHANGES.rst` under a version heading. Fix: single-source the version
(`dynamic = ["version"]` reading `__init__.py`), tag releases, and delete the stale `dist/`.

#### 🟧 STR-8 `[A]` Four top-level `phase*` packages have no analog in any comparator, and the one that is load-bearing is unshippable
`phase1/` (16 tracked files), `phase2/` (31), `phase3/` (4), `phase4/` (51) occupy the top-level import
namespace of a repo that publishes one package. `STRUCTURE.md:51` says phase2/3/4 are "**not** imported by
experiments", yet all 86 files remain tracked, unlinted (CI-1) and — for phase4 — carrying the customer
data risk (LEG-3, GOV-3). Neither Flask nor LangChain nor mem0 keeps superseded evaluation code in-tree:
LangChain's mechanism is a deprecation decorator with `since`/`removal` versions followed by removal;
Flask deletes. The inverse problem applies to `phase1`, which **is** load-bearing — 34 experiment modules
plus `tests/test_diagnostic_playbook.py:28`,`:46-47` import it — while being excluded from the wheel
(`pyproject.toml:60`) and from mypy (`:88`). So the repo has 86 files of dead top-level namespace and one
package that is simultaneously a hard dependency and unshippable. Fix: `phase2-4` → the companion repo of
STR-3; `phase1` → either into `search_as_code` (as `search_as_code.harness.surface`, which also fixes
DOC-1) or into the companion repo with the experiments that import it. `STRUCTURE.md`'s "Proposed cleanup"
section already proposes the first half of this and is stalled pending sign-off.

#### 🟨 STR-9 `[C]` No `CONTRIBUTING`, `SECURITY`, `CODE_OF_CONDUCT` or `CITATION.cff`, though the README documents a contribution workflow
mem0 ships `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` and `SECURITY.md`; LangChain ships `CITATION.cff`;
Flask ships `.editorconfig` and issue templates under `.github/`. This repo's `.github/` contains exactly
two workflow files and nothing else (verified: `find .github -type f` → `ci.yml`, `publish.yml`), while
`README.md:214-231` documents a full contributor workflow (git worktrees, `pip install -e '.[dev]'`, the
green-check command) that exists nowhere as a file a contributor would find. For a repo whose primary
output is *research claims* — a headline benchmark, `open_problems.md` with citations — the missing
`CITATION.cff` is the notable one: there is currently no machine-readable way to cite this work.

#### 🟨 STR-10 `[R]` No `Makefile`/task runner, so the "keep it green" command is duplicated in prose and disagrees with CI
mem0 has a root `Makefile`; LangChain has one per lib (verified: `libs/core/Makefile`). Here the same
command is written out twice — `CLAUDE.md:62` and `README.md:222`, both
`ruff check search_as_code && mypy search_as_code && pytest -q` — and **neither matches CI**, which runs
`pytest --cov --cov-report=term-missing` (`.github/workflows/ci.yml`). Three copies of a build command in
two markdown files and a YAML is exactly the drift a `make check` target removes; it would also give
CI-1's widened lint scope a single place to land.

#### 🟨 STR-11 `[C]` `examples/` are two loose scripts that CI never runs; Flask's examples are installable projects with their own tests
`examples/` holds `demo.py` and `opensearch_quickstart.py` (2 tracked files), neither reachable from
`testpaths = ["tests"]` (`pyproject.toml:63`) — if `demo.py` broke, nothing would report it, and the
README leads with it as the zero-setup entry point (`README.md:63`). Flask's `examples/tutorial/` is a
complete standalone project — its own `pyproject.toml`, `LICENSE.txt`, `README.rst`, `flaskr/` and
`tests/` — so the example is exercised as software rather than displayed as text. Compounding EX-1 (the
examples cover only the oldest API, while the one worked example of the current `agentic_solve` flow is
*untracked* at `experiments/deep_judge/example_connect_explore_deploy.py`). Fix: run the examples in CI as
a smoke test, and promote the connect→explore→deploy example into `examples/`.

#### 🟨 STR-12 `[R]` Two changelogs plus a status file, where each comparator has exactly one record of change
`CHANGELOG.md` (282 lines, mixing a task board with a dated log), `benchmark_changelog.md` (131 lines) and
`docs/STATUS.md` (stale, DOC-4) are three overlapping records of "what happened". Flask has one
`CHANGES.rst`, entries grouped under version headings, no task board. The task board in particular is
state that a tracker or `issues.md` should hold — `CHANGELOG.md:58` still reports "All tracked tasks
complete" alongside a "Phase 3 extensions" wish-list dated 2026-07-22. `STRUCTURE.md`'s cleanup section
already proposes folding `benchmark_changelog.md` in; this logs the third file too.

#### 🟨 STR-13 `[C]` `STRUCTURE.md`'s own import count is stale
`STRUCTURE.md:68-69` proposes the `phase1/ → harness/` move and says it must "update the 24
`from phase1 …` imports across `experiments/`". The actual count is **34**:
```bash
grep -rl 'from phase1\|import phase1' experiments --include=*.py | wc -l   # -> 34
```
Minor, but it is the estimate a future agent would size the refactor from, and it under-counts by 42%.

### What this comparison adds to the diagnosis

§13 concluded that the best methodology in this repo sits in the directories marked deprecated or
internal, and that honesty lives in narrative rather than in construction. The structural comparison points
at the same root from the other side: **every one of the three comparators enforces with topology and
tooling what this repo enforces with prose.** LangChain does not ask integrations to honour a contract, it
ships the test suite that fails them (STR-2). mem0 does not ask contributors to keep benchmarks out of the
distribution, it makes the benchmarks a different repository (STR-3). Flask does not ask anyone to check
the wheel, it makes the source un-importable without installing (STR-1). Here the equivalent rules all
exist — in `README.md`, `soul.md`, `CLAUDE.md` — and the audit above is largely a list of the times a rule
was not followed. The four cheapest structural changes, in order of defects-prevented per hour:
**(1)** a wheel-install smoke job in CI (STR-1 → prevents DOC-1's class), **(2)** a parametrized adapter
conformance suite (STR-2 → SDK-C2/C3/C6, TEST-3), **(3)** `.pre-commit-config.yaml` with
`detect-private-key` + `check-added-large-files` (STR-5 → GOV-2), **(4)** a docs link-checker
(STR-6 → DOC-4/5/6).

---

## 15. Found while executing the fix sweep (branch `fix/audit-sweep`, 2026-08-17)

New defects surfaced *by* the fixes — mostly by the two controls that did not exist before (the
adapter conformance suite and the CI gates). Logged per the standing routine.

#### 🟥 CI-3 `[C]` The branch could not pass its own CI: ruff and mypy were both failing before any fix
`CLAUDE.md:62` and `README.md:222` both state the invariant
`ruff check search_as_code && mypy search_as_code && pytest -q`, and `.github/workflows/ci.yml`
runs the first two. Measured on **committed** `feat/deep-sac` in a clean `git worktree` (so this
predates the audit sweep):
```bash
git worktree add --detach /tmp/base HEAD && cd /tmp/base
python3 -m mypy search_as_code        # -> Found 52 errors in 12 files
python3 -m ruff check search_as_code  # -> Found 37 errors
```
So every "keep it green" instruction in the repo was unenforceable, and any PR from this branch
would have gone red on the first CI run. **FIXED** 2026-08-17 `d27542a` (both to 0; `make check`
now runs the same set locally).

#### 🟥 ADP-1 `[C]` `FaissStore.upsert` is not idempotent by id — duplicate vectors, drifting `count()`
`adapters/faiss_store.py:42-52` unconditionally `self.index.add(...)` and extends `self._ids`,
while the composed `MemoryStore` **replaces** the document. Re-upserting a known id therefore
appends a *second* vector: `count()` (which reads `index.ntotal`) drifts away from the true
document count, and the stale vector stays searchable forever. Reproduce:
`store.upsert([d]); store.upsert([d]); assert store.count() == 1` → fails with 2.
Found by `tests/test_conformance.py::test_upsert_is_idempotent_on_id[faiss]`, which is the first
test any of these backends have ever had (TEST-3). **FIXED** 2026-08-17 `e1bd684`.

#### 🟥 ADP-2 `[C]` `delete()` was unimplemented on two shipped backends
`FaissStore` and `SqliteStore` defined no `delete`, so the call fell through to
`adapters/base.py:84` and raised `NotImplementedError` — on two backends the README badge and
`adapters.available()` advertise as shipped. **FIXED** 2026-08-17 `e1bd684`.

#### 🟧 ADP-3 `[C]` Chroma drops `$`-prefixed filter operators — the same fail-open shape as SDK-C2
`adapters/chroma.py:_to_where` skipped any key starting with `$`, so
`search(filter={"$or": [...]})` ran **unfiltered** and over-returned. Identical class to SDK-C2 on
OpenSearch, in a second adapter, and invisible for the same reason: no conformance suite.
**FIXED** 2026-08-17 `e1bd684` (`$and`/`$or` translated; anything unsupported raises).

#### 🟧 EXP-1 `[C]` `eval_fair.py` swallows every worker exception, so a crashed run looks like an empty one
`experiments/multi_hop_synth_queries/eval_fair.py` runs the per-query work as
`list(as_completed([ex.submit(one, r) for r in rows]))` — it never calls `.result()`, so any
exception inside `one()` is discarded silently. A run that fails on every query produces no
traceback and no output; I hit exactly this and had to re-run in the foreground to see the error.
This is LEG-5's shape at the experiment level: measurement cannot distinguish "the arm scored 0"
from "the arm never ran". Fix: `for fut in as_completed(...): fut.result()`, or collect and report
the exception count beside the metrics.

#### 🟧 EXP-2 `[A]` The multi-hop "SAC beats tool-calling" quality margin does not survive prompt matching
Confirms P1-7 and supersedes §4 of `experiments/multi_hop_synth_queries/RESULTS.md`. With one
`STRATEGY_BRIEF` given verbatim to both LLM arms, and `sac.explore`'s forged primitive supplied to
**both** as knowledge *and* capability (n=30/hop, `rows[200:230]`, forge-disjoint):
`sac_explored − tool_explored` = **−0.100 / −0.067 / −0.075** recall@10 at 2/3/4 hops, all within
noise. The published **+0.06 / +0.08 / +0.13** is not reproduced. Written up as
[`RESULTS.md` §4b](experiments/multi_hop_synth_queries/RESULTS.md); raw
`recall_fair_explore5{,_ci,_perquery}.json(l)`.
**What survives:** the cost axis (1 turn vs ~8–9; ~600 vs ~15,000–19,000 input tokens), and
explore itself (it lifts SAC +0.083/+0.056/+0.025 and cuts searches 5.3→3.3) — but explore lifts
the *tool* arm as much or more, so it is a corpus-knowledge win, not a code-mode win.

#### 🟨 EXP-3 `[A]` My own first attempt at the P1-7 fix was wrong — recorded so it is not repeated
Deleting the worked recipe from `CODE_SYS` made the prompts symmetric but removed the **mechanism**:
the SDK's documented workflow is explore-first, so single-shot SAC with no strategy measures
something the SDK never claims. The corrected design seeds *both* harnesses with the forged
artifact. Noting it because "make the arms equal by taking the good thing away from one of them" is
a tempting and wrong way to fix an unfair comparison — the fair version gives it to both.

#### 🟨 EXP-4 `[C]` `sac_explored` ≈ dense on this corpus, which the dense-default gate should catch
`sac_explored − dense` = +0.033 / +0.089 / +0.017 recall@10, all ns. Consistent with the
`qwen8b_sac` finding that SAC's value is inverse to retriever strength, and with why the
dense-default gate exists — but the gate lives in the explore pipeline, not in `eval_fair.py`, so
this comparison does not actually exercise it. Worth running the multi-hop arms *through* the gate.

#### 🟨 GOV-4 `[C]` `.gitignore` listed `phase4/altera_*.py` but 19 `phase4/run_*.sh` wrappers were tracked
Extends P4-3 with the exact count. `.gitignore` only covered `phase4/run_altera*.sh`, leaving
`run_call.sh`, `run_claude_code.sh`, `run_dump.sh`, `run_eval_explore.sh`, `run_explore.sh`,
`run_fit.sh`, `run_hyde.sh` and 12 others tracked, each `source phase4/.secrets`. Untracked in the
sweep. **FIXED** 2026-08-17 `bf4ed25`; `scripts/check_no_customer_artifacts.py --check-tree` is now
the enforcing control and is green.

#### 🟥 DJ-4 `[C]` The SDK ships a judge prompt that is NOT the tuned one its comment claims, and re-running reproduces the untuned score
`search_as_code/harness/diagnostic_judge.py:18` described `DIAGNOSTIC_PROMPT` as the "round-7
critic revision; held-out balanced-acc 0.721". Two checks say otherwise:
```bash
python3 -m experiments.deep_judge.validate_judge --split test   # fresh run of the SHIPPED judge
```
1. The shipped prompt is **2,964 chars** vs the adopted `best_prompt_ce_same.txt` at **2,992**
   (99.5% similar, not identical).
2. Re-running the shipped judge on the held-out 100 reproduces the **round-0** confusion matrix
   *exactly* — `tp=37 tn=33 fp=14 fn=16`, balanced accuracy **0.700 [0.613, 0.789]** — where
   round 7 recorded `tp=37 tn=35 fp=12 fn=16` / 0.721.
Either the tuned revision was never actually shipped, or the 2-example difference is noise —
and DJ-2 already showed the whole claimed gain **is** those 2 examples. Either way the docstring
asserted a number the shipped artifact does not produce. **FIXED** 2026-08-17 (docstring
corrected to the measured value + interval); which of the two explanations holds is still open.

#### 🟥 DJ-5 `[A]` The LLM judge does not beat a plain logistic model on the same signals
Same run. Reference points measured alongside the judge:

| | balanced accuracy |
|---|---|
| always-PASS baseline | 0.500 |
| **shipped LLM judge** | **0.700 [0.613, 0.789]** |
| logistic regression on the same features, 5-fold | **0.722 ± 0.039** |

The judge clearly beats the trivial baseline, but a nine-feature LogReg over the coverage/score
signals it is *shown* does at least as well **without any LLM call**. So "the judge mimics the
oracle at the signal ceiling" is better stated as: *the ceiling is reachable by a cheap
classifier, and the LLM is not adding accuracy over the features it reads.* This does not make
the component useless — its value is the structured **diagnosis** (`DIAGNOSIS` / `TECHNIQUE` /
`NEXT_QUERY`) that steers the next hop, which a classifier does not produce — but the PASS/FAIL
accuracy framing should be dropped, and a LogReg gate is the obvious cheaper stop-controller to
A/B against. Raw: `experiments/deep_judge/judge_validation_test.json`.

#### 🟥 P2-1-RESULT `[A]` The learned-profile lift does not reproduce — leak-free OR in-sample
Re-ran P2-1 properly (mine on `train`, evaluate on a disjoint `test`, deltas with paired
bootstrap CIs) on both FiQA and HotpotQA. **Every delta is zero or within noise on both splits.**
On the HotpotQA held-out split all four deltas are exactly `+0.0000`. The CHANGELOG's
"+2.7 pts all_found from learned synonyms" is **not reproduced**, and since the lift is absent
*in-sample* too, contamination is not the whole explanation — the mined profile appears inert
under this evaluation. Contributing observation: on HotpotQA/test the normalizer changed **0 of
150** queries. Full write-up: [`experiments/learned_profile_leakfree.md`](experiments/learned_profile_leakfree.md).
Fix landed: `phase2/splits.py` + both passes are split-aware and print intervals. **The claim
should be struck from `CHANGELOG.md` and `MULTI_DATASET_REPORT.md`** unless someone reproduces it
under a stated protocol. `align_prompts.calibrate_judge` has the same leakage shape and is still
unaudited.

#### 🟧 EX-3 `[C]` `SynthesizeStage` reported `ok` after generating zero queries, then `validate` crashed
`explore/stages.py` — `_gen_queries` swallows a parse/API failure and returns `[]` per document,
so a generator whose output cannot be parsed produced **0 queries while the stage still recorded
`ok`**; the next stage then died with the confusing `RuntimeError: no synth queries to validate
on`. Same family as SDK-A5 (a `validate()` gate nothing implemented) plus LEG-5 (a silent
fallback recorded as a result). Hit while writing `examples/03_explore_first.py`.
**FIXED** 2026-08-17 — `SynthesizeStage.validate()` rejects an empty/short output with the real
cause, so the failure surfaces at the stage that caused it.

#### 🟧 EX-4 `[C]` The README's headline workflow was not runnable from an install
The "explore first — the default workflow" section pointed only at
`experiments/deep_judge/run_explore_pipeline.py`, which `pyproject.toml` does not ship, and the
`agentic_solve` / `DiagnosticJudge` / `HarnessForge` entry points it leads with were not exported
at the top level — you had to know the submodule path. So a `pip install` user could not run or
easily import the documented headline workflow. **FIXED** 2026-08-17: entry points exported from
`search_as_code`, plus `examples/03_explore_first.py` and `examples/04_harness_judge_forge.py`
(zero-setup, no API key, executed by CI) and a README "Learning the pieces" table.

#### 🟨 DOC-10 `[C]` Two public docs linked to files that were untracked, so the links break on clone
Found by the new `scripts/check_doc_links.py --public`. `README.md` linked to `STRUCTURE.md` and
`open_problems.md`; `open_problems.md` linked to `experiments/explore_learning/README.md`,
`primitive_selection/*`, `deep_sac/*.json` and four figures — none of them tracked. Same class as
DOC-6, five more instances. Resolved by tracking the genuinely-public docs and **de-linking**
`experiments/browsecomp/RESULTS.md`, which GOV-1 says must not be published (a public doc must not
link to a file we refuse to publish). **FIXED** 2026-08-17 `bf4ed25`.

---

## 16. Found while merging `feat/deep-sac` into `main` (2026-08-17)

#### 🟥 MRG-1 `[C]` The `phase2/ → internal/legacy/phase2/` reorg moved files without updating their imports
`internal/legacy/phase2/{beir_qrels,beir_train,learn_rules,impact_eval}.py` still read
`from phase2 import beir` / `from phase2.splits import pick` after main's 32-file rename, while
their siblings (`beir_run.py:21`, `impact_eval.py:18`) had been updated to
`from internal.legacy.phase2 import beir`. So the moved modules were half-repointed and would
`ModuleNotFoundError` at their new home. Nothing caught it because CI lints and type-checks only
`search_as_code/` (CI-1) and `internal/legacy/` has no tests.
```bash
grep -rn '^\s*from phase2' internal/legacy/     # -> 5 hits before the fix
```
**FIXED** 2026-08-17 in the merge commit. **Underlying issue stands:** a rename of 32 tracked
modules with no import check is exactly what `make check` should cover — widen ruff/mypy beyond
`search_as_code/` (CI-1) or the next reorg breaks silently again.

#### 🟧 MRG-2 `[C]` Deleting `benchmark_changelog.md` broke two public README links
Main folded `benchmark_changelog.md` into `CHANGELOG.md` and deleted it, but `README.md` still
linked to it twice — the same defect as DOC-6, reintroduced by the cleanup that was supposed to
tidy it. Caught by `scripts/check_doc_links.py`, which did not exist when the deletion was made.
**FIXED** 2026-08-17 (relinked to `CHANGELOG.md`). Argues for running the link checker on `main`
too, not only on branches that happen to carry it.

#### 🟥 MRG-3 `[A]` GOV-1 is live on `main`, and `.gitignore` cannot fix it
`origin/main` tracks **26** `experiments/{browsecomp,su_multihop}` files, including
`data/su_multihop_{2,3,4}docs.jsonl` (SearchUnify-derived) — the paths `.gitignore` labels
"INTERNAL, do not push". `feat/deep-sac` had already untracked them, so **the merge pulled them
back in**: a `.gitignore` entry does not untrack a tracked file, and a merge from a branch that
still tracks it re-adds it. Demonstrated by the guard failing on a tree whose `.gitignore` lists
those very paths:
```bash
git check-ignore experiments/browsecomp/RESULTS.md    # -> no output: tracked, so not ignored
```
Handled in the merge with `git rm --cached` on all 26 (files remain on disk; history untouched,
per the untrack-going-forward decision). **Still open:** the data remains in `origin/main`'s
history and in every existing clone. Removing it needs a history rewrite and a force-push — a
decision for the repo owner, not a code change.

#### 🟨 MRG-4 `[R]` Two independent lineages built the same harness, and `main` held the older copy of 17 files
The merge base is `b57b0af`; `feat/agentic-explore` (→ main) and `feat/diagnostic-judge` (→
deep-sac) each added `search_as_code/harness/*` separately, producing add/add conflicts on 45
files. For **17** of them main's copy was byte-identical to pre-sweep deep-sac — the same code
minus the audit fixes — meaning `main` had been shipping the GEN-1 (`out[0]`) and SDK-A4 (bare
`\band\b`) defects this whole time. Long-lived parallel feature branches over a shared subsystem
is the root cause; merging to `main` more often would have surfaced it as a small conflict instead
of a 45-file one.

#### 🟥 MRG-5 `[C]` Merge-time fixes were silently dropped: edits made after staging never reached the commit, and the guards passed anyway
During the `deep-sac → main` merge I repaired the reorg's stale imports
(`internal/legacy/phase2/{learn_rules,impact_eval,beir_qrels,beir_train}.py` →
`from internal.legacy.phase2 ...`) and the path references in
`experiments/learned_profile_leakfree.md`. Both edits were made **after** the corresponding
files were staged, so `git commit` during the merge took the index and left them behind. They
never shipped. `main` therefore still had:
```bash
grep -rn 'from phase2' internal/legacy/ --include='*.py'    # -> 5 hits after the merge
python3 -c "from internal.legacy.phase2.splits import pick" # OK — misleading
# the bad imports are INSIDE functions, so they only fail when called:
#   ModuleNotFoundError: No module named 'phase2.splits'
```
Two compounding faults, both worth fixing rather than just noting:
1. **The guards read the working tree, not the index.** `scripts/check_doc_links.py` reported
   "All relative links resolve" on the merge branch because the unstaged sed *was* on disk. A
   check that passes on uncommitted state gives false assurance precisely when it matters.
   **FIXED**: `--staged` now refuses to report on a dirty tree, and `make check` uses it.
2. **Function-level imports hide breakage from import-time checks.** The modules import fine and
   only fail when the function runs, so neither CI (which does not cover `internal/`, CI-1) nor a
   smoke import catches it.
**FIXED** 2026-08-18 on `fix/post-merge-paths`. Cross-references MRG-1, which this entry shows
was not actually fixed by the commit that claimed it.

---

## 17. Found by an external review pass (2026-08-18) — judge validity, forge/loop integrity, primitives correctness

Four parallel review sweeps (SDK primitives, deep judge, explore→forge, open_problems staleness)
over HEAD `3bec64f`. Everything below is new — not restatements of §1–§16. Reproduction commands
were run where stated.

### Deep judge — the corrected numbers are themselves not clean

#### 🟥 DJ-6 `[A]` The tune/test split leaks at the query level: 52 of ~76 test queries were also tuned on
`experiments/deep_judge/validate_judge.py:34-46` (same in `tune_judge.py:53-70`) shuffles the 200
examples stratified **by label only**. Each query contributes two examples (shallow top-5 + deep
top-10) with the same question, sub-facts and gold — so tune and test share **52 queries** (measured:
76 distinct per split, overlap 52). Every "held-out" number in `deep_judge/README.md` §1, including
the corrected 0.700 [0.613, 0.789], is partially in-sample. The bootstrap also resamples examples
i.i.d., ignoring the 2-per-query clustering, so the ±0.09 intervals are *narrower* than the truth.
Fix: split by query id (GroupShuffleSplit), bootstrap by query.

#### 🟥 DJ-7 `[A]` The shipped prompt's CE thresholds were calibrated on all 200 examples — test half included
`experiments/deep_judge/augment_ce.py:37-42` computes min-CE separability over the **full** eval set
(PASS 1.50 vs FAIL −4.04, reproduced), and `search_as_code/harness/diagnostic_judge.py:39-54` bakes
exactly those cut points (0.1 / −0.5 / −1.5) into the prompt ("# Calibrated ce thresholds."). The
"held-out 100" therefore scores a rule whose parameters saw it. Recalibrate on the tune half only.

#### 🟥 DJ-8 `[C]` The DJ-5 LogReg headline 0.722 ± 0.039 is a row-ordering artifact; grouped CV says it *ties* the judge
`validate_judge.py:143` feeds `tune + test` (a list blocked by split and label) to
`cross_val_score(..., cv=5)`, which uses **unshuffled** StratifiedKFold. Re-running the identical
code: tune+test order → 0.7219 ± 0.0390 (matches the published number exactly); natural file order →
0.7052 ± 0.0750. Under `GroupKFold` by query (fixing the shallow/deep pairing leak) → **0.699 ±
0.067**. So the honest DJ-5 statement is "a cheap classifier **ties** the judge", not beats it — and
a fold-std is being displayed next to a bootstrap 95% CI as if comparable.

#### 🟥 DJ-9 `[A]` The judge is behaviorally a one-feature threshold — and the shipped threshold is mis-set
The shipped judge's TEST confusion matrix (`judge_validation_test.json`: tp=37 tn=33 fp=14 fn=16)
is **identical** to the rule the prompt literally states, `min_i ce_i > 0.1`, applied directly to the
same 100 examples (bal-acc 0.700, same matrix). Sliding that one threshold: −0.5 → 0.719, −1.0 →
0.738, **−1.5 → 0.785** (+0.085, larger than every effect §1 of the README discusses; test-tuned, so
an upper bound — but the direction stands). The LLM adds no PASS/FAIL accuracy over the single
feature it is told to threshold, and the threshold it ships is far from optimal. A tuned `min_ce`
gate is the baseline this component must beat; on present evidence it does not.

#### 🟥 DJ-10 `[C]` Production feeds the judge 1/(rank+1) pseudo-scores, making 3 of its 9 signals constant and pinning `buried` on
`harness/agentic.py:214` and `harness/playbook.py:153` construct candidates with
`score = 1.0/(rank+1)`. `score_signals` (`diagnostic_judge.py:92-102`) on that input is constant
regardless of query: top3_ratio=0.611, cliff=0.5 — so the prompt rule "cliff > 0.3 or top3_ratio
< 0.85 ⇒ buried" (`diagnostic_judge.py:67`) fires on **every hop, forever**. In the validation set
those same signals had real spread (top3_ratio mean 0.906, 119–164 distinct values). The judge was
validated on informative signals and deployed on degenerate ones — a plausible cause of in-loop
`stop_correct` 0.467–0.567 sitting far below the offline 0.700.
**FIXED** 2026-08-18 `60858a1` — both call sites now feed `dj.candidate_scores` (sigmoid of whole-query CE logits) instead of 1/(rank+1).

#### 🟥 DJ-11 `[A]` Stage 3's "validate WITHOUT the oracle" leaks gold coverage into the prompt, and the two arms share memory
`run_explore_pipeline.py:118` passes `gold=r["gold_ids"]` with `judge_stop=True`; inside the loop
`agentic.py:228` writes `"... (covered {got}/{len(goldset)})"` to memory as a finding, and
`agentic.py:104,:179` feeds findings back into the strategist prompt — so the true gold-coverage
count is in-prompt every hop of the "no-oracle" arm, biasing judge-stop **upward** (the honest 45%
oracle-recovery on BrowseComp is an upper bound). Both arms also run concurrently over the same
rows while writing cross-query `skill_win`s into one shared `AgentMemory`
(`run_explore_pipeline.py:106,:121-123,:152-158`), so they are not independent. (Plain
`agentic_solve(..., gold=None)` is unaffected — this is experiment-protocol only.)

#### 🟧 DJ-12 `[A]` "The value is the DIAGNOSIS, not PASS/FAIL" is asserted in five places and tested in none
Stated at `diagnostic_judge.py:9,:20-23,:37-38`, `deep_judge/README.md:72-77`, `README.md:188`,
all pointing at §2 (n=30, no CIs, sign flips: HotpotQA global 0.467 vs diagnostic 0.433, SU 0.333
vs 0.533). The comparison is confounded: the diagnostic arm also gets a different assembly
(`allocate_reserve`, `run_playbook.py:150-153`) which the README separately credits for the
multi-hop fix. The one arm that isolates targeting (`widen`, same reserve assembly, untargeted —
`run_playbook.py:172-179`) ran only at n=12 and shows **diagnostic = global** (`playbook_4hop.json`:
0.333 vs 0.333). The load-bearing claim for keeping the LLM in the loop is currently untested.

#### 🟧 DJ-13 `[C]` The 0.72-ceiling retraction did not propagate: three tree locations still assert it, and README contradicts itself
`README.md:113` ("validated to the ~0.72 signal ceiling") vs `README.md:188` (the correction), 75
lines apart; `run_explore_pipeline.py:6` and `:103` still say "tuned to the ~0.72 signal ceiling".
`README.md:114` ("confirm its recall matches the oracle-stopped run") is contradicted by the repo's
own `explore_pipeline_browsecomp.json` (judge-stop 0.054 vs oracle 0.119 = 45%). Also: the
"supervised ceiling 0.725" row in `deep_judge/README.md:24` has **no producing artifact** anywhere
in the repo.
**FIXED** 2026-08-18 `60858a1` — README pipeline steps 2-3 and run_explore_pipeline.py:6/:103 restated to the measured 0.700 [0.613, 0.789] + honest judge-vs-oracle gaps. The unsourced 'supervised ceiling 0.725' row still stands in deep_judge/README.md.

#### 🟨 DJ-14 `[C]` The fresh validation renders a different input format than the shipped judge, and the critic saw an inverted verdict
`validate_judge.py:49-57` defines its own user-message format, differing from the production
renderer (`diagnostic_judge.py:121-131`) and the tuning renderer (`judge_core.py:52-65`) — so it
validates the shipped system prompt against a non-shipped input format. Separately
`tune_judge.py:94` renders `judge said VERDICT={e['oracle_pass'] and 'FAIL' or 'PASS'}` — inverted —
two lines above the correct `ORACLE TRUTH:` line, so the critic LLM tuned against self-contradictory
transcripts.

#### 🟨 DJ-15 `[C]` DJ-4 resolved: the tuned round-7 prompt was simply never shipped
Diff of `diagnostic_judge.py`'s prompt vs `best_prompt_ce_same.txt`: 2,964 vs 2,992 chars, exactly
two line differences (a typographic apostrophe; a missing clause "or support borderline cases").
Reproducing round-0's confusion matrix is therefore expected, not mysterious. Closes DJ-4's open
question; per DJ-2 the 2-example difference is noise either way.

### Explore→forge — the loop does not close, and its honesty mechanism fell out of the tree

#### 🟥 FRG-1 `[A]` The dense-default gate is absent from HEAD; the README asserts a guarantee no code enforces
Commit `e0b1d89` added the gate (`DENSE_CODE` / `stage4_selected` / `dense_held` in
`run_explore_pipeline.py`), and `git merge-base --is-ancestor e0b1d89 HEAD` confirms ancestry — but
`grep -n "dense_held\|DENSE_CODE\|stage4_selected" experiments/deep_judge/run_explore_pipeline.py`
on HEAD `3bec64f` → no hits: the deep-sac→main merge restored the older lineage's copy (MRG-4/MRG-5
class, previously unlogged instance). HEAD's stage 4 accepts on `mean > 0.0`
(`run_explore_pipeline.py:73`). Meanwhile `README.md:139-142` ("So SAC never underperforms dense"),
`README.md:194`, and `qwen8b_sac/README.md:73-83` present the gate as standing, and the v2/v3
qwen8b artifacts are not reproducible from HEAD. A restoration is in progress on
`fix/dense-default-gate` (uncommitted at review time).

#### 🟥 FRG-2 `[A]` "Structure-emergent" is largely predetermined by the harness, and structure is attributed to the wrong hop
Three compounding causes: (1) `os_first=True` (default, `agentic.py:144`) forces hop 1 to raw-OS
DSL with a deterministic fallback (`agentic.py:186,:73-100`), and the structure classifier reads
**`codes[0]` only** (`run_explore_pipeline.py:133`) via a regex for `re.split|split(|decompose` — so
on any OpenSearch corpus "whole-query" is near-guaranteed regardless of what won
(`qwen8b_sac/explore_pipeline_browsecomp_qwen8b_v3.json`: `"decomposed": "0/50"`); (2) recall may
come from hop 4 while hop 1's code gets the credit; (3) `AUTHOR_SYSTEM` itself instructs "DEFAULT =
DENSE … keep the query WHOLE … Do NOT decompose by default" (`agentic.py:40,:53-57`) — the prompt
dictates the structure the pipeline reports as discovered. The `decomposed=` flag (`agentic.py:235`)
also only tests for `re.split`/`split(` and mislabels manual decomposition. The one genuinely
emergent contrast in `deep_judge/README.md:192-200` is n=3 vs n=5. Fix: classify from the winning
hop, under a structure-neutral author prompt.

#### 🟧 FRG-3 `[C]` Forge acceptance bars are vacuous-to-thin, the forged "skill" is not derived from the winning code, and the subagent is never executed
`forge_from_exploration`'s `min_recall` defaults to **0.0** (`run_explore_pipeline.py:51,:73`); the
SDK path accepts on one held query with ≥1 gold found (`forge.py:279-285`); the only dense-relative
bar accepts up to 10% *worse* than dense (`reforge_and_full.py:82`, `>= 0.9 * dense_held`). The
forged `LearnedSkill` is picked from two **hardcoded** retriever lists by a boolean
(`run_explore_pipeline.py:172-173`) — the discovered structure is flattened into an unordered bag of
RRF pools where "rerank" is a peer pool, not a final stage (`forge.py:50-62`, `skills.py:198-204`).
`LearnedSubagent` is created, saved, loaded — and consulted by **no runtime path**.

#### 🟧 FRG-4 `[C]` Forge artifacts carry no provenance, and `learnings.md` feeds the loop contradictory instructions
`HarnessStore.save` (`forge.py:183-190`) records no timestamp, held-set metric, corpus fingerprint,
source strategies, or version; `create_skill` silently overwrites by name (`forge.py:215-222`).
`learnings.md` dedups exact strings only (`forge.py:242-246`), so
`forge_store_browsecomp_explored/learnings.md` simultaneously asserts "structure = decompose (1/2)",
"decompose (6/6)" and "whole-query (39/274)", and `learnings_block` injects the last 12 rules into
prompts (`forge.py:195-197`). The artifacts on disk are already mutually inconsistent:
`explore_pipeline_browsecomp.json` records `forged_code: null` / stage5 = stage7 = 0 (the pipeline's
forge **failed**) while the store beside it holds a primitive written later by `reforge_and_full.py`
— and `deep_judge/README.md:236-241` narrates stage 4 as if the pipeline forged it, disclosing the
substitution only in a footnote.

#### 🟧 FRG-5 `[C]` The multi-hop "unseeded" control is contaminated, and forge execution swallows errors at load and at run
`eval_fair.py:191-193` puts `forged` in `TOOL_SCHEMAS` unconditionally — described to the model as
"the primitive sac.explore FORGED on this corpus" — and for unseeded arms it silently degrades to
hybrid (`eval_fair.py:97-98`), so baselines are told a forged primitive exists and get an extra
retrieval mode. `HarnessForge.__init__` `exec`s every persisted primitive at construction
(`forge.py:209-213`) in a porous namespace (`forge.py:68-101`: `getattr`, an `__import__` shim, the
live session); `CodePrimitive.to_skill` swallows every exception and returns `[]`
(`forge.py:121-128`), as do pipeline stages 5/7 (`run_explore_pipeline.py:186-189,:198-201`) — so
"broken primitive" and "primitive found nothing" record the same number (LEG-5's shape, again).

### Primitives layer — correctness defects invisible because the layer has almost no tests

#### 🟧 SDK-C15 `[C]` `mmr` never writes MMR scores back, so `.top()` silently undoes the diversification
`primitives.py:342-344` returns hits with **original** scores; `ResultSet.top()` re-sorts by score
(`types.py:113`). Verified: MMR order ['aligned_lowscore','offaxis_highscore'] → after `.top(2)`
reversed. `surface.py:49` teaches exactly this chain to the model.
**FIXED** 2026-08-18 `60858a1` — mmr writes the strictly-decreasing MMR objective back as the score; `tests/test_primitives.py::test_mmr_order_survives_top`.

#### 🟧 SDK-C16 `[C]` The SDK-C13 `.info` fix is half-done: signals still die on the exact path the prompt teaches
`types.py:64-67` claims side signals propagate across chained calls, but only `_derive`-based
methods (`top`/`where`/`dedup`) carry them; `rerank`, `fuse`, `normalize_scores` construct fresh
ResultSets and drop them. Verified: `consensus().agreement` = 1.0 → after `rerank()` = 0.0.
`surface.py:215-216` instructs the model to remember `cons.agreement` and then `sac.rerank(...)`.
**FIXED** 2026-08-18 `60858a1` — `.info` now survives fuse/relative_score_fusion/consensus (merged, degraded summed) and rerank/normalize_scores/score_cutoff/diversity_quota/freshness/mmr (passed through); test `test_info_survives_every_chaining_primitive`.

#### 🟧 SDK-C17 `[C]` `normalize_scores` maps singleton/tied lists to 0.0, collapsing `relative_score_fusion`
`primitives.py:113` (`rng = (hi-lo) or 1.0`) gives a single-hit list score 0.0, so
`relative_score_fusion` of two singleton lists returns everything tied at 0.0 — total ranking
collapse in a primitive documented as "often beats RRF" (`primitives.py:121-123`). Map singletons
to 1.0 (or preserve rank by epsilon).
**FIXED** 2026-08-18 `60858a1` — singleton/all-tied lists normalize to 1.0; regression test on relative_score_fusion of singletons.

#### 🟧 SDK-C18 `[A]` Keyword/regex emulation is not behavior-preserving, and `hybrid` exists four times with two pool sizes
`Session._regex` (`session.py:457-467`) **embeds the regex pattern as a dense query** and scans only
the top max(top_k·20, 200) dense hits — a doc matching at dense rank 5,000 is invisible; regex's
contract is exhaustive exact match. `Session._keyword` (`session.py:442-447`) emulates BM25 with
`|q∩d|/|q|` overlap (no IDF, no length norm) over a dense pool. `hybrid` is implemented in
`memory.py:148-153`, `faiss_store.py:113-117`, `nmslib_store.py:92-96` (byte-identical, top_k·4)
and a fourth time in `session.py:449-455` with **top_k·3** — so native and emulated hybrid provably
return different sets, against the README's "`mode="hybrid"` behaves the same everywhere". Also:
`adapters/memory.py:151`, `faiss_store.py:116`, `nmslib_store.py:95` import `primitives.fuse`
upward with a "avoid cycle" deferred import — the adapter layer reaching into the layer above it.

#### 🟧 SDK-C19 `[C]` The sandbox test asserts an escape is blocked that falls to a one-liner; no timeout/output cap; namespace poisoning persists across hops
`tests/test_units.py:105-112` asserts open/import are blocked, but
`Document.__init__.__globals__['__builtins__']` returns full builtins and
`().__class__.__base__.__subclasses__()` walks out (both verified ok=True) — the classes injected at
`sandbox.py:66-68` hand back `__globals__`. `sandbox.py:8-10` honestly disclaims the boundary; the
test manufactures false confidence in the disclaimed property. Robustness independent of security:
no timeout/memory/recursion/output cap (`sandbox.py:96-97`; truncation is cosmetic, `:41`); the
namespace is built once and only `evidence` is reset per run (`sandbox.py:60,:93`) so `fuse = None`
in hop 1 poisons all later hops; `surface.py:19,:157` promise `query` in scope but
`_build_namespace` never injects it (`phase1/agents.py:165` patches the private `box._globals` from
outside); 9 of 26 primitives are missing from the namespace while `expand`/`decompose` are injected
with an arity the model can't call (`sandbox.py:76-77` — `expand(query)` → TypeError).
**PARTIALLY FIXED** 2026-08-18 `60858a1` — timeout (per-thread trace hook), stdout cap, per-run rebinding of injected names, `query` injection, full primitive set with generator-bound expand/decompose; the misleading escape test replaced with honest robustness tests. Still open: real isolation backend (by design), C-level-call overrun.

#### 🟨 SDK-R8 `[R]` Dead audit-fix APIs and unread capability fields
`ResultSet.mark_degraded`/`.degraded` (`types.py:96-110`) — the LEG-5 fix — have **zero callers**
repo-wide while the 61 bare `except Exception` sites it was built for still swallow silently, and
the same idea is separately implemented on `ctx.degraded` (`explore/templates.py:66,:76-78`).
`Capabilities.multi_vector` is never read; `native_rerank` is set by 7 adapters and consumed by
nobody; `max_top_k` is guarded (`session.py:44`) but **no adapter sets it** — OpenSearch's hard
10,000 `max_result_window` therefore surfaces as a raw backend error instead of the typed one.
`quality_filter` has zero callsites. `surface.py`'s own docstring says it was moved into the package
so pip users get the prompt surface, but `__init__.py` never imports it — `sac.SAC_SYSTEM` →
AttributeError.

#### 🟧 TEST-5 `[C]` 21 of 26 primitives have zero tests; conformance covers 5 of 9 adapters despite its own docstring; no `py.typed`
Grep of every primitive name across `tests/`: only freshness/mmr/fuse/extract/decompose/dedup appear;
zero tests for consensus, score_cutoff, normalize_scores, relative_score_fusion, diversity_quota,
confidence, abstain, rerank, fan_out and 12 more — which is why SDK-C15/C16/C17 survived the audit
sweep. `test_conformance.py:6-7` cites nmslib/milvus "zero tests" as motivation but `BACKENDS`
(`:66-72`) covers only memory/sqlite/faiss/chroma/opensearch — qdrant, pgvector, nmslib, milvus
still have none. `search_as_code/py.typed` does not exist, so the wheel ships no type information
despite the mypy CI gate. `tests/test_diagnostic_playbook.py` loads a real model in `_make()`
(`:43-56`) and hangs >60 s in the default suite; it and `test_genutil.py` import `phase1`, which
the wheel does not ship.

### open_problems.md — staleness audit (the file has never been substantively edited)
**PARTIALLY FIXED** 2026-08-18 `60858a1` — `tests/test_primitives.py` (33 tests) covers the whole primitives layer; `py.typed` ships (PEP 561). Still open: conformance for qdrant/pgvector/nmslib/milvus; the hanging playbook test.

#### 🟨 OPM-1 `[A]` open_problems.md status lines predate the 08-13→08-18 work; three of eight are now wrong or misleading
The file's only commit is `bf4ed25` (tracking/de-linking). Current reality per item: **#3** is DONE
in the trainer (`training.py:544-595`, realized_recall + headroom — supersedes SDK-A2) but
`explore/engine.py:207-212` still writes only `cv_acc`/`vs_fixed` into the pack manifest and
`docs/EXPLORE.md:40,:61-62` still headline CV accuracy; **#6** is half-obsolete — the DiagnosticJudge
*is* the built, validated stop (judge-vs-oracle measured on three corpora), and the proposed
value-gate was **measured** (DJ-5's LogReg) but never built into the loop; **#2** — triage
(`triage.py:61,:108`) is a rule-based **2-value** depth (single|multi), not the proposed 3-class
router, and no experiment evaluates it (cost saved / recall vs always-deep unmeasured); **#1/#8** —
class rebalancing and the orthogonal-template redesign were never run (no
`class_weight|focal|rebalanc` outside `validate_judge.py:117`), though the SDK-A1 availability-gate
fix removed one mechanical confound; agentic_solve supersedes the template router on the recommended
path while `docs/EXPLORE.md` still sends users to the 16-way router with no cross-reference; **#5**
— agentic_solve's pool accumulation (`agentic.py:196-198`) removes the overwrite mechanism but no
deep-vs-one-shot ablation was re-run, and P1-2's asymmetry caveat applies to the old monotone
numbers; **#4** holds and is still the shipped default (`playbook.py:32-39`); **#7** improved
modestly (BrowseComp all-golds 0.029→0.048 at gte-base) but the lever is retriever strength, and
FRG-1 currently unenforces the forged==dense guarantee. Fix: add dated status annotations per item
(append-style, like this file), and cross-link `docs/EXPLORE.md` ↔ `agentic_solve`.

---

## 17. Reproducing explore + forge on `main` (2026-08-18)

Ran the documented default pipeline end to end on HotpotQA to confirm `main` works:
`python -m experiments.deep_judge.run_explore_pipeline hotpot 8 6 12 4 4`.
Exploration reproduced (recall@20 0.875–0.969 across runs), but the forge and the reported
headline did not.

#### 🟥 EXP-5 `[A]` The README's "dense-default gate" guarantee is not implemented anywhere
`README.md:139` states: *"a **dense-default selection gate** adopts the forged primitive only if
it beats plain dense on held queries; otherwise it emits `session.search(mode='dense')`. So SAC
never underperforms dense."*
```bash
grep -rn "dense-default\|dense_default" --include='*.py' search_as_code/ experiments/   # -> no matches
```
There is no such gate. In `run_explore_pipeline.py`, `prim = reg.get(name)` is only populated
when the forge is accepted; on rejection `prim is None`, stages 5 and 7 skip every query, and
`round(float(np.mean(x)) if x else 0, 3)` reports **0** — so the pipeline's headline was
`stage7_test_recall@20: 0` when the truth was "no primitive was produced, and nothing fell back
to dense". First observed run: stage1 explore 0.875, stage7 **0.000**.
**FIXED** 2026-08-18 — stage 4b implements the gate (forged vs dense on the held set, strict `>`
so a tie goes to the cheaper arm), `_run_selected()` deploys the winner, and the report now
carries `selected_strategy` / `gate_forged_on_held@20` / `gate_dense_on_held@20` /
`forge_accepted`, with `None` rather than `0` when a stage did not run. Post-fix on the same
inputs: gate `forged=0.000 vs dense=0.850 -> dense`, stage7 recall@10 **0.729** / @20 **0.750**.

#### 🟥 EXP-6 `[A]` The forge was trained on the FORCED first hop, so its primitive scored 0.000 every time
`run_explore_pipeline.py` captured its exemplar as `best = res["codes"][0]`, commented "FIRST hop
= the initial structural choice". But `agentic.py:186-187` **forces** hop 1 to a raw OpenSearch
DSL query whenever `os_first=True` (the default, and one of the three advertised explore
guarantees). So every exemplar handed to `forge_from_exploration` was the deliberately-naive
opening move, not the strategy in play when gold coverage was reached. The synthesized primitive
faithfully reproduced that opening move and returned ~nothing:
`[stage4] forge validation mean recall@20 over 5 held = 0.000 · accepted=False · code_primitives=[]`.
Confirmed by construction: feeding `forge_from_exploration` hand-written exemplars of the
*fused* shape (raw DSL + dense, `fuse_ids`) yields `ok=True, mean_held_recall=1.000` on the same
corpus and session — the forge itself was never broken, only its training signal.
**FIXED** 2026-08-18 — structure detection still reads hop 1 (that genuinely is the structural
choice), but the code exemplar now comes from the last hop. Post-fix:
`forge validation 0.850 · accepted=True · code_primitives=['hotpot_explored_primitive']`.

#### 🟨 EXP-7 `[C]` `pip install` and the packaging surface verified clean on `main`
Recorded as a positive so a later change that breaks it is visibly a regression. Wheel builds
(49 modules), installs into a clean venv, and the README quickstart runs; every declared extra
resolves (`core / opensearch / qdrant / chroma / faiss / learn / dev / all`); a real
`pip install 'search-as-code[learn]'` brings sklearn and exposes the top-level entry points
(`agentic_solve`, `explore`, `Harness`, `DiagnosticJudge`, `HarnessForge`, `triage`,
`bootstrap_ci`); and `search_as_code/surface.py` is present in the wheel, so DOC-1 stays fixed.

#### 🟥 AGT-1 `[C]` The author prompt commanded raw OpenSearch DSL on every backend — agentic explore scored 0.075 vs dense 0.800 on the memory-backed SU corpus
Found by the first clean pipeline chain (2026-08-18, `ws1_pipeline_su.log`): stage-1 explore
recall@20 = **0.075** while plain dense on the same train rows = **0.800** — the whole agentic
loop was near-zero on SU. Mechanism: hop 1's forced raw-OS probe was correctly guarded
(`agentic.py:186` `hasattr(session.store, "_search")`), but the static `AUTHOR_SYSTEM` prompt
told the model on EVERY hop to call `session.store._search(body)` (with a worked example), and
the judge's diagnosis hint pushed the same escalation — so on `MemoryStore` (no `_search`)
every authored program raised `AttributeError`, `_exec` returned `[]`, and 10 hops burned for
nothing. The dense-default gate absorbed the damage (stage 7 shipped dense: 0.669/0.787), which
is the gate doing its job — but the loop itself was broken on exactly the backend-portability
the product claims. Same class as SDK-A7 (authoring without introspecting what the backend
supports).
**FIXED** 2026-08-18 — `build_author_system(session)`: the raw-DSL call list, worked example,
and diagnosis hint are included only when the store has `_search`; portable backends get a
`mode='keyword'` exact-term escalation instead. Regression test asserts the memory-backend
prompt contains no `_search`. **Confirmed by SU re-run** (`ws1_pipeline_su_agt1fix.log`):
explore recall@20 0.075 → **0.658**, judge-stop 0.675 vs oracle-stop 0.713 (95%), gate still
correctly selects dense (0.800 tie).

#### 🟥 COST-1 `[C]` The cost benchmark's `forged_skill` broke Tools' contract — every seeded forged() call failed silently, so the published seeded rows measured guidance-text lift only
`experiments/cost_tokens/run_cost.py` passed a bare lambda as `forged_skill`, but
`eval_fair.Tools.forged` calls `self.forged_skill.run(session, query, top_k)` — a method.
Every `forged()` call in the seeded arms raised `AttributeError` inside `_retrieve`'s
swallow-all try (LEG-5's shape, again), returning empty results with no trace. Consequence:
the BrowseComp 5-arm "final" table (sac_explored 0.162) measured the guidance TEXT's effect
only — the forged primitive never executed. Caught by the v2 sanity gate (dense=1.00 while
every seeded arm=0.00 on qid 331). Root cause of the miss: the seed smoke tested `forged_fn`
DIRECTLY instead of through `Tools.forged` — the same class of error as DJ-14 (validate
through the interface production uses).
**FIXED** 2026-08-18 — the loader returns the Skill object; the contract is now smoke-tested
through `Tools.forged` itself; v3 chain re-running all numbers.

#### 🟥 COST-2 `[C]` Plain RRF let escalation noise evict a vetted hop-0 gold — judge false-FAIL turned dense recall 1.00 into 0.00
`sac_product` fused hop-0 (the gate-vetted baseline, 50 ids) with up to 5 escalation pools by
unweighted RRF: a gold at dense rank ~8 (1/69) is outranked by every escalation-pool head
(1/61+), so on a judge false-FAIL the fused top-10 dropped the gold the baseline had already
found. Observed on qid 331 (dense 1.00, sac_product 0.00, judge FAILed to the 5-hop cap).
**FIXED** 2026-08-18 — weighted RRF: the vetted baseline pool gets weight 2.0 vs 1.0 for
escalation pools, so escalation can add coverage but cannot evict what the gate already
vetted. The judge's BrowseComp false-FAIL rate itself remains the WS2 item.

#### 🟨 LOG-1 `[C]` Two sections share the number `## 17.`, breaking the append-only log's addressing
`issues.md:1648` (`## 17. Found by an external review pass (2026-08-18) …`) and `issues.md:1891`
(`## 17. Reproducing explore + forge on `main` (2026-08-18)`) were appended independently on the
same day and collide:
```bash
grep -nE '^## 17\.' issues.md    # -> two hits
```
Harmless to content, but the log is referenced by section number throughout (`§7`, `§13`, `§14`
supersede one another), so a duplicate makes "see §17" ambiguous and will keep drifting as more
sections land. The append-only rule forbids renumbering an existing section, so the fix is
forward-only: **the next section is §19**, and cross-references should cite the entry tag
(`EXP-5`, `DJ-14`) rather than the section number. Recording it so the collision is deliberate
and documented rather than silently confusing.

#### 🟥 PROD-1 `[A]` The "never underperform the baseline" guarantee holds at the ARTIFACT level but leaks at runtime — measured 0.234 vs dense 0.257
The gate guarantee is real for what forge ships (`accept_code_primitive` persists max(dense,
hybrid) or better — on BrowseComp qwen8b the shipped primitive IS dense, 0.257 by construction).
The runtime loop leaks it three ways, measured on the v3 BrowseComp qwen8b leg (n=100):
1. **Judge false-FAILs fuse escalation into solved queries** — escalation rate 1.00 under the
   coverage premise, including the ~26% of queries dense already solved at hop 0.
2. **Weighted RRF (2× base) protects membership, not internal order**: an escalation hop that
   RE-FINDS base's rank-30 doc lifts it (2/91 + 1/61 = 0.038) above base's rank-8 gold (2/69 =
   0.029) — the baseline's tail overtakes its head and the gold exits the fused top-10. This is
   the residual 0.234-vs-0.257 gap in `sac_product` (`experiments/cost_tokens/run_cost.py`
   rrf_ids base_weight=2.0).
3. **`sac_explored`'s authored program may add pools around `forged()`** (its guidance permits a
   keyword escalation), reintroducing dilution the gate had eliminated (0.213).
Fixes: (a) the sufficiency premise (feat/ws2-stopgate) removes fusion from solved queries —
PASS returns base verbatim, and FAILed queries have ~0 base recall to lose; (b) a construction-
level floor guard on the FAIL path: candidates in base top-k scoring above the tuned min-CE
threshold cannot be evicted by fusion; (c) claim wording: the guarantee is UNCONDITIONAL at the
artifact level and JUDGE-CONDITIONAL in the loop — the v1 README's "never ships a strategy that
underperforms" is accurate; looser paraphrases ("never underperforms dense", incl. in chat/docs)
must not be used until (a)+(b) are measured. Re-test on browsecomp_qwen8b is queued behind the
v3 chain.
