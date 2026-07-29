# The Exploration Phase — `sac.explore`

A one-time **onboarding step** you run when SAC is installed against a corpus. It studies
the data and writes a versioned **ProfilePack** that a `Session` loads at query time to tune
retrieval to the data — new/optimized primitives, a learned router, a domain ontology,
data-derived prompts and few-shots.

```python
import search_as_code as sac
s = sac.Session("opensearch", index="docs", hosts=[...], embedder=emb, generator=llm)
pack = sac.explore(s, out="docs_pack/")     # run once (and on data drift)
print(pack.report())
# later:  Session(..., profile="docs_pack/")   # loads the pack   (wiring: TODO)
```

## Why
On a heterogeneous corpus, the right retrieval strategy depends on the data shape (exact/regex
for part-numbers & fact-cards, dense/hyde for prose, fielded/phrase for structured fields). The
explorer *discovers* that shape and bakes the decision into an artifact, instead of every query
re-deriving it.

## The pipeline (7 stages)
| # | stage | status | produces |
|---|---|---|---|
| 1 | `sample` | ✅ done | stratified sample (cluster the corpus, keep reps per cluster) |
| 2 | `profile` | ✅ done | schema + content-type mix + **LLM characterization** (overall + per cluster) |
| 3 | `ontology` | planned | LLM-induced entities/relations/synonyms, **web-enriched + reviewed** |
| 4 | `crossdoc` | planned | document links from the ontology (entity co-occurrence / KG edges) |
| 5 | `synthesize` | planned | stratified easy/med/hard queries grounded in sample+ontology (no test leakage) |
| 6 | `router` | planned | combo-exploration → **XGBoost** primitive router (query+profile → best combo) |
| 7 | `codegen` | planned | (situation→chain) templates + (query→code) few-shots + **sandbox-validated new primitives** + prompt overrides |
| — | `validate` | planned | held-out retrieval eval; **keep only tunings that beat baseline**; `REPORT.md` |

## The template router — `explore.fit(...)`
`sac.explore(...)` returns an **`Explorer`** (it delegates to the pack, and adds `.fit()`):

```python
explorer = sac.explore(session, out="pack/")     # sample/profile/synthesize/validate
metrics  = explorer.fit(n=5000)                   # label queries -> train router
print(metrics["cv_accuracy"], metrics["router_lift_over_fixed"])
tmpl = explorer.route("part number for AGFC019")  # predicted template for a query
```

**20 templates** (`sac.TEMPLATE_NAMES`) are named recipes composed from primitives —
`dense`, `keyword`, `hybrid[_dense_heavy|_kw_heavy]`, `hyde[_dense|_hybrid]`,
`decompose[_dense]`, `*_rerank`, `all_rrf`/`all_rerank`, and code-oriented `exact` / `regex` /
`exact_dense` / `code_fusion` / `code_rerank`.

**How `fit` works:**
1. **Collect queries** — either the `queries=[{"query","gold_id"}, ...]` you pass, or ~`n`
   **grounded synthetic** queries (each + `rephrases` paraphrases) generated from the corpus.
2. **Label** — compute the base pools *once* per query (dense/keyword/hyde/decompose/exact/regex),
   then every template is a cheap recombination; the label is the template that ranks the query's
   `gold_id` highest (`none` if no template finds it).
3. **Featurize** — query embedding + lexical signals (length, #part-numbers, has-digit, question-ish…).
4. **Train** — `HistGradientBoosting` (XGB-style) with cross-validation.
5. **Report** — `cv_accuracy`, `oracle_coverage` (any template finds gold), `best_single_template_acc`
   (always-pick-one baseline) and **`router_lift_over_fixed`** (does routing beat a fixed template?).

Writes `router.pkl`, `router_labels.jsonl`, `router_meta.json`. Labeling knobs: `label_llm`
(hyde/decompose pools — an LLM call per query at label time, default off) and `label_rerank`
(cross-encoder pass, default off) trade fidelity for speed at scale.

## ProfilePack (the artifact)
A directory + `manifest.json`. Each stage writes its own file(s); the manifest records every
stage's `status` (`ok` / `rejected` / `error` / `planned` / `skipped`), timing, and a summary.

Key API (`search_as_code.explore.ProfilePack`): `open(path)`, `read_json`/`write_json`,
`read_jsonl`/`write_jsonl`, `is_done(stage)`, `report()`.

## Robustness (baked into the engine)
- **Resumable** — a done stage (artifacts present) is skipped on re-run unless `force=True`.
- **Drift-aware** — the manifest stores a corpus *fingerprint*; when the data changes, stages re-run.
- **Validate-before-keep** — a stage's `validate()` gate can `reject` output that doesn't beat
  baseline; rejected artifacts aren't trusted downstream (the honesty rule).
- **Fault-isolated** — one stage's exception is recorded as `error` and never aborts the pipeline.
- **Dependency-checked** — a stage with unmet `requires` is `skipped` with the reason logged.

## Config knobs (`explore(..., config={...})`)
`pool_size` (random pool to cluster, default 200), `n_clusters` (default ≤8), `per_cluster`
(reps kept per cluster, default 3), `llm` (LLM profiling on/off, default: on iff a generator is
attached), `seed`.

## Extending
Subclass `search_as_code.explore.Stage` (`name`, `produces`, `requires`, `run(ctx)`, optional
`validate(ctx, summary)`), then pass your list to `explore(..., stages=[...])` or edit
`default_pipeline()`. `ctx` exposes `session`, `store`, `embedder`, `generator`, `pack`, `cfg()`.
