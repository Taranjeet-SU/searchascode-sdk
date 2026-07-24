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
