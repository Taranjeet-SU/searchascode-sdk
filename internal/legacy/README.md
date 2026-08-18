# internal/legacy — archived eval phases

`phase2/` (BEIR/qrels benchmark + learned-rules) and `phase3/` (cross-DB relevance) were earlier
evaluation phases, superseded by `experiments/`. Kept for provenance. Internal imports were rewritten
to `internal.legacy.phase2` / `internal.legacy.phase3`. Not used by the SDK or any experiment.
