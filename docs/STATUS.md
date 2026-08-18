# Project status

> **This file is no longer hand-maintained.** It used to carry a progress bar chart and a
> "Services right now" section, and drifted badly: it reported "Database adapters (5) 80%",
> "65 tests passing" and "Learned components 0% — designed" against a reality of 9 registered
> backends, 199 unit tests, and an implemented router (`explore/training.py`). A status file
> that tracks session-scoped service state cannot stay true — issues.md DOC-4.
>
> Anything that can go stale now lives where it is generated or enforced:

| you want | look at | kept true by |
|---|---|---|
| what is built / changed | [`../CHANGELOG.md`](../CHANGELOG.md) | updated per work session (`soul.md` rule 3) |
| what is broken | [`../issues.md`](../issues.md) | the standing audit routine in `CLAUDE.md` |
| what is hard / unsolved | [`../open_problems.md`](../open_problems.md) | 8 dead-ends mapped to the literature |
| which backend supports what | [`DATABASES.md`](DATABASES.md) | `tests/test_conformance.py` runs the contract against every installed backend |
| how many tests pass | `make test` | the number is measured, not written down |
| does the published package work | `make wheel` | builds the wheel, installs it clean, runs the README quickstart |
| where anything lives | [`../STRUCTURE.md`](../STRUCTURE.md) | the repo map |

## Current shape, in one paragraph

The SDK (`search_as_code/`) ships the primitive API, 9 registered backends behind one
`VectorStore` protocol with capability emulation, the sandbox, the exploration/router
subsystem, and the agentic harness (`agentic_solve`, `DiagnosticJudge`, the forge). The
experiments under `experiments/` are self-contained studies; `phase1/` is the benchmark harness
they import. The honest summary of what the numbers do and do not support is
[`../open_problems.md`](../open_problems.md) plus the Tier-1 section of
[`../issues.md`](../issues.md) — read those before quoting any headline figure.
