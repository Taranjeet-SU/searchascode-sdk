# Repository map

This repo couples **a shippable SDK**, **research experiments**, and **agent-assisted learnings** in
one place (the pattern used by ML research monorepos — a shipped library + an `experiments/` section
of trials + captured learnings; see [TR-Labs](https://medium.com/tr-labs-ml-engineering-blog/mono-repos-for-model-delivery-pros-and-cons-11c555405ce5),
[Tweag Python monorepo](https://www.tweag.io/blog/2023-04-04-python-monorepo-1/)). Four buckets:

## 1. 📦 The standard package (shippable SDK)
The generalizable, PyPI-published product. Nothing corpus- or customer-specific lives here.

| path | what |
|---|---|
| `search_as_code/` | the SDK — `Session`, primitives, adapters, `explore/` (router/labeling), rerankers |
| `tests/` | package tests |
| `docs/` | package + primitive/template docs (`docs/TEMPLATES.md`, `docs/RESEARCH.md`) |
| `examples/` | runnable usage examples |
| `pyproject.toml` | single build/config at root |
| `README.md` | package front page (install + quickstart) |

## 2. 🔬 Experiments (research)
Each subfolder is a **self-contained study** with its own runner, results JSON, charts, and a
`RESULTS.md`/`README.md`. Import only from the package + `phase1` harness.

| path | study |
|---|---|
| `experiments/multi_hop_synth_queries/` | SAC vs tool-calling vs dense on synthetic multi-hop (recall/tokens/turns); §11–§15 (deep-mode, rerank ablation, fewshot, combo, monotonicity) |
| `experiments/su_multihop/` | same benchmark on SearchUnify docs |
| `experiments/browsecomp/` | BrowseComp-Plus (100k corpus) 3-arm + explore + deep-SAC |
| `experiments/primitive_selection/` | the learned template router (BEIR + multi-hop), model bake-off |
| `experiments/explore_learning/` | **what `sac.explore` learns** — the consolidated writeup + charts |
| `experiments/deep_sac/` | deep-mode SAC: cost of going deep, fewshot/plan, monotonicity |

## 3. 🧠 Learnings & open problems
The captured knowledge — what we concluded and what remains hard.

| path | what |
|---|---|
| `open_problems.md` | the 8 dead-ends we hit, each mapped to the field's open problem + citations + our status |
| `experiments/explore_learning/README.md` | the router-learning synthesis (metric correction, mode collapse, fewshot+plan) |
| `research.md` | running research log (newest first) — where ideas come from |
| `learnings_standard.md` | learnings promoted into the standard SDK |
| `CHANGELOG.md` | package changelog |

## 4. 🛠 Internal harness & docs-constitution
The agentic harness the experiments drive, plus the doc rules.

| path | what |
|---|---|
| `soul.md` | **read first** — the docs "constitution": what each `.md` is for, the *standard-vs-internal* rule |
| `phase1/` | the agent harness experiments import: `agents.run_sac`, `llm`, `common`, `sac_surface`, sandbox |
| `phase2/`, `phase3/`, `phase4/` | earlier eval phases (BEIR/qrels, answer-gen/metrics) — **not** imported by experiments |
| `benchmarks/`, `benchmark_changelog.md` | benchmark harness + its log |

> **The one rule that governs placement** (`soul.md`): *generalizable* code/learnings → the standard
> SDK (`search_as_code/`, `learnings_standard.md`); *corpus-/customer-specific* work stays internal
> and **never** goes to the public repo (Altera KB/models/secrets are gitignored and excluded).

---

## Where to start reading
1. `README.md` — what the SDK is + quickstart.
2. `experiments/multi_hop_synth_queries/RESULTS.md` — the headline result (code-mode wins multi-hop).
3. `experiments/explore_learning/README.md` — what the exploration phase learns.
4. `open_problems.md` — the honest frontier + how it maps to the literature.

## Proposed cleanup (pending review — moves tracked files other agents/CI import)
These would tidy the tree but **rename tracked paths**, so they need sign-off before executing:
- **Group the harness**: `phase1/ → harness/` (or `internal/harness/`) and update the 24
  `from phase1 …` imports across `experiments/` in one commit. `phase1` is the only phase experiments
  use.
- **Archive dead phases**: move `phase2/ phase3/` (and public `phase4/`) under `internal/legacy/` —
  they aren't experiment deps.
- **One changelog**: fold `benchmark_changelog.md` into `CHANGELOG.md`.
- **Learnings hub**: a `learnings/` dir indexing `open_problems.md` + `explore_learning` +
  `research.md` (index only; keep experiment files in place so result links don't break).

Say the word and I'll execute these as a single, import-updating commit.
