# qwen8b_sac — the standard SAC explore→forge→full pipeline on the best embedding model we can fit

Runs the **standard** search-as-code pipeline (`deep_judge/run_explore_pipeline.py` → `run_forged_on_full.py`
— the documented 7-stage default) end-to-end on **BrowseComp-Plus** and **SU (su_docs)** with
**Qwen3-Embedding-8B** (4096-d) as the retriever instead of the default gte-base (768-d). No bespoke code —
the same pipeline, pointed at a stronger model via env (`BC_EMB`, `SAC_DIM`, `SAC_TAG`, `SAC_EXP_DIR`,
`SAC_MAX_SEQ`). Friction hit along the way is logged in [`issues.md`](issues.md).

- Retriever: `Qwen/Qwen3-Embedding-8B`, docs capped to 512 tokens (paper protocol; also avoids the 8B's
  32K-context OOM — see issues #5). Generation/judge: OpenAI `gpt-4.1-mini` (API).
- BrowseComp: OpenSearch `browsecomp_qwen8b` (100195 docs). SU: 396 docs embedded in-memory, 150 multihop queries.
- **Caveat (issues #4):** the standard pipeline embeds queries **plain** — Qwen3-8B's query instruction is
  never applied (one symmetric `Session.embedder` also serves HyDE + doc indexing). So absolute recall below
  is Qwen3-8B's *plain* mode; its instructed dense ceiling is higher (see the reference row).

## Explore (7-stage pipeline, structure-emergent)

| stage | BrowseComp | SU |
|---|---|---|
| discovered structure | **whole-query** (decomposed 8/50) | **whole-query** (decomposed 2/40) |
| ①  explore recall@20 (oracle-stop) | 0.175 | 0.750 |
| ③  validate: judge-stop / oracle-stop @20 | 0.147 / 0.173 | 0.667 / 0.600 |
| ④  forge accepted? | ✅ | ✅ |
| ⑤  forged primitive on train @20 | 0.137 | 0.769 |
| ⑦  test recall@10 / @20 | 0.098 / 0.117 | 0.606 / 0.669 |

Both corpora **discovered the whole-query structure** (the LLM almost never decomposes) and the deep judge
tracks the oracle ceiling closely (on SU it even edges past it). Forge succeeded on both — artifacts per corpus:
`{corpus}_explored_primitive` (code), `{corpus}_explored_skill`, `{corpus}_explored_agent`, plus the learned
structure rule (in `forge_store_{corpus}_qwen8b_explored/`).

## Full run — forged primitive vs dense baseline (Qwen3-8B, all queries)

| corpus | arm | recall@10 | recall@20 | all-golds@10 |
|---|---|---|---|---|
| **BrowseComp** (830) | dense | **0.149** | **0.208** | **0.075** |
| | forged SAC | 0.103 | 0.158 | 0.046 |
| **SU** (150) | dense | **0.647** | **0.768** | **0.220** |
| | forged SAC | 0.632 | 0.718 | 0.213 |
| *ref: BrowseComp dense, Qwen3-8B **instructed*** | *(exact-kNN)* | *0.277* | *—* | *—* |

## The result

**On a SOTA 8B retriever, the forged SAC primitive does not beat plain dense — on either corpus.** BrowseComp
0.103 < 0.149; SU 0.632 ≈ 0.647 (a whisker below). This is the same conclusion the ReasonIR/Qwen8B
reproduction reached from the other direction: **SAC augmentation value is inverse to retriever strength.**
On weak gte-base the forged primitive *helped* (see `deep_judge/README.md` §4–5); on an 8B retriever the query
already lands its own neighbors, so query-side scaffolding (hybrid/HyDE/rerank fusion) mostly adds noise. SU
is closer (0.632 vs 0.647) because SU is an easier, more lexical corpus where the primitive's structure is
nearly redundant with dense rather than harmful.

Note the pipeline runs Qwen3-8B in **plain** mode; its instructed dense on BrowseComp is ~0.277 R@10 (row
above), so the *real* strong-retriever baseline is even higher and the gap to the forged arm wider. Applying
the query instruction inside SAC is the open SDK item (issues #4).

## Reproduce
```
# BrowseComp (OpenSearch index already built): explore then full
BC_EMB=Qwen/Qwen3-Embedding-8B BC_INDEX=browsecomp_qwen8b SAC_EXP_DIR=experiments/qwen8b_sac SAC_TAG=browsecomp_qwen8b \
  python -m experiments.deep_judge.run_explore_pipeline browsecomp 50 15 50 8 8
BC_EMB=Qwen/Qwen3-Embedding-8B BC_INDEX=browsecomp_qwen8b SAC_EXP_DIR=experiments/qwen8b_sac SAC_TAG=browsecomp_qwen8b \
  python -m experiments.deep_judge.run_forged_on_full browsecomp 8
# SU (in-memory, 4096-d store): explore then full
BC_EMB=Qwen/Qwen3-Embedding-8B SAC_DIM=4096 SAC_EXP_DIR=experiments/qwen8b_sac SAC_TAG=su_qwen8b \
  python -m experiments.deep_judge.run_explore_pipeline su 40 15 40 6 8
BC_EMB=Qwen/Qwen3-Embedding-8B SAC_DIM=4096 SAC_EXP_DIR=experiments/qwen8b_sac SAC_TAG=su_qwen8b \
  python -m experiments.deep_judge.run_forged_on_full su 8
```

Results: `explore_pipeline_{browsecomp,su}_qwen8b.json`, `explore_full_{browsecomp,su}_qwen8b.json`,
`forge_store_{browsecomp,su}_qwen8b_explored/`. Pipeline friction / bugs: [`issues.md`](issues.md).
