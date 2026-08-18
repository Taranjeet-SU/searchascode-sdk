# Standard Learnings

Generalizable things discovered during domain/custom work that belong in the SDK (not forks).
Newest first.

> Two things live here, both required by `soul.md` rule 2 ("improve the SDK, don't fork it"):
> the **narrative learnings** below, and the **promotion ledger** at the end that records which
> learning landed in which SDK module. The file itself was missing for months (`issues.md`
> LEG-2), which is why several capabilities sat unpromoted in `phase2/`, `phase4/` and
> `chatbot/` while the shipped SDK used a weaker version of each.

## Embedding — robust custom-transformer loader (shipped: `get_embedder("transformers", ...)`)
Custom GTE-v1.5 / `model_type="new"` models corrupt non-persistent buffers on meta-device init
(`position_ids`, rotary `cos_cached`/`sin_cached`) → GPU device-assert or NaN outputs. Fix (now in
`search_as_code/embeddings.py`): load `low_cpu_mem_usage=False`, disable the `unpad`/memory-efficient
attention path, and **re-materialize those buffers on the model device**. CLS pooling + L2 normalize.

## Bind a Session to a NON-standard index (unlock the full primitive surface)
The OpenSearch adapter takes `text_field`/`vector_field` + an external embedder, so you can point a
real `sac.Session` at any index (custom field names, custom embedder) and get **all** primitives
(hyde/mmr/prf/compress/dedup/hybrid/rerank/…) instead of hand-wiring a few:
```python
s = sac.Session("opensearch", index=IDX, hosts=[URL],
                text_field=CONTENT_FIELD, vector_field=KNN_FIELD,
                embedder=get_embedder("transformers", model=MODEL), reranker=QwenReranker())
```

## RAG evaluation — measure retrieval AND answer separately
- On a domain the generator already knows, retrieval lifts **citation/source-grounding** far more than
  final-answer text. Always report both; use a **closed-book arm** as the contamination control.
- **Objective citation scoring**: regex-match gold-cited doc ids against retrieved ids → no LLM judge.

## HyDE — use it when closed-book is strong
HyDE pays off exactly when parametric knowledge is good: the hypothetical answer resembles the real
docs, so its embedding/terms retrieve them better than the raw query.

## Agentic retrieval — honest caveats
More machinery (extra hops, critic loops, agglomerative synthesis) does **not** always beat simpler
retrieval; on well-known domains it dilutes context. A **fixed** LLM-authored strategy rarely beats a
good recipe — the value is **per-query adaptive code**, not a better static program.

## Ops — resilient tunnels to remote stores
Long-lived SSH tunnels drop under load/idle. Use an auto-reconnecting keeper with an **active
health-check** (curl the endpoint; kill+reconnect on hung-but-alive states that `ServerAliveInterval`
misses), plus **per-call retry with graceful-degrade**, and cap concurrency to what the channel sustains.

---

## Promoted

| # | learning | found in | now in the SDK | issue |
|---|---|---|---|---|
| 1 | A generator adapter that splits completions into **lines** breaks any consumer that takes `out[0]`. Normalize the contract once instead of at each call site. | `phase1/llm.py` + 9 consumers | `search_as_code/_genutil.py` (`gen_text`, `gen_lines`) | GEN-1/2/3 |
| 2 | `MemoryStore.query_keyword` re-tokenizing the corpus per query is unusable past ~10k docs; cache the DF table and invalidate on write. | `experiments/browsecomp/bc_common.py::FastMemoryStore` | `adapters/memory.py` (`_ensure_kw_index`) | SDK-C11 / BC-3 |
| 3 | An OpenSearch random sample must be **seeded** or every fingerprint/resume mechanism built on it silently re-runs. | `phase4/altera_synth.py::sample_cards` | `adapters/opensearch.py::sample(sample_seed=…)` | SDK-C4 / P4-4 |
| 4 | Report an interval, not a bare mean. SQuAD EM/F1 + a bootstrap CI belong to every experiment, not to one customer phase. | `phase4/metrics.py` | `search_as_code/metrics.py` (`bootstrap_ci`, `compare`, `recall_at_k`, `all_golds_at_k`, `ndcg_at_k`) | P4-8 / DJ-2 |
| 5 | The LLM-facing **prompt surface** is part of the product. If it lives outside the shipped package, `pip install` delivers primitives with no decision rules. | `phase1/sac_surface.py` | `search_as_code/surface.py` (re-exported by the old path) | DOC-1 |
| 6 | Routing must be scored on **realized task recall**, never on classification accuracy — the latter penalises a correct-but-not-cheapest pick and inverts conclusions. | `phase2/router_model.py` | `explore/training.py::realized_recall` | SDK-A2 / open_problems #3 |
| 7 | Online learning needs a **real reward**. "Returned ≥1 id" is not evidence; gate learning on a verifier that can see relevance. | the harness's own failure mode | `harness/loop.default_verify` + `StepResult.verified` gating `reflect()` / `post_write_memory` | SDK-A3 |
| 8 | Keep fallbacks, but **count** them. A silent `except` makes "this strategy lost" indistinguishable from "this strategy crashed". | 49 bare excepts in the SDK | `ResultSet.mark_degraded` / `StrategyContext.degraded` → `degraded_frac` in dataset meta | LEG-5 |
| 9 | A strategy that degrades to another strategy under missing capabilities must be marked **unavailable**, not scored — otherwise the cheapest duplicate wins every tie by construction. | template labeling | `explore/templates.TEMPLATE_REQUIRES` / `available_templates` | SDK-A1 |
| 10 | Non-persistent RoPE buffers (`inv_freq`) can materialise as uninitialised memory under transformers 5.x; repair them explicitly rather than pinning a venv outside the repo. | `experiments/browsecomp/reasonir_encoder.py` | `embeddings._fix_meta_buffers` + a `<5` pin in `requirements/experiments.txt` | BC-4 |
| 11 | Introspect the schema before authoring a backend query. Hardcoding `title` makes the authored body match nothing on a corpus that has no such field. | BrowseComp runs | `harness/os_query.describe_fields` / `build_author_system` | SDK-A6/A7 |

## Known but NOT yet promoted

These are recorded so the next agent does not have to rediscover them.

| learning | found in | why it is still stranded |
|---|---|---|
| **Asymmetric query/passage encoding.** bge/e5/Qwen3 need different prefixes for queries vs passages; getting it wrong costs more than every augmentation measured in this repo (Qwen3-8B BrowseComp R@10 0.149 plain → 0.277 instructed). | `phase2/embed_models.py` | `Session` has ONE symmetric `embedder`, which also serves HyDE and doc indexing. Fixing it is an API change: `Session(query_embedder=…, passage_embedder=…)`. Tracked as P2-5 / qwen8b_sac issues #4. |
| **Judge calibration against a published baseline** — score the vendor's own answers first and check the judge reproduces the known pass-rate before trusting it on new arms. | `phase4/altera_eval.py` | `DiagnosticJudge` has no calibration hook. Worth adding as `judge.calibrate(known_pairs)`. P4-8. |
| **A properly-designed tool-calling baseline** (hybrid → hydrate → cross-encoder rerank *inside* the tool, plus `read_docs`, and hops counted only for retrieval rounds). | `chatbot/toolcalling.py` | The headline benchmark still uses the 3-tool arm. Fixing it means re-running the FiQA and multi-hop comparisons. CB-1 / P1-1. |
