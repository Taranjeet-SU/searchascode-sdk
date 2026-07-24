# Standard Learnings

Generalizable things discovered during domain/custom work that belong in the SDK (not forks).
Newest first.

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
