# Phase 1 — base search vs MCP tool-calling vs Search-as-Code

An apples-to-apples comparison of three ways an agent can retrieve, over a real
labeled corpus (**BEIR FiQA-2018**, 57,638 docs, 648 test queries with qrels)
indexed in **OpenSearch**, using the `search_as_code` SDK.

- **base** — deterministic retrieval (hybrid), no LLM. The cost/latency floor.
- **tool-calling** — MCP-style: the LLM calls discrete search tools; intermediate
  hits flow back through the context on every hop (LangChain tool-calling).
- **SAC** — code-mode: the LLM writes ONE Python program against the SDK, run in
  the sandbox; only the final ids return. Static SDK surface is prompt-cached.

## Components

| File | Role |
|---|---|
| `common.py` | config, key loading (`~/taxonomy/.env`), BEIR loaders, embedder/session factories |
| `ingest_fiqa.py` | download FiQA → embed (`gte-base`, GPU) → bulk-index into OpenSearch |
| `metrics.py` | Recall@k / nDCG@k / MRR@k (BEIR-style) |
| `eval_base.py` | base dense/keyword/hybrid recall over all test queries |
| `llm.py` | OpenAI `gpt-4.1-mini` wrapper with token/cost + cache accounting |
| `sac_surface.py` | the static SAC primitive surface (cached prefix) + tool schemas |
| `agents.py` | the three query paths (`run_base`, `run_sac`, `run_tool_calling`) |
| `benchmark.py` | runs all 3 paths over N queries → per-query traces + summary |
| `ui.py` | Streamlit trace viewer |

## Reproduce

```bash
# 0a. install the phase1 deps (embedder, torch, langchain, requests) on top of the SDK
pip install -e '.[phase1]'

# 0b. OpenSearch must be running (tarball, security off, single node) on :9200
#    cd ~/opensearch_stack/opensearch-2.17.1 && OPENSEARCH_JAVA_OPTS="-Xms4g -Xmx4g" bin/opensearch

# 1. ingest FiQA (idempotent; ~1 min on GPU)
python -m phase1.ingest_fiqa

# 2. base-search recall baseline
python -m phase1.eval_base

# 3. functional test of every primitive on real data
python -m phase1.functional_test

# 4. the 3-way benchmark (writes phase1/runs/)
python -m phase1.benchmark -n 100 --reranker cross-encoder/ms-marco-MiniLM-L-12-v2

# 5. trace UI
streamlit run phase1/ui.py
```

## Notes / findings

- **Embeddings:** `thenlper/gte-base` (768-d), GPU. Base dense R@10≈0.45, nDCG≈0.39
  and BM25 nDCG≈0.24 — in line with published BEIR numbers, so the stack is sound.
- **Reranker caveat:** MS-MARCO cross-encoders *hurt* FiQA recall (domain mismatch);
  the FiQA-appropriate `BAAI/bge-reranker-base` would not download in this env, so
  the SAC recipe leads with **hybrid + query fan-out + RRF fusion** (which helps
  recall) and treats cross-encoder reranking as optional. Any reranker effect
  applies equally to the SAC and tool-calling paths (shared Session), so the
  comparison stays fair.
- **Caching:** the SAC surface is >1024 tokens so OpenAI automatic prompt caching
  bills it at the cached rate after query 1 (see `../docs/CACHING.md`). Measured
  from `usage.prompt_tokens_details.cached_tokens`, reported as `cache_hit_rate`.
- **Cost accounting** folds the Session generator's internal rephrase/expand token
  usage into the owning path, so per-path cost is complete.
