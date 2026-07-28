# FiQA RAG chatbot agent

A retrieval-augmented **chatbot** over cached BEIR-FiQA (57,638 finance docs in
OpenSearch). Retrieval is 100% the `search_as_code` primitives; only the final
answer synthesis uses an LLM (LangChain `ChatOpenAI`, `gpt-4.1-mini`).

```
question ─▶ hybrid retrieval (dense + BM25 → RRF fuse → dedup → cross-encoder rerank)   ← search_as_code
         ─▶ confidence gate (abstain if no clear evidence)                              ← primitives.confidence
         ─▶ LangChain LLM answers, grounded, with [n] citations
```

## Run it

```bash
pip install -e '.[phase1]'          # deps (search_as_code + torch + langchain-openai + …)
# OpenSearch on :9200 with the fiqa index ingested (see phase1/), OPENAI_API_KEY set

# interactive chat
python -m chatbot.agent
# > How do I deposit a cheque made out to my business?

# programmatic
python -c "from chatbot.agent import RagChatbot; print(RagChatbot().answer('...').answer)"
```

## Evaluate on public queries (relevance · speed · cost)

```bash
python -m chatbot.evaluate -n 20 --judge
```

FiQA ships **relevant-doc labels (qrels)**, not gold answers, so *relevance* is
measured as retrieval quality against those labels; `--judge` adds an LLM-judged
answer-faithfulness rate.

### Measured (20 public FiQA queries, gte-base + MS-MARCO reranker)

| axis | metric | value |
|---|---|---|
| **Relevance** | Recall@10 / nDCG@10 / MRR@10 | **0.408 / 0.370 / 0.398** |
| **Relevance** | answer faithfulness (LLM-judged) | **85%** |
| **Speed** | mean / p50 / p95 latency | 4.75 s / **3.57 s** / 9.96 s |
| **Speed** | retrieval-only (mean) | **1.63 s** |
| **Cost** | per query / 20-query total | **$0.00047** / $0.0095 |

Notes: p95 is inflated by first-query model warm-up (~16 s); steady-state p50 is
~3.6 s. The single-query hybrid+rerank here trades some recall for latency vs the
full SAC agent (query fan-out + judge loop → 0.55 Recall@10 on 100 queries) — swap
`bot.retrieve` to `sac.search_many([...variants], mode="hybrid")` to close that gap.
The MS-MARCO reranker is a known FiQA domain-mismatch drag; a finance/bge reranker
would lift relevance.

## Tool-calling counterpart (`chatbot/toolcalling.py`)

Same stack (OpenSearch + hybrid search), but the agent drives retrieval by **calling
discrete tools** (`search_docs` / `read_doc` / `finish`) in a multi-hop loop — no SAC
code-mode. Tools follow Anthropic's "writing tools for agents" (few high-signal tools,
compact structured results, a structured `finish`).

```bash
python -m chatbot.toolcalling                          # interactive
python -m chatbot.evaluate -n 10 --agent toolcalling --judge
```

### Head-to-head — same 10 FiQA queries

| metric | RAG (single-shot) | Tool-calling (multi-hop) |
|---|:--:|:--:|
| Arrival (final answer) | **100%** | 60% |
| avg hops | 1.0 | 4.8 |
| Recall@10 | 0.367 | **0.483** |
| nDCG@10 | **0.373** | 0.358 |
| MRR@10 | **0.420** | 0.362 |
| answer faithfulness | **80%** | 20% |
| latency p50 / mean | **3.4s / 4.6s** | 19.2s / 22.8s |
| cost / query | **$0.00049** | $0.00297 |

Multi-hop tool-calling buys recall (0.48 vs 0.37) but costs ~6× more, runs ~5.6× slower,
converges only 60% of the time within 5 hops, and scores lower faithfulness (it cites
1-2 docs but writes a fuller answer). n=10 → directional; phase1's 100-query benchmark
is the rigorous version.
