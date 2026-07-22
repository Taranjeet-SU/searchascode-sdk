# Multi-dataset retrieval report: dense vs hybrid vs SAC vs tool-calling

Comparative evaluation of **search-as-code (SAC)** — an LLM writing Python against retrieval primitives in a sandbox — against three baselines across BEIR datasets spanning distinct query types. The thesis under test: *agentic code-mode retrieval helps in proportion to how much a query needs computation/decomposition, and can hurt when the raw query is already optimal.*

## 1. Methodology

**Systems compared (identical stack, only the orchestration differs):**
- **dense** — single vector kNN query (HNSW/Lucene in OpenSearch).
- **hybrid** — dense + BM25 fused (weighted RRF, alpha=0.7).
- **SAC** — gpt-4.1-mini writes Python over primitives (search/fan-out/fuse/rerank/decompose/expand/rephrase/mmr/…), executed in a sandbox; an LLM-judge loop retries up to 1x.
- **tool-calling (MCP-style)** — same LLM, same budget, but each primitive is a discrete tool call (LangChain agent) instead of code.

**Fixed components:** embedder = `thenlper/gte-base` (768-d, normalized); reranker = Qwen3-Reranker-0.6B (yes/no logit scoring); LLM = gpt-4.1-mini; store = OpenSearch.

**Metrics:**
- **recall@10** = |gold ∩ top10| / |gold|.
- **all_found@10** = 1 if *every* gold doc is in top10 else 0 — the multi-hop-sensitive metric (a single dense query rarely lands *both* supporting docs).

**Protocol:** each corpus is embedded and indexed once. dense/hybrid are measured on the full labeled query set (stable baseline); SAC/tool run on the first N (LLM cost) with dense/hybrid recomputed on the *same* N for a paired comparison. gte-base on a shared GPU; Qwen reranker capped at max_length 512.

## 2. Dataset characteristics

| dataset | task type | corpus | queries (labeled) | avg gold/q | query character |
|---|---|---|---|---|---|
| hotpotqa | multi-hop QA | 100978 | 500 | 2.0 | needs 2 supporting docs; fan-out/decompose is the lever |
| scifact | scientific claim verification | 5183 | 300 | 1.13 | term-heavy, keyword-favoring, ~1.1 gold/q |
| nfcorpus | medical IR | 3633 | 323 | 38.186 | natural-language queries, MANY gold/q (recall@10 capped) |
| arguana | counter-argument retrieval | 8674 | 1406 | 1.0 | long argumentative queries, 1 gold/q |
| scidocs | citation/related-paper | 25657 | 1000 | 4.928 | title->cited papers, ~5 gold/q |
| trec-covid | broad topical COVID | 171332 | 50 | 493.46 | few rich topics, MANY gold/q |

## 3. Headline results — SAC-subset (paired, N queries)

| dataset | N | dense | hybrid | **SAC** | tool | dense all@10 | **SAC all@10** | LLM $ |
|---|---|---|---|---|---|---|---|---|
| hotpotqa | 50 | 0.790 | 0.950 | **0.960** | 0.750 | 0.620 | **0.920** | — |
| scifact | 40 | 0.857 | 0.876 | **0.876** | 0.866 | 0.825 | **0.825** | — |
| nfcorpus | 40 | 0.189 | 0.170 | **0.179** | 0.177 | 0.025 | **0.025** | — |
| arguana | 40 | 0.850 | 0.750 | **0.725** | 0.500 | 0.850 | **0.725** | — |
| scidocs | 40 | 0.222 | 0.221 | **0.241** | 0.152 | 0.000 | **0.000** | $0.0048 |
| trec-covid | 40 | 0.016 | 0.017 | **0.017** | 0.013 | 0.000 | **0.000** | $0.0053 |

_recall@10 unless noted; **bold** = SAC columns. all@10 = all_found@10._

## 4. Full-query-set baseline (dense/hybrid, stable)

| dataset | N_full | dense r@10 | hybrid r@10 | dense all@10 | hybrid all@10 |
|---|---|---|---|---|---|
| hotpotqa | 500 | 0.787 | 0.890 | 0.626 | 0.788 |
| scifact | 300 | 0.843 | 0.860 | 0.827 | 0.843 |
| nfcorpus | 323 | 0.174 | 0.180 | 0.040 | 0.053 |
| arguana | 1406 | 0.752 | 0.730 | 0.752 | 0.730 |
| scidocs | 1000 | 0.245 | 0.218 | 0.013 | 0.009 |
| trec-covid | 50 | 0.019 | 0.020 | 0.000 | 0.000 |

## 5. Analysis — when does agentic search pay off?

- **Multi-hop (HotpotQA)** — the clear SAC win: SAC recall@10 0.96 / all_found@10 0.92 vs dense 0.79 / 0.62. Decompose + fan-out + fuse retrieves *both* supporting docs, which a single dense query structurally cannot. This is the flagship result.
- **Term-heavy (SciFact)** — SAC ties hybrid (~0.88) and both edge dense (~0.86). SAC's job is to *route to hybrid/keyword*, not to add hops; the win is small because dense is already strong.
- **Long-argument (ArguAna)** — **anti-result**: dense 0.85 > hybrid 0.75 > SAC 0.73 > tool 0.50. The query is a full argument and *is* the ideal retrieval key; rephrasing/decomposing degrades it. Agentic manipulation must be applied *conditionally*.
- **Many-gold (NFCorpus, SciDocs, TREC-COVID)** — recall@10 is structurally capped (>10 gold), so pool-expansion + rerank matters more than hops; SAC ≈ hybrid.
- **Semantic single-hop (FiQA)** — SAC ties dense; learned rules net-neutral (no routing structure to exploit — see phase2 ceiling/impact analysis).

**Takeaway:** there is no single best retriever across query types (consistent with BEIR's own finding that the best dense model beat BM25 on only 8/18 datasets). SAC's value is that *one code policy can pick the right strategy per query* — decisively so on multi-hop, neutrally on easy/semantic sets, and it must learn to *abstain from manipulation* on tasks like ArguAna.

## 5b. Learned-profile impact (deterministic, no LLM at query time)

Rules (aliases/synonyms) are mined offline from each dataset's dense-misses and applied to the *dense* path (normalize + synonym-expand + fuse). This measures whether learning helps a cheap retriever close the gap toward the agentic ceiling.

| dataset | queries expanded | base r@10 | learned r@10 | Δ r@10 | base all@10 | learned all@10 | Δ all@10 |
|---|---|---|---|---|---|---|---|
| hotpotqa | 24 | 0.803 | 0.817 | +0.013 | 0.633 | 0.660 | +0.027 |
| scifact | 5 | 0.863 | 0.863 | +0.000 | 0.840 | 0.840 | +0.000 |
| nfcorpus | 2 | 0.167 | 0.167 | +0.000 | 0.040 | 0.040 | +0.000 |
| arguana | 0 | 0.793 | 0.793 | +0.000 | 0.793 | 0.793 | +0.000 |
| scidocs | 17 | 0.230 | 0.232 | +0.003 | 0.007 | 0.007 | +0.000 |

**Finding:** the learned-synonym benefit scales with multi-hop/entity structure — HotpotQA gains +2.7 pts all_found@10, while single-hop/many-gold sets are ~neutral. ArguAna mines **zero** rules (its dense-misses are stance-based, not lexical), so learning is a *safe no-op* there rather than a regression. Learning should be applied conditionally; the miner self-limits where there is no lexical structure to exploit.

## 6. Research grounding

- BEIR: heterogeneous zero-shot IR benchmark; dense ≠ universally best (Thakur et al., 2104.08663).
- HotpotQA: multi-hop QA requiring 2+ supporting facts (Yang et al., 2018).
- SciFact: scientific claim verification; BM25 nDCG@10 ≈ 0.66, term-heavy (Wadden et al., 2020).
- Adaptive/agentic retrieval parallels: 'Think Before You Retrieve: Test-Time Adaptive Search with Small LMs' (2511.07581); 'Claim-Aware Scientific RAG: Evidence-First Retrieval and Abstention'. These motivate SAC's route/decompose/confidence-abstain primitives.

## 7. Reproduction

```bash
# ingest + eval any dataset (dense/hybrid full set; SAC/tool on first N)
python -m phase2.beir_run --dataset scifact --ingest --n 40
# full 5-dataset campaign (serial, shared GPU)
bash phase2/run_campaign.sh
# regenerate this report from runs/*.json
python -m phase2.make_report
```

_Auto-generated from phase2/runs/*.json — 6 datasets rendered: hotpotqa, scifact, nfcorpus, arguana, scidocs, trec-covid._
