# Why SAC wins on HotpotQA (multi-hop)

HotpotQA questions require **two** supporting documents (a "bridge" entity links them). A single
dense query embeds the whole question and lands *one* of them; **search-as-code (SAC)** decomposes
the question, fans out per sub-question, fuses, and reranks — so it retrieves **both**.

## Retrieval (100k-doc corpus)
| method | recall@10 | all_found@10 |
|---|---|---|
| dense | 0.79 | 0.62 |
| hybrid | 0.95 | 0.90 |
| tool-calling | 0.75 | 0.68 |
| **SAC** | **0.96** | **0.92** |

`all_found@10` (every gold doc in the top-10) is the multi-hop-sensitive metric — **SAC +0.30 over dense**.

## Answer generation (n=200, gpt-4.1-mini, same corpus, SQuAD EM/F1 + bootstrap CIs)
| arm | EM | F1 |
|---|---|---|
| closed-book (no retrieval) | 0.310 | 0.423 |
| vanilla-RAG | 0.470 | 0.626 |
| tool-calling RAG | 0.500 | 0.659 |
| **SAC** | **0.520** | **0.673** |

SAC tops answer quality; **retrieval adds +0.25 F1 over the closed-book contamination control**, and
SAC beats vanilla single-shot RAG by +0.05 EM.

## The mechanism (why, not just what)
1. **decompose** — split the 2-hop question into focused sub-questions.
2. **fan-out** — retrieve each sub-question (dense + keyword).
3. **fuse (RRF) + rerank** — merge the sub-results and re-score → surface *both* supporting docs.

A single dense query is biased toward one entity and misses the bridge doc; SAC's *per-hop* retrieval
structurally captures both — which is exactly why `all_found@10` jumps 0.62 → 0.92 and the answers follow.

## Reproduce
```bash
python -m phase2.hotpot_eval --n 60            # retrieval (recall@10 / all_found@10)
python -m phase4.answer_gen --dataset hotpotqa --n 200   # answer generation (EM / F1)
```
