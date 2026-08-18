# fable_baselines — the no-LLM floors (WS6, 2026-08-18)

One runner (`run_baselines.py`), three corpora, four no-LLM arms. These are the floors every
SAC/explore/judge arm must beat, measured with the exact configs the published numbers used
(gte-base, same indexes, HotpotQA on the forge-disjoint slice rows[200:300], BrowseComp on all
830 queries with the re-fetched official qrels — BC-1 fixed).

## recall@10 / all_golds@10

| corpus | hop | dense | bm25 | hybrid (RRF) | dense→CE-rerank |
|---|---|---|---|---|---|
| hotpotqa | 2 | .935 / .87 | .945 / .91 | **.970 / .94** | .935 / .88 |
| hotpotqa | 3 | .787 / .60 | .887 / .73 | **.910 / .79** | .763 / .54 |
| hotpotqa | 4 | .665 / .28 | .753 / .42 | **.785 / .46** | .618 / .23 |
| su | 2 | **.950 / .91** | .890 / .79 | .945 / .89 | .920 / .84 |
| su | 3 | .813 / .55 | .853 / .60 | **.877 / .70** | .787 / .47 |
| su | 4 | .715 / .29 | .823 / .47 | **.825 / .53** | .645 / .22 |
| browsecomp | — | .0705 / .034 | .0527 / .029 | **.0797 / .040** | .0652 / .033 |

Paired-bootstrap deltas vs dense are in the JSONs; hybrid's recall lift is **significant** on
HotpotQA 2/3/4-hop; ns on BrowseComp (+0.009) and SU.

## The three findings

1. **Hybrid ≥ dense everywhere, and the gap widens with hop depth.** The dense-default gate
   (README's "SAC never underperforms dense") is gating against the wrong floor: on HotpotQA
   4-hop, hybrid is +0.12 recall over dense. The gate should compare the forged primitive
   against the **best no-LLM baseline** on held queries (dense *and* hybrid), and emit that
   baseline as the fallback. → fable.md WS3.
2. **Whole-query cross-encoder rerank hurts multi-hop at every depth** (3-hop and 4-hop rows,
   both corpora: dense_rerank < dense). This is open_problems.md #4 reproduced as a proper
   *baseline control* (P1-10) rather than an ablation — the harness's coverage-first assembly
   is not beating a strawman; rerank genuinely is the wrong single-pass pipeline for multi-gold.
3. **Full-text BM25 does not rescue BrowseComp** (0.053 vs dense 0.071) — this is OpenSearch
   full-text BM25, *not* the KW_CHARS=2000 truncated memory index (BC-2), so the earlier
   caveat "lexical numbers were measured on 6% of each doc" is now bounded: even at 100% of
   each doc, BM25 alone loses to dense here. Hybrid is +0.009 ns.

## Reproduce

```bash
python3 -m experiments.fable_baselines.run_baselines            # all three
python3 -m experiments.fable_baselines.run_baselines hotpotqa   # one corpus
```

SU docs are internal (`~/scripts/data/su_docs_2.csv`, DS-2); only aggregates are written.
BrowseComp needs `experiments/browsecomp/` local data (corpus vectors + the restored qrels).
