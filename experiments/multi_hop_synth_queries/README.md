# multi_hop_synth_queries

## Goal
Generate synthetic multi-hop queries over the HotpotQA corpus where answering **requires N
specific documents together**. A query is "solved" only if **all** its gold docs appear in
**recall@10**. Purpose: a fair test bed for whether multi-strategy retrieval (`decompose`/fan-out)
beats a single dense pass — single dense tends to surface *one* hop, not all, so this is where
routing *should* win (unlike single-hop prose IR, where dense already wins).

**Three distinct datasets** (≥1,000 queries each): **N=2, N=3, N=4** documents required.
Increasing N should make single-dense recall@k progressively worse and multi-strategy retrieval
progressively more necessary — a difficulty gradient.

## Approach
1. **Seed** — sample a document from the HotpotQA corpus (`hotpotqa` index on OpenSearch :9200).
2. **Chain** — build a chain of N related docs: seed → BM25 neighbor → neighbor-of-neighbor → …
   (each consecutive pair shares keywords but covers different content; skip near-identical titles).
   If the chain can't be extended to length N, skip that seed.
3. **Generate** — ask the LLM to write ONE question answerable **only by using ALL N** docs; no
   subset suffices. **Don't force it** — if the docs lack common ground, the LLM returns `NONE`
   and we skip.
4. **Record** valid queries as `{query, gold_ids:[...N ids...], titles, facts, n_docs}`.
5. Repeat until ≥1,000 per N.

## Requirements
- OpenSearch :9200 with the `hotpotqa` index (fields: `title`, `text`, `vector`; ~100,978 docs).
- LLM via `phase1.llm.LLM` (gpt-4.1-mini) — `OPENAI_API_KEY` in `~/taxonomy/.env`.
- Python env: `requests`, `openai` (repo `.venv`).

## Run
```
# args: target workers n_docs
bash experiments/multi_hop_synth_queries/run.sh 1000 8 2   # 2-hop
bash experiments/multi_hop_synth_queries/run.sh 1000 8 3   # 3-hop
bash experiments/multi_hop_synth_queries/run.sh 1000 8 4   # 4-hop
```

## Output — `data/multihop_{N}docs_queries.jsonl`
One JSON per line:
```
{"query": "...", "gold_ids": ["id1", ... N ids],
 "titles": ["...", ...], "facts": ["fact from doc1", ...], "n_docs": N}
```

## Success criterion (for downstream labeling)
A retrieval strategy **solves** a query iff **BOTH `gold_ids` are in top-k (recall@k)** — this is
the *all-golds* criterion (not *any-gold*). Downstream, label the 16 SAC templates with this
all-golds@10 rule; expectation: `decompose_rerank`/fan-out beat single dense, which usually
retrieves only one hop.

## Notes for future agents
- **Yield is low per pair** (many `NONE`) — oversample pairs; the generator keeps going until the
  target is met.
- **Partner band matters**: too similar (near-dupe) → trivial/degenerate question; unrelated → no
  bridge. BM25 rank ~1–4 is a good "shares terms, different content" band; tune if yield is poor.
- No automatic verification that the question *strictly* needs both docs — the `NONE`-skip + the
  `fact_from_A`/`fact_from_B` fields are the quality signal. A stricter filter (verify neither doc
  alone answers it) is a possible v2.
