# AUDIT.md — what the numbers do and don't support

This project's claims are maintained under a standing, adversarial self-audit. The full
trail — 170+ logged findings, methodology corrections, retracted claims, and every
experiment runner — lives in the companion research repository. This file is the honest
summary a user should read before quoting anything.

## Claims that hold (and how they were earned)

| claim | status |
|---|---|
| **Code-mode cost win vs tool-calling: ~31× fewer input tokens, ~1.7× lower latency, 1 turn vs ~9.5** | **Holds, structural.** Matched tools + budget, both arms gpt-4.1-mini, BrowseComp-Plus n=100 on Qwen3-Embedding-8B. Widens with hop depth. The one result that has survived every re-measurement. |
| **The never-below-baseline gate** | **Holds by construction, with receipts.** Forged primitives ship only after beating max(dense, hybrid) on held queries (paired bootstrap CI); otherwise the baseline ships. Published gate decisions include it *selecting the baseline* — on strong retrievers, authored strategies often don't beat plain dense, and the system says so. |
| **Judge quality: 0.771 [0.666, 0.870] balanced accuracy vs oracle** | **Holds, leak-free.** Query-grouped split (an earlier 0.700 came from a leaky split + a non-shipped render format). Judge-stopped runs recover 95–103% of oracle-stopped recall on HotpotQA/SU. No-LLM references published alongside: tuned threshold 0.738, logistic 0.749 — the judge leads within CI at n=100. |
| **Explore lifts retrieval on the corpora it learned** | **Holds — as corpus knowledge.** Seeding the forged strategy lifts *both* code-mode and tool-calling arms. It is a corpus-knowledge win, not a code-mode win, and we say so. |
| **Hybrid ≥ dense on every corpus tested, gap widens with hop depth** | **Holds** (3 corpora, paired CIs) — which is why the gate's floor is max(dense, hybrid), not dense. |

## Claims we retracted (so you don't have to discover them)

- **"SAC beats tool-calling on retrieval *quality*"** — did not survive matched prompts.
  With identical guidance and budgets, quality deltas are within noise. The win is cost,
  latency, and the learning loop.
- **"The judge sits at a 0.72 signal ceiling"** — a measurement artifact (leaky split,
  wrong renderer). The corrected, leak-free number is above it.
- **"Whole-query reranking helps multi-hop"** — the opposite: a dense→cross-encoder
  control sits *below* plain dense on every 3/4-hop cell we measured. Rerank sharpens
  single-gold pools; coverage-first fusion is the multi-hop default.
- **A learned query-profile lift of +2.7 pts** — did not reproduce under leak-free
  splits; struck from the record.
- **A seeded-agent benchmark row (0.169 r@10) measured with the wrong artifact** — the
  arms were seeded with the forge store's *candidate* strategy (a multi-mode fusion the
  acceptance gate had rejected) instead of the gate's recorded selection (`dense`). The
  fusion diluted recall on every query. Corrected the same day; the deploy rule it
  produced — **consume the gate's selection, never the rejected candidate** — is now in
  the README, and `HarnessForge.accept_code_primitive` persists the winning side under
  the requested name so gate-written stores cannot reproduce the mistake.

## How to check us

Every measured delta in this repo's docs carries a paired-bootstrap confidence interval
(`sac.metrics.compare`); a difference whose CI includes zero is not a result. The eval
harness gives both agent arms identical tools and budgets, evaluates on forge-disjoint
query slices, and re-raises worker exceptions so a crashed arm can't score as an empty
one. If you want the raw per-query records or the defect log, they're in the research
repository — nothing was cleaned for this release except the directory tree.
