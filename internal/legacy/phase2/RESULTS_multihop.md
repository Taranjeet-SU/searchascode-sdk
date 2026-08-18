# Multi-hop results — where code-as-search beats dense

## Real HotpotQA (BeIR, reduced 100k-doc corpus: 995 gold + 100k distractors, gte-base)
dense/hybrid over 500 queries; SAC/tool over 40:

| method | recall@10 | all_found@10 (both gold docs) |
|---|---|---|
| dense (single query) | 0.775 | 0.575 |
| hybrid (RRF)          | 0.938 | 0.875 |
| **SAC (code-as-search)** | 0.925 | **0.850** |
| tool-calling (MCP)    | 0.775 | 0.650 |

- **SAC beats dense +27 pts all_found (0.575→0.850)** and beats tool-calling (0.650) — it recovers
  the 2nd hop a single dense query can't reach.
- hybrid ties SAC here because HotpotQA questions are entity-rich (keyword matches the bridge doc).

## Synthetic pure-bridge (bridge entity NOT in the query — hybrid can't help)
| method | recall@10 | all_found@10 |
|---|---|---|
| dense | 0.729 | 0.458 |
| **SAC (decompose→read bridge→search)** | **1.000** | **1.000** |

## Takeaway
FiQA (simple semantic): SAC ≈ dense ≈ hybrid (no headroom).
Multi-hop: SAC ≫ dense/tool; ≈ hybrid on entity-rich HotpotQA, and **uniquely wins on true bridge
queries** where no fixed strategy reaches the second hop. SAC's value = per-query adaptivity + the
ability to *read an intermediate result and chain the next retrieval* — retrieval + computation.
