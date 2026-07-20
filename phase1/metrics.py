"""Standard IR metrics for the recall benchmark (doc-level, BEIR-style)."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(ranked_ids: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    if not gold:
        return 0.0
    return len(set(ranked_ids[:k]) & gold) / len(gold)


def mrr_at_k(ranked_ids: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    for i, d in enumerate(ranked_ids[:k], start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], gold: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, d in enumerate(ranked_ids[:k]):
        rel = gold.get(d, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 2)
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal) if r > 0)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(rankings: dict[str, Sequence[str]], qrels: dict[str, dict[str, int]],
             k: int = 10) -> dict[str, float]:
    """Average Recall@k, nDCG@k, MRR@k over the queries present in ``rankings``."""
    rec, ndcg, mrr, n = 0.0, 0.0, 0.0, 0
    for qid, ranked in rankings.items():
        gold = {d: s for d, s in qrels.get(qid, {}).items() if s > 0}
        if not gold:
            continue
        rec += recall_at_k(ranked, gold, k)
        ndcg += ndcg_at_k(ranked, gold, k)
        mrr += mrr_at_k(ranked, gold, k)
        n += 1
    n = n or 1
    return {f"recall@{k}": rec / n, f"ndcg@{k}": ndcg / n, f"mrr@{k}": mrr / n, "n_queries": n}
