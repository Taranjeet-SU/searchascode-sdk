"""Retrieval and answer metrics, with honest uncertainty.

Promoted from ``phase4/metrics.py`` (issues.md P4-8). That file held SQuAD-standard EM /
token-F1 **and ``bootstrap_ci`` — the only significance testing anywhere in this repository** —
while every headline number elsewhere is a bare mean. It sat unpromoted in the customer phase,
which is exactly the gap ``soul.md`` rule 2 ("improve the SDK, don't fork it") exists to close
and which the audit traced to the missing ``learnings_standard.md`` workflow (LEG-2).

Concretely, DJ-2 records a judge "improvement" of +0.011 balanced accuracy at n=100, where the
95% interval is ±0.088 — eight times the claimed gain. :func:`bootstrap_ci` and
:func:`compare` make that visible at the point the number is produced.

    from search_as_code.metrics import recall_at_k, all_golds_at_k, bootstrap_ci, compare

    per_query = [recall_at_k(ids, gold) for ids, gold in runs]
    mean, lo, hi = bootstrap_ci(per_query)
"""
from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
from typing import Iterable, Sequence

__all__ = [
    "normalize_answer", "exact_match", "token_f1", "score",
    "recall_at_k", "all_golds_at_k", "reciprocal_rank", "ndcg_at_k",
    "bootstrap_ci", "compare", "format_ci",
]


# --------------------------------------------------------------------------- #
# answer metrics (SQuAD / HotpotQA standard, so numbers compare to leaderboards)
# --------------------------------------------------------------------------- #
def normalize_answer(s: str) -> str:
    """SQuAD normalization: lowercase, strip punctuation, articles, extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec, rec = num_same / len(p), num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def score(pred: str, golds) -> tuple[float, float]:
    """Best EM/F1 over a list of acceptable gold answers (or a single string)."""
    if isinstance(golds, str):
        golds = [golds]
    em = max((exact_match(pred, g) for g in golds), default=0.0)
    f1 = max((token_f1(pred, g) for g in golds), default=0.0)
    return em, f1


# --------------------------------------------------------------------------- #
# retrieval metrics                                                             #
# --------------------------------------------------------------------------- #
def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int | None = None) -> float:
    """|gold ∩ top-k| / |gold| — the fraction of required documents that were found."""
    g = {str(x) for x in gold}
    if not g:
        return 0.0
    top = [str(x) for x in (retrieved[:k] if k else retrieved)]
    return len(g & set(top)) / len(g)


def all_golds_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int | None = None) -> float:
    """1.0 iff EVERY gold document is in the top-k — the multi-hop success criterion.

    Distinct from recall: a 4-hop query with 3 of 4 golds scores recall 0.75 but all-golds 0.
    """
    g = {str(x) for x in gold}
    if not g:
        return 0.0
    top = {str(x) for x in (retrieved[:k] if k else retrieved)}
    return float(g <= top)


def reciprocal_rank(retrieved: Sequence[str], gold: Iterable[str]) -> float:
    g = {str(x) for x in gold}
    for i, doc_id in enumerate(retrieved):
        if str(doc_id) in g:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold, k: int = 10) -> float:
    """nDCG@k with binary or graded relevance (``gold`` may be a set or an {id: gain} dict)."""
    gains = gold if isinstance(gold, dict) else {str(x): 1.0 for x in gold}
    dcg = sum(float(gains.get(str(d), 0.0)) / math.log2(i + 2)
              for i, d in enumerate(retrieved[:k]))
    ideal = sorted((float(v) for v in gains.values()), reverse=True)[:k]
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


# --------------------------------------------------------------------------- #
# uncertainty — attach it to the number, not to a paragraph of prose            #
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: Sequence[float], n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """``(mean, lo, hi)`` with a percentile bootstrap CI (95% by default)."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    mean = sum(values) / n
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return mean, lo, hi


def compare(a: Sequence[float], b: Sequence[float], n_boot: int = 2000,
            seed: int = 0, paired: bool = True) -> dict:
    """Compare two arms and say whether the difference is distinguishable from noise.

    ``paired=True`` (default) bootstraps the per-query DIFFERENCE, which is the right test
    when both arms ran on the same queries — the case for every arm comparison in this repo.

    Returns ``delta``, its CI, and ``significant`` (the CI excludes 0). Use it before writing
    "arm A beats arm B": DJ-2 is the cautionary example — a +0.011 delta reported as a finding
    when the interval was ±0.088.
    """
    if not a or not b:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "significant": False, "n": 0}
    if paired and len(a) != len(b):
        paired = False
    rng = random.Random(seed)
    if paired:
        diffs = [x - y for x, y in zip(a, b)]
        mean, lo, hi = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
        n = len(diffs)
    else:
        na, nb = len(a), len(b)
        deltas = []
        for _ in range(n_boot):
            ma = sum(a[rng.randrange(na)] for _ in range(na)) / na
            mb = sum(b[rng.randrange(nb)] for _ in range(nb)) / nb
            deltas.append(ma - mb)
        deltas.sort()
        mean = sum(a) / na - sum(b) / nb
        lo, hi = deltas[int(0.025 * n_boot)], deltas[min(n_boot - 1, int(0.975 * n_boot))]
        n = min(na, nb)
    return {"delta": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "significant": bool(lo > 0 or hi < 0), "n": n, "paired": paired}


def format_ci(mean: float, lo: float, hi: float, places: int = 3) -> str:
    """``0.549 [0.482, 0.615]`` — the form every headline number should be reported in."""
    return f"{mean:.{places}f} [{lo:.{places}f}, {hi:.{places}f}]"
