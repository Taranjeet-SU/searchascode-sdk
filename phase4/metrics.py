"""Answer-generation metrics — SQuAD/HotpotQA-standard Exact Match and token-F1,
so our numbers are directly comparable to published leaderboards. Plus a bootstrap
95% CI helper for honest significance.
"""
from __future__ import annotations

import random
import re
import string
from collections import Counter


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


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% bootstrap CI."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    mean = sum(values) / n
    return mean, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]
