"""Template router — learn which :mod:`templates` template to fire per query.

Pipeline: featurize a query (embedding + lexical signals) -> label it with the template
that best retrieves its gold doc -> train an XGB-style classifier -> predict per query.
"""

from __future__ import annotations

import pickle
import re
from typing import Optional

import numpy as np

from .templates import TEMPLATE_NAMES, apply_template, extract_codes

_QWORDS = ("how", "what", "which", "where", "why", "when", "who", "can", "does", "is")


def lexical_features(query: str) -> list[float]:
    q = query.strip()
    toks = q.split()
    low = q.lower()
    codes = extract_codes(q)
    return [
        float(len(toks)),                                   # length
        float(len(codes)),                                  # #part-number tokens
        float(bool(codes)),                                 # has part-number
        float(any(ch.isdigit() for ch in q)),               # has digit
        float(bool(re.search(r"[A-Z]{2,}", q))),            # has acronym/caps
        float(low.endswith("?") or low.split()[:1] in ([w] for w in _QWORDS)),  # question-ish
        float(any(low.startswith(w + " ") for w in _QWORDS)),
        float(len(q)),                                      # chars
    ]


def featurize(query: str, emb: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(emb, dtype=np.float32), lexical_features(query)])


def label_from_pools(pools: dict, rr: Optional[dict], gold_id: str, k: int = 10):
    """Return (best_template, hits) where hits[name]=1 if gold in template's top-k.
    best = the template that ranks gold highest; 'none' if no template finds it."""
    hits, best, best_rank = {}, "none", 1e9
    for name in TEMPLATE_NAMES:
        ids = apply_template(name, pools, rr, top_k=k)
        if gold_id in ids:
            hits[name] = 1
            rank = ids.index(gold_id)
            if rank < best_rank:
                best_rank, best = rank, name
        else:
            hits[name] = 0
    return best, hits


class TemplateRouter:
    """Fitted classifier: query features -> template name. Persists to the pack."""

    def __init__(self, model=None, classes=None, emb_dim: int = 0, metrics: Optional[dict] = None):
        self.model = model
        self.classes = classes or []
        self.emb_dim = emb_dim
        self.metrics = metrics or {}

    def predict(self, query: str, emb: np.ndarray) -> str:
        if self.model is None:
            return "all_rerank"
        x = featurize(query, emb).reshape(1, -1)
        return str(self.model.predict(x)[0])

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "classes": self.classes,
                         "emb_dim": self.emb_dim, "metrics": self.metrics}, f)

    @classmethod
    def load(cls, path) -> "TemplateRouter":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["model"], d["classes"], d["emb_dim"], d.get("metrics", {}))


def train_router(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Train HistGradientBoosting (XGB-style) with CV. Returns model + metrics."""
    from collections import Counter

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    counts = Counter(y)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         max_depth=6, random_state=seed)
    # CV needs >=2 per class; fold count bounded by the rarest class
    min_class = min(counts.values()) if counts else 0
    folds = max(2, min(5, min_class))
    cv_acc = None
    if len(counts) >= 2 and min_class >= 2:
        scores = cross_val_score(clf, X, y, cv=folds, scoring="accuracy")
        cv_acc = (float(scores.mean()), float(scores.std()))
    clf.fit(X, y)
    return {"model": clf, "classes": sorted(counts),
            "label_counts": dict(counts), "cv_folds": folds,
            "cv_accuracy": cv_acc[0] if cv_acc else None,
            "cv_std": cv_acc[1] if cv_acc else None,
            "train_accuracy": float((clf.predict(X) == y).mean())}
