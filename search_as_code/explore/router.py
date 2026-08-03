"""Template router — learn which :mod:`templates` template to fire per query.

Pipeline: featurize a query (embedding + lexical signals) -> label it with the template
that best retrieves its gold doc -> train an XGB-style classifier -> predict per query.
"""

from __future__ import annotations

import pickle
import re
from typing import Optional

import numpy as np

from .templates import TEMPLATE_COST, TEMPLATE_NAMES, extract_codes, run_template

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


def best_from_hits(hits: dict) -> str:
    """Winner policy over recall@k hits: the **cheapest-effort** template that retrieved the
    gold doc (so the router learns the lightest strategy that works). 'none' if all missed."""
    winners = [t for t in TEMPLATE_NAMES if hits.get(t)]
    if not winners:
        return "none"
    return min(winners, key=lambda t: (TEMPLATE_COST.get(t, 9), TEMPLATE_NAMES.index(t)))


def label_via_templates(ctx, gold, k: int = 10, cascade: bool = True, all_golds: bool = True):
    """Return (best_template, hits) where best = the **cheapest template that returns ALL the gold
    answers** in its top-k, and hits[name]=1 iff that template met the success criterion.

    ``gold``: a single id or a set/list of relevant ids (qrels / multi-hop can have several).

    ``all_golds`` (**default True** = cheapest-to-all-golds): a template "solves" a query iff **ALL**
    gold ids are in its top-k (``gold_set ⊆ top_k``) — the correct criterion when the gold is a SET
    of N documents that must ALL be retrieved (multi-hop). This **reduces exactly to single-gold
    recall@k when the gold set has one element** (``{g} ⊆ got`` ≡ ``g ∈ got``), so single-gold
    corpora (BEIR) are unaffected. Set ``all_golds=False`` for the older "ANY gold in top-k" gate.

    ``cascade`` (default): evaluate templates cheapest-first and **stop at the first cost group
    that solves the query** — so the expensive LLM strategies only run on the queries the cheap
    ones miss (which is exactly the unsolved/gap subset). Exact for the winner label; it only
    under-measures the hit-rate of dear templates on easy queries.
    """
    from itertools import groupby

    golds = {gold} if isinstance(gold, str) else set(gold)

    def _hit(name):
        got = set(run_template(name, ctx, top_k=k))
        return int(golds <= got) if all_golds else int(bool(golds & got))

    order = sorted(TEMPLATE_NAMES, key=lambda t: (TEMPLATE_COST.get(t, 99), TEMPLATE_NAMES.index(t)))
    hits: dict = {}
    if not cascade:
        for name in order:
            hits[name] = _hit(name)
        return best_from_hits(hits), hits
    for _cost, grp in groupby(order, key=lambda t: TEMPLATE_COST.get(t, 99)):
        grp = list(grp)
        for name in grp:
            hits[name] = _hit(name)
        if any(hits[n] for n in grp):
            return best_from_hits(hits), hits     # cheapest solver is in this group
    return "none", hits


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

    def route_plan(self, query: str, emb: np.ndarray, top_k: int = 3) -> list[str]:
        """A ranked strategy CASCADE for one query: the top_k templates by predicted probability
        (cheapest-first among near-ties). The code agent executes it as t1 → escalate to t2 → t3,
        which is where routing actually pays off (chaining, not betting on a single template).
        Falls back to a single predict when the head has no ``predict_proba``."""
        if self.model is None:
            return ["all_rerank"]
        x = featurize(query, emb).reshape(1, -1)
        if not hasattr(self.model, "predict_proba"):
            return [str(self.model.predict(x)[0])]
        from .templates import TEMPLATE_COST
        proba = self.model.predict_proba(x)[0]
        classes = list(getattr(self.model, "classes_", self.classes))
        order = sorted(range(len(classes)),
                       key=lambda i: (-float(proba[i]), TEMPLATE_COST.get(str(classes[i]), 99)))
        return [str(classes[i]) for i in order[:top_k]]

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "classes": self.classes,
                         "emb_dim": self.emb_dim, "metrics": self.metrics}, f)

    @classmethod
    def load(cls, path) -> "TemplateRouter":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["model"], d["classes"], d["emb_dim"], d.get("metrics", {}))


def format_route_plan(plan: list[str]) -> str:
    """Render a :meth:`TemplateRouter.route_plan` cascade as a per-query prompt hint the code
    agent can follow — the learned ``t1 → t2 → t3`` plan, with what each strategy does."""
    from .templates import TEMPLATE_DOCS
    if not plan:
        return ""
    steps = " → ".join(plan)
    detail = "; ".join(f"{t} ({TEMPLATE_DOCS.get(t, {}).get('does', '')})" for t in plan)
    return (f"Learned strategy plan for THIS query (router-ranked, cheapest-effective first): "
            f"{steps}. Execute as a cascade — start with the first; escalate to the next only if "
            f"the result looks weak. What each does: {detail}.")


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
