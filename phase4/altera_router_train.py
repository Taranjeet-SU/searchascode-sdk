"""XGB step 2: train the learned primitive router on the exploration data.

Features = gte query embedding (+ facet). Label = best arm that retrieved the gold.
Uses sklearn HistGradientBoosting (XGB-style gradient boosting; no extra dep). Reports
5-fold CV accuracy vs the majority-arm baseline — if CV >> majority, there is real
routing structure to exploit; if not, one primitive dominates (no routing needed).

    python -m phase4.altera_router_train
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from phase1 import common

EXP = Path(common.REPO) / "phase4" / "runs" / "router_explore_altera.json"
OUT = Path(common.REPO) / "phase4" / "runs" / "router_model_altera.json"
ARMS = ["dense", "keyword", "kb", "kb_expanded", "hybrid"]


def main():
    data = json.loads(EXP.read_text())
    solved = [d for d in data if d["best"] != "none"]
    print(f"[router] {len(data)} queries, {len(solved)} solved (gold retrieved by some arm)")
    # per-arm hit@10 + oracle (any arm) + how often arms disagree
    per_arm = {a: np.mean([d["hits"][a] for d in data]) for a in ARMS}
    oracle = np.mean([1 if d["best"] != "none" else 0 for d in data])
    disagree = np.mean([1 if 0 < sum(d["hits"].values()) < len(ARMS) else 0 for d in data])
    print("  per-arm hit@10:", {a: round(per_arm[a], 3) for a in ARMS})
    print(f"  oracle (best-per-query) = {oracle:.3f}   single-best-arm = {max(per_arm.values()):.3f}")
    print(f"  arms disagree on {disagree:.1%} of queries  <- routing headroom")

    if len(solved) < 40:
        print("[router] too few solved queries to train a router."); return
    X = np.array([d["emb"] for d in solved], dtype=np.float32)
    y = np.array([d["best"] for d in solved])
    maj = Counter(y).most_common(1)[0]
    majority_acc = maj[1] / len(y)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6)
    k = min(5, Counter(y).most_common()[-1][1])       # folds <= smallest class
    cv = cross_val_score(clf, X, y, cv=max(2, k), scoring="accuracy")
    print(f"\n[router] label dist: {dict(Counter(y))}")
    print(f"[router] majority-arm baseline acc = {majority_acc:.3f} (always '{maj[0]}')")
    print(f"[router] XGB router  CV acc = {cv.mean():.3f} +/- {cv.std():.3f}")
    verdict = ("ROUTING STRUCTURE: router beats majority -> per-query primitive choice helps"
               if cv.mean() > majority_acc + 0.03 else
               "NO routing structure: one primitive dominates -> just use the single best arm")
    print(f"[router] => {verdict}")
    clf.fit(X, y)
    OUT.write_text(json.dumps({"n_solved": len(solved), "per_arm_hit": per_arm, "oracle": oracle,
                               "single_best": max(per_arm.values()), "disagree_rate": disagree,
                               "majority_acc": majority_acc, "cv_acc": float(cv.mean()),
                               "cv_std": float(cv.std()), "labels": dict(Counter(y)),
                               "verdict": verdict}, indent=2))
    print(f"[router] saved {OUT}")


if __name__ == "__main__":
    main()
