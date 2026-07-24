"""XGB router training on the combo-ranking exploration (expanded router step 2).

Reports: per-combo all_found, best-combo distribution, and the per-DIFFICULTY best combo
(the real routing signal). Trains HistGradientBoosting (XGB-style) on query embedding
(+ difficulty/src) -> best combo, with CV vs the majority-combo baseline. If CV >> majority,
there is learnable routing structure; the trained model is saved for router-conditioned use.

    python -m phase4.altera_router_train2
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from phase1 import common

EXP = Path(common.REPO) / "phase4" / "runs" / "router_explore2_altera.json"
OUT = Path(common.REPO) / "phase4" / "runs" / "router_model2_altera.json"
COMBOS = ["dense", "keyword", "kb", "hybrid", "kb_expanded", "expand_fuse", "fanout", "fanout_rerank"]


def main():
    data = json.loads(EXP.read_text())
    solved = [d for d in data if d["best"] != "none"]
    print(f"[router2] {len(data)} queries, {len(solved)} solved")
    per = {c: np.mean([d["hits"][c] for d in data]) for c in COMBOS}
    print("  per-combo all_found:", {c: round(per[c], 3) for c in COMBOS})
    oracle = np.mean([1 if d["best"] != "none" else 0 for d in data])
    print(f"  oracle(any combo)={oracle:.3f}  single-best-combo={max(per.values()):.3f}")
    # per-difficulty best combo (the routing story)
    print("  best-combo by difficulty:")
    for diff in ["easy", "medium", "hard"]:
        sub = [d["best"] for d in solved if d["difficulty"] == diff]
        if sub:
            print(f"    {diff:7s} (n={len(sub)}): {dict(Counter(sub).most_common(4))}")
    print("  best-combo by source:")
    for src in ["doc", "kg"]:
        sub = [d["best"] for d in solved if d["src"] == src]
        if sub:
            print(f"    {src:4s} (n={len(sub)}): {dict(Counter(sub).most_common(4))}")

    if len(solved) < 60:
        print("[router2] too few solved to train."); return
    difs = {"easy": 0, "medium": 1, "hard": 2}; srcs = {"doc": 0, "kg": 1}
    X = np.array([d["emb"] + [difs.get(d["difficulty"], 0), srcs.get(d["src"], 0)] for d in solved], dtype=np.float32)
    y = np.array([d["best"] for d in solved])
    maj = Counter(y).most_common(1)[0]; majority = maj[1] / len(y)
    kmin = min(5, Counter(y).most_common()[-1][1])
    cv = cross_val_score(HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07),
                         X, y, cv=max(2, kmin), scoring="accuracy")
    print(f"\n[router2] labels: {dict(Counter(y))}")
    print(f"[router2] majority-combo baseline = {majority:.3f} (always '{maj[0]}')")
    print(f"[router2] XGB router CV acc = {cv.mean():.3f} +/- {cv.std():.3f}")
    verdict = ("ROUTING STRUCTURE: per-query combo selection beats always-one-combo"
               if cv.mean() > majority + 0.03 else
               "WEAK/NO routing structure: one combo dominates")
    print(f"[router2] => {verdict}")
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07).fit(X, y)
    import pickle
    pickle.dump(clf, open(Path(common.REPO) / "phase4" / "runs" / "router2.pkl", "wb"))
    OUT.write_text(json.dumps({"n_solved": len(solved), "per_combo": per, "oracle": oracle,
                               "single_best": max(per.values()), "majority": majority,
                               "cv_acc": float(cv.mean()), "cv_std": float(cv.std()),
                               "labels": dict(Counter(y)), "verdict": verdict}, indent=2))
    print(f"[router2] saved {OUT} + router2.pkl")


if __name__ == "__main__":
    main()
