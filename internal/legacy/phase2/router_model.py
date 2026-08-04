"""Router Step 2 — model the query→primitive preference from the tagged data.

Learns to pick the arm per query. Evaluated OUT-OF-FOLD (5-fold CV) so numbers are
held-out. Compares: always-dense (best fixed) vs learned router vs oracle ceiling.
Two feature sets: probe-only (13 cheap signals) and probe+qvec (768-d text repr).
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict

from phase1 import common

ARMS = ["dense", "keyword", "hybrid_.8", "prf", "dense+rerank", "hybrid+rerank",
        "expand_fuse", "expand_fuse+rerank"]


def main():
    data = json.loads((common.REPO / "phase2" / "runs" / "router_data.json").read_text())
    qids = list(data)
    feat_names = list(data[qids[0]]["feats"])
    Xp = np.array([[data[q]["feats"][f] for f in feat_names] for q in qids], dtype=np.float32)
    Xq = np.array([data[q]["qvec"] for q in qids], dtype=np.float32)
    Xf = np.hstack([Xp, Xq])
    Y = np.array([[data[q]["arms"][a] for a in ARMS] for q in qids], dtype=np.float32)  # recall per arm
    best = Y.argmax(1)
    dense_i = ARMS.index("dense")

    always_dense = Y[:, dense_i].mean()
    oracle = Y.max(1).mean()
    print(f"n={len(qids)}  arms={len(ARMS)}  probe_feats={len(feat_names)}")
    print(f"always-dense (best fixed): {always_dense:.4f}")
    print(f"oracle (routing ceiling) : {oracle:.4f}   headroom +{oracle-always_dense:.4f}\n")

    cv = KFold(5, shuffle=True, random_state=0)

    def router_recall_reg(X):
        # per-arm recall regressor -> pick argmax predicted -> actual recall of that arm
        pred = np.zeros_like(Y)
        for a in range(len(ARMS)):
            pred[:, a] = cross_val_predict(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05),
                                           X, Y[:, a], cv=cv)
        pick = pred.argmax(1)
        return Y[np.arange(len(qids)), pick].mean(), pick

    def router_recall_clf(X):
        pick = cross_val_predict(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05),
                                 X, best, cv=cv)
        return Y[np.arange(len(qids)), pick].mean(), pick

    for name, X in [("probe-only (13)", Xp), ("probe+qvec (781)", Xf)]:
        rr, pick = router_recall_reg(X)
        rc, _ = router_recall_clf(X)
        dev = int((pick != dense_i).sum())
        print(f"[{name}]")
        print(f"  router (per-arm regression) recall@10 = {rr:.4f}   "
              f"(+{rr-always_dense:.4f} vs dense, captures {100*(rr-always_dense)/(oracle-always_dense):.0f}% of headroom)")
        print(f"  router (classifier)         recall@10 = {rc:.4f}   (+{rc-always_dense:.4f} vs dense)")
        print(f"  router deviates from dense on {dev}/{len(qids)} queries\n")

    # which probe features drive routing (permutation importance on best-arm classifier)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(Xp, best)
    imp = permutation_importance(clf, Xp, best, n_repeats=10, random_state=0)
    order = imp.importances_mean.argsort()[::-1]
    print("top probe features driving arm choice:")
    for i in order[:8]:
        print(f"  {feat_names[i]:18s} {imp.importances_mean[i]:.4f}")


if __name__ == "__main__":
    main()
