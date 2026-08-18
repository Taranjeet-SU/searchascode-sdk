"""Pool every per-dataset qrels-labeled shard set into ONE global template router.

All datasets are embedded with the same model (gte-base), so the query feature vectors live in
one space and a single classifier learns to route across domains.

    python -m phase2.global_router
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

from search_as_code.explore import ProfilePack, TemplateRouter, load_dataset
from search_as_code.explore.training import RouterDataset, train_router_model


def main():
    packs = sorted(glob.glob("phase2/runs/*_qrels_pack"))
    if not packs:
        print("no *_qrels_pack found — run phase2.beir_qrels first")
        return
    Xs, ys, per = [], [], {}
    for pd in packs:
        ds = load_dataset(ProfilePack.open(pd))
        if len(ds):
            Xs.append(ds.X); ys.extend(ds.y)
            per[Path(pd).name.replace("_qrels_pack", "")] = {
                "n": ds.meta["n"], "oracle": ds.meta["oracle_coverage"],
                "labels": ds.meta["label_distribution"]}
    X = np.concatenate(Xs)
    solved = sum(1 for lab in ys if lab != "none")
    meta = {"n": len(ys), "solved": solved, "unsolved": len(ys) - solved,
            "oracle_coverage": round(solved / len(ys), 4),
            "datasets": per,
            "label_distribution": dict(Counter(lab for lab in ys if lab != "none")),
            "n_templates": 16, "template_hit_rate@k": {}}
    ds = RouterDataset(X=X, y=ys, queries=[], meta=meta)

    model, m = train_router_model(ds, "hist_gb", cv=5, max_iter=400, learning_rate=0.07)
    out = Path("phase2/runs/global_router")
    out.mkdir(parents=True, exist_ok=True)
    if model is not None:
        TemplateRouter(model, classes=sorted(set(ys) - {"none"}),
                       emb_dim=X.shape[1] - 8, metrics=m).save(out / "router.pkl")
    (out / "global_meta.json").write_text(json.dumps(m, indent=2))

    print("===== GLOBAL TEMPLATE ROUTER =====")
    print(f"  datasets pooled : {len(per)}  -> {list(per)}")
    print(f"  total queries   : {m['n']}  (solved {m['solved']}, oracle {m['oracle_coverage']:.3f})")
    print(f"  label dist      : {m['label_distribution']}")
    if m.get("cv_accuracy") is not None:
        print(f"  best single tmpl: {m['best_single_template_acc']:.3f}")
        print(f"  GLOBAL CV ACC   : {m['cv_accuracy']:.3f} +/- {m['cv_std']:.3f}")
        print(f"  lift vs fixed   : {m['router_lift_over_fixed']:+.3f}")
    else:
        print(f"  (no router: {m.get('note')})")
    print("  per-dataset:")
    for name, d in per.items():
        print(f"    {name:14s} n={d['n']:5d} oracle={d['oracle']:.3f} labels={d['labels']}")
    print(f"[global] saved -> {out}")


if __name__ == "__main__":
    main()
