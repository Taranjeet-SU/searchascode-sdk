"""Charts for the sac.explore LEARNING writeup (README.md).

Palette (dataviz skill, CVD-validated categorical slots):
  dense=#2a78d6 (blue)  router=#1baf7a (aqua)  oracle=#eda100 (amber)
Sources: experiments/primitive_selection/model_bakeoff.json + csv_*/template_recall.csv + labels.csv
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
PS = HERE.parent / "primitive_selection"
BAKE = json.loads((PS / "model_bakeoff.json").read_text())

DENSE, ROUTER, ORACLE = "#2a78d6", "#1baf7a", "#eda100"
INK, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e3"
DS = [("hotpotqa_multihop", "HotpotQA multi-hop"), ("su_multihop", "SU multi-hop")]


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def chart_routed_recall():
    """THE headline: always-dense vs learned router vs oracle, per dataset."""
    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=140); fig.patch.set_facecolor(SURF)
    labels = [t for _, t in DS]
    dense = [BAKE[k]["baselines"]["always_dense"] for k, _ in DS]
    router = [BAKE[k]["routed_recall"]["hist_gb_grid"]["routed_recall"] for k, _ in DS]
    oracle = [BAKE[k]["baselines"]["oracle"] for k, _ in DS]
    x = range(len(DS)); w = 0.26
    for j, (ys, c, lab) in enumerate([(dense, DENSE, "always-dense"),
                                      (router, ROUTER, "learned router (hist_gb, grid)"),
                                      (oracle, ORACLE, "oracle (any template)")]):
        xs = [i + (j - 1) * w for i in x]
        ax.bar(xs, ys, w * 0.92, color=c, label=lab, zorder=3)
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + 0.012, f"{yi:.3f}", ha="center", va="bottom", fontsize=7.6, color=INK)
    _style(ax)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0); ax.set_ylabel("realized all-golds@10 recall", color=MUTED)
    ax.set_title("What the router learned: routed recall beats always-dense\n(and the headroom to oracle)",
                 color=INK, fontsize=11, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=1)
    fig.tight_layout(); fig.savefig(FIG / "routed_recall_vs_dense.png", facecolor=SURF); plt.close(fig)


def chart_model_bakeoff():
    """CV classification accuracy per model head, per dataset — no model dominates."""
    models = ["hist_gb", "xgb", "logreg", "random_forest", "mlp"]
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=140); fig.patch.set_facecolor(SURF)
    w = 0.38
    cols = [ROUTER, DENSE]
    for j, (k, t) in enumerate(DS):
        ys = [BAKE[k]["cv_accuracy"][m]["cv_acc"] for m in models]
        xs = [i + (j - 0.5) * w for i in range(len(models))]
        ax.bar(xs, ys, w * 0.92, color=cols[j], label=t, zorder=3)
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + 0.004, f"{yi:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    _style(ax)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(0.5, 0.72); ax.set_ylabel("5-fold CV classification accuracy", color=MUTED)
    ax.set_title("Model bake-off: all heads cluster ~0.62–0.64 — no model breaks the ceiling",
                 color=INK, fontsize=10.5, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "model_bakeoff.png", facecolor=SURF); plt.close(fig)


def _winner_dist(csv_path, top=8):
    c = Counter()
    for r in csv.DictReader(open(csv_path)):
        if r["winner"] != "none":
            c[r["winner"]] += 1
    return c.most_common(top)


def chart_winner_distribution():
    """Winning-template distribution per dataset (contrast BEIR's ~84% light_dense)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=140); fig.patch.set_facecolor(SURF)
    for ax, (k, t) in zip(axes, DS):
        dist = _winner_dist(PS / f"csv_{k}" / "labels.csv")
        names = [n for n, _ in dist][::-1]; vals = [v for _, v in dist][::-1]
        ax.barh(range(len(names)), vals, color=ROUTER, zorder=3, height=0.7)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.01, i, str(v), va="center", fontsize=7.5, color=INK)
        _style(ax); ax.grid(axis="x", color=GRID, linewidth=0.8); ax.grid(axis="y", visible=False)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        ax.set_title(t, color=INK, fontsize=10, loc="left")
    fig.suptitle("Winning-template distribution (cheapest all-golds solver) — the label spreads "
                 "off light_dense on multi-hop", color=INK, fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "winner_distribution.png", facecolor=SURF); plt.close(fig)


def chart_template_recall():
    """Per-template all-golds@10 recall (top templates), per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=140); fig.patch.set_facecolor(SURF)
    for ax, (k, t) in zip(axes, DS):
        rows = list(csv.DictReader(open(PS / f"csv_{k}" / "template_recall.csv")))
        rows = sorted(rows, key=lambda r: -float(r["recall@k"]))[:8]
        names = [r["template"] for r in rows][::-1]
        vals = [float(r["recall@k"]) for r in rows][::-1]
        ax.barh(range(len(names)), vals, color=DENSE, zorder=3, height=0.7)
        for i, v in enumerate(vals):
            ax.text(v + 0.008, i, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
        _style(ax); ax.grid(axis="x", color=GRID, linewidth=0.8); ax.grid(axis="y", visible=False)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0, 1.0); ax.set_title(t, color=INK, fontsize=10, loc="left")
    fig.suptitle("Per-template all-golds@10 recall (cascade-labeled: dear templates measured on the "
                 "residual only)", color=INK, fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "template_recall.png", facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    chart_routed_recall()
    chart_model_bakeoff()
    chart_winner_distribution()
    chart_template_recall()
    print(f"charts -> {FIG}")
