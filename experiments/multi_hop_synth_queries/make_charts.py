"""Render the benchmark charts (PNG) for RESULTS.md from recall_fair.json + per-query jsonl.

Palette (dataviz skill, first 3 categorical slots — CVD all-pairs validated):
  dense=#2a78d6 (blue)  tool=#eb6834 (orange)  sac=#1baf7a (aqua)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
COL = {"dense": "#2a78d6", "tool": "#eb6834", "sac": "#1baf7a"}
ARMS = ["dense", "tool", "sac"]
HOPS = [2, 3, 4]
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

agg = json.loads((BASE / "recall_fair.json").read_text())
pq = [json.loads(x) for x in (BASE / "recall_fair_perquery.jsonl").read_text().splitlines() if x.strip()]


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)


def bars(metric, title, fname):
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=140); fig.patch.set_facecolor(SURF)
    w = 0.26
    for j, a in enumerate(ARMS):
        xs = [i + (j - 1) * w for i in range(len(HOPS))]
        ys = [agg[f"{h}hop"][a][metric] for h in HOPS]
        ax.bar(xs, ys, w * 0.92, color=COL[a], label=a, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    _style(ax)
    ax.set_xticks(range(len(HOPS))); ax.set_xticklabels([f"{h}-hop" for h in HOPS])
    ax.set_ylim(0, 1.05); ax.set_ylabel(metric, color=MUTED)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / fname, facecolor=SURF); plt.close(fig)


def dist(field, title, fname, logy=False):
    """Distribution of a per-query field across the n queries (box), faceted by hop."""
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), dpi=140, sharey=True)
    fig.patch.set_facecolor(SURF)
    for hi, h in enumerate(HOPS):
        ax = axes[hi]
        data = [[r[field] for r in pq if r["hop"] == h and r["arm"] == a] for a in ARMS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.4))
        for patch, a in zip(bp["boxes"], ARMS):
            patch.set_facecolor(COL[a]); patch.set_edgecolor(MUTED); patch.set_alpha(0.9)
        _style(ax)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(ARMS, fontsize=8)
        ax.set_title(f"{h}-hop", color=MUTED, fontsize=9)
    axes[0].set_ylabel(field, color=MUTED)
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / fname, facecolor=SURF); plt.close(fig)


def tokens_vs_hops(fname):
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=140); fig.patch.set_facecolor(SURF)
    for a in ("tool", "sac"):
        ys = [agg[f"{h}hop"][a]["avg_in_tokens"] for h in HOPS]
        ax.plot(HOPS, ys, "-o", color=COL[a], linewidth=2, markersize=7, label=a, zorder=3)
        for x, y in zip(HOPS, ys):
            ax.text(x, y * 1.12, f"{y:,}", ha="center", fontsize=8, color=INK)
    _style(ax); ax.set_yscale("log")
    ax.set_xticks(HOPS); ax.set_xticklabels([f"{h}-hop" for h in HOPS])
    ax.set_ylabel("avg input tokens / query (log)", color=MUTED)
    ax.set_title("Context cost vs hop depth — SAC stays flat, tool-calling grows", color=INK,
                 fontsize=10.5, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / fname, facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    bars("recall@10", "recall@10 by hop count (n per hop)", "recall_by_hop.png")
    bars("all_golds@10", "all_golds@10 (all N docs retrieved) by hop count", "allgolds_by_hop.png")
    dist("in_tok", "Input-token distribution over queries (log y)", "tokens_dist.png", logy=True)
    dist("searches", "Searches (hops) distribution over queries", "searches_dist.png")
    tokens_vs_hops("tokens_vs_hops.png")
    print(f"charts -> {FIG}")
