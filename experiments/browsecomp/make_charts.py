"""Render BrowseComp-Plus 3-arm charts with the standard data-viz palette.

Two figures (light surface):
  bc_recall.png  — grouped bars: recall@10 / recall@20 per arm (accuracy)
  bc_tokens.png  — avg input tokens per query per arm (efficiency; the Hornet story)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
d = json.loads((HERE / "bc_recall.json").read_text())
arms = d["arms"]
cfg = d["config"]

# --- palette (reference instance, light surface) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECOND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
CAT = {"dense": "#2a78d6", "tool": "#eb6834", "sac": "#1baf7a"}  # slots 1,2,3
LABEL = {"dense": "Dense", "tool": "Tool-calling", "sac": "SAC code-mode"}
ORDER = ["dense", "tool", "sac"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": SECOND,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(length=0)


# ---------- Figure 1: recall@10 / recall@20 ----------
fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=140)
style(ax)
metrics = [("recall@10", "recall@10"), ("recall@20", "recall@20"), ("all_golds@10", "all-golds@10")]
x = range(len(metrics))
w = 0.26
for i, a in enumerate(ORDER):
    vals = [arms[a][m[0]] for m in metrics]
    xs = [xi + (i - 1) * w for xi in x]
    bars = ax.bar(xs, vals, w, color=CAT[a], label=LABEL[a], zorder=3,
                  edgecolor=SURFACE, linewidth=1.5)
    for xb, v in zip(xs, vals):
        ax.text(xb, v + 0.006, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8, color=INK, fontweight="medium")
ax.set_xticks(list(x))
ax.set_xticklabels([m[1] for m in metrics], color=SECOND, fontsize=10)
ax.set_ylabel("score (fraction of gold docs found)", fontsize=9)
ax.set_ylim(0, max(0.35, max(arms[a]["recall@20"] for a in ORDER) * 1.25))
ax.set_title("BrowseComp-Plus — retrieval quality by harness",
             fontsize=12, color=INK, fontweight="bold", loc="left", pad=12)
ax.text(0, 1.02, f"n={cfg['n_sample']} queries · gte-base · budget={cfg['budget']} · corpus={cfg['corpus_size']:,}",
        transform=ax.transAxes, fontsize=8.5, color=MUTED)
ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=3, columnspacing=1.2)
fig.tight_layout()
fig.savefig(FIG / "bc_recall.png", facecolor=SURFACE)
print("wrote figures/bc_recall.png")

# ---------- Figure 2: cost — input tokens + model turns (small multiples) ----------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 4.3), dpi=140)
xs = range(len(ORDER))

# left: avg input tokens / query
style(axL)
tok = [arms[a]["avg_in_tokens"] for a in ORDER]
axL.bar(xs, tok, 0.55, color=[CAT[a] for a in ORDER], zorder=3,
        edgecolor=SURFACE, linewidth=1.5)
for xb, v in zip(xs, tok):
    axL.text(xb, v + max(tok) * 0.01, f"{v:,}", ha="center", va="bottom",
             fontsize=9, color=INK, fontweight="medium")
axL.set_xticks(list(xs)); axL.set_xticklabels([LABEL[a] for a in ORDER], color=SECOND, fontsize=9)
axL.set_ylabel("avg input tokens / query", fontsize=9)
axL.set_ylim(0, max(tok) * 1.18)
axL.set_title("prompt-token cost", fontsize=10.5, color=INK, fontweight="bold", loc="left", pad=8)

# right: avg model turns / query
style(axR)
turns = [arms[a]["avg_turns"] for a in ORDER]
axR.bar(xs, turns, 0.55, color=[CAT[a] for a in ORDER], zorder=3,
        edgecolor=SURFACE, linewidth=1.5)
for xb, v in zip(xs, turns):
    axR.text(xb, v + max(turns) * 0.01, f"{v:.1f}", ha="center", va="bottom",
             fontsize=9, color=INK, fontweight="medium")
axR.set_xticks(list(xs)); axR.set_xticklabels([LABEL[a] for a in ORDER], color=SECOND, fontsize=9)
axR.set_ylabel("avg model turns / query", fontsize=9)
axR.set_ylim(0, max(turns) * 1.18)
axR.set_title("model turns", fontsize=10.5, color=INK, fontweight="bold", loc="left", pad=8)

fig.suptitle("BrowseComp-Plus — cost by harness (code-mode: 1 program, results out of context)",
             fontsize=12, color=INK, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIG / "bc_cost.png", facecolor=SURFACE)
print("wrote figures/bc_cost.png")

# ---------- Figure 3: per-query recall@10 distribution (from bc_perquery.jsonl) ----------
pq = [json.loads(l) for l in (HERE / "bc_perquery.jsonl").open()]
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=140)
style(ax)
import numpy as np
bins = np.linspace(0, 1, 11)
for i, a in enumerate(ORDER):
    vals = [r[a]["recall@10"] for r in pq]
    ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=CAT[a], label=LABEL[a], zorder=3)
ax.set_xlabel("per-query recall@10", fontsize=9)
ax.set_ylabel("# queries", fontsize=9)
ax.set_title("BrowseComp-Plus — per-query recall@10 distribution",
             fontsize=12, color=INK, fontweight="bold", loc="left", pad=12)
ax.text(0, 1.02, f"n={len(pq)} queries — most queries score 0 on this hard benchmark",
        transform=ax.transAxes, fontsize=8.5, color=MUTED)
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "bc_recall_dist.png", facecolor=SURFACE)
print("wrote figures/bc_recall_dist.png")
