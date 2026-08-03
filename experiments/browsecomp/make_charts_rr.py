"""Render BrowseComp-Plus rerank+deep charts (the fair code-mode rerun) from bc_recall_rr.json.

Palette (per spec): dense=#2a78d6, tool=#eb6834, sac=#1baf7a, oracle=#eda100, deep(llm)=#4a3aa7.
Figures (light surface):
  bc_recall_rr.png  — grouped bars: recall@10 / recall@20 / all_golds@10 per arm
  bc_cost_rr.png    — avg input tokens + model turns per arm (efficiency)
  bc_recall_dist_rr.png — per-query recall@10 distribution
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
d = json.loads((HERE / "bc_recall_rr.json").read_text())
arms = d["arms"]
cfg = d["config"]

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; SECOND = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
CAT = {"dense": "#2a78d6", "tool": "#eb6834", "sac": "#1baf7a", "sac_rerank": "#eda100",
       "sac_deep_oracle": "#9a8fd8", "sac_deep_llm": "#4a3aa7"}
LABEL = {"dense": "Dense", "tool": "Tool-calling", "sac": "SAC code (coverage)",
         "sac_rerank": "SAC code (rerank-fwd)",
         "sac_deep_oracle": "SAC deep (oracle-judge)", "sac_deep_llm": "SAC deep (LLM-judge)"}
ORDER = ["dense", "tool", "sac", "sac_rerank", "sac_deep_oracle", "sac_deep_llm"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": SECOND, "xtick.color": MUTED, "ytick.color": MUTED,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(length=0)


# ---------- Figure 1: recall@10 / recall@20 / all_golds@10 ----------
fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=140)
style(ax)
metrics = [("recall@10", "recall@10"), ("recall@20", "recall@20"), ("all_golds@10", "all-golds@10")]
x = np.arange(len(metrics))
w = 0.13
for i, a in enumerate(ORDER):
    vals = [arms[a][m[0]] for m in metrics]
    xs = x + (i - (len(ORDER) - 1) / 2) * w
    ax.bar(xs, vals, w, color=CAT[a], label=LABEL[a], zorder=3, edgecolor=SURFACE, linewidth=1.2)
    for xb, v in zip(xs, vals):
        ax.text(xb, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
ax.set_xticks(list(x))
ax.set_xticklabels([m[1] for m in metrics], color=SECOND, fontsize=10)
ax.set_ylabel("score (fraction of gold docs found)", fontsize=9)
ax.set_ylim(0, max(0.2, max(arms[a]["recall@20"] for a in ORDER) * 1.3))
ax.set_title("BrowseComp-Plus — retrieval quality by harness (rerank + deep-mode)",
             fontsize=12, color=INK, fontweight="bold", loc="left", pad=12)
ax.text(0, 1.02, f"n={cfg['n_sample']} queries · gte-base + cross-encoder rerank · budget={cfg['budget']} · corpus={cfg['corpus_size']:,}",
        transform=ax.transAxes, fontsize=8.5, color=MUTED)
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2, columnspacing=1.0)
fig.tight_layout()
fig.savefig(FIG / "bc_recall_rr.png", facecolor=SURFACE)
print("wrote figures/bc_recall_rr.png")

# ---------- Figure 2: cost — input tokens + model turns ----------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=140)
xs = np.arange(len(ORDER))
style(axL)
tok = [arms[a]["avg_in_tokens"] for a in ORDER]
axL.bar(xs, tok, 0.6, color=[CAT[a] for a in ORDER], zorder=3, edgecolor=SURFACE, linewidth=1.2)
for xb, v in zip(xs, tok):
    axL.text(xb, v + max(tok) * 0.01, f"{v:,}", ha="center", va="bottom", fontsize=8, color=INK)
axL.set_xticks(list(xs)); axL.set_xticklabels([LABEL[a] for a in ORDER], color=SECOND, fontsize=7.5, rotation=18, ha="right")
axL.set_ylabel("avg input tokens / query", fontsize=9)
axL.set_ylim(0, max(tok) * 1.18 or 1)
axL.set_title("prompt-token cost", fontsize=10.5, color=INK, fontweight="bold", loc="left", pad=8)

style(axR)
turns = [arms[a]["avg_turns"] for a in ORDER]
axR.bar(xs, turns, 0.6, color=[CAT[a] for a in ORDER], zorder=3, edgecolor=SURFACE, linewidth=1.2)
for xb, v in zip(xs, turns):
    axR.text(xb, v + (max(turns) or 1) * 0.01, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
axR.set_xticks(list(xs)); axR.set_xticklabels([LABEL[a] for a in ORDER], color=SECOND, fontsize=7.5, rotation=18, ha="right")
axR.set_ylabel("avg model turns / query", fontsize=9)
axR.set_ylim(0, (max(turns) or 1) * 1.18)
axR.set_title("model turns (deepening hops)", fontsize=10.5, color=INK, fontweight="bold", loc="left", pad=8)

fig.suptitle("BrowseComp-Plus — cost by harness (oracle-judge = minimal deep-mode cost)",
             fontsize=12, color=INK, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIG / "bc_cost_rr.png", facecolor=SURFACE)
print("wrote figures/bc_cost_rr.png")

# ---------- Figure 3: per-query recall@10 distribution ----------
pq = [json.loads(l) for l in (HERE / "bc_perquery_rr.jsonl").open()]
fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=140)
style(ax)
bins = np.linspace(0, 1, 11)
for a in ORDER:
    vals = [r[a]["recall@10"] for r in pq]
    ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=CAT[a], label=LABEL[a], zorder=3)
ax.set_xlabel("per-query recall@10", fontsize=9)
ax.set_ylabel("# queries", fontsize=9)
ax.set_title("BrowseComp-Plus — per-query recall@10 distribution (rerank + deep)",
             fontsize=12, color=INK, fontweight="bold", loc="left", pad=12)
ax.text(0, 1.02, f"n={len(pq)} queries — a hard deep-research benchmark",
        transform=ax.transAxes, fontsize=8.5, color=MUTED)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "bc_recall_dist_rr.png", facecolor=SURFACE)
print("wrote figures/bc_recall_dist_rr.png")
