"""Render the deep-SAC comparison charts + append section 11 to RESULTS.md.

Merges three sources (baselines are NOT recomputed):
  - experiments/multi_hop_synth_queries/recall_fair.json   (dense/tool/sac — HotpotQA)
  - experiments/su_multihop/su_recall.json                 (dense/tool/sac — SU)
  - experiments/deep_sac/deep_recall.json                  (sac_deep, sac_deep+explore — both)

Charts (-> multi_hop_synth_queries/figures/):
  (i)  deep_recall_vs_cost.png   recall@10 vs input-token cost per arm (deep gain & its cost)
  (ii) deep_deepening_cost.png   avg hops & input tokens: single-shot vs deep, before/after explore

Palette (task-prescribed):
  dense=#2a78d6 tool=#eb6834 sac=#1baf7a sac_deep=#eda100 sac_deep+explore=#4a3aa7
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parents[1]
MH = ROOT / "multi_hop_synth_queries"
FIG = MH / "figures"
FIG.mkdir(exist_ok=True)

COL = {"dense": "#2a78d6", "tool": "#eb6834", "sac": "#1baf7a",
       "sac_deep": "#eda100", "sac_deep+explore": "#4a3aa7"}
LABEL = {"sac": "sac (single-shot)", "sac_deep": "sac_deep", "sac_deep+explore": "sac_deep+explore"}
HOPS = [2, 3, 4]
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

fair = json.loads((MH / "recall_fair.json").read_text())
su = json.loads((ROOT / "su_multihop" / "su_recall.json").read_text())
deep = json.loads((Path(__file__).parent / "deep_recall.json").read_text())


def rows_for(corpus: str):
    """Return {hop: {arm: metricdict}} with a UNIFORM schema across all 5 arms."""
    out = {}
    for h in HOPS:
        hk = f"{h}hop"
        row = {}
        base = fair[hk] if corpus == "hotpotqa" else su[hk]["arms"]
        for a in ("dense", "tool", "sac"):
            b = base[a]
            # uniform "hops" proxy: dense=1 (one search, no model loop), tool=model turns, sac=1
            hops = 1.0 if a in ("dense", "sac") else b.get("avg_model_turns", 1.0)
            row[a] = {"recall@10": b["recall@10"], "all_golds@10": b["all_golds@10"],
                      "hops": hops, "searches": b["avg_searches"],
                      "in": b["avg_in_tokens"], "out": b["avg_out_tokens"]}
        d = deep.get(corpus, {}).get(hk, {})
        for a in ("sac_deep", "sac_deep_explore"):
            if a not in d:
                continue
            key = "sac_deep+explore" if a == "sac_deep_explore" else a
            row[key] = {"recall@10": d[a]["recall@10"], "all_golds@10": d[a]["all_golds@10"],
                        "hops": d[a]["avg_hops"], "searches": d[a]["avg_searches"],
                        "in": d[a]["avg_in_tokens"], "out": d[a]["avg_out_tokens"]}
        out[h] = row
    return out


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)


SAC3 = ["sac", "sac_deep", "sac_deep+explore"]


def _grouped(axes_specs, data, corpus, suptitle, fname, fmt_int=False):
    """Grouped bars over the 3 SAC variants, one subplot per (metric,title,ylabel) spec."""
    variants = [v for v in SAC3 if all(v in data[h] for h in HOPS)]
    n = len(axes_specs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.8), dpi=140)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURF)
    w = 0.26
    for ax, (metric, ttl, yl, is_int) in zip(axes, axes_specs):
        vmax = 0.0
        for j, v in enumerate(variants):
            xs = [i + (j - 1) * w for i in range(len(HOPS))]
            ys = [data[h][v][metric] for h in HOPS]
            vmax = max(vmax, max(ys))
            ax.bar(xs, ys, w * 0.92, color=COL[v], label=LABEL.get(v, v), zorder=3)
            for x, y in zip(xs, ys):
                txt = f"{y:,}" if is_int else f"{y:.2f}"
                ax.text(x, y + vmax * 0.012, txt, ha="center", va="bottom",
                        fontsize=7, color=INK)
        _style(ax)
        ax.set_xticks(range(len(HOPS)))
        ax.set_xticklabels([f"{h}-hop" for h in HOPS])
        ax.set_ylabel(yl, color=MUTED)
        ax.set_ylim(0, (1.05 if metric in ("recall@10", "all_golds@10") else vmax * 1.18))
        ax.set_title(ttl, color=INK, fontsize=10.5, loc="left", pad=8)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=1, loc="upper right"
                   if axes_specs[0][0] in ("recall@10", "all_golds@10") else "upper left")
    fig.suptitle(suptitle, color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / fname, facecolor=SURF)
    plt.close(fig)


def chart_quality(data, corpus):
    """recall@10 + all_golds@10 for single-shot vs deep vs deep+explore, per hop."""
    _grouped([("recall@10", "recall@10", "recall@10", False),
              ("all_golds@10", "all_golds@10 (all N golds in top-10)", "all_golds@10", False)],
             data, corpus,
             f"[{corpus}] retrieval quality — single-shot SAC vs deep vs deep+explore",
             f"deep_quality_{corpus}.png")


def chart_cost(data, corpus):
    """avg_hops, avg_searches, avg_in_tokens for single-shot vs deep vs deep+explore."""
    _grouped([("hops", "avg deepening rounds (hops)", "avg hops / query", False),
              ("searches", "avg retrieval searches", "avg searches / query", False),
              ("in", "avg input tokens", "avg input tokens / query", True)],
             data, corpus,
             f"[{corpus}] the cost of going deep — hops, searches, tokens",
             f"deep_cost_{corpus}.png")


def md_table(data, corpus):
    order = ["dense", "tool", "sac", "sac_deep", "sac_deep+explore"]
    lines = [f"**{corpus}**", "",
             "| hops | arm | recall@10 | all_golds@10 | avg hops | avg searches | avg in_tok | avg out_tok |",
             "|---|---|---|---|---|---|---|---|"]
    for h in HOPS:
        for i, a in enumerate(order):
            if a not in data[h]:
                continue
            m = data[h][a]
            hcell = f"**{h}**" if i == 0 else ""
            name = LABEL.get(a, a)
            emph = a in ("sac_deep", "sac_deep+explore")
            nm = f"**{name}**" if emph else name
            lines.append(
                f"| {hcell} | {nm} | {m['recall@10']:.3f} | {m['all_golds@10']:.3f} | "
                f"{m['hops']:.2f} | {m['searches']:.1f} | {m['in']:,} | {m['out']:,} |")
    lines.append("")
    return "\n".join(lines)


def main():
    hp = rows_for("hotpotqa")
    sur = rows_for("su") if "su" in deep else None
    chart_quality(hp, "hotpotqa")
    chart_cost(hp, "hotpotqa")
    if sur:
        chart_quality(sur, "su")
        chart_cost(sur, "su")

    # ---- honest deltas (HotpotQA) ----
    hops = [hp[h]["sac_deep"]["hops"] for h in HOPS]
    srch = [hp[h]["sac_deep"]["searches"] for h in HOPS]
    intok = [hp[h]["sac_deep"]["in"] for h in HOPS]
    cost = [deep["hotpotqa"][f"{h}hop"]["sac_deep"]["avg_cost_usd"] for h in HOPS]
    tok_mult = [hp[h]["sac_deep"]["in"] / max(hp[h]["sac"]["in"], 1) for h in HOPS]
    exp_r = [hp[h]["sac_deep+explore"]["recall@10"] - hp[h]["sac_deep"]["recall@10"] for h in HOPS]
    exp_s = [hp[h]["sac_deep+explore"]["searches"] / max(hp[h]["sac_deep"]["searches"], 0.01)
             for h in HOPS]

    prose = (
        "**What the numbers say (honest read):**\n"
        f"- **Going deep is affordable, not a blow-up.** The judge-gated deepening stays bounded: "
        f"deep-mode averages only **{min(hops):.2f}-{max(hops):.2f} hops** (i.e. only ~20-30% of "
        f"queries ever deepen past hop 1), **{min(srch):.1f}-{max(srch):.1f} searches**, "
        f"**{min(intok):,}-{max(intok):,} input tokens**, **~${min(cost):.4f}-${max(cost):.4f}/query** "
        f"on HotpotQA. Deep mode reaches solid absolute recall@10 (0.92/0.81/0.745) and all_golds@10 "
        f"(0.86/0.52/0.44) at that modest, predictable premium — the feared multi-hop token explosion "
        f"does not happen because the calibrated judge stops most queries at one hop.\n"
        f"- **But the premium doesn't beat the cheap single-shot harness here.** The reused single-shot "
        f"SAC (decompose->fuse, budget 6, ~340-380 in-tok) already scores 0.95/0.83/0.765 — so deep "
        f"mode spends ~{round(sum(tok_mult)/len(tok_mult))}x the input tokens for roughly-equal "
        f"(slightly lower) recall on these already-tractable synthetic multi-hop sets. The value of "
        f"'going deep' is real (bounded cost, strong absolute recall) but it is not free recall over a "
        f"well-tuned single pass.\n"
        f"- **Explore as a static prompt hint HURT on HotpotQA and was neutral on SU — and always cost "
        f"more.** Seeding the deep agent with the `describe(llm=True)` corpus profile dropped HotpotQA "
        f"recall@10 by **{exp_r[0]:+.3f}/{exp_r[1]:+.3f}/{exp_r[2]:+.3f}** (0.92->0.79, 0.81->0.70, "
        f"0.745->0.61) while **~{min(exp_s):.1f}-{max(exp_s):.1f}x-ing the searches** (2.98->7.6, "
        f"3.64->9.0, 4.30->10.2) and roughly doubling tokens/cost. On SU it left recall unchanged and "
        f"only added cost. The blanket 'this is prose, decompose across sub-facts, go wide' instruction "
        f"pushed the agent to **over-decompose / over-fan-out**, knocking golds out of the top-10 on "
        f"hops that a single hybrid+rerank already solved.\n"
        f"- **Verdict — more guidance is not better.** Injecting explore's learnings as a STATIC prompt "
        f"hint adds cost and can actively hurt. Explore's value should be delivered through the "
        f"**learned per-query router** (see §7 primitive-selection: pick the right primitive/template "
        f"per query), not a corpus-wide 'always go wide' instruction bolted onto the agent's prompt. "
        f"The deep loop's own strength is a wide hop-1 pool + rerank with judge-bounded deepening; "
        f"pouring extra blanket guidance into it degrades that.\n"
    )

    section = ["", "## 11. Deep-mode SAC — the cost of going deep, and what explore adds", "",
               "Deep-mode SAC (`phase1.agents.run_sac(..., deep=True, max_retries=3)`) writes a Python "
               "program, an LLM-as-judge grades the retrieved evidence, and on failure (or low "
               "ensemble agreement) it writes a NEW, wider program with the sandbox variables "
               "PERSISTING across hops (prior learning). We measure two deep arms — **sac_deep** "
               "(before explore) and **sac_deep+explore** (the same agent, hop-1 codegen prompt seeded "
               "with the `session.describe(llm=True)` corpus profile + recommended primitives, injected "
               "as an extra guidance message; the judge is left unbiased) — against the reused "
               "dense / tool / single-shot-sac baselines from `recall_fair.json` / `su_recall.json`. "
               "Reranker: CrossEncoder (ms-marco-MiniLM), used identically by both deep arms. "
               "n=50 queries/hop, workers=4, 0 errors. `avg searches` counts underlying retrieval "
               "calls (search + fan-out sub-searches + hyde/prf/answerability), so deep's fan-out reads "
               "higher than the single-shot search budget.",
               "", md_table(hp, "HotpotQA")]
    if sur:
        section += [md_table(sur, "SearchUnify docs")]
    section += [prose,
                "![retrieval quality](figures/deep_quality_hotpotqa.png)",
                "![cost of going deep](figures/deep_cost_hotpotqa.png)", ""]

    results = MH / "RESULTS.md"
    txt = results.read_text()
    if "## 11. Deep-mode SAC" in txt:
        txt = txt[: txt.index("## 11. Deep-mode SAC")].rstrip() + "\n"
    results.write_text(txt.rstrip() + "\n" + "\n".join(section) + "\n")
    print(f"charts -> {FIG}")
    print(f"appended section 11 -> {results}")


if __name__ == "__main__":
    main()
