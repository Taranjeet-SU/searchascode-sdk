"""Auto-assemble the DETAILED, paper-ready multi-dataset report from phase2/runs/*.json.
Backfills corpus/query/gold stats from phase2/data for older run JSONs. Renders whatever
datasets have completed. Run any time:

    python -m phase2.make_report
"""
from __future__ import annotations

import json
from pathlib import Path

from phase1 import common
from phase2 import beir

RUNS = Path(common.REPO) / "phase2" / "runs"
DATA = Path(common.REPO) / "phase2" / "data"
OUT = Path(common.REPO) / "phase2" / "MULTI_DATASET_REPORT.md"

CHAR = {
    "fiqa": ("financial QA", "semantic, vocabulary-mismatch, single-hop, ~1.7 gold/q"),
    "hotpotqa": ("multi-hop QA", "needs 2 supporting docs; fan-out/decompose is the lever"),
    "scifact": ("scientific claim verification", "term-heavy, keyword-favoring, ~1.1 gold/q"),
    "nfcorpus": ("medical IR", "natural-language queries, MANY gold/q (recall@10 capped)"),
    "arguana": ("counter-argument retrieval", "long argumentative queries, 1 gold/q"),
    "scidocs": ("citation/related-paper", "title->cited papers, ~5 gold/q"),
    "trec-covid": ("broad topical COVID", "few rich topics, MANY gold/q"),
}
ORDER = ["fiqa", "hotpotqa", "scifact", "nfcorpus", "arguana", "scidocs", "trec-covid"]


def backfill_stats(name):
    """corpus size + avg gold from the on-disk BEIR files (cheap: line counts + qrels)."""
    d = DATA / name
    n_corpus = avg_gold = None
    try:
        with open(d / "corpus.jsonl") as f:
            n_corpus = sum(1 for _ in f)
        split = beir.DATASETS.get(name, "test")
        golds = []
        cur = None; c = 0
        with open(d / "qrels" / f"{split}.tsv") as f:
            f.readline()
            per = {}
            for line in f:
                qid, cid, sc = line.rstrip("\n").split("\t")
                if int(sc) > 0:
                    per[qid] = per.get(qid, 0) + 1
            golds = list(per.values())
        avg_gold = round(sum(golds) / len(golds), 3) if golds else None
    except Exception:
        pass
    return n_corpus, avg_gold


def load_all():
    rows = {}
    hp = RUNS / "hotpot.json"
    if hp.exists():
        d = json.loads(hp.read_text())
        rows["hotpotqa"] = {"dataset": "hotpotqa", "n_corpus": 100978, "n_full": d["n_full"],
                            "n_sub": d["n_sub"], "avg_gold_per_query": 2.0,
                            "full": d["full"], "sub": d["sub"], "llm_cost_usd": None}
    for name in ["scifact", "nfcorpus", "arguana", "scidocs", "trec-covid"]:
        p = RUNS / f"{name}.json"
        if p.exists():
            r = json.loads(p.read_text())
            if not r.get("n_corpus") or not r.get("avg_gold_per_query"):
                nc, ag = backfill_stats(name)
                r.setdefault("n_corpus", nc); r["n_corpus"] = r.get("n_corpus") or nc
                r["avg_gold_per_query"] = r.get("avg_gold_per_query") or ag
            rows[name] = r
    return rows


def f(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def main():
    rows = load_all()
    L = []
    L += ["# Multi-dataset retrieval report: dense vs hybrid vs SAC vs tool-calling", ""]
    L += ["Comparative evaluation of **search-as-code (SAC)** — an LLM writing Python against retrieval "
          "primitives in a sandbox — against three baselines across BEIR datasets spanning distinct query "
          "types. The thesis under test: *agentic code-mode retrieval helps in proportion to how much a "
          "query needs computation/decomposition, and can hurt when the raw query is already optimal.*", ""]

    L += ["## 1. Methodology", ""]
    L += ["**Systems compared (identical stack, only the orchestration differs):**",
          "- **dense** — single vector kNN query (HNSW/Lucene in OpenSearch).",
          "- **hybrid** — dense + BM25 fused (weighted RRF, alpha=0.7).",
          "- **SAC** — gpt-4.1-mini writes Python over primitives (search/fan-out/fuse/rerank/decompose/"
          "expand/rephrase/mmr/…), executed in a sandbox; an LLM-judge loop retries up to 1x.",
          "- **tool-calling (MCP-style)** — same LLM, same budget, but each primitive is a discrete tool "
          "call (LangChain agent) instead of code.", ""]
    L += ["**Fixed components:** embedder = `" + common.EMB_MODEL + "` (768-d, normalized); reranker = "
          "Qwen3-Reranker-0.6B (yes/no logit scoring); LLM = gpt-4.1-mini; store = OpenSearch.", ""]
    L += ["**Metrics:**",
          "- **recall@10** = |gold ∩ top10| / |gold|.",
          "- **all_found@10** = 1 if *every* gold doc is in top10 else 0 — the multi-hop-sensitive metric "
          "(a single dense query rarely lands *both* supporting docs).", ""]
    L += ["**Protocol:** each corpus is embedded and indexed once. dense/hybrid are measured on the full "
          "labeled query set (stable baseline); SAC/tool run on the first N (LLM cost) with dense/hybrid "
          "recomputed on the *same* N for a paired comparison. gte-base on a shared GPU; Qwen reranker "
          "capped at max_length 512.", ""]

    L += ["## 2. Dataset characteristics", ""]
    L += ["| dataset | task type | corpus | queries (labeled) | avg gold/q | query character |",
          "|---|---|---|---|---|---|"]
    for name in ORDER:
        if name not in rows:
            continue
        r = rows[name]; tt, ch = CHAR.get(name, ("", ""))
        L.append(f"| {name} | {tt} | {r.get('n_corpus','—')} | {r.get('n_full','—')} | "
                 f"{r.get('avg_gold_per_query','—')} | {ch} |")
    L.append("")

    L += ["## 3. Headline results — SAC-subset (paired, N queries)", ""]
    L += ["| dataset | N | dense | hybrid | **SAC** | tool | dense all@10 | **SAC all@10** | LLM $ |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name in ORDER:
        if name not in rows:
            continue
        r = rows[name]; s = r.get("sub", {})
        cost = r.get("llm_cost_usd")
        L.append(f"| {name} | {r.get('n_sub','—')} | {f(s.get('dense_r'))} | {f(s.get('hybrid_r'))} | "
                 f"**{f(s.get('sac_r'))}** | {f(s.get('tool_r'))} | {f(s.get('dense_a'))} | "
                 f"**{f(s.get('sac_a'))}** | {('$'+format(cost,'.4f')) if isinstance(cost,(int,float)) else '—'} |")
    L.append("")
    L += ["_recall@10 unless noted; **bold** = SAC columns. all@10 = all_found@10._", ""]

    L += ["## 4. Full-query-set baseline (dense/hybrid, stable)", ""]
    L += ["| dataset | N_full | dense r@10 | hybrid r@10 | dense all@10 | hybrid all@10 |",
          "|---|---|---|---|---|---|"]
    for name in ORDER:
        if name not in rows:
            continue
        r = rows[name]; fu = r.get("full", {})
        L.append(f"| {name} | {r.get('n_full','—')} | {f(fu.get('dense_r'))} | {f(fu.get('hybrid_r'))} | "
                 f"{f(fu.get('dense_a'))} | {f(fu.get('hybrid_a'))} |")
    L.append("")

    L += ["## 5. Analysis — when does agentic search pay off?", ""]
    L += ["- **Multi-hop (HotpotQA)** — the clear SAC win: SAC recall@10 0.96 / all_found@10 0.92 vs dense "
          "0.79 / 0.62. Decompose + fan-out + fuse retrieves *both* supporting docs, which a single dense "
          "query structurally cannot. This is the flagship result.",
          "- **Term-heavy (SciFact)** — SAC ties hybrid (~0.88) and both edge dense (~0.86). SAC's job is to "
          "*route to hybrid/keyword*, not to add hops; the win is small because dense is already strong.",
          "- **Long-argument (ArguAna)** — **anti-result**: dense 0.85 > hybrid 0.75 > SAC 0.73 > tool 0.50. "
          "The query is a full argument and *is* the ideal retrieval key; rephrasing/decomposing degrades it. "
          "Agentic manipulation must be applied *conditionally*.",
          "- **Many-gold (NFCorpus, SciDocs, TREC-COVID)** — recall@10 is structurally capped (>10 gold), so "
          "pool-expansion + rerank matters more than hops; SAC ≈ hybrid.",
          "- **Semantic single-hop (FiQA)** — SAC ties dense; learned rules net-neutral (no routing structure "
          "to exploit — see phase2 ceiling/impact analysis).", ""]
    L += ["**Takeaway:** there is no single best retriever across query types (consistent with BEIR's own "
          "finding that the best dense model beat BM25 on only 8/18 datasets). SAC's value is that *one code "
          "policy can pick the right strategy per query* — decisively so on multi-hop, neutrally on easy/"
          "semantic sets, and it must learn to *abstain from manipulation* on tasks like ArguAna.", ""]

    # learned-profile impact section (from impact_<ds>.json)
    imp = {}
    for name in ORDER:
        p = RUNS / f"impact_{name}.json"
        if p.exists():
            imp[name] = json.loads(p.read_text())
    if imp:
        L += ["## 5b. Learned-profile impact (deterministic, no LLM at query time)", ""]
        L += ["Rules (aliases/synonyms) are mined offline from each dataset's dense-misses and applied to the "
              "*dense* path (normalize + synonym-expand + fuse). This measures whether learning helps a cheap "
              "retriever close the gap toward the agentic ceiling.", ""]
        L += ["| dataset | queries expanded | base r@10 | learned r@10 | Δ r@10 | base all@10 | learned all@10 | Δ all@10 |",
              "|---|---|---|---|---|---|---|---|"]
        for name in ORDER:
            if name not in imp:
                continue
            r = imp[name]; b, s = r["base"], r["synonym_expand"]
            L.append(f"| {name} | {s.get('changed','—')} | {f(b['r'])} | {f(s['r'])} | "
                     f"{s['r']-b['r']:+.3f} | {f(b['a'])} | {f(s['a'])} | {s['a']-b['a']:+.3f} |")
        L += ["", "**Finding:** the learned-synonym benefit scales with multi-hop/entity structure — HotpotQA "
              "gains +2.7 pts all_found@10, while single-hop/many-gold sets are ~neutral. ArguAna mines **zero** "
              "rules (its dense-misses are stance-based, not lexical), so learning is a *safe no-op* there rather "
              "than a regression. Learning should be applied conditionally; the miner self-limits where there is "
              "no lexical structure to exploit.", ""]

    L += ["## 6. Research grounding", ""]
    L += ["- BEIR: heterogeneous zero-shot IR benchmark; dense ≠ universally best (Thakur et al., 2104.08663).",
          "- HotpotQA: multi-hop QA requiring 2+ supporting facts (Yang et al., 2018).",
          "- SciFact: scientific claim verification; BM25 nDCG@10 ≈ 0.66, term-heavy (Wadden et al., 2020).",
          "- Adaptive/agentic retrieval parallels: 'Think Before You Retrieve: Test-Time Adaptive Search "
          "with Small LMs' (2511.07581); 'Claim-Aware Scientific RAG: Evidence-First Retrieval and "
          "Abstention'. These motivate SAC's route/decompose/confidence-abstain primitives.", ""]

    L += ["## 7. Reproduction", ""]
    L += ["```bash",
          "# ingest + eval any dataset (dense/hybrid full set; SAC/tool on first N)",
          "python -m phase2.beir_run --dataset scifact --ingest --n 40",
          "# full 5-dataset campaign (serial, shared GPU)",
          "bash phase2/run_campaign.sh",
          "# regenerate this report from runs/*.json",
          "python -m phase2.make_report",
          "```", ""]
    L += [f"_Auto-generated from phase2/runs/*.json — {len(rows)} datasets rendered: {', '.join(rows)}._", ""]

    OUT.write_text("\n".join(L))
    print(f"wrote {OUT} ({len(rows)} datasets: {', '.join(rows)})")


if __name__ == "__main__":
    main()
