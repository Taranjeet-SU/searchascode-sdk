"""Run the learned PRIMITIVE/TEMPLATE-SELECTION ROUTER on the MULTI-HOP synth datasets.

Question: on multi-hop queries — where the gold is a SET of N docs that must ALL be
retrieved and the templates are genuinely complementary — does routing beat "always-dense",
unlike single-hop BEIR where it only tied (+0.006)?

CRITICAL: multi-hop uses an ALL-GOLDS gate — a template "solves" a query iff ALL N gold
docs are in its top-k (all_golds@k = gold_set ⊆ top_k), NOT single-gold recall. We pass the
purely-additive ``all_golds=True`` flag threaded through Explorer.dataset -> build_dataset ->
label_via_templates (shipped single-gold behavior is untouched).

Corpora (one Explorer/pack each, 3 hop-files pooled with an n_docs tag):
  - HotpotQA 2/3/4-hop over the OpenSearch "hotpotqa" index (cap 200/hop = 600).
  - SU 2/3/4-hop over a MEMORY session built from ~/scripts/data/su_docs_2.csv (all 450).

Both use gte-base (SentenceTransformer, normalized), QwenReranker, and LLM().as_generator()
so the deep templates (hyde/decompose/rephrase/expand + rerank) actually fire.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from search_as_code.explore import ProfilePack
from search_as_code.explore.engine import Explorer
from search_as_code.explore.router import best_from_hits, label_via_templates
from search_as_code.explore.templates import (
    TEMPLATE_COST,
    TEMPLATE_DOCS,
    TEMPLATE_NAMES,
    StrategyContext,
    run_template,
)
from search_as_code.explore.training import (
    analyze_failures,
    load_dataset,
    train_router_model,
    write_dataset_csv,
    _read_jsonl,
)

HERE = Path(__file__).parent
REPO = HERE.parent.parent
K = 10
WORKERS = 4
HOPS = (2, 3, 4)
CAP_HOTPOT_PER_HOP = 200          # -> 600 balanced
SU_CSV = Path.home() / "scripts" / "data" / "su_docs_2.csv"


# --------------------------------------------------------------------------- #
# shared embedder                                                               #
# --------------------------------------------------------------------------- #
def make_embed():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    print(f"[embed] {common.EMB_MODEL} on {dev}", flush=True)

    def embed(texts):
        return em.encode(list(texts), normalize_embeddings=True, batch_size=128).tolist()

    return embed


def load_pool(files):
    """files = [(n_docs, path)]; concatenate rows tagging each with its hop as 'dataset'."""
    out = []
    for n, path, cap in files:
        rows = [json.loads(l) for l in Path(path).open() if l.strip()]
        if cap:
            rows = rows[:cap]
        for r in rows:
            out.append({"query": r["query"], "gold_ids": r["gold_ids"], "dataset": f"{n}hop"})
    return out


# --------------------------------------------------------------------------- #
# gate verification — prove the all-golds gate is stricter than single-gold     #
# --------------------------------------------------------------------------- #
def verify_gate(session, embed, queries, name, n=3):
    print(f"\n[verify {name}] checking ALL-GOLDS gate on {n} queries...", flush=True)
    proven = False
    for row in queries[:n]:
        q, golds = row["query"], set(row["gold_ids"])
        emb = embed([q])[0]
        ctx = StrategyContext(session, q, P_pool=25, emb=emb, use_llm=True,
                              use_rerank=True, top_k=K)
        hits_single, hits_all = {}, {}
        for t in TEMPLATE_NAMES:
            ids = set(run_template(t, ctx, top_k=K))
            hits_single[t] = int(bool(golds & ids))
            hits_all[t] = int(golds <= ids)
        # all-golds must be <= single-gold for every template (strictly stricter criterion)
        assert all(hits_all[t] <= hits_single[t] for t in TEMPLATE_NAMES), \
            f"all-golds gate not stricter than single-gold on {q[:60]!r}"
        # cross-check against the shipped labeler on a fresh (recomputed) context
        ctx2 = StrategyContext(session, q, P_pool=25, emb=emb, use_llm=True,
                               use_rerank=True, top_k=K)
        best_ag, hv_ag = label_via_templates(ctx2, golds, k=K, cascade=False, all_golds=True)
        assert hv_ag == hits_all, "label_via_templates(all_golds=True) disagrees with manual gate"
        diff = [t for t in TEMPLATE_NAMES if hits_single[t] == 1 and hits_all[t] == 0]
        n_gold = len(golds)
        print(f"  n_gold={n_gold} single-solvers={sum(hits_single.values())} "
              f"all-golds-solvers={sum(hits_all.values())} "
              f"templates-that-got-SOME-not-ALL={diff[:4]}", flush=True)
        if diff:
            proven = True
    if proven:
        print(f"[verify {name}] OK — found templates that retrieve SOME but not ALL golds "
              f"=> the all-golds gate genuinely differs from single-gold.", flush=True)
    else:
        print(f"[verify {name}] WARNING — no divergence observed on the sample "
              f"(all templates either got all or none); gate logic still correct.", flush=True)


# --------------------------------------------------------------------------- #
# per-corpus run                                                                #
# --------------------------------------------------------------------------- #
def per_hop_breakdown(pack):
    """Read shards -> per-hop oracle coverage (any template all-golds@10) + label dist."""
    sdir = pack.root / "dataset" / "shards"
    by_hop = defaultdict(lambda: {"n": 0, "solved": 0, "winners": Counter()})
    total = {"n": 0, "solved": 0, "winners": Counter()}
    for f in sorted(sdir.glob("lab_*.jsonl")):
        for r in _read_jsonl(f):
            hop = r.get("dataset", "?")
            w = best_from_hits(r.get("hits") or {})
            for bucket in (by_hop[hop], total):
                bucket["n"] += 1
                if w != "none":
                    bucket["solved"] += 1
                    bucket["winners"][w] += 1
    def fmt(b):
        return {"n": b["n"], "solved": b["solved"],
                "oracle_coverage": round(b["solved"] / b["n"], 4) if b["n"] else 0.0,
                "winners": dict(b["winners"])}
    return {hop: fmt(b) for hop, b in sorted(by_hop.items())}, fmt(total)


def run_corpus(name, session, embed, queries):
    print(f"\n{'='*70}\n[corpus {name}] {len(queries)} queries "
          f"({Counter(q['dataset'] for q in queries)})\n{'='*70}", flush=True)
    verify_gate(session, embed, queries, name)

    pack = ProfilePack.open(str(HERE / f"pack_{name}"))
    explorer = Explorer(session, pack)
    t0 = time.time()
    explorer.dataset(queries=queries, k=K, P=25, label_llm=True, label_rerank=True,
                     workers=WORKERS, batch_size=100, resume=True, all_golds=True,
                     progress_every=1)
    print(f"[corpus {name}] labeling done in {time.time()-t0:.0f}s", flush=True)

    ds = load_dataset(pack)
    meta = ds.meta
    n = meta["n"]
    solved = meta["solved"]
    label_dist = meta["label_distribution"]
    tmpl_recall = meta["template_hit_rate@k"]

    # baselines (accuracy of a FIXED policy over the cheapest-solver labels, among solved)
    always_dense = round(label_dist.get("light_dense", 0) / solved, 4) if solved else 0.0
    best_single_name = max(label_dist, key=label_dist.get) if label_dist else "none"
    best_single = round(label_dist.get(best_single_name, 0) / solved, 4) if solved else 0.0

    models = {}
    for spec in ("hist_gb", "logreg"):
        _, m = train_router_model(ds, spec, cv=5)
        models[spec] = {"cv_accuracy": m.get("cv_accuracy"), "cv_std": m.get("cv_std"),
                        "cv_folds": m.get("cv_folds"), "train_accuracy": m.get("train_accuracy"),
                        "best_single_template_acc": m.get("best_single_template_acc"),
                        "router_lift_over_fixed": m.get("router_lift_over_fixed")}
        cv = m.get("cv_accuracy")
        print(f"[corpus {name}] {spec}: cv_acc={cv} best_single={m.get('best_single_template_acc')} "
              f"lift_vs_best_single={m.get('router_lift_over_fixed')}", flush=True)

    csvs = write_dataset_csv(pack, out_dir=HERE / f"csv_{name}")

    # per-hop + augmented per-query CSV with the n_docs column
    per_hop, total = per_hop_breakdown(pack)
    _write_perquery_ndocs_csv(pack, HERE / f"csv_{name}" / "labels_with_ndocs.csv")

    # failure taxonomy on the unsolved (all-golds misses)
    unsolved_items = []
    for f in sorted((pack.root / "dataset" / "shards").glob("lab_*.jsonl")):
        for r in _read_jsonl(f):
            if best_from_hits(r.get("hits") or {}) == "none":
                unsolved_items.append({"query": r["query"], "gold_id": r["gold_id"],
                                       "gold_ids": r.get("gold_ids") or [r["gold_id"]]})
    fails = analyze_failures(session, unsolved_items, sample=300) if unsolved_items else {}

    hist_cv = models["hist_gb"]["cv_accuracy"]
    result = {
        "corpus": name, "n": n, "solved": solved,
        "oracle_coverage": meta["oracle_coverage"],
        "always_dense_acc": always_dense,
        "best_single_template": best_single_name, "best_single_acc": best_single,
        "router_cv": models,
        "headline_lift_vs_always_dense": (round(hist_cv - always_dense, 4)
                                          if hist_cv is not None else None),
        "label_distribution": label_dist,
        "template_all_golds@10_recall": tmpl_recall,
        "per_hop": per_hop,
        "failure_taxonomy": fails,
        "csv_paths": csvs,
    }
    return result


def _write_perquery_ndocs_csv(pack, out_path):
    import csv
    sdir = pack.root / "dataset" / "shards"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["query", "n_docs", "gold_ids", "winner", "solved"]
                   + [f"hit_{t}" for t in TEMPLATE_NAMES])
        for f in sorted(sdir.glob("lab_*.jsonl")):
            for r in _read_jsonl(f):
                hits = r.get("hits") or {}
                winner = best_from_hits(hits)
                golds = r.get("gold_ids") or [r.get("gold_id")]
                w.writerow([r.get("query", ""), r.get("dataset", ""), "|".join(map(str, golds)),
                            winner, int(winner != "none")]
                           + [int(hits.get(t, 0)) for t in TEMPLATE_NAMES])


# --------------------------------------------------------------------------- #
# SU corpus loading                                                             #
# --------------------------------------------------------------------------- #
def load_su_docs():
    df = pd.read_csv(SU_CSV)
    docs = []
    for _, row in df.iterrows():
        content = row.get("content")
        if pd.isna(content) or not str(content).strip():
            continue
        title = "" if pd.isna(row.get("title")) else str(row.get("title"))
        docs.append({"id": str(row["id"]), "text": (title + ". " + str(content)).strip()})
    print(f"[su] {len(docs)} docs with content", flush=True)
    return docs


# --------------------------------------------------------------------------- #
# reporting                                                                     #
# --------------------------------------------------------------------------- #
def append_markdown(results):
    md = HERE / "results_primitive_selection.md"
    lines = []
    lines.append("\n## 7. Template routing on multi-hop synth data "
                 "(does the router beat dense where templates are complementary?)\n")
    lines.append(
        "We re-ran the learned template router on the **newer multi-hop synthetic datasets**, "
        "where the gold is a **SET of N documents** (n_docs=2/3/4) that must **ALL** be retrieved. "
        "The success gate is therefore **all_golds@10** (a template solves iff `gold_set ⊆ top_k`), "
        "not the single-gold recall@10 used for single-hop BEIR (added as the purely-additive "
        "`all_golds=True` flag; shipped behavior untouched). Deep templates fire for real "
        "(gte-base dense, QwenReranker, gpt-4.1-mini generator for hyde/decompose/rephrase/expand).\n")

    lines.append("### 7a. Headline — router CV vs always-dense vs best-single (the LIFT)\n")
    lines.append("| corpus | n | oracle (any all-golds@10) | router CV (hist_gb) | best-single-template | always-dense | **lift vs always-dense** |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        cv = r["router_cv"]["hist_gb"]["cv_accuracy"]
        std = r["router_cv"]["hist_gb"]["cv_std"]
        cvs = f"{cv:.3f} ± {std:.3f}" if cv is not None else "n/a"
        lift = r["headline_lift_vs_always_dense"]
        lifts = f"**{lift:+.3f}**" if lift is not None else "n/a"
        lines.append(f"| {r['corpus']} | {r['n']} | {r['oracle_coverage']:.3f} | {cvs} | "
                     f"{r['best_single_template']} {r['best_single_acc']:.3f} | "
                     f"{r['always_dense_acc']:.3f} | {lifts} |")
    lines.append("")

    lines.append("### 7b. Winning-template distribution (cheapest all-golds solver)\n")
    lines.append("Contrast with **BEIR single-hop, where ~84% of solved queries were won by "
                 "`light_dense`** (no complementarity => nothing to route to).\n")
    for r in results:
        dist = r["label_distribution"]
        tot = sum(dist.values()) or 1
        top = sorted(dist.items(), key=lambda kv: -kv[1])
        frag = ", ".join(f"`{t}` {v}({v/tot*100:.0f}%)" for t, v in top)
        ld = dist.get("light_dense", 0)
        lines.append(f"- **{r['corpus']}** (solved={r['solved']}/{r['n']}): light_dense="
                     f"{ld/tot*100:.0f}% of winners. Full: {frag}")
    lines.append("")

    lines.append("### 7c. Per-template all_golds@10 recall (top strategies)\n")
    for r in results:
        tr = r["template_all_golds@10_recall"]
        top = sorted(tr.items(), key=lambda kv: -kv[1])[:6]
        frag = ", ".join(f"`{t}` {v:.3f}" for t, v in top)
        lines.append(f"- **{r['corpus']}**: {frag} "
                     f"(caveat: cascade labeling under-measures dear templates on queries a "
                     f"cheaper one already solved).")
    lines.append("")

    lines.append("### 7d. Failure taxonomy on unsolved (no template got all N golds@10)\n")
    for r in results:
        f = r["failure_taxonomy"]
        if not f:
            lines.append(f"- **{r['corpus']}**: 0 unsolved.")
            continue
        fr = f.get("fractions", {})
        frag = ", ".join(f"{c} {v*100:.0f}%" for c, v in sorted(fr.items(), key=lambda kv: -kv[1]))
        lines.append(f"- **{r['corpus']}** ({f.get('checked',0)} unsolved checked): {frag}")
    lines.append("")

    lines.append("### 7e. Verdict\n")
    lines.append(_verdict(results))
    lines.append("")
    with md.open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[report] appended section 7 to {md}", flush=True)


def _verdict(results):
    bits = []
    for r in results:
        cv = r["router_cv"]["hist_gb"]["cv_accuracy"]
        ad = r["always_dense_acc"]
        lift = r["headline_lift_vs_always_dense"]
        dist = r["label_distribution"]
        tot = sum(dist.values()) or 1
        ld_frac = dist.get("light_dense", 0) / tot
        if lift is None:
            bits.append(f"**{r['corpus']}**: too few solved/classes to train a router "
                        f"(oracle {r['oracle_coverage']:.2f}).")
            continue
        collapsed = ld_frac > 0.6
        if lift > 0.02 and not collapsed:
            v = (f"**{r['corpus']}**: router **beats always-dense by {lift:+.3f}** "
                 f"(CV {cv:.3f} vs {ad:.3f}); winners are spread across templates "
                 f"(light_dense only {ld_frac*100:.0f}%) — multi-hop gives real routing headroom.")
        elif lift > 0.0:
            v = (f"**{r['corpus']}**: router edges always-dense by {lift:+.3f} "
                 f"(CV {cv:.3f} vs {ad:.3f}) — a real but modest gain"
                 + (f"; label still leans light_dense ({ld_frac*100:.0f}%)." if collapsed
                    else "; winners are meaningfully spread across templates."))
        else:
            v = (f"**{r['corpus']}**: router does NOT beat always-dense ({lift:+.3f}; "
                 f"CV {cv:.3f} vs {ad:.3f})"
                 + (f" — label collapsed onto light_dense ({ld_frac*100:.0f}%), same story as BEIR."
                    if collapsed else " — winners spread but not predictable from cheap features."))
        bits.append(v)
    return "\n\n".join(bits)


# --------------------------------------------------------------------------- #
def main():
    common.load_env()
    embed = make_embed()
    results = []

    # ---- HotpotQA (OpenSearch) ----
    hp_files = [(n, REPO / "experiments/multi_hop_synth_queries/data"
                 / f"multihop_{n}docs_queries.jsonl", CAP_HOTPOT_PER_HOP) for n in HOPS]
    hp_queries = load_pool(hp_files)
    hp_session = sac.Session("opensearch", index="hotpotqa", dim=common.DIM,
                             hosts=[common.OS_HOST], text_field="text", vector_field="vector",
                             embedder=embed, reranker=sac.QwenReranker(),
                             generator=LLM().as_generator())
    results.append(run_corpus("hotpotqa_multihop", hp_session, embed, hp_queries))

    # ---- SU (Memory) ----
    su_files = [(n, REPO / "experiments/su_multihop/data" / f"su_multihop_{n}docs.jsonl", None)
                for n in HOPS]
    su_queries = load_pool(su_files)
    su_docs = load_su_docs()
    su_session = sac.Session("memory", dim=common.DIM, embedder=embed,
                             reranker=sac.QwenReranker(), generator=LLM().as_generator())
    su_session.add(su_docs)
    print(f"[su] session has {su_session.store.count()} docs", flush=True)
    results.append(run_corpus("su_multihop", su_session, embed, su_queries))

    # ---- save raw + markdown ----
    (HERE / "multihop_router.json").write_text(json.dumps(results, indent=2))
    print(f"[done] wrote multihop_router.json", flush=True)
    append_markdown(results)

    # ---- final table to stdout ----
    print("\n" + "=" * 70 + "\nFINAL LIFT TABLE\n" + "=" * 70)
    print(f"{'corpus':22s} {'oracle':>7s} {'router_cv':>10s} {'always_dense':>13s} {'lift':>8s}")
    for r in results:
        cv = r["router_cv"]["hist_gb"]["cv_accuracy"]
        print(f"{r['corpus']:22s} {r['oracle_coverage']:>7.3f} "
              f"{(cv if cv is not None else float('nan')):>10.3f} "
              f"{r['always_dense_acc']:>13.3f} "
              f"{(r['headline_lift_vs_always_dense'] or 0):>+8.3f}")


if __name__ == "__main__":
    main()
