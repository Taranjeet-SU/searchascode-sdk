"""Complete the learned profile: (1) mine few-shot exemplars (query -> winning arm)
from the exploration tagged data, (2) calibrate the judge confidence threshold
against qrels. Merge both into the profile (DB + file).

    python -m phase2.align_prompts --dataset fiqa --n 120
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import agents, common
from phase1.llm import LLM

LEARN_INDEX = "sac_learned"
RECIPE = {
    "dense": "sac.search(query, mode='dense')",
    "keyword": "sac.search(query, mode='keyword')",
    "hybrid_.8": "sac.search(query, mode='hybrid', alpha=0.8)",
    "prf": "sac.prf_search(query)",
    "dense+rerank": "sac.retrieve_rerank(query, pool_k=500)",
    "hybrid+rerank": "sac.rerank(query, sac.search(query, mode='hybrid'))",
    "expand_fuse": "sac.expand_search(query)",
    "expand_fuse+rerank": "sac.rerank(query, sac.expand_search(query))",
}


def mine_exemplars(dataset):
    from phase2 import beir
    rd = common.REPO / "phase2" / "runs" / f"router_data_{dataset}.json"
    if not rd.exists():
        rd = common.REPO / "phase2" / "runs" / "router_data.json"  # legacy fiqa
    if not rd.exists() or dataset not in ("fiqa",):
        return []  # no exploration-tagged data for this dataset -> no exemplars
    data = json.loads(rd.read_text())
    queries, _, _ = beir.eval_data(dataset)
    ex = []
    for qid, d in data.items():
        arms = d["arms"]
        best = max(arms, key=arms.get)
        # prioritize interesting routes: best arm beats dense by a margin
        if arms[best] - arms.get("dense", 0) >= 0.2 and qid in queries:
            ex.append({"query": queries[qid], "arm": best, "recipe": RECIPE.get(best, best),
                       "gain": round(arms[best] - arms["dense"], 3)})
    ex.sort(key=lambda e: -e["gain"])
    return ex[:15]


def calibrate_judge(dataset, n):
    from phase2 import beir
    queries, qr, index = beir.eval_data(dataset)
    qids = [x for x in qr if any(v > 0 for v in qr[x].values())][:n]
    em = SentenceTransformer(common.EMB_MODEL, device="cuda")
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).tolist()
    s = sac.Session("opensearch", index=index, dim=common.DIM, hosts=[common.OS_HOST], embedder=embed)
    jchat = agents.lc_chat()
    llm_usage = agents.Usage()
    rows = []
    for x in qids:
        g = {d for d, v in qr[x].items() if v > 0}
        ids = s.search(queries[x], top_k=10, mode="dense").ids()
        rec = len(set(ids) & g) / len(g) if g else 0.0
        _, _, conf = agents.judge(jchat, queries[x], ids, s, llm_usage)
        rows.append((conf, 1 if rec > 0 else 0))
    # find confidence threshold maximizing F1 of "conf>=t predicts recall>0"
    best_t, best_f1 = 0.5, -1.0
    for t in [i / 20 for i in range(21)]:
        tp = sum(1 for c, y in rows if c >= t and y == 1)
        fp = sum(1 for c, y in rows if c >= t and y == 0)
        fn = sum(1 for c, y in rows if c < t and y == 1)
        prec = tp / (tp + fp) if tp + fp else 0
        rec_ = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    base_pass = np.mean([1 for c, _ in rows if c >= 0.5]) if rows else 0
    align_default = np.mean([1 for c, y in rows if (c >= 0.5) == (y == 1)])
    align_tuned = np.mean([1 for c, y in rows if (c >= best_t) == (y == 1)])
    return best_t, best_f1, align_default, align_tuned, llm_usage.cost_usd


def main(dataset="fiqa", n=120):
    ex = mine_exemplars(dataset)
    print(f"[fewshot] mined {len(ex)} exemplars (best-arm beats dense by >=0.2)")
    for e in ex[:6]:
        print(f"  +{e['gain']} {e['arm']:18s} | {e['query'][:60]}")

    t, f1, ad, at, cost = calibrate_judge(dataset, n)
    print(f"\n[judge] calibrated confidence threshold = {t}  (F1 {f1:.2f})")
    print(f"[judge] PASS<->recall alignment: default(0.5)={ad:.2f} -> tuned({t})={at:.2f}  (llm ${cost:.4f})")

    # merge into profile
    store = sac.connect("opensearch", index="_meta", hosts=[common.OS_HOST])
    prof = store.client.get(index=LEARN_INDEX, id=dataset)["_source"]
    prof["exemplars"] = ex
    prof["judge_threshold"] = {"min_conf": t}
    store.client.index(index=LEARN_INDEX, id=dataset, body=prof, refresh=True)
    (common.REPO / "phase2" / "runs" / f"learned_{dataset}.json").write_text(json.dumps(prof, indent=2))
    print(f"\n[align] updated profile in DB '{LEARN_INDEX}' id='{dataset}' "
          f"(+{len(ex)} exemplars, judge_threshold={prof['judge_threshold']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiqa"); ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args(); main(a.dataset, a.n)
