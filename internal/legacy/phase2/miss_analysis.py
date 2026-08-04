"""For FiQA gold docs missed at rank 100, find their ACTUAL dense rank and diagnose
why (semantic gap, vocabulary mismatch) so remedies are grounded.

    python -m phase2.miss_analysis --n 150
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common

BIG = 10000


def words(t):
    return set(re.findall(r"[a-z]{3,}", (t or "").lower()))


def main(n=150):
    q = json.loads((common.DATA_DIR / "queries.json").read_text())
    qr = json.loads((common.DATA_DIR / "qrels.json").read_text())
    qids = [x for x in qr if any(v > 0 for v in qr[x].values())][:n]
    em = SentenceTransformer(common.EMB_MODEL, device="cuda")
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    store = sac.connect("opensearch", index="fiqa", dim=768, hosts=[common.OS_HOST])
    client = store.client

    def knn(qv, k):
        body = {"size": k, "_source": False, "query": {"knn": {"vector": {"vector": list(qv), "k": k}}}}
        r = client.search(index="fiqa", body=body)
        return [h["_id"] for h in r["hits"]["hits"]], [h["_score"] for h in r["hits"]["hits"]]

    def gold_vec(gid):
        d = client.mget(index="fiqa", body={"ids": [gid]}, _source=["vector"])["docs"][0]
        v = d.get("_source", {}).get("vector")
        return np.asarray(v, dtype=np.float32) if v else None

    buckets = {"100-500": 0, "500-2000": 0, "2000-10000": 0, ">10000": 0}
    missed = []
    n_gold_total = n_miss = 0
    for x in qids:
        gold = {d for d, v in qr[x].items() if v > 0}
        n_gold_total += len(gold)
        qv = embed([q[x]])[0]
        top100, _ = knn(qv, 100)
        miss = gold - set(top100)
        if not miss:
            continue
        ids, sc = knn(qv, BIG)
        pos = {i: r for r, i in enumerate(ids, 1)}
        top1 = sc[0]
        for g in miss:
            n_miss += 1
            r = pos.get(g)
            gv = gold_vec(g)
            gcos = float(qv @ (gv / (np.linalg.norm(gv) or 1))) if gv is not None else None
            if r is None:
                buckets[">10000"] += 1
                rank = ">10000"
            else:
                rank = r
                buckets["100-500" if r <= 500 else "500-2000" if r <= 2000 else "2000-10000"] += 1
            gtext = (store.get([g]) or [None])[0]
            gtext = gtext.text if gtext else ""
            missed.append((rank if r else 10001, x, g, rank, gcos, top1, len(words(q[x]) & words(gtext)), q[x], gtext))

    missed.sort()
    print(f"queries analyzed: {len(qids)}  | gold docs: {n_gold_total}  | missed@100: {n_miss} "
          f"({100*n_miss/max(1,n_gold_total):.0f}%)")
    print("\nactual-rank distribution of missed golds:")
    for b, c in buckets.items():
        print(f"  {b:12s} {c}")
    tail = [m for m in missed if m[3] == '>10000' or (isinstance(m[3], int) and m[3] > 2000)]
    print(f"\ndeep misses (rank>2000 or unfound): {len(tail)}/{n_miss}")
    print("\nexamples (rank | gold_cos vs top1_cos | shared_words | query -> gold):")
    for _, x, g, rank, gcos, top1, ov, qt, gt in missed[:12]:
        gc = f"{gcos:.3f}" if gcos is not None else "n/a"
        print(f"  rank {str(rank):>6} | cos {gc} vs top1 {top1:.3f} | shared {ov} | "
              f"{qt[:52]!r} -> {gt[:60]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=150)
    main(ap.parse_args().n)
