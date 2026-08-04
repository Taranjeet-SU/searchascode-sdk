"""Dump individual missed-at-100 cases with FULL text for case-by-case reasoning."""
from __future__ import annotations

import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common

BIG = 10000
def words(t): return set(re.findall(r"[a-z]{3,}", (t or "").lower()))


def main(n=120):
    q = json.loads((common.DATA_DIR / "queries.json").read_text())
    qr = json.loads((common.DATA_DIR / "qrels.json").read_text())
    qids = [x for x in qr if any(v > 0 for v in qr[x].values())][:n]
    em = SentenceTransformer(common.EMB_MODEL, device="cuda")
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    store = sac.connect("opensearch", index="fiqa", dim=768, hosts=[common.OS_HOST])
    client = store.client

    def knn(qv, k):
        r = client.search(index="fiqa", body={"size": k, "_source": False,
                          "query": {"knn": {"vector": {"vector": list(qv), "k": k}}}})
        return [h["_id"] for h in r["hits"]["hits"]], [h["_score"] for h in r["hits"]["hits"]]

    cases = []
    for x in qids:
        gold = {d for d, v in qr[x].items() if v > 0}
        qv = embed([q[x]])[0]
        top100, _ = knn(qv, 100)
        miss = gold - set(top100)
        if not miss:
            continue
        ids, sc = knn(qv, BIG)
        pos = {i: r for r, i in enumerate(ids, 1)}
        for g in miss:
            r = pos.get(g, 10001)
            gt = (store.get([g]) or [None])[0]
            gt = gt.text if gt else ""
            shared = words(q[x]) & words(gt)
            cases.append((r, x, g, sc[0], q[x], gt, shared))
    cases.sort()
    # spread across buckets
    def band(r): return 0 if r <= 500 else 1 if r <= 2000 else 2 if r <= 10000 else 3
    picks, seen = [], {0: 0, 1: 0, 2: 0, 3: 0}
    quota = {0: 5, 1: 4, 2: 3, 3: 3}
    for c in cases:
        b = band(c[0])
        if seen[b] < quota[b]:
            picks.append(c); seen[b] += 1
    print(f"total missed: {len(cases)} | showing {len(picks)} across buckets\n")
    for r, x, g, top1, qt, gt, shared in picks:
        rank = r if r <= 10000 else ">10000"
        print("="*100)
        print(f"RANK {rank}  (top1_cos={top1:.3f})  shared_terms={sorted(shared)}")
        print(f"QUERY : {qt}")
        print(f"GOLD  : {gt[:400]}")


if __name__ == "__main__":
    main()
