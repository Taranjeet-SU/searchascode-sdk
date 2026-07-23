"""Test the ANN-tuning hypothesis: rebuild the HotpotQA HNSW graph with higher
m / ef_construction and see if dense recall closes the gap to exact (0.925).

Creates `hotpotqa_tuned` (m=48, ef_construction=512), reindexes vectors from the
existing index (no re-embedding), then measures dense recall@10 / all_found@10.

    python -m phase3.reindex_tuned --n 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import search_as_code as sac
from phase1 import common

DATA = Path(common.REPO) / "phase2" / "data"
SRC, DST = "hotpotqa", "hotpotqa_tuned"
DIM = 768


def main(n=60, m=48, ef_construction=512):
    osx = sac.connect("opensearch", index=SRC, dim=DIM, hosts=[common.OS_HOST])
    cl = osx.client
    cl.indices.delete(index=DST, ignore=[404])
    cl.indices.create(index=DST, body={
        "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {"properties": {
            "vector": {"type": "knn_vector", "dimension": DIM, "method": {
                "name": "hnsw", "engine": "lucene", "space_type": "cosinesimil",
                "parameters": {"m": m, "ef_construction": ef_construction}}},
            "text": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 32766}}},
        }}})
    print(f"[reindex] {SRC} -> {DST} (m={m}, ef_construction={ef_construction})...", flush=True)
    t0 = time.time()
    cl.reindex(body={"source": {"index": SRC}, "dest": {"index": DST}},
               wait_for_completion=True, request_timeout=1200)
    cl.indices.refresh(index=DST)
    print(f"[reindex] done, count={cl.count(index=DST)['count']} ({time.time()-t0:.0f}s)", flush=True)

    import torch
    from sentence_transformers import SentenceTransformer
    queries = json.loads((DATA / "hotpot_queries.json").read_text())
    qrels = json.loads((DATA / "hotpot_qrels.json").read_text())
    qids = list(queries)[:n]
    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    tuned = sac.connect("opensearch", index=DST, dim=DIM, hosts=[common.OS_HOST])

    def rec(ids, g): return len(set(ids[:10]) & g) / len(g) if g else 0.0
    def allf(ids, g): return 1.0 if g and g <= set(ids[:10]) else 0.0
    R, A = [], []
    for q in qids:
        g = {c for c, s in qrels[q].items() if s > 0}
        v = em.encode(queries[q], normalize_embeddings=True, convert_to_numpy=True).tolist()
        ids = tuned.query_vector(v, top_k=10).ids()
        R.append(rec(ids, g)); A.append(allf(ids, g))
    out = {"index": DST, "m": m, "ef_construction": ef_construction, "n": len(qids),
           "recall@10": float(np.mean(R)), "all_found@10": float(np.mean(A))}
    print(f"[tuned] recall@10={out['recall@10']:.4f} all_found@10={out['all_found@10']:.4f}", flush=True)
    print("  reference: default HNSW 0.792/0.617 | exact 0.925/0.850", flush=True)
    (Path(common.REPO) / "phase3" / "reindex_tuned.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--m", type=int, default=48)
    ap.add_argument("--ef-construction", type=int, default=512)
    a = ap.parse_args(); main(a.n, a.m, a.ef_construction)
