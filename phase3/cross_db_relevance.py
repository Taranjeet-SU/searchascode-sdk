"""Phase 3 — cross-DB relevance: run the SAME HotpotQA dense retrieval on multiple
backends through the ONE primitive API, and compare relevance.

Reuses the vectors already indexed in OpenSearch (scroll them out — no re-embedding)
and loads them into the in-process backends (FAISS, SQLite, memory). Then embeds the
HotpotQA queries once and measures dense recall@10 / all_found@10 per backend.

    python -m phase3.cross_db_relevance --n 60 --max-docs 100978

Near-parity is the expected (and desired) result: same vectors + same metric => same
relevance, regardless of backend. Divergence only from ANN approximation (OpenSearch
HNSW) vs exact (FAISS/SQLite brute force).
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
OUT = Path(common.REPO) / "phase3" / "cross_db_relevance.json"
INDEX = "hotpotqa"
DIM = 768


def scroll_vectors(store, max_docs):
    """Yield Documents (id, text, vector, title) from the OpenSearch index."""
    cl = store.client
    body = {"size": 1000, "query": {"match_all": {}},
            "_source": ["vector", "text", "title"]}
    res = cl.search(index=INDEX, body=body, scroll="5m")
    sid = res["_scroll_id"]; got = 0
    while True:
        hits = res["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            s = h["_source"]
            if "vector" not in s:
                continue
            yield sac.Document(id=h["_id"], text=s.get("text", ""),
                               vector=s["vector"], metadata={"title": s.get("title", "")})
            got += 1
        if got >= max_docs:
            break
        res = cl.scroll(scroll_id=sid, scroll="5m")
        sid = res["_scroll_id"]
    cl.clear_scroll(scroll_id=sid, ignore=(404,))


def main(n=60, max_docs=100978, extra=False):
    import torch
    from sentence_transformers import SentenceTransformer
    queries = json.loads((DATA / "hotpot_queries.json").read_text())
    qrels = json.loads((DATA / "hotpot_qrels.json").read_text())
    qids = list(queries)[:n]

    osx = sac.connect("opensearch", index=INDEX, dim=DIM, hosts=[common.OS_HOST])
    t0 = time.time()
    print(f"[xdb] scrolling up to {max_docs} vectors out of OpenSearch...", flush=True)
    docs = list(scroll_vectors(osx, max_docs))
    print(f"[xdb] pulled {len(docs)} docs ({time.time()-t0:.0f}s). Building backends...", flush=True)

    backends = {"opensearch": osx}
    specs = [("faiss", {}), ("sqlite", {}), ("memory", {})]
    if extra:
        specs += [("chroma", {"collection": "xdb"}), ("qdrant", {"collection": "xdb"})]
    for b, opts in specs:
        try:
            st = sac.connect(b, dim=DIM, **opts)
            B = 5000
            for i in range(0, len(docs), B):
                st.upsert(docs[i:i + B])
            backends[b] = st
            print(f"[xdb] {b}: count={st.count()} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[xdb] {b}: SKIP ({type(e).__name__}: {e})", flush=True)

    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    qvecs = {q: em.encode(queries[q], normalize_embeddings=True, convert_to_numpy=True).tolist() for q in qids}

    def rec(ids, g): return len(set(ids[:10]) & g) / len(g) if g else 0.0
    def allf(ids, g): return 1.0 if g and g <= set(ids[:10]) else 0.0

    results = {}
    for name, st in backends.items():
        try:
            R, A, lat = [], [], []
            for q in qids:
                g = {c for c, s in qrels[q].items() if s > 0}
                t = time.time()
                ids = st.query_vector(qvecs[q], top_k=10).ids()
                lat.append(time.time() - t)
                R.append(rec(ids, g)); A.append(allf(ids, g))
            results[name] = {"recall@10": float(np.mean(R)), "all_found@10": float(np.mean(A)),
                             "avg_latency_ms": float(np.mean(lat) * 1000), "count": st.count()}
            print(f"[xdb] {name:10s} recall@10={results[name]['recall@10']:.4f} "
                  f"all_found@10={results[name]['all_found@10']:.4f} "
                  f"lat={results[name]['avg_latency_ms']:.1f}ms", flush=True)
        except Exception as e:
            print(f"[xdb] {name:10s} QUERY FAILED ({type(e).__name__}: {e})", flush=True)

    OUT.write_text(json.dumps({"n_queries": len(qids), "n_docs": len(docs),
                               "backends": results}, indent=2))
    print(f"[xdb] saved {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-docs", type=int, default=100978)
    ap.add_argument("--extra", action="store_true", help="also test chroma + qdrant (in-process)")
    a = ap.parse_args(); main(a.n, a.max_docs, a.extra)
