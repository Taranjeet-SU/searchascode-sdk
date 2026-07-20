"""Functional test of every Phase 1 primitive against the live FiQA index.

Proves each primitive runs end-to-end on real data, and does a small quality
check that cross-encoder reranking improves recall over dense alone.
"""

from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

import search_as_code as sac
from phase1 import common, metrics
from phase1.llm import LLM


def main() -> None:
    queries = json.loads((common.DATA_DIR / "queries.json").read_text())
    qrels = json.loads((common.DATA_DIR / "qrels.json").read_text())
    qids = [q for q in qrels if any(s > 0 for s in qrels[q].values())]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_model = SentenceTransformer(common.EMB_MODEL, device=device)
    embed = lambda ts: emb_model.encode(list(ts), normalize_embeddings=True,
                                        convert_to_numpy=True, show_progress_bar=False).tolist()

    reranker = sac.CrossEncoderReranker()
    llm = LLM()
    s = sac.Session("opensearch", index=common.INDEX, dim=common.DIM, hosts=[common.OS_HOST],
                    embedder=embed, reranker=reranker, generator=llm.as_generator())

    q0 = queries[qids[0]]
    print(f"\n=== single-query primitive demo ===\nquery: {q0!r}\n")
    dense = s.search(q0, top_k=8)
    print("dense   :", dense.ids()[:5])
    print("keyword :", s.search(q0, top_k=8, mode="keyword").ids()[:5])
    print("hybrid  :", s.search(q0, top_k=8, mode="hybrid").ids()[:5])
    print("rerank  :", s.rerank(q0, dense, top_k=5).ids())
    print("dedup   :", len(dense.dedup()), "unique of", len(dense))
    print("mmr     :", s.mmr(q0, dense, top_k=5).ids())
    print("filter  :", s.search(q0, top_k=8).where(lambda h: bool(h.get("title"))).ids()[:5])
    agg = s.store.aggregate({"titles": {"cardinality": {"field": "title.keyword"}}})
    print("aggregate: distinct titles =", agg["titles"]["value"])

    print("\n=== rephraser (gpt-4.1-mini) ===")
    for qid in qids[:3]:
        better = sac.rephrase(queries[qid], llm.as_generator())
        print(f"  {queries[qid][:60]!r}\n   -> {better[:70]!r}")

    # --- quality check: dense@10 vs dense(top50)->cross-encoder rerank@10 ---
    print("\n=== rerank quality on 40 queries (recall@10) ===")
    sample = qids[:40]
    q_emb = embed([queries[q] for q in sample])
    dense_rank, rerank_rank = {}, {}
    for qid, vec in zip(sample, q_emb):
        pool = s.store.query_vector(np.asarray(vec, dtype=np.float32), top_k=50)
        dense_rank[qid] = pool.ids()[:10]
        rerank_rank[qid] = sac.rerank(queries[qid], s.hydrate(pool), reranker=reranker, top_k=10).ids()
    md = metrics.evaluate(dense_rank, qrels, k=10)
    mr = metrics.evaluate(rerank_rank, qrels, k=10)
    print(f"  dense        recall@10={md['recall@10']:.4f}  ndcg@10={md['ndcg@10']:.4f}")
    print(f"  dense+rerank recall@10={mr['recall@10']:.4f}  ndcg@10={mr['ndcg@10']:.4f}")
    print(f"  llm usage: {llm.usage.as_dict()}")
    print("\n[functional] all primitives executed OK")


if __name__ == "__main__":
    main()
