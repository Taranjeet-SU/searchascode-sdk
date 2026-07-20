"""Base-search recall benchmark over FiQA test queries (no LLM).

Runs dense / keyword / hybrid retrieval for all qrels-bearing queries and reports
Recall@10, nDCG@10, MRR@10. Validates the retrieval stack and produces the
'base search' baseline the SAC vs tool-calling comparison is measured against.
"""

from __future__ import annotations

import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

import search_as_code as sac
from phase1 import common
from phase1 import metrics


def main(k: int = 10) -> None:
    queries = json.loads((common.DATA_DIR / "queries.json").read_text())
    qrels = json.loads((common.DATA_DIR / "qrels.json").read_text())
    qids = [q for q in qrels if any(s > 0 for s in qrels[q].values())]
    q_texts = [queries[q] for q in qids]
    print(f"[base] {len(qids)} test queries")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(common.EMB_MODEL, device=device)
    q_emb = model.encode(q_texts, batch_size=128, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    store = sac.connect("opensearch", index=common.INDEX, dim=common.DIM, hosts=[common.OS_HOST])

    results = {}
    for mode in ("dense", "keyword", "hybrid"):
        rankings = {}
        t0 = time.time()
        for qid, text, vec in zip(qids, q_texts, q_emb):
            if mode == "dense":
                hits = store.query_vector(vec, top_k=k)
            elif mode == "keyword":
                hits = store.query_keyword(text, top_k=k)
            else:
                hits = store.query_hybrid(vec, text, top_k=k)
            rankings[qid] = hits.ids()
        m = metrics.evaluate(rankings, qrels, k=k)
        m["latency_ms_per_query"] = (time.time() - t0) * 1000 / len(qids)
        results[mode] = m
        print(f"[{mode:8s}] recall@{k}={m[f'recall@{k}']:.4f}  "
              f"ndcg@{k}={m[f'ndcg@{k}']:.4f}  mrr@{k}={m[f'mrr@{k}']:.4f}  "
              f"lat={m['latency_ms_per_query']:.1f}ms")

    (common.RUNS_DIR / "base_eval.json").write_text(json.dumps(results, indent=2))
    print("[base] wrote", common.RUNS_DIR / "base_eval.json")


if __name__ == "__main__":
    main()
