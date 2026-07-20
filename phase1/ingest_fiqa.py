"""Ingest BEIR FiQA into OpenSearch through the SAC adapter.

Download → embed corpus (gte-base, GPU) → bulk index. Idempotent: skips if the
index already has the full corpus. Saves queries+qrels alongside for the benchmark.
"""

from __future__ import annotations

import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

import search_as_code as sac
from phase1 import common


def main() -> None:
    beir = common.download_beir()
    corpus = common.load_corpus(beir)
    queries = common.load_queries(beir)
    qrels = common.load_qrels(beir)
    print(f"[fiqa] corpus={len(corpus)} queries={len(queries)} qrels_queries={len(qrels)}")

    # persist queries/qrels for the benchmark
    (common.DATA_DIR / "queries.json").write_text(json.dumps(queries))
    (common.DATA_DIR / "qrels.json").write_text(json.dumps(qrels))

    store = sac.connect("opensearch", index=common.INDEX, dim=common.DIM, hosts=[common.OS_HOST])
    have = store.count()
    if have >= len(corpus):
        print(f"[fiqa] index already has {have} docs >= {len(corpus)}; skipping ingest")
        return

    ids = list(corpus)
    texts = [((corpus[i]["title"] + ". ") if corpus[i]["title"] else "") + corpus[i]["text"] for i in ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embed] {common.EMB_MODEL} on {device}")
    model = SentenceTransformer(common.EMB_MODEL, device=device)

    t0 = time.time()
    batch = 2000
    total = 0
    for start in range(0, len(ids), batch):
        chunk_ids = ids[start:start + batch]
        chunk_txt = texts[start:start + batch]
        vecs = model.encode(chunk_txt, batch_size=128, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        docs = [
            sac.Document(id=cid, text=corpus[cid]["text"],
                         vector=vec.astype(np.float32).tolist(),
                         metadata={"title": corpus[cid]["title"]})
            for cid, vec in zip(chunk_ids, vecs)
        ]
        store.upsert(docs)
        total += len(docs)
        print(f"[index] {total}/{len(ids)}  ({(time.time()-t0):.0f}s)")

    time.sleep(1)
    print(f"[fiqa] done. index count = {store.count()} in {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
