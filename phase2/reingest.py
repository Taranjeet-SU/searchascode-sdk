"""Re-ingest FiQA into a per-embedder OpenSearch index (passage embeddings).

    python -m phase2.reingest --model bge-large
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import search_as_code as sac
from phase1 import common
from phase2 import embed_models


def main(key: str):
    _, passage_embed, dim, index = embed_models.build(key)
    print(f"[reingest] model={key} dim={dim} index={index}")
    beir = common.download_beir()
    corpus = common.load_corpus(beir)
    ids = list(corpus)
    texts = [((corpus[i]["title"] + ". ") if corpus[i]["title"] else "") + corpus[i]["text"] for i in ids]

    store = sac.connect("opensearch", index=index, dim=dim, hosts=[common.OS_HOST])
    if store.count() >= len(corpus):
        print(f"[reingest] {index} already has {store.count()} docs; skip"); return

    t0 = time.time(); total = 0; B = 2000
    for s in range(0, len(ids), B):
        chunk = ids[s:s + B]
        vecs = passage_embed(texts[s:s + B])
        docs = [sac.Document(id=c, text=corpus[c]["text"],
                             vector=np.asarray(v, dtype=np.float32).tolist(),
                             metadata={"title": corpus[c]["title"]})
                for c, v in zip(chunk, vecs)]
        store.upsert(docs); total += len(docs)
        print(f"[reingest] {total}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
    import time as _t; _t.sleep(1)
    print(f"[reingest] done: {store.count()} docs in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
    main(ap.parse_args().model)
