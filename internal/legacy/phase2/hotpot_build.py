"""Build a tractable HotpotQA multi-hop retrieval corpus from the LOCAL HF cache
(BeIR/hotpotqa, full 5.2M) — no download. Fixed corpus = gold supporting docs for a
query sample + a random distractor pool. Preserves the multi-hop challenge (2 gold
docs per question) while staying small enough to index.

    python -m phase2.hotpot_build --queries 500 --distractors 100000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common

DATA = Path(common.REPO) / "phase2" / "data"
INDEX = "hotpotqa"
DIM = 768


def main(n_queries=500, n_distract=100000, seed=42):
    t0 = time.time()
    corpus = load_dataset("BeIR/hotpotqa", "corpus", split="corpus")
    qds = load_dataset("BeIR/hotpotqa", "queries", split="queries")
    qrels_rows = load_dataset("BeIR/hotpotqa-qrels", split="test")
    qtext = {r["_id"]: r["text"] for r in qds}
    qrels = {}
    for r in qrels_rows:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    qids = [q for q in qrels if any(s > 0 for s in qrels[q].values()) and q in qtext]
    random.seed(seed)
    qids = random.sample(qids, min(n_queries, len(qids)))
    gold = set()
    for q in qids:
        gold |= {c for c, s in qrels[q].items() if s > 0}
    print(f"[hotpot] {len(qids)} queries, {len(gold)} gold docs (load {time.time()-t0:.0f}s)", flush=True)

    ids_col = corpus["_id"]                                  # 5.2M ids (fast Arrow column)
    id2row = {cid: i for i, cid in enumerate(ids_col)}
    gold_rows = [id2row[g] for g in gold if g in id2row]
    distract_rows = random.sample(range(len(corpus)), n_distract)
    rows = sorted(set(gold_rows) | set(distract_rows))
    sub = corpus.select(rows)
    print(f"[hotpot] corpus subset: {len(sub)} docs (gold {len(gold_rows)} + distract {n_distract}) "
          f"({time.time()-t0:.0f}s)", flush=True)

    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    store = sac.connect("opensearch", index=INDEX, dim=DIM, hosts=[common.OS_HOST])
    store.client.indices.delete(index=INDEX, ignore=[404])
    store.ensure_index(DIM)
    ids_s, titles, texts = sub["_id"], sub["title"], sub["text"]
    B = 2000
    for s in range(0, len(sub), B):
        e = min(s + B, len(sub))
        embtxt = [(titles[i] + ". " + texts[i]) if titles[i] else texts[i] for i in range(s, e)]
        vecs = em.encode(embtxt, normalize_embeddings=True, convert_to_numpy=True,
                         batch_size=256, show_progress_bar=False)
        store.upsert([sac.Document(id=ids_s[i], text=texts[i],
                                   vector=vecs[j].astype(np.float32).tolist(),
                                   metadata={"title": titles[i]})
                      for j, i in enumerate(range(s, e))])
        if (s // B) % 10 == 0:
            print(f"[hotpot] indexed {e}/{len(sub)} ({time.time()-t0:.0f}s)", flush=True)

    (DATA / "hotpot_queries.json").write_text(json.dumps({q: qtext[q] for q in qids}))
    (DATA / "hotpot_qrels.json").write_text(json.dumps({q: qrels[q] for q in qids}))
    time.sleep(1)
    print(f"[hotpot] done. index count = {store.count()} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--distractors", type=int, default=100000)
    main(ap.parse_args().queries, ap.parse_args().distractors)
