"""Build the BrowseComp-Plus corpus index: load corpus (NON-streaming), embed
with gte-base on GPU, and persist vecs/ids/texts locally for the eval harness.

If CAP env is set (e.g. CAP=40000), the corpus is capped to
union(all gold docids) + a random sample up to CAP docs (gold guaranteed present).

    python -m experiments.browsecomp.build_index
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from phase1 import common
from experiments.browsecomp import bc_common as B


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    cap = int(os.environ.get("CAP", "0"))
    golds = B.load_golds()
    gold_ids = {d for ds in golds.values() for d in ds}
    log(f"gold docids: {len(gold_ids)} across {len(golds)} queries")

    log("loading corpus Tevatron/browsecomp-plus-corpus (non-streaming)...")
    ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    log(f"corpus loaded: {len(ds)} docs; columns={ds.column_names}")

    # figure out id/text field names
    cols = ds.column_names
    id_field = "docid" if "docid" in cols else ("id" if "id" in cols else cols[0])
    text_field = "text" if "text" in cols else ("contents" if "contents" in cols else cols[1])
    log(f"using id_field={id_field} text_field={text_field}")

    ids = [str(x) for x in ds[id_field]]
    texts = ds[text_field]

    keep_idx = list(range(len(ids)))
    if cap and len(ids) > cap:
        idset = set(ids)
        missing = [g for g in gold_ids if g not in idset]
        log(f"CAP={cap}: gold docids missing from corpus (will be absent): {len(missing)}")
        gold_pos = [k for k, i in enumerate(ids) if i in gold_ids]
        rest = [k for k in range(len(ids)) if ids[k] not in gold_ids]
        random.seed(0)
        n_extra = max(0, cap - len(gold_pos))
        keep_idx = sorted(gold_pos + random.sample(rest, min(n_extra, len(rest))))
        log(f"capped to {len(keep_idx)} docs ({len(gold_pos)} gold + sample)")

    kept_ids = [ids[k] for k in keep_idx]
    kept_texts = [texts[k] for k in keep_idx]
    present_gold = len(set(kept_ids) & gold_ids)
    log(f"gold docids present in kept corpus: {present_gold}/{len(gold_ids)}")

    em = SentenceTransformer(common.EMB_MODEL,
                             device="cuda" if torch.cuda.is_available() else "cpu")
    N = len(kept_texts)
    dim = common.DIM
    vecs = np.zeros((N, dim), dtype=np.float32)
    bs = 256
    t0 = time.time()
    for start in range(0, N, bs):
        chunk = kept_texts[start:start + bs]
        v = em.encode(chunk, normalize_embeddings=True, batch_size=bs,
                      show_progress_bar=False)
        vecs[start:start + len(chunk)] = v
        if (start // bs) % 20 == 0:
            done = start + len(chunk)
            rate = done / max(1e-6, time.time() - t0)
            eta = (N - done) / max(1e-6, rate)
            log(f"embedded {done}/{N} ({100*done/N:.1f}%) {rate:.0f} docs/s ETA {eta:.0f}s")
    log(f"embedding done in {time.time()-t0:.0f}s")

    np.save(B.VECS_NPY, vecs)
    B.IDS_JSON.write_text(json.dumps(kept_ids))
    with B.TEXTS_JSONL.open("w") as f:
        for i, t in zip(kept_ids, kept_texts):
            f.write(json.dumps({"id": i, "text": t}) + "\n")
    log(f"saved: {B.VECS_NPY.name} {vecs.shape}, {B.IDS_JSON.name}, {B.TEXTS_JSONL.name}")


if __name__ == "__main__":
    main()
