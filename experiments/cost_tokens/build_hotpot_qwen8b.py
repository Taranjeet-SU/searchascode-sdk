"""Re-embed the hotpotqa OpenSearch corpus with Qwen3-Embedding-8B -> index `hotpotqa_qwen8b`.

Same recipe as experiments/browsecomp/embed_and_index.py (docs plain — the instruction prefix
is query-side only; 512-token cap for tractable 8B throughput), but sourced by scrolling the
existing `hotpotqa` index instead of a corpus file. Resumable: skips ids already indexed.

    python -m experiments.cost_tokens.build_hotpot_qwen8b [batch=16]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common
from search_as_code.types import Document

SRC, DST, DIM = "hotpotqa", "hotpotqa_qwen8b", 4096
MODEL = "Qwen/Qwen3-Embedding-8B"


def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    dst = sac.connect("opensearch", index=DST, dim=DIM, hosts=[common.OS_HOST],
                      text_field="text", vector_field="vector")
    src = sac.connect("opensearch", index=SRC, dim=common.DIM, hosts=[common.OS_HOST],
                      text_field="text", vector_field="vector")
    have = set()
    try:
        if dst.count() > 0:                    # resume: collect already-indexed ids
            from opensearchpy.helpers import scan
            have = {h["_id"] for h in scan(dst.client, index=DST, query={"query": {"match_all": {}}},
                                           _source=False, size=2000)}
            print(f"[build] resuming: {len(have)} docs already in {DST}", flush=True)
    except Exception:
        pass

    em = SentenceTransformer(MODEL, device="cuda" if torch.cuda.is_available() else "cpu",
                             trust_remote_code=True)
    em.max_seq_length = 512

    from opensearchpy.helpers import scan
    t0, n_done, buf = time.time(), 0, []

    def flush():
        nonlocal buf, n_done
        if not buf:
            return
        texts = [t[:4000] for _, t in buf]
        vecs = em.encode(texts, normalize_embeddings=True, batch_size=batch).tolist()
        dst.upsert([Document(id=i, text=t, vector=v) for (i, t), v in zip(buf, vecs)])
        n_done += len(buf)
        if n_done % 2000 < len(buf):
            rate = n_done / max(1, time.time() - t0)
            print(f"[build] {n_done} docs ({rate:.0f}/s)", flush=True)
        buf = []

    for h in scan(src.client, index=SRC, query={"query": {"match_all": {}}},
                  _source=["text"], size=1000):
        did = str(h["_id"])
        if did in have:
            continue
        text = (h["_source"].get("text") or "").strip()
        if not text:
            continue
        buf.append((did, text))
        if len(buf) >= 256:
            flush()
    flush()
    print(f"[build] DONE: {n_done} newly embedded -> {DST} (total {dst.count()}) "
          f"in {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
