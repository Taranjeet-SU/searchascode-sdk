"""Dump SAC-retrieved context for every sheet question so a frontier LLM (Claude, in the
session) can answer them by hand. Retrieval only (no answer/judge LLM) -> lighter tunnel
load. Resumable: appends one JSONL row per question, skips already-done QIDs.

    ALTERA_OS=... HF_TOKEN=... python -m phase4.altera_dump_context --workers 3
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from phase4 import altera
from phase4.altera_eval import ctx_text, decompose, rrf
from phase4.altera_eval_tuned import expand_query

SHEET = Path(common.REPO) / "SearchUnify_Evaluation_Package_2026_06_27 (1).xlsx - 5. All Questions.csv"
OUT = Path(common.REPO) / "phase4" / "runs" / "agent_context.jsonl"
_lock = threading.Lock()


def retrieve(gen, reranker, rr_lock, q, k=8):
    """Tunnel-LIGHT: KG cards only (2 BM25 calls, no embed/decompose/rerank) -> ~10x faster.
    The curated altera_kg cards are the best content; Claude reasons over them."""
    fused = rrf([altera.bm25_kg(q, 14), altera.bm25_kg(expand_query(q), 10),
                 altera.bm25_doc(q, 8)])          # +a few doc chunks for coverage
    top = fused[:k]
    ctx = ctx_text(top, k)
    srcs = [{"title": d.get("title", ""), "url": d.get("url") or d.get("id")} for d in top]
    return ctx, srcs


def main(workers=3):
    rows = [r for r in csv.DictReader(open(SHEET)) if r.get("Question") and r.get("Expected Answer")]
    done = set()
    if OUT.exists():
        for line in OUT.open():
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass
    todo = [(i, r) for i, r in enumerate(rows) if i not in done]
    print(f"[dump] {len(rows)} questions, {len(done)} already done, {len(todo)} to do", flush=True)
    gen = LLM(); reranker = sac.QwenReranker(); reranker("warm", ["a", "b"]); altera.embedder()
    rr_lock = threading.Lock()

    def work(i, r):
        ctx, srcs = retrieve(gen, reranker, rr_lock, r["Question"])
        return {"idx": i, "device": r.get("Device"), "topic": r.get("Topic"),
                "complexity": r.get("Complexity"), "verdict": r.get("Verdict"),
                "question": r["Question"], "context": ctx, "sources": srcs,
                "gold": r["Expected Answer"], "vendor": r.get("Vendor Answer", ""),
                "gold_cites": r.get("Citations", "")}

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, i, r) for i, r in todo]
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception:
                continue
            with _lock, OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            n += 1
            if n % 10 == 0:
                print(f"[dump] {n}/{len(todo)}", flush=True)
    print(f"[dump] done, wrote {n} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3)
    main(ap.parse_args().workers)
