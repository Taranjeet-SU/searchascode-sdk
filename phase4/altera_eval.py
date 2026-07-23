"""Altera sandbox eval (INTERNAL — gitignored): can SAC beat the vendor's answer
quality on the 195-question sheet, using their own OpenSearch KB?

Retrieval arms (all against the tunneled Altera OpenSearch, one primitive style):
  closed_book | bm25 | dense(gte-alt-v1) | hybrid(RRF) | sac(decompose+fan-out+RRF+rerank)
Each -> top-k context -> gpt-4.1-mini answer -> LLM judge PASS/FAIL vs Expected Answer.
The judge is first calibrated by scoring the Vendor answers (should ~match the sheet's
52% PASS) so our pass rates are comparable.

    ALTERA_OS=http://localhost:8056 HF_TOKEN=... python -m phase4.altera_eval --n 40
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from phase4 import altera

SHEET = Path(common.REPO) / "SearchUnify_Evaluation_Package_2026_06_27 (1).xlsx - 5. All Questions.csv"
RUNS = Path(common.REPO) / "phase4" / "runs"

ANS_SYS = ("You are an Altera/Intel FPGA technical support assistant. Answer the question "
           "accurately and specifically using ONLY the provided context (device names, part "
           "numbers, specs, steps). Be concise and factual. If the context is insufficient, "
           "say what is known and what is missing. Do not invent part numbers or specs.")
JUDGE_SYS = ("You are grading an FPGA support answer against a ground-truth answer. Output JSON "
             '{"verdict":"PASS"|"FAIL","reason":"..."}. PASS only if the candidate is factually '
             "consistent with the ground truth on the key specs/values/steps asked; FAIL if it "
             "contradicts, omits the core answer, or hallucinates. Ignore style differences.")
JUDGE_CITE_SYS = ("You compare cited SOURCES for an FPGA question. Given the reference source URLs "
                  "and a candidate's retrieved sources (document titles + urls), output JSON "
                  '{"verdict":"PASS"|"FAIL","reason":"..."}. PASS if the candidate retrieved at least '
                  "one document that clearly corresponds to a reference source — match by document "
                  "title/topic/slug (e.g. 'stratix-10-tx-device-overview' == 'Stratix 10 TX Device "
                  "Overview'), ignoring URL-scheme/id differences. FAIL if none correspond.")


def rrf(lists, k=60):
    scores = {}
    docs = {}
    for lst in lists:
        for rank, d in enumerate(lst):
            scores[d["id"]] = scores.get(d["id"], 0.0) + 1.0 / (k + rank + 1)
            docs[d["id"]] = d
    return [docs[i] for i in sorted(scores, key=lambda x: -scores[x])]


def ctx_text(docs, k):
    out = []
    for d in docs[:k]:
        t = (d.get("title") or "").strip()
        c = (d.get("text") or "").strip().replace("\n", " ")
        if c:
            out.append(f"[{t}] {c[:700]}")
    return out


def decompose(gen, q):
    r = gen.complete(f"Break this FPGA support question into 1-3 focused search queries "
                     f"(one per line, no numbering). If already atomic, return it unchanged.\n\nQ: {q}",
                     system="You expand questions into retrieval sub-queries.")
    subs = [s.strip("-• ").strip() for s in r.splitlines() if s.strip()]
    return subs[:3] or [q]


def retrieve(arm, q, gen, reranker):
    """Return ranked docs (list of dicts) for the arm; [] for closed_book."""
    if arm == "closed_book":
        return []
    if arm == "bm25":
        return rrf([altera.bm25_doc(q, 10), altera.bm25_kg(q, 10)])
    if arm == "dense":
        return altera.dense(q, 12)
    if arm == "hybrid":
        return rrf([altera.dense(q, 12), altera.bm25_doc(q, 12), altera.bm25_kg(q, 10)])
    if arm == "sac":
        subs = decompose(gen, q)
        pools = []
        for sq in subs:
            pools.append(altera.dense(sq, 10)); pools.append(altera.bm25_doc(sq, 10))
        pools.append(altera.bm25_kg(q, 10))
        fused = rrf(pools)[:30]
        texts = [(d.get("title", "") + ". " + (d.get("text") or ""))[:800] for d in fused]
        if texts:
            scores = reranker(q, texts)
            fused = [d for _, d in sorted(zip(scores, fused), key=lambda x: -x[0])]
        return fused
    raise ValueError(arm)


def sources_str(docs, k):
    return "\n".join(f"- {(d.get('title') or '?')[:80]}  ({d.get('url') or d.get('id')})"
                     for d in docs[:k]) or "(none)"


def judge_cite(gen, q, gold_cites, docs, k):
    if not gold_cites.strip() or not docs:
        return None  # no gold citations or no retrieval -> not scored
    r = gen.complete(f"Question: {q}\n\nReference sources (ground truth):\n{gold_cites[:1200]}\n\n"
                     f"Candidate retrieved sources:\n{sources_str(docs, k)}\n\nGrade:",
                     system=JUDGE_CITE_SYS)
    m = re.search(r"\{.*\}", r, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)).get("verdict", "FAIL").upper() == "PASS"
        except Exception:
            pass
    return "PASS" in r.upper()[:20]


def answer(gen, q, ctx):
    if ctx:
        c = "\n\n".join(ctx)
        prompt = f"Context:\n{c}\n\nQuestion: {q}\n\nAnswer:"
    else:
        prompt = f"Question: {q}\n\nAnswer:"
    return gen.complete(prompt, system=ANS_SYS).strip()


def judge(gen, q, expected, cand):
    r = gen.complete(f"Question: {q}\n\nGround-truth answer:\n{expected[:1200]}\n\n"
                     f"Candidate answer:\n{cand[:1200]}\n\nGrade:", system=JUDGE_SYS)
    m = re.search(r"\{.*\}", r, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)).get("verdict", "FAIL").upper() == "PASS"
        except Exception:
            pass
    return "PASS" in r.upper()[:20]


ARMS = ["closed_book", "bm25", "dense", "hybrid", "sac"]


def process_one(r, gen, reranker, rr_lock, k):
    """Full per-question pipeline (all arms + judges). Thread-safe."""
    q, exp, gold_cites = r["Question"], r["Expected Answer"], r.get("Citations", "")
    out = {"ans": {}, "cite": {}, "vendor": 0, "vendor_sheet": 0, "has_vendor": False}
    if r.get("Vendor Answer"):
        out["has_vendor"] = True
        out["vendor"] = int(judge(gen, q, exp, r["Vendor Answer"]))
        out["vendor_sheet"] = int(r.get("Verdict", "").upper() == "PASS")
    for a in ARMS:
        if a == "sac":
            with rr_lock:                      # GPU reranker: serialize the forward pass
                docs = retrieve(a, q, gen, reranker)
        else:
            docs = retrieve(a, q, gen, reranker)
        ctx = ctx_text(docs, k)
        out["ans"][a] = int(judge(gen, q, exp, answer(gen, q, ctx)))
        out["cite"][a] = judge_cite(gen, q, gold_cites, docs, k)   # None if n/a
    return out


def main(n=40, k=6, workers=8):
    rows = [r for r in csv.DictReader(open(SHEET)) if r.get("Question") and r.get("Expected Answer")]
    rows = rows[:n]
    gen = LLM()
    reranker = sac.QwenReranker()
    reranker("warm", ["a", "b"])
    altera.embedder()  # warm (loads gte-alt-v1 on CPU)
    rr_lock = threading.Lock()

    arms = ARMS
    ans = {a: 0 for a in arms}
    cite = {a: 0 for a in arms}; cite_n = {a: 0 for a in arms}
    vendor_pass = vendor_sheet_pass = 0
    done = 0
    print(f"[altera] running {len(rows)} questions with {workers} parallel workers...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_one, r, gen, reranker, rr_lock, k) for r in rows]
        for f in as_completed(futs):
            o = f.result(); done += 1
            if o["has_vendor"]:
                vendor_pass += o["vendor"]; vendor_sheet_pass += o["vendor_sheet"]
            for a in arms:
                ans[a] += o["ans"][a]
                if o["cite"][a] is not None:
                    cite_n[a] += 1; cite[a] += o["cite"][a]
            if done % 5 == 0:
                la = " ".join(f"{a}={ans[a]/done:.2f}" for a in arms)
                lc = " ".join(f"{a}={cite[a]/cite_n[a]:.2f}" for a in arms if cite_n[a])
                print(f"[altera] {done}/{len(rows)}  ANS[{la}]  vendor={vendor_pass/done:.2f}", flush=True)
                print(f"            CITE[{lc}]", flush=True)

    N = len(rows)
    print(f"\n===== Altera eval (n={N}, gen=gpt-4.1-mini, KB=altera sandbox) =====")
    print(f"  vendor answer (our judge)     PASS={vendor_pass/N:.3f}")
    print(f"  vendor answer (sheet verdict) PASS={vendor_sheet_pass/N:.3f}  <- judge calibration target")
    print(f"  {'arm':11s} {'answer_PASS':>12s} {'citation_PASS':>14s}")
    for a in arms:
        cp = f"{cite[a]/cite_n[a]:.3f}" if cite_n[a] else "  n/a"
        print(f"  {a:11s} {ans[a]/N:>12.3f} {cp:>14s}")
    print(f"  llm cost ${gen.usage.cost_usd:.4f}")
    RUNS.mkdir(exist_ok=True)
    (RUNS / "altera_eval.json").write_text(json.dumps(
        {"n": N, "k": k, "vendor_our_judge": vendor_pass / N, "vendor_sheet": vendor_sheet_pass / N,
         "answer": {a: ans[a] / N for a in arms},
         "citation": {a: (cite[a] / cite_n[a] if cite_n[a] else None) for a in arms},
         "citation_n": cite_n, "llm_cost_usd": round(gen.usage.cost_usd, 4)}, indent=2))
    print("[altera] saved runs/altera_eval.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); main(a.n, a.k, a.workers)
