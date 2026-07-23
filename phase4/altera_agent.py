"""Agentic finetuned SAC for Altera: multi-hop with a GOLD-FREE critic in the loop.

Loop per question:
  decompose -> retrieve (KG-first + glossary) -> draft answer ->
  CRITIC (gold-free): is it grounded/complete? if not, emit NEW queries ->
  retrieve those -> refine ... up to max_hops.

The critic never sees the gold answer (no leakage). The separate eval judge (with gold)
only scores. Returns (docs, answer, n_hops) so we can report how many hops were used.
"""
from __future__ import annotations

import json
import re

from phase4 import altera
from phase4.altera_eval import ctx_text, decompose, rrf
from phase4.altera_eval_tuned import ANS_SYS_TUNED, expand_query

CRITIC_SYS = (
    "You are a retrieval critic for an Altera/Intel FPGA assistant. You see the question, the "
    "sources retrieved so far, and a draft answer. Decide if the sources are SUFFICIENT to fully "
    "and correctly answer (every specific value grounded). You do NOT see any gold answer. "
    'Output ONLY JSON: {"sufficient": true|false, "missing": "what is still needed", '
    '"queries": ["new search query 1", ...]}. If sufficient, queries=[]. Otherwise give 1-3 '
    "specific NEW search queries targeting the gaps (exact part numbers, specs, feature names).")


def _fanout(queries, k=10):
    pools = []
    for q in queries:
        qe = expand_query(q)
        kg = altera.bm25_kg(qe, k)
        pools += [kg, kg, altera.dense(qe, k), altera.bm25_doc(q, k)]   # KG weighted 2x
    return pools


def _dedup_merge(pool, new_lists):
    for lst in new_lists:
        for d in lst:
            pool.setdefault(d["id"], d)
    return pool


def agentic_sac(gen, reranker, rr_lock, question, k=6, max_hops=3):
    queries = decompose(gen, question)                 # hop-1 sub-queries
    pool, docs, draft, hops = {}, [], "", 0
    for hop in range(max_hops):
        hops = hop + 1
        _dedup_merge(pool, _fanout(queries))
        cand = list(pool.values())
        texts = [(d.get("title", "") + ". " + (d.get("text") or ""))[:800] for d in cand]
        if texts:
            with rr_lock:
                scores = reranker(question, texts)
            cand = [d for _, d in sorted(zip(scores, cand), key=lambda x: -x[0])]
        docs = cand[:30]
        ctx = ctx_text(docs, k)
        draft = gen.complete(("Context:\n" + "\n\n".join(ctx) + f"\n\nQuestion: {question}\n\nAnswer:"),
                             system=ANS_SYS_TUNED).strip()
        if hop == max_hops - 1:
            break
        # gold-free critic -> feedback queries for the next hop
        summ = "\n".join(f"- {str(d.get('title'))[:70]}: {str(d.get('text'))[:120]}" for d in docs[:k])
        raw = gen.complete(f"Question: {question}\n\nSources so far:\n{summ}\n\nDraft answer:\n{draft}\n\nCritique:",
                           system=CRITIC_SYS)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            break
        try:
            j = json.loads(m.group(0))
        except Exception:
            break
        if j.get("sufficient") or not j.get("queries"):
            break
        queries = [str(x) for x in j["queries"]][:3]     # critic feedback -> next queries
    return docs, draft, hops
