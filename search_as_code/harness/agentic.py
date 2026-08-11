"""`agentic_solve` — the RECOMMENDED retrieval pipeline: free-form, structure-emergent.

Unlike `diagnostic_solve` (which hardcodes decompose -> per-sub-fact arsenal), here the LLM AUTHORS the
retrieval STRATEGY itself each hop, as code over the OpenSearch query surface — it decides whether to keep
the query WHOLE or DECOMPOSE, whether to use dense/keyword/hybrid/phrase/field-boost/raw-DSL, and whether
to rerank. The playbook only *guides*: the DiagnosticJudge diagnoses what's still missing and the
RAG-Techniques SkillLookup suggests techniques; the model is free to follow or ignore them. Winning
strategies are captured so the forge can bottle the structure that actually worked, per corpus — instead
of assuming decomposition everywhere.

    from search_as_code.harness import agentic_solve
    res = agentic_solve(session, query, generator=llm, reranker=rr, embedder=embed)   # gold=None: judge stops
"""
from __future__ import annotations

import re

import numpy as np

from . import diagnostic_judge as dj
from .forge import _safe_globals
from .playbook import _coverage, _reserve, _rrf, sf_arsenal   # reuse coverage + fusion helpers
from .rag_techniques import SkillLookup

_CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

AUTHOR_SYSTEM = """You are a retrieval STRATEGIST. Write ONE Python function that retrieves the documents \
answering the QUESTION from an OpenSearch index. Signature EXACTLY: def run(session, query, top_k):

Use ONLY these calls (each IS an OpenSearch query); each search returns a ResultSet with `.ids()`:
  session.search(q, top_k=k, mode='dense'|'keyword'|'hybrid')      # kNN / BM25 / both
  session.hyde_search(q, top_k=k)                                  # hallucinate an answer doc, embed it
  session.store.query_fielded(q, ['title^2','text'], top_k=k)     # multi_match + boosts (guard: hasattr(session.store,'query_fielded'))
  session.store._search(body)                                     # RAW OpenSearch DSL (match_phrase, bool,
      # function_score, knn, boosts). Returns a dict; get ids with: [h['_id'] for h in r['hits']['hits']]
These helpers are already in scope (do NOT import them):
  fuse_ids([ids_a, ids_b, ...]) -> list           # RRF-fuse several id lists
  rerank(session, query, ids, top_k=k) -> list    # cross-encoder rerank an id list

YOU choose the STRATEGY. Some questions are ONE entity satisfying MANY constraints — keep the query WHOLE \
(dense/hybrid, then rerank). Others need SEVERAL different documents — DECOMPOSE into sub-questions and \
fuse_ids. Do NOT decompose by default. The deep judge may DIAGNOSE what's missing and suggest a technique \
for that gap — treat it as a HINT for the missing aspect, but YOU still decide the overall structure \
(whole-query vs decompose); do NOT decompose just because the judge mentioned it. ALWAYS return a list of \
ids. Return ONLY one ```python block```."""


def _author(gen, query, diagnosis, suggestions, prior, memory_wins="", findings=""):
    prompt = (f"QUESTION: {query}\n\n"
              f"MEMORY — strategies that WORKED on similar past queries (reuse the winning structure):\n{memory_wins or '(none yet)'}\n\n"
              f"FINDINGS so far — the deep judge's read of THIS query's earlier hops:\n{findings or '(first hop)'}\n\n"
              f"DIAGNOSIS from the deep judge (covered / missing sub-fact / why / suggested technique / suggested next query):\n{diagnosis or '(first hop — decide the strategy)'}\n\n"
              f"SUGGESTED techniques (RAG playbook — optional):\n{suggestions or '(none)'}\n\n"
              f"PRIOR results (id: snippet) — improve on these:\n{prior or '(none yet)'}\n\n"
              "Write def run(session, query, top_k): choosing the strategy that fits THIS question, "
              "acting on the judge's diagnosis (target the MISSING sub-fact with its SUGGESTED technique).")
    raw = gen.complete(prompt, system=AUTHOR_SYSTEM)
    m = _CODE.search(raw)
    return (m.group(1) if m else raw).strip()


def _rerank_helper(session, query, ids, top_k=10):
    docs = session.store.get(list(ids)[:60])
    texts = [d.text or "" for d in docs]
    if not texts:
        return list(ids)[:top_k]
    scores = session.reranker(query, texts) if getattr(session, "reranker", None) else list(range(len(texts), 0, -1))
    order = sorted(range(len(docs)), key=lambda i: -scores[i])
    return [str(docs[i].id) for i in order[:top_k]]


def _exec(code, session, query, top_k):
    g = _safe_globals()
    g["fuse_ids"] = lambda lists: _rrf([list(l) for l in lists])       # RRF over id lists
    g["rerank"] = _rerank_helper                                        # cross-encoder rerank helper
    ns: dict = {}
    exec(compile(code, "<agentic>", "exec"), g, ns)  # noqa: S102 — restricted sandbox
    fn = ns.get("run")
    if not callable(fn):
        raise ValueError("authored code has no run(session, query, top_k)")
    out = fn(session, query, top_k)
    if out is None:
        return []
    return out.ids()[:top_k] if hasattr(out, "ids") else [str(x) for x in out][:top_k]


def agentic_solve(session, query, *, gold=None, max_hops=4, generator=None, judge=None,
                  skill_lookup=None, reranker=None, embedder=None, judge_stop=None, top_k=10,
                  capture=None, memory=None):
    """LLM authors the retrieval strategy each hop (free structure); the DEEP JUDGE guides EVERY hop —
    its covered/missing/diagnosis/technique/next_query become the next hop's instructions — and MEMORY
    carries findings across hops (working) and winning strategies across queries (long-term, for skill
    building). `capture`: per-hop log for the forge. Returns {ids, hops, stopped_by, codes, all_recall, solved}."""
    reranker = reranker or getattr(session, "reranker", None)
    embedder = embedder or getattr(session, "embedder", None)
    if reranker is None:
        from .. import CrossEncoderReranker
        reranker = CrossEncoderReranker()
    if judge is None and generator is not None:
        judge = dj.DiagnosticJudge(generator)
    if skill_lookup is None and embedder is not None:
        skill_lookup = SkillLookup(embedder)
    if memory is None:
        from .memory import AgentMemory
        memory = AgentMemory()                       # transient — still gives cross-hop findings within this query
    if judge_stop is None:
        judge_stop = gold is None
    goldset = set(str(g) for g in (gold or []))

    # cross-query memory: strategies that worked on similar past queries (skill building)
    recalled = memory.recall(query, k=3, kind="skill_win")
    mem_wins = "\n".join(f"- {m.content}" for m in recalled) or ""

    pooled, codes, diagnosis, prior = [], [], "", ""
    fused, got, stopped_by = [], 0, None
    for hop in range(1, max_hops + 1):
        subfacts = [query]                           # decomposition here is ONLY the judge's coverage lens
        try:
            from .. import primitives as P
            subfacts = [s for s in (P.decompose(query, session._require_generator()) or [query]) if s.strip()][:6] or [query]
        except Exception:
            pass
        findings = memory.working_context(max_chars=600, kinds={"finding"})   # cross-hop memory
        key = diagnosis or query
        suggestions = ", ".join(f"{n} ({t})" for n, t in (skill_lookup.suggest(key)[:3] if skill_lookup else []))
        code = _author(generator, query, diagnosis, suggestions, prior, mem_wins, findings)
        codes.append(code)
        try:
            ids = _exec(code, session, query, max(top_k, 30))
        except Exception as e:
            ids = []
            diagnosis = f"previous strategy errored: {type(e).__name__}: {str(e)[:120]} — try a simpler one."
        if ids:
            pooled.append(ids)
        fused = _reserve(pooled, top_k) if len(pooled) > 1 else (pooled[0][:top_k] if pooled else [])
        got = len(goldset & set(fused[:top_k])) if goldset else 0
        if capture is not None:
            capture.append({"hop": hop, "code": code, "n_ids": len(ids),
                            "won": bool(goldset and (goldset & set(fused[:top_k])))})
        if hop == max_hops:
            stopped_by = "maxhops"; break
        if not judge_stop and goldset and got == len(goldset):
            stopped_by = "oracle"; break
        # ALWAYS run the deep judge for its structured suggestion (guides the next hop in BOTH modes)
        sub_vecs = np.asarray(embedder(subfacts), dtype=np.float32)
        cov, cids, ctexts = _coverage(session, embedder, reranker, subfacts, sub_vecs, fused)
        v = {"verdict": "FAIL", "covered": "", "missing": "", "diagnosis": "", "technique": "", "next_query": ""}
        if judge is not None:
            cands = [{"id": i, "score": 1.0 / (r + 1), "snippet": t} for r, (i, t) in enumerate(zip(cids, ctexts))]
            v = judge.judge(query, subfacts, cands, cov)
            if judge_stop and v["verdict"] == "PASS":
                stopped_by = "judge_pass"; break
        weakest = min((c["ce_best"] for c in cov), default=0.0)
        diagnosis = (f"COVERAGE: the current set does NOT fully answer the question yet "
                     f"(weakest ce={weakest:.1f}). Covered aspects: {v.get('covered') or '?'}; still MISSING: "
                     f"aspect {v.get('missing') or '?'} (why: {v.get('diagnosis') or 'weak coverage'}). "
                     f"Optional hint for the missing aspect: {v.get('technique') or 'hyde'} / query "
                     f"'{v.get('next_query') or subfacts[0]}'. You choose whether to keep the query WHOLE or decompose.")
        memory.observe(f"hop{hop}: {diagnosis}  -> retrieved {len(ids)} (covered {got}/{len(goldset) or '?'})",
                       kind="finding")               # cross-hop memory write
        prior = "\n".join(f"{i}: {t[:80]}" for i, t in zip(cids[:5], ctexts[:5]))

    # cross-query memory write: remember the winning strategy so LATER queries can reuse it (skill building)
    if got > 0 and codes:
        memory.remember(f"query \"{query[:70]}\" reached coverage {got}: winning strategy authored "
                        f"(decomposed={'yes' if 're.split' in codes[-1] or 'split(' in codes[-1] else 'no'})",
                        kind="skill_win", code=codes[-1], recall=got)
    return {"ids": fused, "hops": hop, "stopped_by": stopped_by, "codes": codes,
            "all_recall": (got / len(goldset)) if goldset else None,
            "solved": int(bool(goldset) and got == len(goldset)) if goldset else None}
