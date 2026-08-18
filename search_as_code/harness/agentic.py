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
from .playbook import _coverage, _reserve, _rrf  # reuse coverage + fusion helpers
from .rag_techniques import SkillLookup

_CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

AUTHOR_SYSTEM = """You are a retrieval STRATEGIST. Write ONE Python function that retrieves the documents \
answering the QUESTION from an OpenSearch index. Signature EXACTLY: def run(session, query, top_k):

Use ONLY these calls (each IS an OpenSearch query); each search returns a ResultSet with `.ids()`:
  session.search(q, top_k=k, mode='dense'|'keyword'|'hybrid')      # kNN / BM25 / both
  session.hyde_search(q, top_k=k)                                  # hallucinate an answer doc, embed it
  session.store.query_fielded(q, ['title^2','text'], top_k=k)     # multi_match + boosts (guard: hasattr(session.store,'query_fielded'))
  session.store._search(body)                                     # RAW OpenSearch DSL. Returns a dict;
      # get ids with: [h['_id'] for h in r['hits']['hits']]
These helpers are already in scope (do NOT import them):
  fuse_ids([ids_a, ids_b, ...]) -> list           # RRF-fuse several id lists
  rerank(session, query, ids, top_k=k) -> list    # cross-encoder rerank an id list

DEFAULT = DENSE. For most questions a whole-query dense search is the single strongest move — start there:
    ids = session.search(query, top_k=top_k, mode='dense').ids()
ESCALATE only when the judge says dense coverage is WEAK on an EXACT constraint (a year, a day, a phrase, a
proper name) — dense BLURS exact tokens, so author a RAW OpenSearch DSL query on the `text` field targeting
those literal terms and FUSE it with the dense hits. This works even with NO metadata fields — everything is
on `text`. Worked example (adapt the terms to the missing constraint):
    r = session.store._search({"size": top_k, "query": {"bool": {
            "must":   [{"match_phrase": {"text": "three-day event"}}],
            "should": [{"term": {"text": "2002"}}, {"match": {"text": "Thursday Saturday"}}],
            "minimum_should_match": 1}}})
    dsl_ids = [h["_id"] for h in r["hits"]["hits"]]
    ids = fuse_ids([session.search(query, top_k=top_k, mode='dense').ids(), dsl_ids])

YOU choose the STRATEGY. Most questions are ONE entity satisfying MANY constraints — keep the query WHOLE \
(dense, escalate to raw DSL on the weak exact constraint, then rerank). DECOMPOSE into sub-questions + \
fuse_ids ONLY when the answer genuinely needs SEVERAL different documents. Do NOT decompose by default just \
because the judge mentioned a missing aspect — prefer dense + a targeted raw-DSL escalation for the exact \
constraint. When the judge names a SUGGESTED technique with its recipe, APPLY that recipe (esp. the \
os_query / phrase-bool ones). ALWAYS return a list of ids. Return ONLY one ```python block```."""


AUTHOR_OS_SYSTEM = """You author the FIRST retrieval step as ONE raw OpenSearch query. Signature EXACTLY:
def run(session, query, top_k):
You MUST call session.store._search(body) with a RAW OpenSearch DSL body over the `text` field ONLY (no
dense / hybrid / hyde on this first step). Build a bool query: the DISTINCTIVE exact spans (a proper name, a
title, a quoted phrase) as {"match_phrase": {"text": "..."}} in `should`; the rare content terms / years /
dates as {"match": {"text": "..."}} in `should`; set "minimum_should_match": 1 and "size": top_k*3. Get ids
with [h["_id"] for h in r["hits"]["hits"]] and return that list. Return ONLY one ```python block```."""

_STOP = {"the", "and", "for", "that", "with", "which", "this", "who", "was", "were", "has", "had", "from",
         "what", "name", "please", "tell", "individual", "following", "criteria", "provide", "could", "would"}


def _raw_os_body(query, top_k):
    """Deterministic raw-OS fallback: bool-should of the query's salient tokens over `text` (guaranteed valid)."""
    terms, seen = [], set()
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", query):
        lw = w.lower()
        if lw in _STOP or lw in seen:
            continue
        seen.add(lw)
        terms.append(w)
    terms = terms[:12] or [query[:40]]
    return {"size": top_k * 3, "query": {"bool": {"should": [{"match": {"text": t}} for t in terms],
                                                  "minimum_should_match": 1}}}


def _author_os_first(gen, query, top_k):
    """Hop-1 raw-OpenSearch-query authoring (guaranteed): LLM writes a _search DSL step; if it doesn't comply,
    fall back to a deterministic bool/phrase body. Either way hop 1 IS a raw OS query."""
    try:
        raw = gen.complete(f"QUESTION: {query}\n\nWrite the raw-OpenSearch-DSL first step.", system=AUTHOR_OS_SYSTEM)
        m = _CODE.search(raw)
        code = (m.group(1) if m else raw).strip()
        if "_search(" in code and "def run" in code:
            return code
    except Exception:
        pass
    return ("def run(session, query, top_k):\n"
            f"    r = session.store._search({_raw_os_body(query, top_k)!r})\n"
            "    return [h['_id'] for h in r['hits']['hits']]")


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
    g["fuse_ids"] = lambda lists: _rrf([list(x) for x in lists])       # RRF over id lists
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


def agentic_solve(session, query, *, gold=None, max_hops=10, generator=None, judge=None,
                  skill_lookup=None, reranker=None, embedder=None, judge_stop=None, top_k=10,
                  capture=None, memory=None, os_first=True):
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
    fused: list = []
    got, stopped_by = 0, None
    for hop in range(1, max_hops + 1):
        subfacts = [query]                           # decomposition here is ONLY the judge's coverage lens
        try:
            from .. import primitives as P
            subfacts = [s for s in (P.decompose(query, session._require_generator()) or [query]) if s.strip()][:6] or [query]
        except Exception:
            pass
        findings = memory.working_context(max_chars=600, kinds={"finding"})   # cross-hop memory
        key = diagnosis or query
        # inject each suggested skill's RECIPE (when_to_use), not just its name, so the model can APPLY it
        sug = skill_lookup.suggest(key)[:3] if skill_lookup else []
        suggestions = "\n".join(f"- {n} [{t}]: {recipe}" for n, t, recipe in sug)
        # GUARANTEE: the FIRST step of explore is a raw OpenSearch query (when the store supports raw DSL);
        # later hops are free-form (dense default + escalate). Makes "raw OS query first" a hard invariant.
        if hop == 1 and os_first and hasattr(session.store, "_search"):
            code = _author_os_first(generator, query, max(top_k, 30))
        else:
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
            stopped_by = "maxhops"
            break
        if not judge_stop and goldset and got == len(goldset):
            stopped_by = "oracle"
            break
        # ALWAYS run the deep judge for its structured suggestion (guides the next hop in BOTH modes)
        sub_vecs = np.asarray(embedder(subfacts), dtype=np.float32)
        cov, cids, ctexts = _coverage(session, embedder, reranker, subfacts, sub_vecs, fused)
        v = {"verdict": "FAIL", "covered": "", "missing": "", "diagnosis": "", "technique": "", "next_query": ""}
        if judge is not None:
            csc = dj.candidate_scores(reranker, query, ctexts)   # real spread, not 1/(rank+1) (DJ-10)
            cands = [{"id": i, "score": s, "snippet": t} for (i, t), s in zip(zip(cids, ctexts), csc)]
            v = judge.judge(query, subfacts, cands, cov)
            if judge_stop and v["verdict"] == "PASS":
                stopped_by = "judge_pass"
                break
        weakest = min((c["ce_best"] for c in cov), default=0.0)
        diagnosis = (f"COVERAGE: the current set does NOT fully answer the question yet "
                     f"(weakest ce={weakest:.1f}). Covered aspects: {v.get('covered') or '?'}; still MISSING: "
                     f"aspect {v.get('missing') or '?'} (why: {v.get('diagnosis') or 'weak coverage'}). "
                     f"Hint for the missing aspect: {v.get('technique') or 'hyde'} / query "
                     f"'{v.get('next_query') or subfacts[0]}'. Dense is your default; if the MISSING aspect is an "
                     f"EXACT constraint (year/date/phrase/proper-name), ESCALATE — author a raw "
                     f"session.store._search bool/match_phrase over `text` on those literal terms and fuse with "
                     f"dense. Keep the query WHOLE unless it genuinely needs several different documents.")
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
