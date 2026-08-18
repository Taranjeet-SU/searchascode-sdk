"""Diagnostic multi-hop retrieval loop — the reusable SAC playbook.

`diagnostic_solve` runs a multi-hop retrieval as: decompose → per-sub-fact arsenal (hybrid + HyDE +
fielded, RRF-fused) → assemble by reserving one slot per sub-fact then filling by RRF (coverage-first,
avoids the multi-hop dilution where late broad hops evict earlier golds) → the DiagnosticJudge reads the
coverage signals and, for each weak sub-fact, the RAG-technique SkillLookup prescribes the next technique
(HyDE / fielded / rerank / decompose / PRF / LLM-authored os_query), applied and fused.

Stopping: with a gold set it can oracle-stop (measurement upper bound); otherwise the judge decides
(autonomous). Pass `forged=[primitive.run, ...]` to retrieve through FORGED authored primitives instead
of the raw arsenal — the "SAC-replicate" mode that reproduces raw-query relevance from bottled skills.

Returns {ids, all_recall, solved, hops, stopped_by, ...}. `gold` is used only for scoring / oracle-stop.
"""
from __future__ import annotations

import numpy as np

from . import diagnostic_judge as dj
from .loop import fuse_ids
from .os_query import author_os_query
from .rag_techniques import SkillLookup

CE_WEAK = 0.0   # a sub-fact whose best candidate cross-encoder score is below this is treated as missing


def _rrf(lists, k=60):
    """Delegates to the one id-list RRF implementation (SDK-R2)."""
    return fuse_ids(lists, k=k)


def _reserve(sf_lists, k=10):
    """Reserve each sub-fact's best candidate a slot, then fill by global RRF (coverage-guaranteed)."""
    reserved, seen = [], set()
    for lst in sf_lists:
        if lst and lst[0] not in seen:
            reserved.append(lst[0])
            seen.add(lst[0])
    return (reserved + [i for i in _rrf(sf_lists) if i not in seen])[:k]


def _snip(text, n=220):
    return " ".join((text or "").split())[:n]


def sf_arsenal(session, sub, k=30):
    """Per-sub-fact arsenal (hybrid + HyDE + fielded), RRF-fused -> one ranked id list."""
    lists = [session.search(sub, top_k=k, mode="hybrid").ids()]
    try:
        lists.append(session.hyde_search(sub, top_k=k).ids())
    except Exception:
        pass
    f = getattr(session.store, "query_fielded", None)
    try:
        lists.append([h.id for h in f(sub, ["title", "text"], top_k=k)] if f
                     else session.search(sub, top_k=k, mode="keyword").ids())
    except Exception:
        pass
    return _rrf([x for x in lists if x])


def apply_technique(session, reranker, technique, nq, pool_ids, generator=None):
    """Targeted retrieval for a missing sub-fact -> ranked id list."""
    try:
        if technique == "os_query" and generator is not None:
            ids, _b, _ok = author_os_query(session.store, generator, nq, top_k=30)
            return ids or session.search(nq, top_k=30, mode="keyword").ids()
        if technique == "arsenal":
            return sf_arsenal(session, nq)
        if technique == "hyde":
            return session.hyde_search(nq, top_k=30).ids()
        if technique == "fielded":
            f = getattr(session.store, "query_fielded", None)
            return [h.id for h in f(nq, ["title", "text"], top_k=30)] if f else \
                session.search(nq, top_k=30, mode="keyword").ids()
        if technique == "prf":
            return session.prf_search(nq, top_k=30).ids()
        if technique == "decompose":
            from .. import primitives as P
            subs = P.decompose(nq, session._require_generator()) or [nq]
            return _rrf([session.search(s, top_k=30, mode="hybrid").ids() for s in subs])
        if technique == "rerank":
            docs = session.store.get(pool_ids[:40])
            texts = [d.text or "" for d in docs]
            order: list = list(np.argsort(reranker(nq, texts))[::-1]) if texts else []
            return [docs[i].id for i in order]
    except Exception:
        pass
    return session.search(nq, top_k=30, mode="hybrid").ids()


def _coverage(session, embedder, reranker, subfacts, sub_vecs, fused_ids):
    ids = fused_ids[:10]
    docs = {d.id: d for d in session.store.get(ids)}
    texts = [_snip(docs[i].text) if i in docs else "" for i in ids]
    dim = len(sub_vecs[0]) if len(sub_vecs) else 768
    cand_vecs = np.asarray(embedder(texts), dtype=np.float32) if texts else np.zeros((0, dim), np.float32)
    return dj.coverage_signals(subfacts, sub_vecs, texts, cand_vecs, reranker), ids, texts


def _decompose(session, query):
    from .. import primitives as P
    subs = [s for s in (P.decompose(query, session._require_generator()) or [query]) if s.strip()][:6]
    return subs or [query]


def diagnostic_solve(session, query, *, gold=None, max_hops=6, generator=None, judge=None,
                     skill_lookup=None, reranker=None, embedder=None, forged=None, judge_stop=None, top_k=10):
    """One multi-hop solve. `generator` = LLM-like with .complete (needed for the judge / os_query).
    `forged` = optional list of retriever callables run(session, text, k)->ids (SAC-replicate mode).
    `judge_stop`: None auto (judge-stop iff gold is None); True/False to force."""
    reranker = reranker or getattr(session, "reranker", None)
    embedder = embedder or getattr(session, "embedder", None)
    if reranker is None:
        from .. import CrossEncoderReranker
        reranker = CrossEncoderReranker()
    if judge is None and generator is not None:
        judge = dj.DiagnosticJudge(generator)
    if skill_lookup is None and embedder is not None:
        skill_lookup = SkillLookup(embedder)
    if judge_stop is None:
        judge_stop = gold is None

    goldset = set(str(g) for g in (gold or []))
    subfacts = _decompose(session, query)
    sub_vecs = np.asarray(embedder(subfacts), dtype=np.float32)

    def retrieve(fn, text):                          # forged-primitive call, tolerant of ResultSet/ids
        try:
            out = fn(session, text, top_k=30)
            return out if isinstance(out, list) else list(out.ids())
        except Exception:
            return sf_arsenal(session, text)

    base = (lambda s, t, k=30: retrieve(forged[0], t)) if forged else sf_arsenal
    sf_lists = [base(session, s) for s in subfacts]

    got, stopped_by = 0, None
    for hop in range(1, max_hops + 1):
        fused = _reserve(sf_lists, top_k)
        got = len(goldset & set(fused[:top_k])) if goldset else 0
        oracle_complete = bool(goldset) and got == len(goldset)
        if hop == max_hops:
            stopped_by = "maxhops"
            break
        if not judge_stop and oracle_complete:
            stopped_by = "oracle"
            break

        cov, ids, texts = _coverage(session, embedder, reranker, subfacts, sub_vecs, fused)
        weak = [j for j, c in enumerate(cov) if c["ce_best"] < CE_WEAK]
        if judge_stop:
            cands = [{"id": i, "score": 1.0 / (r + 1), "snippet": t} for r, (i, t) in enumerate(zip(ids, texts))]
            v = judge.judge(query, subfacts, cands, cov) if judge else {"verdict": "FAIL", "technique": "", "missing": "", "next_query": ""}
            if v["verdict"] == "PASS":
                stopped_by = "judge_pass"
                break
            if not weak:
                weak = [int(np.argmin([c["ce_best"] for c in cov]))]
        else:
            v = {"technique": "", "missing": "", "next_query": ""}
            if not weak:
                weak = [int(np.argmin([c["ce_best"] for c in cov]))]

        for j in weak:
            if (v.get("missing") or "").isdigit() and int(v["missing"]) - 1 == j and v.get("technique"):
                tech, nq = v["technique"], (v.get("next_query") or subfacts[j])
            elif skill_lookup is not None:
                tech, nq = skill_lookup.suggest(subfacts[j])[0][1], subfacts[j]
            else:
                tech, nq = "hyde", subfacts[j]
            if forged:                               # SAC mode: fix via forged primitives
                fix = retrieve(forged[min(hop, len(forged) - 1)], nq)
            else:
                fix = apply_technique(session, reranker, tech, nq, fused, generator=generator)
            sf_lists[j] = _rrf([sf_lists[j], fix])

    fused = _reserve(sf_lists, top_k)
    got = len(goldset & set(fused[:top_k])) if goldset else got
    return {"ids": fused, "hops": hop, "stopped_by": stopped_by, "subfacts": subfacts,
            "all_recall": (got / len(goldset)) if goldset else None,
            "solved": int(bool(goldset) and got == len(goldset)) if goldset else None}
