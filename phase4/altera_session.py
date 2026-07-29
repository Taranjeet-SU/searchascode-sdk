"""Bind a real sac.Session to the Altera KB so the code-mode agent gets the FULL SDK
primitive surface (not the 6 hand-wired ones):

  sac  : Session over ft_document (custom fields + gte-alt-v1 embedder + Qwen reranker) ->
         search(dense/keyword/hybrid/regex), hyde_search, prf_search, expand_search,
         decompose_search, rephrase_search, rerank, retrieve_rerank, mmr, semantic_dedup,
         compress, adaptive_search, smart_search, hydrate, ...
  kb   : altera_kg curated cards, returned as a ResultSet (composes with the above)
  + fuse / mmr / diversity_quota / confidence / abstain / score_cutoff / normalize_scores
"""
from __future__ import annotations

import search_as_code as sac
from search_as_code import primitives as P
from search_as_code.types import Document, Hit, ResultSet
from phase1.llm import LLM
from phase4 import altera

_session = None


def altera_session():
    global _session
    if _session is None:
        altera.embedder()  # warm gte-alt-v1 (via the standard SDK embedder)
        # TRUE batch (one forward per batch) — the SDK embedder handles the gte-alt-v1
        # meta-buffer/device quirks; far faster than one-at-a-time for the router fit.
        def embed_batch(ts):
            return altera.embedder().embed(list(ts))
        _session = sac.Session(
            "opensearch", index=altera.FT_DOC, hosts=[altera.OS_URL], dim=768,
            text_field=altera.FTP + "content", vector_field=altera.FT_VECTOR,
            embedder=embed_batch, reranker=sac.QwenReranker(), generator=LLM().as_generator())
    return _session


def kb(query: str, k: int = 10) -> ResultSet:
    """altera_kg curated cards as a ResultSet, so they compose with Session primitives."""
    hits = []
    for d in altera.bm25_kg(query, k):
        hits.append(Hit(id=d["id"], score=float(d["score"]), store="altera_kg",
                        document=Document(id=d["id"], text=d.get("text", ""),
                                          metadata={"title": d.get("title", ""), "url": d.get("url", "")})))
    return ResultSet(hits)


def agent_namespace(question: str) -> dict:
    """Namespace exposing the FULL primitive surface for LLM-authored code-mode retrieval."""
    s = altera_session()
    return {"sac": s, "kb": kb, "question": question, "results": None,
            "fuse": P.fuse, "rrf": getattr(P, "rrf", P.fuse), "mmr": P.mmr,
            "diversity_quota": P.diversity_quota, "confidence": P.confidence, "abstain": P.abstain,
            "score_cutoff": P.score_cutoff, "normalize_scores": P.normalize_scores}


def _self_test():
    ns = agent_namespace("Agilex 7 transceiver channel count and speed grade")
    s, kbf = ns["sac"], ns["kb"]
    q = ns["question"]
    print("dense:      ", len(s.search(q, 3, mode="dense").ids()), "hits")
    print("hybrid:     ", len(s.search(q, 3, mode="hybrid", alpha=0.5).ids()), "hits")
    print("hyde_search:", len(s.hyde_search(q, 3).ids()), "hits")
    print("kb(cards):  ", len(kbf(q, 3).ids()), "hits")
    fused = ns["fuse"]([s.search(q, 10, mode="hybrid"), s.hyde_search(q, 10), kbf(q, 10)])
    print("fuse+rerank:", len(s.rerank(q, fused, top_k=5).ids()), "hits")
    print("=> FULL primitive surface active over the Altera KB")


if __name__ == "__main__":
    _self_test()
