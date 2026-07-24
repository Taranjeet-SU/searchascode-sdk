"""Agglomerative multi-evidence SAC for Altera (retrieval is cheap -> cast a wide net).

For multi-document / multi-evidence questions:
  1. decompose -> ATOMIC sub-questions, and REPHRASE each (synonyms/glossary) -> many queries
  2. broad retrieve per sub-question (dense+bm25+kg over all rephrasings) -> rerank -> top evidence
  3. write an evidence-grounded PARTIAL answer per sub-question         (map)
  4. AGGLOMERATE the partials into one final answer, resolving conflicts (reduce)

Returns (all_evidence_docs, final_answer, n_atomic) for scoring + citation.
"""
from __future__ import annotations

from phase4 import altera
from phase4.altera_eval import ctx_text, decompose, rrf
from phase4.altera_eval_tuned import ANS_SYS_TUNED, expand_query

REPHRASE_SYS = ("Rephrase the FPGA search query in 2 different ways using domain synonyms and "
                "alternate phrasings (keep exact part numbers/specs). One per line, no numbering.")
PARTIAL_SYS = ("You are answering ONE focused FPGA sub-question from the context. Give a short, "
               "factual answer grounded ONLY in the context; cite specific values (part numbers, "
               "specs). If the context does not answer it, say 'not found in sources'.")
AGGLOM_SYS = ("You are synthesizing the FINAL answer to an FPGA question from several evidence-grounded "
              "partial findings. Combine them into one complete, accurate answer to the ORIGINAL question. "
              "Use every relevant finding; resolve conflicts by preferring specific knowledge-card facts; "
              "do not invent values not present in the findings. Be precise and well-organized.")


def rephrase(gen, q, n=2):
    r = gen.complete(f"Query: {q}", system=REPHRASE_SYS)
    outs = [s.strip("-• ").strip() for s in r.splitlines() if s.strip()]
    return outs[:n]


def _fanout(sub, rephrasings, k=8):
    """Wide net, cheap: dense (embed) ONLY on the core sub-question; BM25/KG (no embed)
    on the sub-question AND every rephrasing -> many queries without CPU-embed blowup."""
    pools = [altera.dense(expand_query(sub), k)]           # 1 embed
    for q in [sub] + rephrasings:
        qe = expand_query(q)
        kg = altera.bm25_kg(qe, k)
        pools += [kg, kg, altera.bm25_doc(q, k)]           # KG weighted 2x, no embed
    return pools


def agglomerative_sac(gen, reranker, rr_lock, question, k=4, n_rephrase=3):
    subs = decompose(gen, question)                       # atomic sub-questions
    findings, all_docs = [], {}
    for sub in subs:
        reps = rephrase(gen, sub, n_rephrase)             # atomic + rephrased (wide net)
        fused = rrf(_fanout(sub, reps))[:20]
        texts = [(d.get("title", "") + ". " + (d.get("text") or ""))[:800] for d in fused]
        if texts:
            with rr_lock:
                scores = reranker(sub, texts)             # rerank against the SUB-question
            fused = [d for _, d in sorted(zip(scores, fused), key=lambda x: -x[0])]
        top = fused[:k]
        for d in top:
            all_docs.setdefault(d["id"], d)
        ctx = ctx_text(top, k)
        partial = gen.complete(("Context:\n" + "\n\n".join(ctx) + f"\n\nSub-question: {sub}\n\nAnswer:")
                               if ctx else f"Sub-question: {sub}\n\nAnswer:", system=PARTIAL_SYS).strip()
        findings.append((sub, partial))

    block = "\n\n".join(f"Sub-question: {s}\nFinding: {p}" for s, p in findings)
    final = gen.complete(f"Original question: {question}\n\nEvidence-grounded findings:\n{block}\n\n"
                         f"Final answer:", system=AGGLOM_SYS).strip()
    return list(all_docs.values()), final, len(subs)
