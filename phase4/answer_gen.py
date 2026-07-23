"""Phase 4 — answer-generation benchmark: does SAC produce better ANSWERS than
closed-book / vanilla-RAG / tool-calling-RAG, at equal budget?

All arms share the SAME generator (gpt-4.1-mini), the SAME corpus, and the SAME
answer prompt — only the retrieval strategy differs — so any EM/F1 delta is
attributable to retrieval. Closed-book (no context) is the contamination control:
the real signal is the LIFT over closed-book.

    python -m phase4.answer_gen --dataset hotpotqa --n 200 --k 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import search_as_code as sac
from phase1 import agents, common
from phase1.llm import LLM
from phase4 import metrics

DATA = Path(common.REPO) / "phase4" / "data"
RUNS = Path(common.REPO) / "phase4" / "runs"
DIM = 768

ANS_SYS = (
    "You are a precise question-answering system. Answer with the SHORTEST exact phrase "
    "that answers the question (a name, entity, number, date, or yes/no). Do NOT write "
    "a sentence or explanation. If the answer is not determinable, give your best guess."
)


def load(dataset):
    """Return (queries {qid:question}, golds {qid:answer}, opensearch_index)."""
    if dataset == "hotpotqa":
        q = json.loads((Path(common.REPO) / "phase2" / "data" / "hotpot_queries.json").read_text())
        golds = json.loads((DATA / "hotpot_golds.json").read_text())
        qids = [x for x in q if x in golds]
        return {x: q[x] for x in qids}, {x: golds[x] for x in qids}, "hotpotqa"
    raise ValueError(f"no answer-gold loader for {dataset} yet")


def answer(gen: LLM, question: str, contexts: list[str]) -> str:
    if contexts:
        ctx = "\n\n".join(f"[{i+1}] {c[:700]}" for i, c in enumerate(contexts))
        prompt = f"Context:\n{ctx}\n\nQuestion: {question}\nShort answer:"
    else:
        prompt = f"Question: {question}\nShort answer:"
    return gen.complete(prompt, system=ANS_SYS).strip()


def main(dataset="hotpotqa", n=200, k=5):
    queries, golds, index = load(dataset)
    qids = list(queries)[:n]
    from sentence_transformers import SentenceTransformer
    import torch
    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).tolist()
    gen = LLM()
    s = sac.Session("opensearch", index=index, dim=DIM, hosts=[common.OS_HOST], embedder=embed,
                    reranker=sac.QwenReranker(), generator=gen.as_generator())
    store = sac.connect("opensearch", index=index, dim=DIM, hosts=[common.OS_HOST])
    chat = agents.lc_chat()
    s.reranker("warm", ["a", "b"])

    def texts(ids):
        docs = {d.id: d for d in store.get(list(ids)[:k])}
        return [docs[i].text for i in list(ids)[:k] if i in docs and docs[i].text]

    arms = ["closed_book", "vanilla_rag", "tool_rag", "sac"]
    res = {a: {"em": [], "f1": []} for a in arms}
    for i, qid in enumerate(qids):
        qn, gold = queries[qid], golds[qid]
        # each arm -> ranked ids -> top-k context -> same answer prompt
        ctxs = {
            "closed_book": [],
            "vanilla_rag": texts(s.search(qn, k, mode="dense").ids()),
            "tool_rag": texts(agents.run_tool_calling(s, qn, chat=chat, max_retries=1)["ids"]),
            "sac": texts(agents.run_sac(s, qn, chat=chat, max_retries=1)["ids"]),
        }
        for a in arms:
            pred = answer(gen, qn, ctxs[a])
            e, f = metrics.score(pred, gold)
            res[a]["em"].append(e); res[a]["f1"].append(f)
        if (i + 1) % 10 == 0:
            line = "  ".join(f"{a}:F1={sum(res[a]['f1'])/len(res[a]['f1']):.3f}" for a in arms)
            print(f"[{dataset}] {i+1}/{len(qids)}  {line}", flush=True)

    RUNS.mkdir(exist_ok=True)
    out = {"dataset": dataset, "n": len(qids), "k": k, "generator": "gpt-4.1-mini",
           "llm_cost_usd": round(gen.usage.cost_usd, 4), "arms": {}}
    print(f"\n===== {dataset} answer generation (n={len(qids)}, gen=gpt-4.1-mini) =====")
    for a in arms:
        em_m, em_lo, em_hi = metrics.bootstrap_ci(res[a]["em"])
        f1_m, f1_lo, f1_hi = metrics.bootstrap_ci(res[a]["f1"])
        out["arms"][a] = {"EM": em_m, "EM_ci": [em_lo, em_hi], "F1": f1_m, "F1_ci": [f1_lo, f1_hi]}
        print(f"  {a:12s} EM={em_m:.3f} [{em_lo:.3f},{em_hi:.3f}]  F1={f1_m:.3f} [{f1_lo:.3f},{f1_hi:.3f}]")
    cb = out["arms"]["closed_book"]["F1"]
    print(f"\n  lift over closed-book (F1):  vanilla +{out['arms']['vanilla_rag']['F1']-cb:.3f}  "
          f"tool +{out['arms']['tool_rag']['F1']-cb:.3f}  SAC +{out['arms']['sac']['F1']-cb:.3f}")
    print(f"  llm cost ${gen.usage.cost_usd:.4f}")
    (RUNS / f"answergen_{dataset}.json").write_text(json.dumps(out, indent=2))
    print(f"[{dataset}] saved runs/answergen_{dataset}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args(); main(a.dataset, a.n, a.k)
