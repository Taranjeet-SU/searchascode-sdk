"""Evaluate the RAG chatbot on public BEIR-FiQA test queries.

    python -m chatbot.evaluate -n 20

Reports the three axes asked for:
- RELEVANCE  — retrieval quality vs the public qrels (Recall@10 / nDCG@10 / MRR@10),
               plus an optional LLM-judged answer-faithfulness rate.
- SPEED      — mean / p50 / p95 end-to-end latency and retrieval-only latency.
- COST       — total and per-query USD (gpt-4.1-mini pricing).

Retrieval relevance is the honest headline: FiQA ships relevant-doc labels, not gold
answers, so answer quality is bounded by whether we surface the right passages.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time

from phase1 import common, metrics


def _build(agent: str, deep: bool = True):
    if agent == "toolcalling":
        from chatbot.toolcalling import ToolCallingChatbot
        return ToolCallingChatbot()
    if agent == "sac":
        from chatbot.sac_bot import SacChatbot
        return SacChatbot(deep=deep)
    from chatbot.agent import RagChatbot
    return RagChatbot()


def _p(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))]


def main(n: int = 20, judge: bool = False, agent: str = "rag", deep: bool = True) -> None:
    queries = json.loads((common.DATA_DIR / "queries.json").read_text())
    qrels = json.loads((common.DATA_DIR / "qrels.json").read_text())
    qids = [q for q in qrels if any(s > 0 for s in qrels[q].values())][:n]

    bot = _build(agent, deep=deep)
    rankings: dict[str, list[str]] = {}
    lat, cost, faith, hops = [], [], [], []
    arrived = 0
    print(f"[eval] {len(qids)} FiQA queries → {agent} chatbot\n")

    for i, qid in enumerate(qids):
        q = queries[qid]
        a = bot.answer(q)
        rankings[qid] = a.ids[:10]
        lat.append(a.latency_s)
        cost.append(a.usage.get("cost_usd", 0.0))
        hops.append(a.hops)
        arrived += int(a.arrived)
        if judge and not a.abstained and a.answer:
            faith.append(_judge_faithful(bot, q, a))
        if i < 2:
            print(f"Q: {q[:88]}\nA: {a.answer[:220]}\n   {a.hops} hop(s) · sources {a.ids[:5]} · {a.latency_s}s · ${a.usage.get('cost_usd',0):.5f}\n")

    m = metrics.evaluate(rankings, qrels, k=10)
    print("=" * 66)
    print(f"AGENT      {agent}   ({len(qids)} FiQA queries)")
    print(f"ARRIVAL    reached final answer = {arrived}/{len(qids)} ({arrived/len(qids):.0%})   "
          f"avg hops = {st.fmean(hops):.1f}  (min {min(hops)}, max {max(hops)})")
    print(f"RELEVANCE  Recall@10={m['recall@10']:.4f}  nDCG@10={m['ndcg@10']:.4f}  MRR@10={m['mrr@10']:.4f}")
    if faith:
        print(f"           answer-faithfulness (LLM-judged) = {sum(faith)/len(faith):.0%}")
    print(f"SPEED      mean={st.fmean(lat):.2f}s  p50={_p(lat,50):.2f}s  p95={_p(lat,95):.2f}s")
    print(f"COST       ${sum(cost):.4f} total  ·  ${sum(cost)/len(cost):.5f}/query")
    print("=" * 66)


def _judge_faithful(bot, q: str, a) -> float:
    llm = getattr(bot, "base", None) or getattr(bot, "llm", None) or getattr(bot, "chat", None)
    ids = a.ids[:6]
    texts = {s["id"]: s.get("text", "") for s in a.sources if s.get("text")}
    if not texts:                                     # tool-calling bot cites ids only → fetch text
        texts = {d.id: (d.text or "")[:220] for d in bot.sac.store.get(ids)}
    src = "\n".join(f"[{i + 1}] {texts.get(_id, '')}" for i, _id in enumerate(ids))
    prompt = (f"Question: {q}\nSources:\n{src}\n\nAnswer: {a.answer}\n\n"
              "Is every claim in the Answer supported by the Sources? Reply only YES or NO.")
    out = (llm.invoke(prompt).content or "").strip().upper()
    return 1.0 if out.startswith("YES") else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--agent", choices=["rag", "toolcalling", "sac"], default="rag")
    ap.add_argument("--judge", action="store_true", help="add LLM-judged answer faithfulness")
    ap.add_argument("--shallow", action="store_true", help="SAC: use the lean prompt instead of deep ensemble+consensus")
    args = ap.parse_args()
    t = time.time()
    main(n=args.n, judge=args.judge, agent=args.agent, deep=not args.shallow)
    print(f"[eval] done in {time.time()-t:.0f}s")
