"""A **search-as-code (SAC)** chatbot over cached FiQA (OpenSearch).

Retrieval is the SAC code-mode agent (``phase1.agents.run_sac``): the LLM writes a
Python program against the primitive surface, runs it in a sandbox, and a judge loop
deepens it — only the final evidence ids come back. This bot then synthesizes a
grounded, cited answer from those ids. Contrast with ``chatbot.toolcalling`` (discrete
tool calls) for the SAC-vs-tool-calling comparison.
"""

from __future__ import annotations

import copy
import time

import search_as_code as sac
from chatbot.agent import SYSTEM, Answer
from phase1 import agents, common
from phase1.llm import LLM


class SacChatbot:
    def __init__(self, k: int = 6, max_retries: int = 2, use_reranker: bool = True, deep: bool = True):
        common.load_env()
        self.k = k
        self.max_retries = max_retries
        self.deep = deep  # default: ensemble+consensus SAC prompt
        embed = common.get_embedder()
        reranker = sac.CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2") if use_reranker else None
        self.gen = LLM()  # generator for query-side primitives AND final answer synthesis (usage-tracked)
        self.session = sac.Session(
            "opensearch", index=common.INDEX, dim=common.DIM, hosts=[common.OS_HOST],
            embedder=embed, reranker=reranker, generator=self.gen.as_generator(),
        )
        self.sac = self.session       # alias so eval helpers find the store uniformly
        self.chat = agents.lc_chat()  # LangChain ChatOpenAI for code-gen + judge
        self._price = common.LLM_PRICE

    def answer(self, question: str, deep: bool | None = None) -> Answer:
        t0 = time.perf_counter()
        before = copy.copy(self.gen.usage)
        # 1. SAC code-mode retrieval (writes+runs a program, judge loop) → evidence ids
        r = agents.run_sac(self.session, question, chat=self.chat, k=self.k,
                           max_retries=self.max_retries, deep=self.deep if deep is None else deep)
        ids = r["ids"]

        # 2. ground + synthesize a cited answer from the retrieved docs
        byid = {d.id: d for d in self.session.store.get(ids)}
        docs = [byid[i] for i in ids if i in byid]
        if not docs:
            return self._done(question, "I don't have enough information to answer that.", [],
                              r, before, t0, abstained=True)
        context = "\n\n".join(f"[{i + 1}] {(d.text or '')[:600]}" for i, d in enumerate(docs))
        answer = self.gen.complete(f"Sources:\n{context}\n\nQuestion: {question}\nAnswer (cite [n]):", system=SYSTEM)
        return self._done(question, answer, [d.id for d in docs], r, before, t0)

    def _done(self, q, answer, ids, r, before, t0, abstained=False) -> Answer:
        # cost/tokens = SAC code-gen + judge (r["usage"]) + generator delta (query-side + answer)
        ru = r["usage"]
        gen_delta = self.gen.usage.cost_usd - before.cost_usd
        pin = ru.get("input_tokens", 0) + (self.gen.usage.input_tokens - before.input_tokens)
        pout = ru.get("output_tokens", 0) + (self.gen.usage.output_tokens - before.output_tokens)
        trace = {
            "kind": "sac",
            "code": r.get("code", ""),
            "reasoning": r.get("reasoning", ""),
            "agreement": self.session.recall("agreement"),
            "hops_detail": [{"hop": a["hop"], "judge": a.get("judge"), "agreement": a.get("agreement")}
                            for a in r.get("attempts", [])],
            "attempts": [{"hop": a["hop"], "judge": a.get("judge"), "code": a.get("code", ""),
                          "reasoning": a.get("reasoning", ""), "ids": a.get("ids", [])[:6]}
                         for a in r.get("attempts", [])],
        }
        return Answer(
            question=q, answer=(answer or "").strip(),
            sources=[{"id": i} for i in ids], ids=ids,
            hops=r.get("hops", 1), arrived=bool(ids), abstained=abstained,
            latency_s=round(time.perf_counter() - t0, 3),
            usage={"cost_usd": round(ru.get("cost_usd", 0.0) + gen_delta, 6),
                   "retrieval_cost": ru.get("cost_usd", 0.0),
                   "prompt_tokens": pin, "completion_tokens": pout},
            trace=trace,
        )


if __name__ == "__main__":
    import sys

    bot = SacChatbot()
    print("FiQA SAC (code-mode) chatbot — ask a finance question (Ctrl-C to exit)\n")
    for line in sys.stdin:
        q = line.strip()
        if not q:
            continue
        a = bot.answer(q)
        print(f"\n{a.answer}\n  sources: {a.ids[:5]}  ({a.hops} hop(s) · {a.latency_s}s · ${a.usage['cost_usd']:.5f})\n")
