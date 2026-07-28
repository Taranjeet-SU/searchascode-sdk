"""A RAG chatbot agent over cached BEIR-FiQA (OpenSearch) that retrieves with the
``search_as_code`` primitives (hybrid + fan-out + fuse + dedup + rerank + confidence
gating) and answers with citations via LangChain.

    from chatbot.agent import RagChatbot
    bot = RagChatbot()
    ans = bot.answer("How do I deposit a cheque made out to my business?")
    print(ans.answer)          # grounded answer with [n] citations
    print(ans.sources)         # the passages it cited
    print(ans.latency_s, ans.usage["cost_usd"])

Retrieval is 100% our SDK; only the final answer synthesis uses the LLM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import search_as_code as sac
from search_as_code import primitives as P
from phase1 import common

SYSTEM = (
    "You are a precise financial-QA assistant. Answer the user's question USING ONLY the "
    "numbered sources below. Cite every claim with its source number like [2]. If the sources "
    "do not contain enough information to answer, say exactly: \"I don't have enough information "
    "to answer that.\" Be concise (2-4 sentences)."
)


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    retrieval_s: float = 0.0
    usage: dict = field(default_factory=dict)
    abstained: bool = False
    hops: int = 1          # tool-calling rounds (1 for the single-shot RAG bot)
    arrived: bool = True   # did the agent reach a final answer within the hop budget
    trace: dict = field(default_factory=dict)  # agent-specific: generated code / tool-call steps


class RagChatbot:
    """Hybrid-search RAG agent. Retrieval = search_as_code primitives; generation = LangChain."""

    def __init__(self, k: int = 6, pool: int = 40, use_reranker: bool = True,
                 min_gap: float = 0.0):
        common.load_env()
        self.k = k
        self.pool = pool
        self.min_gap = min_gap
        embed = common.get_embedder()
        reranker = sac.CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2") if use_reranker else None
        self.sac = sac.Session(
            "opensearch", index=common.INDEX, dim=common.DIM, hosts=[common.OS_HOST],
            embedder=embed, reranker=reranker,
        )
        from langchain_openai import ChatOpenAI

        self.llm = ChatOpenAI(model=common.LLM_MODEL, temperature=0.0)
        self._price = common.LLM_PRICE

    # ---- retrieval: pure search_as_code primitives -----------------------
    def retrieve(self, question: str, k: int | None = None) -> sac.ResultSet:
        """Hybrid retrieval: dense + BM25 fanned out, RRF-fused, deduped, then
        cross-encoder reranked to the top-k. Showcases the primitive stack."""
        k = k or self.k
        dense = self.sac.search(question, top_k=self.pool, mode="dense")
        keyword = self.sac.search(question, top_k=self.pool, mode="keyword")
        pool = self.sac.fuse([dense, keyword]).dedup()           # hybrid via RRF
        pool = self.sac.hydrate(pool.top(self.pool))
        if self.sac.reranker is not None:
            return self.sac.rerank(question, pool, top_k=k)      # two-stage rerank
        return pool.top(k)

    # ---- answer: retrieve → ground → generate with citations -------------
    def answer(self, question: str) -> Answer:
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        retrieval_s = time.perf_counter() - t0

        # confidence gate: if the top result is weak / no clear winner, abstain
        conf = P.confidence(hits)
        if conf["n"] == 0 or (self.min_gap and conf["gap"] < self.min_gap):
            return Answer(
                question=question, answer="I don't have enough information to answer that.",
                sources=[], ids=hits.ids(), retrieval_s=round(retrieval_s, 3),
                latency_s=round(time.perf_counter() - t0, 3), usage={"cost_usd": 0.0},
                abstained=True,
            )

        context = "\n\n".join(f"[{i + 1}] {(h.text or '')[:600]}" for i, h in enumerate(hits))
        prompt = f"{SYSTEM}\n\nSources:\n{context}\n\nQuestion: {question}\nAnswer (cite sources as [n]):"
        resp = self.llm.invoke(prompt)
        pin, pout = _tokens(resp)
        cost = (pin * self._price["input"] + pout * self._price["output"]) / 1_000_000
        return Answer(
            question=question,
            answer=(resp.content or "").strip(),
            sources=[{"n": i + 1, "id": h.id, "score": round(h.score, 3), "text": (h.text or "")[:220]}
                     for i, h in enumerate(hits)],
            ids=hits.ids(),
            retrieval_s=round(retrieval_s, 3),
            latency_s=round(time.perf_counter() - t0, 3),
            usage={"prompt_tokens": pin, "completion_tokens": pout, "cost_usd": round(cost, 6)},
        )


def _tokens(resp: Any) -> tuple[int, int]:
    um = getattr(resp, "usage_metadata", None)
    if um:
        return int(um.get("input_tokens", 0)), int(um.get("output_tokens", 0))
    tu = (getattr(resp, "response_metadata", {}) or {}).get("token_usage", {}) or {}
    return int(tu.get("prompt_tokens", 0)), int(tu.get("completion_tokens", 0))


if __name__ == "__main__":  # tiny interactive REPL
    import sys

    bot = RagChatbot()
    print("FiQA RAG chatbot — ask a finance question (Ctrl-C to exit)\n")
    for line in sys.stdin:
        q = line.strip()
        if not q:
            continue
        a = bot.answer(q)
        print(f"\n{a.answer}\n")
        print("sources:", ", ".join(f"[{s['n']}] doc {s['id']}" for s in a.sources))
        print(f"({a.latency_s}s · ${a.usage['cost_usd']:.5f})\n")
