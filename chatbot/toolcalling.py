"""A **tool-calling** RAG chatbot over cached FiQA (OpenSearch) — the MCP-style
counterpart to ``chatbot.agent.RagChatbot``. Same stack (OpenSearch + hybrid search
via ``search_as_code`` as the retrieval backend), but the agent drives retrieval by
**calling discrete tools** in a multi-hop loop instead of writing code (no SAC
code-mode).

Tool design follows Anthropic's "writing tools for agents" guidance:
- few, high-signal tools (`search_docs`, `read_doc`, `finish`);
- clear names + descriptions written "as if for a new hire";
- compact, structured results (id + short snippet — token-efficient);
- a dedicated structured `finish` tool for the final grounded answer;
- actionable guidance in descriptions (prefer 2-3 focused searches over one broad one).

Each question is resolved as a short "conversation" of 2-5 tool-calling hops.
"""

from __future__ import annotations

import json
import time
from typing import Any

import search_as_code as sac
from chatbot.agent import Answer
from phase1 import common

SYSTEM = (
    "You are a financial-QA assistant answering from a corpus of finance Q&A documents. "
    "Use the tools EFFICIENTLY — stop as soon as you can answer:\n"
    "1. Call `search_docs` (start with mode=\"hybrid\"). INSPECT the returned snippets.\n"
    "2. If the top snippets already answer the question, call `finish` IMMEDIATELY — one search is "
    "enough; do NOT keep searching for its own sake.\n"
    "3. ONLY if the results are insufficient or off-topic, either search again with a different "
    "phrasing/mode OR `read_docs` the TOP FEW promising ids together (batch — never one at a time), "
    "then finish. You may call search_docs and read_docs in the SAME turn.\n"
    "`finish` takes a concise (2-4 sentence) answer grounded ONLY in what you found, citing the "
    "document ids you used in `source_ids`. If the corpus lacks the answer, finish with "
    "\"I don't have enough information to answer that.\" and an empty source list. "
    "Do NOT answer from prior knowledge."
)

TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "search_docs",
        "description": ("Search the FiQA finance corpus. Returns up to top_k compact hits as "
                        "{id, snippet}. Prefer several focused searches with different phrasings "
                        "or modes over one broad query."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the search text"},
            "mode": {"type": "string", "enum": ["dense", "keyword", "hybrid"],
                     "description": "hybrid = semantic+BM25 (best default); keyword = exact terms; dense = semantic"},
            "top_k": {"type": "integer", "description": "number of hits, 1-10 (default 6)"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_docs",
        "description": ("Read the FULL text of one or more documents by id (batch). Pass the TOP few "
                        "candidate ids from your searches TOGETHER — do not read one at a time. You may "
                        "also call search_docs and read_docs in the SAME turn."),
        "parameters": {"type": "object", "properties": {
            "doc_ids": {"type": "array", "items": {"type": "string"},
                        "description": "the most promising ids from search_docs hits (2-5 at once)"}},
            "required": ["doc_ids"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Submit the final grounded answer with the document ids you used as citations. Call once you have enough evidence.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string"},
            "source_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["answer", "source_ids"]}}},
]


class ToolCallingChatbot:
    def __init__(self, k: int = 6, max_hops: int = 5, use_reranker: bool = True):
        common.load_env()
        self.k = k
        self.max_hops = max_hops
        embed = common.get_embedder()
        reranker = sac.CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2") if use_reranker else None
        self.sac = sac.Session("opensearch", index=common.INDEX, dim=common.DIM,
                               hosts=[common.OS_HOST], embedder=embed, reranker=reranker)
        from langchain_openai import ChatOpenAI

        base = ChatOpenAI(model=common.LLM_MODEL, temperature=0.0)
        self.llm = base.bind_tools(TOOLS)
        self.base = base  # for the forced final answer
        self._price = common.LLM_PRICE

    # ---- tool backends (hybrid search via search_as_code) ----------------
    def _search_docs(self, query: str, mode: str = "hybrid", top_k: int = 6) -> tuple[str, list[str]]:
        top_k = max(1, min(int(top_k or self.k), 10))
        rs = self.sac.search(query, top_k=max(top_k * 4, 20), mode=mode if mode in ("dense", "keyword", "hybrid") else "hybrid")
        rs = self.sac.hydrate(rs)
        if self.sac.reranker is not None:
            rs = self.sac.rerank(query, rs, top_k=top_k)
        else:
            rs = rs.top(top_k)
        obs = [{"id": h.id, "snippet": (h.text or "")[:200]} for h in rs]
        return json.dumps(obs), rs.ids()

    def _read_docs(self, doc_ids) -> str:
        ids = [str(i) for i in (doc_ids or [])][:5]
        if not ids:
            return "ERROR: pass doc_ids (a list of ids from search_docs)."
        byid = {d.id: d for d in self.sac.store.get(ids)}
        out = [f"[{i}] {(byid[i].text or '')[:700]}" if i in byid else f"[{i}] (not found)" for i in ids]
        return "\n\n".join(out)

    # ---- multi-hop tool-calling loop -------------------------------------
    def answer(self, question: str) -> Answer:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        t0 = time.perf_counter()
        msgs: list[Any] = [SystemMessage(content=SYSTEM), HumanMessage(content=question)]
        pin = pout = hops = 0
        seen_ids: list[str] = []
        steps: list[dict] = []

        for _ in range(self.max_hops):
            resp: AIMessage = self.llm.invoke(msgs)
            pin, pout = pin + _in(resp), pout + _out(resp)
            msgs.append(resp)
            if not resp.tool_calls:
                # answered without finish → accept the content as the final answer
                return self._done(question, resp.content or "", seen_ids[:10], hops, True, t0, pin, pout, steps)
            finished = None
            # count a hop only for a RETRIEVAL round (search/read); the finish round isn't a hop
            if any(tc["name"] in ("search_docs", "read_docs") for tc in resp.tool_calls):
                hops += 1
            for tc in resp.tool_calls:
                name, args, tcid = tc["name"], tc.get("args", {}), tc["id"]
                steps.append({"hop": hops, "tool": name,
                              "args": {k: str(v)[:80] for k, v in args.items() if k != "answer"}})
                if name == "search_docs":
                    obs, ids = self._search_docs(args.get("query", question), args.get("mode", "hybrid"), args.get("top_k", self.k))
                    seen_ids = ids + [i for i in seen_ids if i not in ids]
                elif name == "read_docs":
                    obs = self._read_docs(args.get("doc_ids", []))
                elif name == "finish":
                    finished, obs = args, "ok"
                else:
                    obs = f"ERROR: unknown tool {name!r}"
                msgs.append(ToolMessage(content=obs, tool_call_id=tcid))
            if finished is not None:
                cited = [str(x) for x in (finished.get("source_ids") or [])] or seen_ids[:10]
                return self._done(question, finished.get("answer", ""), cited, hops, True, t0, pin, pout, steps)

        # hop budget exhausted → force a best-effort final answer
        from langchain_core.messages import HumanMessage as HM

        force = self.base.invoke(msgs + [HM(content="Give your best final answer now, grounded only in the sources above.")])
        pin, pout = pin + _in(force), pout + _out(force)
        return self._done(question, force.content or "", seen_ids[:10], hops, False, t0, pin, pout, steps)

    def _done(self, q, answer, ids, hops, arrived, t0, pin, pout, steps=None) -> Answer:
        cost = (pin * self._price["input"] + pout * self._price["output"]) / 1_000_000
        return Answer(
            question=q, answer=(answer or "").strip(), ids=ids, hops=hops, arrived=arrived,
            latency_s=round(time.perf_counter() - t0, 3),
            usage={"prompt_tokens": pin, "completion_tokens": pout, "cost_usd": round(cost, 6)},
            sources=[{"id": i} for i in ids],
            trace={"kind": "toolcalling", "steps": steps or []},
        )


def _in(resp) -> int:
    um = getattr(resp, "usage_metadata", None) or {}
    return int(um.get("input_tokens", 0))


def _out(resp) -> int:
    um = getattr(resp, "usage_metadata", None) or {}
    return int(um.get("output_tokens", 0))


if __name__ == "__main__":
    import sys

    bot = ToolCallingChatbot()
    print("FiQA tool-calling chatbot — ask a finance question (Ctrl-C to exit)\n")
    for line in sys.stdin:
        q = line.strip()
        if not q:
            continue
        a = bot.answer(q)
        print(f"\n{a.answer}\n  sources: {a.ids[:5]}  ({a.hops} hops · {a.latency_s}s · ${a.usage['cost_usd']:.5f})\n")
