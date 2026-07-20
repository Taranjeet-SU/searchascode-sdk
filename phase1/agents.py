"""The three query paths compared in Phase 1, all sharing one SAC Session.

- run_base:        deterministic retrieval, no LLM (the cost/latency floor).
- run_tool_calling: MCP-style — LLM calls discrete search tools; intermediate
  hits flow back through context each hop (LangChain tool-calling).
- run_sac:         code-mode — LLM writes ONE Python program against the SAC SDK,
  executed in the sandbox; only the final ids return (LangChain code-gen).

Each returns a uniform dict: ids, latency_s, usage(dict), and a trace for the UI.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from search_as_code.sandbox import LocalExecutor
from phase1 import common, sac_surface
from phase1.llm import Usage

common.load_env()

_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def lc_chat(model: str = common.LLM_MODEL, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=temperature)


def _account(usage: Usage, msg) -> None:
    um = getattr(msg, "usage_metadata", None) or {}
    cached = (um.get("input_token_details") or {}).get("cache_read", 0) or 0
    usage.add(um.get("input_tokens", 0), um.get("output_tokens", 0), cached)


# ---------------------------------------------------------------- base
def run_base(session, query: str, k: int = 10, mode: str = "hybrid") -> dict:
    t0 = time.time()
    ids = session.search(query, top_k=k, mode=mode).ids()
    return {"path": "base", "ids": ids, "latency_s": time.time() - t0,
            "usage": Usage().as_dict(), "trace": [{"op": f"search(mode={mode})", "returned": len(ids)}]}


# ---------------------------------------------------------------- SAC code-gen
def run_sac(session, query: str, chat=None, k: int = 10) -> dict:
    from langchain_core.messages import SystemMessage, HumanMessage

    chat = chat or lc_chat()
    usage = Usage()
    t0 = time.time()
    # STATIC prefix first (cached), dynamic query last — see docs/CACHING.md
    resp = chat.invoke([SystemMessage(content=sac_surface.SAC_SYSTEM),
                        HumanMessage(content=f"Query: {query}")])
    _account(usage, resp)
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    m = _CODE_RE.search(raw)
    code = (m.group(1) if m else raw).strip()

    box = LocalExecutor(session)
    box._globals["query"] = query
    result = box.run(code)
    ids = result.evidence if isinstance(result.evidence, list) else []
    ids = [str(x) for x in ids][:k]
    return {"path": "sac", "ids": ids, "latency_s": time.time() - t0, "usage": usage.as_dict(),
            "code": code, "ok": result.ok, "error": result.error,
            "trace": [{"op": "generate_code", "chars": len(code)},
                      {"op": "sandbox_execute", "ok": result.ok, "returned": len(ids),
                       "stdout": (result.stdout or "")[:300]}]}


# ---------------------------------------------------------------- tool-calling
def _tool_search(session, args: dict, k: int) -> list[dict]:
    hits = session.search(args["query"], top_k=args.get("top_k", k), mode=args.get("mode", "hybrid"))
    return [{"id": h.id, "snippet": (h.text or "")[:120]} for h in hits]


def run_tool_calling(session, query: str, chat=None, k: int = 10, max_steps: int = 6) -> dict:
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

    chat = chat or lc_chat()
    bound = chat.bind_tools(sac_surface.TOOLCALL_TOOLS)
    usage = Usage()
    trace: list[dict] = []
    msgs = [SystemMessage(content=sac_surface.TOOLCALL_SYSTEM), HumanMessage(content=f"Query: {query}")]
    t0 = time.time()
    ids: list[str] = []
    for _ in range(max_steps):
        resp = bound.invoke(msgs)
        _account(usage, resp)
        msgs.append(resp)
        calls = resp.tool_calls or []
        if not calls:
            break
        done = False
        for call in calls:
            name, args, cid = call["name"], call["args"], call["id"]
            if name == "finish":
                ids = [str(x) for x in args.get("doc_ids", [])][:k]
                trace.append({"op": "finish", "returned": len(ids)})
                msgs.append(ToolMessage(content="ok", tool_call_id=cid))
                done = True
            elif name == "search":
                hits = _tool_search(session, args, k)
                trace.append({"op": f"search({args.get('mode','hybrid')})", "returned": len(hits)})
                import json
                msgs.append(ToolMessage(content=json.dumps(hits), tool_call_id=cid))
            elif name == "rephrase":
                import search_as_code as sac
                better = sac.rephrase(args["query"], session._require_generator())
                trace.append({"op": "rephrase", "out": better[:80]})
                msgs.append(ToolMessage(content=better, tool_call_id=cid))
            else:
                msgs.append(ToolMessage(content="unknown tool", tool_call_id=cid))
        if done:
            break
    return {"path": "tool_calling", "ids": ids, "latency_s": time.time() - t0,
            "usage": usage.as_dict(), "trace": trace}
