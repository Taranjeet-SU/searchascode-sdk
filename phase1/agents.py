"""The three query paths compared in Phase 1, all sharing one SAC Session.

- run_base:        deterministic retrieval, no LLM (the cost/latency floor).
- run_tool_calling: MCP-style — LLM calls discrete tools (expand/search); intermediate
  hits flow back through context each hop (LangChain tool-calling).
- run_sac:         code-mode — LLM writes ONE Python program against the SAC SDK,
  executed in the sandbox; only the final ids return (LangChain code-gen).

Both LLM paths reformulate the query into exactly `N_QUERY_VARIANTS` formulations,
emit reasoning, and are gated by an LLM-as-judge that can trigger up to
`max_retries` refinement hops. SAC carries its sandbox state across hops (bulky
candidates persist; only compact judge feedback crosses the context boundary) —
the code-execution-with-MCP / search-as-code multi-turn pattern.

Each returns a uniform dict: ids, latency_s, usage, reasoning, hops, attempts, trace.
"""

from __future__ import annotations

import json
import re
import time

from search_as_code.sandbox import LocalExecutor
from phase1 import common, sac_surface
from phase1.llm import Usage

common.load_env()

_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
N = common.N_QUERY_VARIANTS


def lc_chat(model: str = common.LLM_MODEL, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=temperature)


def _account(usage: Usage, msg) -> None:
    um = getattr(msg, "usage_metadata", None) or {}
    cached = (um.get("input_token_details") or {}).get("cache_read", 0) or 0
    usage.add(um.get("input_tokens", 0), um.get("output_tokens", 0), cached)


def _text(msg) -> str:
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def _split_reasoning_code(raw: str) -> tuple[str, str]:
    m = _CODE_RE.search(raw)
    code = (m.group(1) if m else raw).strip()
    pre = raw[: m.start()] if m else ""
    reasoning = ""
    for line in pre.splitlines():
        if line.strip().upper().startswith("REASONING"):
            reasoning = line.split(":", 1)[-1].strip()
            break
    return (reasoning or " ".join(pre.split())[:300]), code


def judge(judge_chat, query: str, ids, session, usage: Usage) -> tuple[bool, str, float]:
    """Calibrated LLM-as-judge. Returns (accept, feedback, confidence 0-1)."""
    from langchain_core.messages import SystemMessage, HumanMessage

    docs = session.store.get([i for i in ids[:6]])
    body = "\n".join(f"[{d.id}] {(d.text or '')[:200]}" for d in docs) or "(no results returned)"
    resp = judge_chat.invoke([SystemMessage(content=sac_surface.JUDGE_SYSTEM),
                              HumanMessage(content=f"Query: {query}\n\nResults:\n{body}")])
    _account(usage, resp)
    text = _text(resp)
    accept, conf, fb = True, 0.5, ""
    for line in text.splitlines():
        s = line.strip()
        u = s.upper()
        if u.startswith("VERDICT"):
            accept = "PASS" in u
        elif u.startswith("CONFIDENCE"):
            try:
                conf = float(s.split(":", 1)[-1].strip().split()[0])
            except Exception:
                pass
        elif u.startswith("FEEDBACK"):
            fb = s.split(":", 1)[-1].strip()
    if not ids:
        accept, conf = False, 0.0
    return accept, fb, conf


# ---------------------------------------------------------------- base
def run_base(session, query: str, k: int = 10, mode: str = "hybrid") -> dict:
    t0 = time.time()
    ids = session.search(query, top_k=k, mode=mode).ids()
    return {"path": "base", "ids": ids, "latency_s": time.time() - t0, "reasoning": "",
            "hops": 1, "attempts": [], "usage": Usage().as_dict(),
            "trace": [{"op": f"search(mode={mode})", "returned": len(ids)}]}


# ---------------------------------------------------------------- SAC code-gen (+judge loop)
def _samples(session, ids, n: int = 5) -> str:
    docs = session.store.get([i for i in ids[:n]])
    return "\n".join(f"[{d.id}] {(d.text or '')[:140]}" for d in docs) or "(none)"


def run_sac(session, query: str, chat=None, k: int = 10, judge_chat=None, max_retries: int = 3) -> dict:
    from langchain_core.messages import SystemMessage, HumanMessage

    chat = chat or lc_chat()
    judge_chat = judge_chat or chat
    usage = Usage()
    t0 = time.time()
    box = LocalExecutor(session)          # ONE executor → namespace persists across hops
    box._globals["query"] = query
    attempts, feedback, ids, code, reasoning = [], None, [], "", ""
    prev_stdout, prev_samples, verdict = "", "", ""
    best_ids, best_conf = [], -1.0            # keep the highest-confidence hop, never lose a good one

    for hop in range(max_retries + 1):
        user_msg = f"Query: {query}"
        if hop > 0:  # feed samples + stdout + judge verdict back so the model deepens its exploration
            user_msg = sac_surface.SAC_RETRY_TEMPLATE.format(
                verdict=verdict, feedback=feedback, samples=prev_samples,
                stdout=prev_stdout or "(none)", code=code)
        msgs = [SystemMessage(content=sac_surface.SAC_SYSTEM), HumanMessage(content=user_msg)]
        resp = chat.invoke(msgs)
        _account(usage, resp)
        reasoning, code = _split_reasoning_code(_text(resp))
        result = box.run(code)            # prior variables (dense, kw, pool, ...) still live here
        ids = [str(x) for x in (result.evidence or [])][:k] if isinstance(result.evidence, list) else []
        if max_retries == 0:              # fast mode: single pass, skip the judge LLM call
            accept, feedback, verdict, conf = True, "", "SKIPPED", 1.0
        else:
            accept, feedback, conf = judge(judge_chat, query, ids, session, usage)
            verdict = "PASS" if accept else "FAIL"
        if ids and conf > best_conf:      # keep the best hop so refinement can't destroy a good result
            best_conf, best_ids = conf, ids
        prev_stdout = (result.stdout or "")[:1200]
        prev_samples = _samples(session, ids)
        attempts.append({"hop": hop, "reasoning": reasoning, "code": code, "ok": result.ok,
                         "error": result.error, "stdout": prev_stdout, "samples": prev_samples,
                         "prompt": user_msg, "ids": ids, "judge": verdict,
                         "confidence": round(conf, 2), "feedback": feedback})
        if accept or conf >= 0.75 or hop == max_retries:   # short-circuit when confident
            break

    ids = best_ids or ids                 # return the highest-confidence hop, not necessarily the last
    return {"path": "sac", "ids": ids, "latency_s": time.time() - t0, "usage": usage.as_dict(),
            "code": code, "reasoning": reasoning, "ok": attempts[-1]["ok"], "hops": len(attempts),
            "attempts": attempts, "system_prompt": sac_surface.SAC_SYSTEM,
            "trace": [{"op": f"hop {a['hop']} · judge {a['judge']}", "returned": len(a["ids"])} for a in attempts]}


# ---------------------------------------------------------------- tool-calling (+judge loop)
def _tool_search(session, args: dict, k: int) -> list[dict]:
    hits = session.search(args["query"], top_k=args.get("top_k", k), mode=args.get("mode", "hybrid"))
    return [{"id": h.id, "snippet": (h.text or "")[:120]} for h in hits]


def run_tool_calling(session, query: str, chat=None, k: int = 10, judge_chat=None,
                     max_retries: int = 3, max_steps: int = 8) -> dict:
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
    import search_as_code as sac

    chat = chat or lc_chat()
    judge_chat = judge_chat or chat
    bound = chat.bind_tools(sac_surface.TOOLCALL_TOOLS)
    usage = Usage()
    trace, attempts, reasoning_acc = [], [], []
    msgs = [SystemMessage(content=sac_surface.TOOLCALL_SYSTEM), HumanMessage(content=f"Query: {query}")]
    t0 = time.time()
    ids, retries = [], 0

    for _ in range(max_steps * (max_retries + 1)):
        resp = bound.invoke(msgs)
        _account(usage, resp)
        msgs.append(resp)
        if _text(resp).strip():
            reasoning_acc.append(_text(resp).strip())
        calls = resp.tool_calls or []
        if not calls:
            break
        finished = False
        for call in calls:
            name, args, cid = call["name"], call["args"], call["id"]
            if name == "finish":
                ids = [str(x) for x in args.get("doc_ids", [])][:k]
                msgs.append(ToolMessage(content="ok", tool_call_id=cid))
                if max_retries == 0:      # fast mode: accept first finish, skip judge LLM call
                    accept, fb = True, ""
                else:
                    accept, fb, _ = judge(judge_chat, query, ids, session, usage)
                attempts.append({"hop": retries, "ids": ids, "judge": "PASS" if accept else "FAIL",
                                 "feedback": fb, "reasoning": " ".join(reasoning_acc)[:500],
                                 "steps": [t["op"] for t in trace]})
                trace.append({"op": f"finish · judge {'PASS' if accept else 'FAIL'}", "returned": len(ids)})
                reasoning_acc = []
                if accept or retries >= max_retries:
                    finished = True
                else:
                    retries += 1
                    msgs.append(HumanMessage(content=sac_surface.TOOLCALL_RETRY_TEMPLATE.format(feedback=fb)))
            elif name == "expand":
                variants = sac.expand(args["query"], session._require_generator(), n=N - 1)
                trace.append({"op": f"expand → {len(variants)} variants", "out": variants})
                msgs.append(ToolMessage(content=json.dumps(variants), tool_call_id=cid))
            elif name == "search":
                hits = _tool_search(session, args, k)
                trace.append({"op": f"search({args.get('mode', 'hybrid')})", "returned": len(hits)})
                msgs.append(ToolMessage(content=json.dumps(hits), tool_call_id=cid))
            else:
                msgs.append(ToolMessage(content="unknown tool", tool_call_id=cid))
        if finished:
            break

    return {"path": "tool_calling", "ids": ids, "latency_s": time.time() - t0, "usage": usage.as_dict(),
            "reasoning": attempts[-1]["reasoning"] if attempts else " ".join(reasoning_acc)[:500],
            "hops": len(attempts) or 1, "attempts": attempts, "trace": trace,
            "system_prompt": sac_surface.TOOLCALL_SYSTEM,
            "tools": [t["function"]["name"] for t in sac_surface.TOOLCALL_TOOLS]}
