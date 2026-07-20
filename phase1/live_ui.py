"""Live analytics UI — type a query, run all three modes on demand, see traces.

    streamlit run phase1/live_ui.py --server.port 8501 --server.address 0.0.0.0

Two tabs:
- Live query: run base / tool-calling / SAC now; see reasoning, generated code,
  tool steps, LLM-as-judge verdicts, retry hops, ids, latency, cost.
- History: click a past search and inspect exactly what happened at each hop.

Requires OpenSearch on :9200 with FiQA ingested, and OPENAI_API_KEY (from
~/taxonomy/.env). Models load once (cached) on first query.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

import search_as_code as sac
from phase1 import agents, common
from phase1.benchmark import _merge_gen_usage
from phase1.llm import LLM

HISTORY = common.RUNS_DIR / "live_history.jsonl"

st.set_page_config(page_title="Search-as-Code — live traces", layout="wide")
st.title("🔎 Search-as-Code — live query, 3 modes, judge loop")
st.caption("base (hybrid, no LLM) · MCP tool-calling · SAC code-mode — over BEIR FiQA in OpenSearch. "
           "Both LLM paths use 4 query formulations and an LLM-as-judge retry loop (max 3 hops).")


@st.cache_resource(show_spinner="Loading embedder + reranker + LLM …")
def backend():
    common.load_env()
    from sentence_transformers import SentenceTransformer
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).tolist()
    gen = LLM()
    session = sac.Session("opensearch", index=common.INDEX, dim=common.DIM, hosts=[common.OS_HOST],
                          embedder=embed, reranker=sac.CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2"),
                          generator=gen.as_generator())
    chat = agents.lc_chat()
    q = json.loads((common.DATA_DIR / "queries.json").read_text()) if (common.DATA_DIR / "queries.json").exists() else {}
    qr = json.loads((common.DATA_DIR / "qrels.json").read_text()) if (common.DATA_DIR / "qrels.json").exists() else {}
    text2qid = {v: k for k, v in q.items()}
    samples = [q[k] for k in list(qr)[:15] if k in q]
    return session, chat, gen, qr, text2qid, samples


session, chat, gen, qrels, text2qid, samples = backend()


def _recall(ids, gold):
    return round(len(set(ids[:10]) & gold) / len(gold), 3) if gold else None


def _snippets(ids, gold):
    docs = {d.id: d for d in session.store.get(list(ids)[:10])}
    for did in list(ids)[:10]:
        d = docs.get(did)
        txt = (d.text[:150] + "…") if d and d.text else ""
        st.markdown(("✅ " if did in gold else "▫️ ") + f"`{did}` {txt}")


def _judge_badge(verdict):
    return "🟢 PASS" if verdict == "PASS" else "🔴 FAIL"


def _render_attempts(r, gold):
    """Render each judge hop for a path (used in both tabs)."""
    for a in r.get("attempts", []):
        st.markdown(f"**Hop {a['hop']} — judge {_judge_badge(a['judge'])}**"
                    + (f" · _{a['feedback']}_" if a.get("feedback") else ""))
        if a.get("reasoning"):
            st.caption("reasoning: " + a["reasoning"])
        if a.get("code") is not None:
            st.code(a["code"], language="python")
            if a.get("stdout"):
                st.caption("sandbox stdout: " + a["stdout"])
        if a.get("steps"):
            st.caption("tool steps: " + " → ".join(a["steps"]))
        st.caption(f"ids: {', '.join(str(x) for x in a.get('ids', [])[:10])}  "
                   f"(recall@10 {_recall(a.get('ids', []), gold)})")
        st.divider()


def _save_history(rec):
    with open(HISTORY, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _run(fn, query):
    before = copy.copy(gen.usage)
    r = fn(session, query, chat=chat)
    _merge_gen_usage(r, gen, before)
    return r


tab_live, tab_hist = st.tabs(["🔎 Live query", "🕘 History"])

# ============================================================ LIVE
with tab_live:
    c1, c2 = st.columns([3, 2])
    with c2:
        pick = st.selectbox("…or pick a labeled FiQA query", ["(type your own)"] + samples)
    with c1:
        default = "" if pick == "(type your own)" else pick
        query = st.text_input("Query", value=default,
                              placeholder="e.g. How to deposit a cheque issued to my business?")
    go = st.button("▶ Run all 3 modes", type="primary")

    if go and query.strip():
        gold = set(qrels.get(text2qid.get(query, ""), {}).keys())
        if gold:
            st.caption(f"Labeled query — {len(gold)} gold docs; ✅ marks a hit.")
        with st.spinner("running base → SAC → tool-calling (with judge loop)…"):
            base = agents.run_base(session, query)
            sacr = _run(agents.run_sac, query)
            tool = _run(agents.run_tool_calling, query)

        cols = st.columns(3)
        for col, r, name in ((cols[0], base, "base"), (cols[1], sacr, "SAC"), (cols[2], tool, "tool-calling")):
            with col:
                rc = _recall(r["ids"], gold)
                st.metric(f"{name} · Recall@10", rc if rc is not None else "—")
                st.caption(f"{r['latency_s']:.2f}s · ${r['usage']['cost_usd']:.4f} · "
                           f"{r['usage']['calls']} LLM call(s) · {r.get('hops',1)} hop(s)")

        cb, cs, ct = st.columns(3)
        with cb:
            st.subheader("base (hybrid)")
            _snippets(base["ids"], gold)
        with cs:
            st.subheader("SAC (code-mode)")
            if sacr.get("reasoning"):
                st.info("💡 " + sacr["reasoning"])
            with st.expander(f"{sacr['hops']} hop(s) — reasoning, code, judge", expanded=True):
                _render_attempts(sacr, gold)
            _snippets(sacr["ids"], gold)
        with ct:
            st.subheader("tool-calling (MCP)")
            if tool.get("reasoning"):
                st.info("💡 " + tool["reasoning"])
            with st.expander(f"{tool['hops']} hop(s) — reasoning, tool steps, judge", expanded=True):
                _render_attempts(tool, gold)
            _snippets(tool["ids"], gold)

        # persist to history
        _save_history({
            "ts": datetime.now().isoformat(timespec="seconds"), "query": query, "gold": list(gold),
            "base": {"ids": base["ids"], "latency_s": base["latency_s"], "recall": _recall(base["ids"], gold)},
            "sac": {**{kk: sacr[kk] for kk in ("ids", "latency_s", "usage", "hops", "attempts", "reasoning")},
                    "recall": _recall(sacr["ids"], gold)},
            "tool_calling": {**{kk: tool[kk] for kk in ("ids", "latency_s", "usage", "hops", "attempts", "reasoning")},
                             "recall": _recall(tool["ids"], gold)},
        })
        st.success("Saved to History tab.")

# ============================================================ HISTORY
with tab_hist:
    if not HISTORY.exists():
        st.info("No history yet — run a query in the Live tab.")
    else:
        recs = [json.loads(l) for l in HISTORY.read_text().splitlines() if l.strip()][::-1]
        st.caption(f"{len(recs)} past searches")
        idx = st.selectbox("Past search", range(len(recs)),
                           format_func=lambda i: f"{recs[i]['ts']} — {recs[i]['query'][:70]}")
        rec = recs[idx]
        gold = set(rec.get("gold", []))
        st.markdown(f"**Query:** {rec['query']}")

        m = st.columns(3)
        m[0].metric("base Recall@10", rec["base"].get("recall") or "—")
        m[1].metric("SAC Recall@10", rec["sac"].get("recall") or "—",
                    help=f"{rec['sac']['hops']} hop(s) · ${rec['sac']['usage']['cost_usd']:.4f}")
        m[2].metric("tool-calling Recall@10", rec["tool_calling"].get("recall") or "—",
                    help=f"{rec['tool_calling']['hops']} hop(s) · ${rec['tool_calling']['usage']['cost_usd']:.4f}")

        cs, ct = st.columns(2)
        with cs:
            st.subheader("SAC — what happened at each hop")
            if rec["sac"].get("reasoning"):
                st.info("💡 " + rec["sac"]["reasoning"])
            _render_attempts(rec["sac"], gold)
        with ct:
            st.subheader("tool-calling — what happened at each hop")
            if rec["tool_calling"].get("reasoning"):
                st.info("💡 " + rec["tool_calling"]["reasoning"])
            _render_attempts(rec["tool_calling"], gold)

st.divider()
st.caption("Static 100-query benchmark view: `streamlit run phase1/ui.py`")
