"""Live analytics UI — type a query, run all three modes on demand, see traces.

    streamlit run phase1/live_ui.py --server.port 8501 --server.address 0.0.0.0

Requires: OpenSearch on :9200 with FiQA ingested, and OPENAI_API_KEY (loaded from
~/taxonomy/.env). Models load once (cached) on first query.
"""

from __future__ import annotations

import copy
import json
import os
import sys

# Streamlit puts phase1/ (the script dir) on sys.path, not the repo root — add it
# so `from phase1 import ...` resolves when launched via `streamlit run`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

import search_as_code as sac
from phase1 import agents, common
from phase1.benchmark import _merge_gen_usage
from phase1.llm import LLM

st.set_page_config(page_title="Search-as-Code — live traces", layout="wide")
st.title("🔎 Search-as-Code — live query, 3 modes side by side")
st.caption("base (hybrid, no LLM) · MCP tool-calling · SAC code-mode — over BEIR FiQA in OpenSearch")


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

col_in, col_pick = st.columns([3, 2])
with col_pick:
    pick = st.selectbox("…or pick a labeled FiQA query", ["(type your own)"] + samples)
with col_in:
    default = "" if pick == "(type your own)" else pick
    query = st.text_input("Query", value=default, placeholder="e.g. How to deposit a cheque issued to my business?")

run = st.button("▶ Run all 3 modes", type="primary")


def _snippets(ids, gold):
    docs = {d.id: d for d in session.store.get(ids[:10])}
    for did in ids[:10]:
        d = docs.get(did)
        txt = (d.text[:160] + "…") if d and d.text else ""
        st.markdown(("✅ " if did in gold else "▫️ ") + f"`{did}` {txt}")


def _run(fn, **kw):
    before = copy.copy(gen.usage)
    r = fn(session, query, chat=chat, **kw)
    _merge_gen_usage(r, gen, before)
    return r


if run and query.strip():
    gold = set(qrels.get(text2qid.get(query, ""), {}).keys())
    if gold:
        st.caption(f"Labeled query — {len(gold)} gold docs; ✅ marks a hit. Recall@10 shown per mode.")

    with st.spinner("running all three…"):
        base = agents.run_base(session, query)
        sacr = _run(agents.run_sac)
        tool = _run(agents.run_tool_calling)

    def recall(r):
        return round(len(set(r["ids"][:10]) & gold) / len(gold), 3) if gold else None

    m1, m2, m3 = st.columns(3)
    for c, r, name in ((m1, base, "base"), (m2, sacr, "SAC"), (m3, tool, "tool-calling")):
        with c:
            st.metric(f"{name} · latency", f"{r['latency_s']:.2f}s")
            rc = recall(r)
            st.caption(f"{'Recall@10 ' + str(rc) + ' · ' if rc is not None else ''}"
                       f"${r['usage']['cost_usd']:.4f} · {r['usage']['calls']} LLM call(s) · "
                       f"{r['usage']['input_tokens']:,} in-tok")

    cb, cs, ct = st.columns(3)
    with cb:
        st.subheader("base (hybrid)")
        _snippets(base["ids"], gold)
    with cs:
        st.subheader("SAC (code-mode)")
        st.code(sacr.get("code", ""), language="python")
        if sacr.get("trace"):
            with st.expander("sandbox trace"):
                for s in sacr["trace"]:
                    st.text("• " + s.get("op", "") + (f"  {s['stdout']}" if s.get("stdout") else ""))
        _snippets(sacr["ids"], gold)
    with ct:
        st.subheader("tool-calling (MCP)")
        with st.expander("tool-call steps", expanded=True):
            for s in tool.get("trace", []):
                st.text("• " + s.get("op", "") + (f"  → {s.get('returned', s.get('out',''))}"))
        _snippets(tool["ids"], gold)

st.divider()
st.caption("Static 100-query benchmark view: `streamlit run phase1/ui.py`")
