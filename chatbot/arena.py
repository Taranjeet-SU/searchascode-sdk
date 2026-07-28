"""LLM-arena UI: one query → SAC (code-mode) vs tool-calling, side by side.

    streamlit run chatbot/arena.py --server.address 0.0.0.0 --server.port 8502

Shows for each agent: the answer, hops, latency, tokens, cost, cited sources, and
the agent's *work* — SAC's generated Python (per hop) vs the tool-calling steps.
Both retrieve from the same cached FiQA index in OpenSearch via hybrid search.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Load the embedder/reranker from local HF cache (no network — the models are already
# cached; the online metadata check crashes under Streamlit's httpx lifecycle).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(page_title="SAC vs Tool-calling — arena", layout="wide")
st.title("⚔️ SAC (code-mode) vs Tool-calling — RAG arena")
st.caption("One query → both agents answer over the same FiQA / OpenSearch hybrid index. "
           "See the answer, hops, latency, tokens, cost, and each agent's actual work.")


@st.cache_resource(show_spinner="Loading embedder + reranker + LLMs (once) …")
def _bots():
    from chatbot.sac_bot import SacChatbot
    from chatbot.toolcalling import ToolCallingChatbot
    return SacChatbot(), ToolCallingChatbot()


sac_bot, tool_bot = _bots()

samples = [
    "How do I deposit a cheque issued to my business into my business account?",
    "Can I send a money order from USPS as a business?",
    "What are the tax implications of selling stock I received as an RSU?",
]
c1, c2 = st.columns([3, 2])
with c2:
    pick = st.selectbox("…or pick a sample", ["(type your own)"] + samples)
with c1:
    query = st.text_input("Query", value="" if pick == "(type your own)" else pick,
                          placeholder="Ask a finance question…")
bc1, bc2 = st.columns([1, 3])
deep = bc1.checkbox("deep SAC (ensemble+consensus)", value=True)
go = bc2.button("▶ Run both", type="primary")


def _metrics(col, a, name):
    u = a.usage
    col.subheader(name)
    m = col.columns(4)
    m[0].metric("hops", a.hops)
    m[1].metric("latency", f"{a.latency_s:.1f}s")
    m[2].metric("tokens", f"{u.get('prompt_tokens',0)+u.get('completion_tokens',0):,}")
    m[3].metric("cost", f"${u.get('cost_usd',0):.5f}")
    if not a.arrived:
        col.warning("did not reach a final answer within the hop budget (forced answer)")
    col.markdown("**Answer**")
    col.write(a.answer or "_(empty)_")
    col.caption("cited sources: " + ", ".join(a.ids[:8]))


def _work(col, a):
    tr = a.trace or {}
    if tr.get("kind") == "sac":
        col.markdown("**🧑‍💻 Generated code** (final hop)")
        col.code(tr.get("code", "") or "(none)", language="python")
        atts = tr.get("attempts", [])
        if len(atts) > 1:
            with col.expander(f"all {len(atts)} hop(s) — code + judge verdict"):
                for at in atts:
                    st.caption(f"hop {at['hop']} · judge {at.get('judge')} · ids {at.get('ids')}")
                    st.code(at.get("code", ""), language="python")
    elif tr.get("kind") == "toolcalling":
        col.markdown("**🔧 Tool calls**")
        steps = tr.get("steps", [])
        if not steps:
            col.write("_(none)_")
        for s in steps:
            col.markdown(f"- hop {s['hop']}: `{s['tool']}` &nbsp; `{s.get('args', {})}`")


if go and query.strip():
    with st.status("Running both agents…", expanded=False):
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_sac = ex.submit(sac_bot.answer, query, deep)
            f_tool = ex.submit(tool_bot.answer, query)
            a_sac, a_tool = f_sac.result(), f_tool.result()

    left, right = st.columns(2)
    _metrics(left, a_sac, "🟦 SAC (code-mode)")
    _metrics(right, a_tool, "🟧 Tool-calling")
    st.divider()
    lw, rw = st.columns(2)
    _work(lw, a_sac)
    _work(rw, a_tool)

    # quick head-to-head deltas
    st.divider()
    st.markdown(
        f"**Head-to-head** — latency: SAC {a_sac.latency_s:.1f}s vs tool {a_tool.latency_s:.1f}s · "
        f"tokens: {sum(a_sac.usage.get(k,0) for k in ('prompt_tokens','completion_tokens')):,} vs "
        f"{sum(a_tool.usage.get(k,0) for k in ('prompt_tokens','completion_tokens')):,} · "
        f"cost: ${a_sac.usage.get('cost_usd',0):.5f} vs ${a_tool.usage.get('cost_usd',0):.5f} · "
        f"hops: {a_sac.hops} vs {a_tool.hops}"
    )
