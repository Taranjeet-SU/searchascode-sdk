"""Trace UI for the Phase 1 benchmark.

    streamlit run phase1/ui.py

Shows the base-vs-tool-calling-vs-SAC comparison summary, then a per-query
browser: the SAC-generated code, sandbox stdout, tool-calling steps, returned
ids, per-path latency, token cost, and recall@10.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

RUNS = Path(__file__).resolve().parent / "runs"

st.set_page_config(page_title="Search-as-Code — Phase 1 traces", layout="wide")
st.title("🔎 Search-as-Code — base vs tool-calling vs SAC")


def _hits(p, t):
    gold = set(t["gold"])
    for did in p.get("ids", [])[:10]:
        st.text(("✅ " if did in gold else "▫️ ") + str(did))


@st.cache_data
def load():
    summary = json.loads((RUNS / "bench_summary.json").read_text()) if (RUNS / "bench_summary.json").exists() else {}
    traces = []
    tp = RUNS / "bench_traces.jsonl"
    if tp.exists():
        traces = [json.loads(l) for l in tp.read_text().splitlines() if l.strip()]
    return summary, traces


summary, traces = load()
if not traces:
    st.warning("No benchmark run found. Run:  `python -m phase1.benchmark -n 100`")
    st.stop()

# ---- summary ----
st.header("Comparison summary")
df = pd.DataFrame(summary).T
pretty = {"recall@10": "Recall@10", "ndcg@10": "nDCG@10", "mrr@10": "MRR@10",
          "avg_latency_s": "Latency (s)", "avg_calls": "LLM calls/q",
          "total_input_tokens": "Input tokens", "cached_input_tokens": "Cached tokens",
          "cache_hit_rate": "Cache hit", "total_cost_usd": "Cost (USD)"}
cols = [c for c in pretty if c in df.columns]
st.dataframe(df[cols].rename(columns=pretty), use_container_width=True)

c1, c2, c3 = st.columns(3)
if "sac" in summary and "base" in summary:
    c1.metric("SAC Recall@10", summary["sac"]["recall@10"],
              round(summary["sac"]["recall@10"] - summary["base"]["recall@10"], 4))
if "sac" in summary and "tool_calling" in summary:
    tc, sc = summary["tool_calling"]["total_cost_usd"], summary["sac"]["total_cost_usd"]
    c2.metric("SAC cost vs tool-calling", f"${sc}", f"{(sc-tc):.4f} vs ${tc}", delta_color="inverse")
    ti, si = summary["tool_calling"]["total_input_tokens"], summary["sac"]["total_input_tokens"]
    c3.metric("SAC input tokens vs tool-calling", f"{si:,}", f"{si-ti:,}", delta_color="inverse")

# ---- per-query browser ----
st.header("Per-query traces")
labels = [f"{i+1}. {t['query'][:70]}" for i, t in enumerate(traces)]
idx = st.selectbox("Query", range(len(traces)), format_func=lambda i: labels[i])
t = traces[idx]
st.markdown(f"**Query:** {t['query']}")
st.caption(f"gold docs: {', '.join(t['gold'][:15])}{' …' if len(t['gold'])>15 else ''}")

col_base, col_sac, col_tool = st.columns(3)

with col_base:
    p = t["paths"]["base"]
    st.subheader("Base")
    st.metric("Recall@10", p.get("recall@10", 0.0))
    st.caption(f"{p['latency_s']*1000:.0f} ms · $0")
    _hits(p, t)

with col_sac:
    p = t["paths"]["sac"]
    st.subheader("SAC (code-mode)")
    st.metric("Recall@10", p.get("recall@10", 0.0))
    st.caption(f"{p['latency_s']:.2f} s · ${p['usage']['cost_usd']} · {p['usage']['calls']} call(s)")
    st.code(p.get("code", ""), language="python")
    if p.get("trace"):
        for step in p["trace"]:
            st.text("• " + step.get("op", "") + (f"  stdout: {step['stdout']}" if step.get("stdout") else ""))
    _hits(p, t)

with col_tool:
    p = t["paths"]["tool_calling"]
    st.subheader("Tool-calling (MCP)")
    st.metric("Recall@10", p.get("recall@10", 0.0))
    st.caption(f"{p['latency_s']:.2f} s · ${p['usage']['cost_usd']} · {p['usage']['calls']} call(s)")
    for step in p.get("trace", []):
        st.text("• " + step.get("op", "") + (f"  → {step.get('returned', step.get('out',''))}"))
    _hits(p, t)
