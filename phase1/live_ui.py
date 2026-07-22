"""Live analytics UI — type a query, run all three modes on demand, see traces.

    streamlit run phase1/live_ui.py --server.port 8501 --server.address 0.0.0.0

Three tabs:
- Live query: run base / tool-calling / SAC now; see reasoning, generated code,
  tool steps, LLM-as-judge verdicts, retry hops, ids, latency, cost.
- Primitives lab: run the model-free SDK primitives directly (smart_search,
  prf_search, adaptive_search, semantic_dedup, mmr, diversity_quota) and see the
  retrieval-confidence / abstain signal — no LLM, no cost.
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
from search_as_code import primitives as P
from phase1 import agents, common, sac_surface
from phase1.benchmark import _merge_gen_usage
from phase1.llm import LLM

HISTORY = common.RUNS_DIR / "live_history.jsonl"

st.set_page_config(page_title="Search-as-Code — live traces", layout="wide")
st.title("🔎 Search-as-Code — live query, 3 modes, judge loop")
st.caption("base (hybrid, no LLM) · MCP tool-calling · SAC code-mode — over BEIR FiQA in OpenSearch. "
           "Both LLM paths use 4 query formulations and an LLM-as-judge retry loop (max 3 hops).")

with st.expander("🔧 View the exact code-prompt sent to OpenAI (the primitive surface)"):
    st.markdown("**SAC system prompt** — the code-API surface the LLM writes against "
                "(a stable, prompt-cached prefix):")
    st.code(sac_surface.SAC_SYSTEM, language="markdown")
    st.markdown("**Tool-calling system prompt** + tools exposed:")
    st.code(sac_surface.TOOLCALL_SYSTEM + "\n\ntools: "
            + ", ".join(t["function"]["name"] for t in sac_surface.TOOLCALL_TOOLS), language="markdown")


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
    # warm up the GPU models once so the first real query isn't slow
    try:
        embed(["warm up"])
        session.reranker("warm up", ["a", "b"])
    except Exception:
        pass
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
                st.caption("🖥️ sandbox stdout (samples/diagnostics the model saw next hop):")
                st.code(a["stdout"], language="text")
        if a.get("samples"):
            st.caption("retrieved samples fed to the judge / next hop:")
            st.code(a["samples"], language="text")
        if a.get("steps"):
            st.caption("tool steps: " + " → ".join(a["steps"]))
        if a.get("prompt"):
            st.caption("↳ exact prompt sent to OpenAI this hop:")
            st.code(a["prompt"], language="markdown")
        st.caption(f"ids: {', '.join(str(x) for x in a.get('ids', [])[:10])}  "
                   f"(recall@10 {_recall(a.get('ids', []), gold)})")
        st.divider()


def _save_history(rec):
    with open(HISTORY, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _run(fn, query, max_retries):
    before = copy.copy(gen.usage)
    r = fn(session, query, chat=chat, max_retries=max_retries)
    _merge_gen_usage(r, gen, before)
    return r


tab_live, tab_lab, tab_hist = st.tabs(["🔎 Live query", "🧪 Primitives lab", "🕘 History"])

# ============================================================ LIVE
with tab_live:
    c1, c2 = st.columns([3, 2])
    with c2:
        pick = st.selectbox("…or pick a labeled FiQA query", ["(type your own)"] + samples)
    with c1:
        default = "" if pick == "(type your own)" else pick
        query = st.text_input("Query", value=default,
                              placeholder="e.g. How to deposit a cheque issued to my business?")
    sc1, sc2 = st.columns([1, 3])
    with sc1:
        retries = st.slider("Judge retries", 0, 3, 1,
                            help="0 = fast single pass (no judge). Higher = deeper judge-driven refinement, slower.")
    go = st.button("▶ Run all 3 modes", type="primary")

    if go and query.strip():
        gold = set(qrels.get(text2qid.get(query, ""), {}).keys())
        if gold:
            st.caption(f"Labeled query — {len(gold)} gold docs; ✅ marks a hit.")
        with st.status("Running…", expanded=True) as status:
            status.update(label="base (hybrid, no LLM)…")
            base = agents.run_base(session, query)
            status.write(f"✅ base — {base['latency_s']:.2f}s")
            status.update(label="SAC (code-mode)…")
            sacr = _run(agents.run_sac, query, retries)
            status.write(f"✅ SAC — {sacr['latency_s']:.1f}s · {sacr['hops']} hop(s) · ${sacr['usage']['cost_usd']:.4f}")
            status.update(label="tool-calling (MCP)…")
            tool = _run(agents.run_tool_calling, query, retries)
            status.write(f"✅ tool-calling — {tool['latency_s']:.1f}s · {tool['hops']} hop(s) · ${tool['usage']['cost_usd']:.4f}")
            status.update(label="done", state="complete", expanded=False)

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

# ============================================================ PRIMITIVES LAB
with tab_lab:
    st.markdown("Run the **model-free** SDK primitives directly on the live FiQA index — "
                "no LLM, no cost. Pick a retrieval strategy, chain post-processors, and see "
                "the effect plus the retrieval-**confidence** signal.")

    lc1, lc2 = st.columns([3, 2])
    with lc2:
        lab_pick = st.selectbox("…or pick a labeled FiQA query", ["(type your own)"] + samples,
                                key="lab_pick")
    with lc1:
        lab_default = "" if lab_pick == "(type your own)" else lab_pick
        lab_query = st.text_input("Query", value=lab_default, key="lab_query",
                                  placeholder="e.g. Can I deposit an IRA CD into a Roth 401k?")

    strat = st.radio(
        "Retrieval strategy",
        ["dense", "hybrid", "smart_search", "prf_search", "adaptive_search"],
        horizontal=True,
        help="smart_search = normalize + boost rare exact tokens · prf_search = Rocchio pseudo-"
             "relevance feedback · adaptive_search = size the result set from the score curve.",
    )
    pc = st.columns(4)
    pool_k = pc[0].slider("pool / top_k", 10, 100, 30, key="lab_pool")
    alpha_h = pc[1].slider("hybrid alpha (dense weight)", 0.0, 1.0, 0.8, 0.1,
                           disabled=strat != "hybrid")
    if strat == "prf_search":
        fb_k = pc[2].slider("prf feedback_k", 3, 20, 5)
        prf_beta = pc[3].slider("prf beta (centroid pull)", 0.0, 2.0, 0.7, 0.1)
    elif strat == "adaptive_search":
        adaptive_method = pc[2].selectbox("cutoff method", ["band", "knee"])
        rel_band = pc[3].slider("band width", 0.02, 0.5, 0.1, 0.02)

    st.markdown("**Post-processors** (chained in order):")
    xc = st.columns(3)
    do_dedup = xc[0].checkbox("semantic_dedup", help="collapse near-duplicate hits (SemDeDup)")
    dedup_th = xc[0].slider("dedup threshold", 0.70, 0.99, 0.85, 0.01, disabled=not do_dedup)
    do_mmr = xc[1].checkbox("mmr diversify", help="relevance vs redundancy (Carbonell & Goldstein)")
    mmr_lambda = xc[1].slider("mmr λ (relevance↔diversity)", 0.0, 1.0, 0.5, 0.1, disabled=not do_mmr)
    do_div = xc[2].checkbox("diversity_quota", help="cap hits per metadata group (Vespa)")
    div_field = xc[2].text_input("group-by metadata field", value="",
                                 disabled=not do_div,
                                 help="e.g. 'source' or 'author'. FiQA has no such field, so this "
                                      "is best shown on a backend with source/topic metadata.")
    div_max = xc[2].slider("max per group", 1, 5, 1, disabled=not do_div)

    gc = st.columns(2)
    min_top = gc[0].slider("abstain if top score <", 0.0, 1.0, 0.0, 0.05,
                           help="R³AG confidence gating: flag results too weak to trust.")
    min_gap = gc[1].slider("abstain if top↔#2 gap <", 0.0, 0.5, 0.0, 0.02)

    if st.button("▶ Run primitives", type="primary", key="lab_go") and lab_query.strip():
        q = lab_query.strip()
        gold = set(qrels.get(text2qid.get(q, ""), {}).keys())
        steps: list[str] = []
        with st.spinner("Running primitives (embedder on GPU)…"):
            t0 = time.time()
            if strat == "dense":
                res = session.search(q, top_k=pool_k, mode="dense"); steps.append(f'search(q, top_k={pool_k}, mode="dense")')
            elif strat == "hybrid":
                res = session.search(q, top_k=pool_k, mode="hybrid", alpha=alpha_h); steps.append(f'search(q, top_k={pool_k}, mode="hybrid", alpha={alpha_h})')
            elif strat == "smart_search":
                res = session.smart_search(q, top_k=pool_k); steps.append(f"smart_search(q, top_k={pool_k})")
            elif strat == "prf_search":
                res = session.prf_search(q, top_k=pool_k, feedback_k=fb_k, beta=prf_beta); steps.append(f"prf_search(q, top_k={pool_k}, feedback_k={fb_k}, beta={prf_beta})")
            else:  # adaptive_search
                res = session.adaptive_search(q, method=adaptive_method, rel_band=rel_band, max_k=pool_k); steps.append(f'adaptive_search(q, method="{adaptive_method}", rel_band={rel_band}, max_k={pool_k})')
            res = session.hydrate(res)
            if do_dedup:
                res = session.semantic_dedup(res, threshold=dedup_th); steps.append(f"semantic_dedup(res, threshold={dedup_th})")
            if do_mmr:
                res = session.mmr(q, res, lambda_=mmr_lambda, top_k=pool_k); steps.append(f"mmr(q, res, lambda_={mmr_lambda}, top_k={pool_k})")
            if do_div and div_field.strip():
                f = div_field.strip()
                res = P.diversity_quota(res, key=lambda h: h.get(f), max_per_group=div_max); steps.append(f'diversity_quota(res, key=lambda h: h.get("{f}"), max_per_group={div_max})')
            dt = time.time() - t0
            conf = P.confidence(res)
            weak = P.abstain(res, min_top=min_top, min_gap=min_gap)

        st.caption("Equivalent SAC code (what the agent would write):")
        st.code("\n".join(f"res = sac.{s}" for s in steps), language="python")

        mcols = st.columns(4)
        mcols[0].metric("results", conf["n"])
        mcols[1].metric("top score", f'{conf["top"]:.3f}')
        mcols[2].metric("top↔#2 gap", f'{conf["gap"]:.3f}')
        rc = _recall(res.ids(), gold)
        mcols[3].metric("Recall@10", rc if rc is not None else "—")
        st.caption(f"⏱️ {dt*1000:.0f} ms · model-free · $0.00")

        if weak:
            st.warning("⚠️ **abstain** — results are below the confidence threshold "
                       "(weak top score or small score gap). The agent should reformulate "
                       "or report 'insufficient evidence' rather than answer from noise.")
        else:
            st.success("✅ confidence gate passed.")

        if gold:
            st.caption(f"Labeled query — {len(gold)} gold docs; ✅ marks a hit.")
        st.subheader(f"Top results ({strat})")
        _snippets(res.ids(), gold)


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
