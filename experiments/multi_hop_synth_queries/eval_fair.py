"""FAIR harness comparison: dense vs tool-calling vs SAC code-mode — IDENTICAL toolset, matched
search budget. Only the harness differs (one tool per turn, with results in context) vs (one
program that CHAINS the same tools, results out of context).

Shared tools over the hotpotqa Session (each retrieval counts against a per-query budget):
  search(query, mode)  hyde(query)  prf(query)  decompose(query)  rephrase(query)
  fuse(id_lists)  rerank(query, ids=None)  mmr(query, ids=None)
Retrieval primitives pool a WIDE candidate set (POOL) in the sandbox — only the final top-K
leaves as model context; rerank/mmr/fuse reorder that pool (free, they don't cost budget). A
real cross-encoder reranker is attached, so the code-mode arm can retrieve-wide-then-rerank.
Metric: recall@10 = |gold ∩ top10|/N and all_golds@10, over the multi-hop datasets.

    python -m experiments.multi_hop_synth_queries.eval_fair [per_dataset=120] [workers=5] [budget=6]
"""
from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import agents, common
from phase1.llm import LLM

DATA = Path(__file__).parent / "data"
K = 10
POOL = 50   # wide candidate pool kept in the sandbox; rerank/mmr operate on it, only top-K is returned


def _rrf(lists, k=60):
    s = {}
    for lst in lists:
        for r, i in enumerate(lst):
            s[i] = s.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(s, key=lambda i: -s[i])


class Tools:
    """The shared toolset both harnesses use; enforces a per-query retrieval budget.

    Retrieval primitives (search/hyde/prf) each cost 1 of the budget and pool ~POOL candidates
    (with text) in the sandbox — only a small preview leaves as model context. rerank/mmr/fuse
    reorder the accumulated pool and are FREE (they issue no new retrieval). A real cross-encoder
    reranker is attached to the Session, so `rerank` is a genuine precision step, not a no-op."""
    def __init__(self, session, gen, budget):
        self.s, self.gen, self.budget = session, gen, budget
        self.searches = 0
        self.pool = []          # every retrieval's ranked ids (for RRF fallback)
        self.docs = {}          # id -> title
        self.hits = {}          # id -> Hit (with text, so rerank/mmr can score)

    def _absorb(self, rs):
        ids = []
        for h in rs:
            ids.append(h.id)
            self.docs[h.id] = ((h.document.metadata or {}).get("title", "") if h.document else "")
            self.hits[h.id] = h
        self.pool.append(ids)
        return ids

    def _retrieve(self, fn):
        if self.searches >= self.budget:
            return []
        self.searches += 1
        try:
            rs = fn()
            try:
                rs = self.s.hydrate(rs)      # ensure candidate text is present for reranking
            except Exception:
                pass
        except Exception:
            return []
        ids = self._absorb(rs)
        return [{"id": i, "title": self.docs[i]} for i in ids[:8]]

    def search(self, query, mode="dense"):
        m = mode if mode in ("dense", "keyword", "hybrid") else "dense"
        return self._retrieve(lambda: self.s.search(query, top_k=POOL, mode=m))

    def hyde(self, query):
        return self._retrieve(lambda: self.s.hyde_search(query, top_k=POOL))

    def prf(self, query):
        return self._retrieve(lambda: self.s.prf_search(query, top_k=POOL))

    def decompose(self, query):
        try:
            return self._subqs(query)
        except Exception:
            return [query]

    def _subqs(self, query):
        r = self.gen.complete(f"Break this question into the distinct factual sub-questions needed "
                              f"to answer it — each targets a DIFFERENT entity/document. One per line, 2-6.\n\nQ: {query}",
                              system="You decompose multi-hop questions.")
        return [x.strip("-•* ").strip() for x in r.splitlines() if x.strip()][:6] or [query]

    def rephrase(self, query):
        r = self.gen.complete(f"Rewrite this query in 3 different ways, same meaning. One per line.\n\nQ: {query}",
                              system="You paraphrase search queries.")
        return [x.strip("-•* ").strip() for x in r.splitlines() if x.strip()][:3] or [query]

    def _resultset(self, ids):
        from search_as_code.types import ResultSet
        return ResultSet([self.hits[i] for i in ids if i in self.hits])

    def _poolset(self):
        seen, order = set(), []
        for lst in self.pool:
            for i in lst:
                if i not in seen:
                    seen.add(i); order.append(i)
        return self._resultset(order)

    def rerank(self, query, ids=None, top_k=None):
        """Cross-encoder rerank of `ids` (or the WHOLE accumulated pool if ids is None) -> ranked ids."""
        rs = self._resultset(ids) if ids else self._poolset()
        if not rs:
            return list(ids or [])
        try:
            return self.s.rerank(query, rs, top_k=top_k or K).ids()
        except Exception:
            return [h.id for h in rs][:(top_k or K)]

    def mmr(self, query, ids=None, top_k=None):
        """MMR-diversify `ids` (or the whole pool) -> ranked ids."""
        rs = self._resultset(ids) if ids else self._poolset()
        if not rs:
            return list(ids or [])
        try:
            return self.s.mmr(query, rs, top_k=top_k or K).ids()
        except Exception:
            return [h.id for h in rs][:(top_k or K)]

    def final(self, ids=None):
        cand = [i for i in (ids or []) if i in self.docs]
        return (cand or _rrf(self.pool))[:K]


# --------------------------------------------------------------------------- #
# harness 1: tool-calling (one tool per turn; results returned to the model)    #
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "search",
        "description": "Search the corpus. Returns ranked {id,title}. Costs 1 of your search budget.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"},
            "mode": {"type": "string", "enum": ["dense", "keyword", "hybrid"]}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "hyde",
        "description": "Retrieve on a hypothetical-answer embedding (bridges vocabulary gap). Costs 1 budget.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "prf",
        "description": "Retrieve with pseudo-relevance-feedback expansion. Costs 1 budget.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "decompose",
        "description": "Split the question into sub-questions (each targets a DIFFERENT document). Returns a list.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "rephrase",
        "description": "Get 3 paraphrases of a query. Returns a list.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "rerank",
        "description": "Cross-encoder rerank of everything retrieved so far for `query`. Returns the top {id,title}. Free.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "mmr",
        "description": "MMR-diversify everything retrieved so far for `query` (kills near-duplicates). Returns top {id,title}. Free.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "finish",
        "description": "Submit the final ranked list of up to 10 document ids that together answer the question.",
        "parameters": {"type": "object", "properties": {"doc_ids": {"type": "array", "items": {"type": "string"}}},
                       "required": ["doc_ids"]}}},
]
TOOL_SYS = ("You retrieve the SET of documents needed to answer a multi-hop question — it needs "
            "SEVERAL different documents. Use `decompose` to split it, `search`/`hyde`/`prf` each "
            "sub-question (limited retrieval budget), `rephrase` if a search misses, `rerank` to "
            "pull the best candidates forward, then `finish` with the ~10 ids that TOGETHER cover "
            "all parts. Call one or few tools per turn.")


def tool_harness(chat, tools: Tools, q, max_steps=10):
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    bound = chat.bind_tools(TOOL_SCHEMAS)
    msgs = [SystemMessage(content=TOOL_SYS), HumanMessage(content=f"Question: {q}")]
    picked, steps, lc_in, lc_out = None, 0, 0, 0
    for _ in range(max_steps):
        resp = bound.invoke(msgs); msgs.append(resp)
        um = getattr(resp, "usage_metadata", None) or {}
        lc_in += um.get("input_tokens", 0); lc_out += um.get("output_tokens", 0)
        calls = resp.tool_calls or []
        if not calls:
            break
        steps += 1                       # one model turn that issued tool call(s)
        stop = False
        for c in calls:
            name, args, cid = c["name"], c["args"], c["id"]
            if name == "search":
                obs = tools.search(args.get("query", q), args.get("mode", "dense"))
            elif name == "hyde":
                obs = tools.hyde(args.get("query", q))
            elif name == "prf":
                obs = tools.prf(args.get("query", q))
            elif name == "decompose":
                obs = tools.decompose(args.get("query", q))
            elif name == "rephrase":
                obs = tools.rephrase(args.get("query", q))
            elif name in ("rerank", "mmr"):
                fn = tools.rerank if name == "rerank" else tools.mmr
                obs = [{"id": i, "title": tools.docs.get(i, "")} for i in fn(args.get("query", q))[:8]]
            elif name == "finish":
                picked = [str(x) for x in args.get("doc_ids", [])]; obs = "ok"; stop = True
            else:
                obs = "unknown"
            msgs.append(ToolMessage(content=json.dumps(obs)[:1500], tool_call_id=cid))
        if stop:
            break
    return tools.final(picked), {"steps": steps, "lc_in": lc_in, "lc_out": lc_out}


# --------------------------------------------------------------------------- #
# harness 2: SAC code-mode (ONE program that chains the same tools)             #
# --------------------------------------------------------------------------- #
CODE_SYS = ("You are a search-as-code agent. Write ONE Python program (no prose) to retrieve the "
            "documents needed for a MULTI-HOP question that needs SEVERAL different documents. "
            "Available functions (search/hyde/prf each cost 1 of a limited retrieval budget; "
            "rerank/mmr/fuse are FREE reorders of what you've already retrieved):\n"
            "  search(query, mode='dense'|'keyword'|'hybrid') -> [{'id','title'}]  # pools ~50, shows top 8\n"
            "  hyde(query) -> like search but on a hypothetical-answer embedding (vocabulary gap)\n"
            "  prf(query) -> like search but pseudo-relevance-feedback expanded\n"
            "  decompose(query) -> [subquestions]      rephrase(query) -> [paraphrases]\n"
            "  fuse(list_of_id_lists) -> merged id list (RRF)\n"
            "  rerank(query, ids) -> ids re-scored by a cross-encoder for a SINGLE query\n"
            "  mmr(query, ids=None) -> ids diversified (kills near-duplicates)\n"
            "COVER all the parts: decompose the question, search each sub-question (+ the full "
            "question), and FUSE the pools. Fusing preserves coverage of the SEVERAL golds a "
            "multi-hop question needs. NOTE: `rerank(question, ...)` scores whole-question relevance, "
            "so it can DROP docs that satisfy only ONE sub-fact — use it to sharpen a SINGLE "
            "sub-pool, not over the fused union. Set `results` to up to 10 ids covering all parts:\n"
            "```\nsubs = decompose(question)\npools = [[h['id'] for h in search(s)] for s in subs]\n"
            "pools.append([h['id'] for h in search(question)])\nresults = fuse(pools)[:10]\n```\n"
            "(hyde/prf give extra recall on a hard sub-fact; rerank(s, [h['id'] for h in search(s)]) "
            "sharpens one sub-pool before fusing.) Return ONLY the code in a ``` block.")


def code_harness(gen, tools: Tools, q):
    r = gen.complete(f"Question: {q}\n\nWrite the program.", system=CODE_SYS)
    m = re.search(r"```(?:python)?\s*(.*?)```", r, re.DOTALL)
    code = m.group(1) if m else r
    ns = {"question": q, "search": tools.search, "hyde": tools.hyde, "prf": tools.prf,
          "decompose": tools.decompose, "rephrase": tools.rephrase,
          "rerank": tools.rerank, "mmr": tools.mmr, "fuse": _rrf, "results": None}
    try:
        exec(compile(code, "<sac>", "exec"), ns)  # noqa: S102
    except Exception:
        pass
    res = ns.get("results")
    ids = [str(x) for x in res] if isinstance(res, list) else None
    return tools.final(ids), {"steps": 1}   # one program = one model turn


def recall(gold, ids):
    g = set(gold); return len(g & set(ids[:K])) / len(g), int(g <= set(ids[:K]))


def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    def embed(texts):
        return em.encode(list(texts), normalize_embeddings=True, batch_size=128).tolist()

    gen = LLM()
    session = sac.Session("opensearch", index="hotpotqa", dim=common.DIM, hosts=[common.OS_HOST],
                          text_field="text", vector_field="vector", embedder=embed,
                          generator=gen.as_generator())
    session.reranker = sac.QwenReranker()   # Qwen3-Reranker (paper-caliber LLM reranker), not the weak MiniLM CE
    chat = agents.lc_chat()
    arms = ["dense", "tool", "sac"]
    keys = ["recall", "all", "n", "searches", "steps", "in", "out"]
    lock = threading.Lock()
    out = {}
    records = []          # per-query rows for distribution plots
    for ds in (2, 3, 4):
        rows = [json.loads(l) for l in (DATA / f"multihop_{ds}docs_queries.jsonl").open()][:per]
        agg = {a: dict.fromkeys(keys, 0.0) for a in arms}

        def one(r):
            q, gold = r["query"], r["gold_ids"]
            m = {}
            dids = session.search(q, top_k=K, mode="dense").ids()
            m["dense"] = (recall(gold, dids), {"searches": 1, "steps": 0, "in": 0, "out": 0})
            tgen = LLM(); tt = Tools(session, tgen, budget)
            tids, tm = tool_harness(chat, tt, q)
            m["tool"] = (recall(gold, tids), {"searches": tt.searches, "steps": tm["steps"],
                         "in": tm["lc_in"] + tgen.usage.input_tokens, "out": tm["lc_out"] + tgen.usage.output_tokens})
            sgen = LLM(); st = Tools(session, sgen, budget)
            sids, sm = code_harness(sgen, st, q)
            m["sac"] = (recall(gold, sids), {"searches": st.searches, "steps": sm["steps"],
                        "in": sgen.usage.input_tokens, "out": sgen.usage.output_tokens})
            with lock:
                for a in arms:
                    (rc, al), meta = m[a]
                    agg[a]["recall"] += rc; agg[a]["all"] += al; agg[a]["n"] += 1
                    for kk in ("searches", "steps", "in", "out"):
                        agg[a][kk] += meta[kk]
                    (rc, al), meta = m[a]
                    records.append({"hop": ds, "arm": a, "recall": rc, "all": al,
                                    "searches": meta["searches"], "turns": meta["steps"],
                                    "in_tok": meta["in"], "out_tok": meta["out"]})
                n = int(agg["dense"]["n"])
                if n % 20 == 0:
                    print(f"[fair {ds}hop] {n}/{len(rows)} " +
                          " ".join(f"{a}=r{agg[a]['recall']/n:.2f}/all{agg[a]['all']/n:.2f}/"
                                   f"srch{agg[a]['searches']/n:.1f}" for a in arms), flush=True)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(as_completed([ex.submit(one, r) for r in rows]))
        n = int(agg["dense"]["n"])
        out[f"{ds}hop"] = {a: {"recall@10": round(agg[a]["recall"] / n, 4),
                               "all_golds@10": round(agg[a]["all"] / n, 4),
                               "avg_searches": round(agg[a]["searches"] / n, 2),
                               "avg_model_turns": round(agg[a]["steps"] / n, 2),
                               "avg_in_tokens": int(agg[a]["in"] / n),
                               "avg_out_tokens": int(agg[a]["out"] / n)} for a in arms}
        print(f"\n===== {ds}-hop (n={n}, budget={budget}) =====")
        print(f"  {'arm':6s} {'recall@10':>9s} {'all@10':>7s} {'searches':>9s} {'turns':>6s} {'in_tok':>7s} {'out_tok':>8s}")
        for a in arms:
            r = out[f"{ds}hop"][a]
            print(f"  {a:6s} {r['recall@10']:>9.3f} {r['all_golds@10']:>7.3f} {r['avg_searches']:>9.1f} "
                  f"{r['avg_model_turns']:>6.1f} {r['avg_in_tokens']:>7d} {r['avg_out_tokens']:>8d}")

    (DATA.parent / "recall_fair.json").write_text(json.dumps(out, indent=2))
    with (DATA.parent / "recall_fair_perquery.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n[fair] saved recall_fair.json + recall_fair_perquery.jsonl ({len(records)} rows)")


if __name__ == "__main__":
    main()
