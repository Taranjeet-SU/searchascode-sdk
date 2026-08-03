"""Multi-hop retrieval benchmark on the (INTERNAL) SearchUnify docs corpus.

Dense vs tool-calling vs SAC code-mode, IDENTICAL toolset + matched search budget — the same
FAIR harness used for the HotpotQA experiment, reused verbatim from
``experiments.multi_hop_synth_queries.eval_fair``.

Pipeline (single process, in-memory Session so gold_ids stay consistent):
  1. load ~/scripts/data/su_docs_2.csv -> docs=[{id, text=title+". "+content}]
  2. build a MEMORY-backed Session, embed + add all docs
  3. generate_multihop (STANDARD fn) for n_docs in (2,3,4) -> data/su_multihop_{n}docs.jsonl
  4. 3-arm benchmark (up to 100 queries/hop), ThreadPoolExecutor(5)
  5. write su_recall.json + print per-hop table

Keep ALL outputs local under experiments/su_multihop/ — internal customer data, never pushed.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import agents, common
from phase1.llm import LLM
from search_as_code.explore import generate_multihop

# reuse the FAIR harness verbatim
from experiments.multi_hop_synth_queries.eval_fair import (
    Tools, tool_harness, code_harness, recall, K,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
CSV = Path.home() / "scripts" / "data" / "su_docs_2.csv"
GEN_TARGET = 150
GEN_SAMPLE_CHUNK = 400     # memory store samples first-n deterministically; corpus is ~396 docs -> use it all in one pass
GEN_WORKERS = 6
PER_HOP = 100
BENCH_WORKERS = 5
BUDGET = 6
HOPS = (2, 3, 4)


def load_docs():
    df = pd.read_csv(CSV)
    docs = []
    for _, row in df.iterrows():
        content = row.get("content")
        if pd.isna(content) or not str(content).strip():
            continue
        title = "" if pd.isna(row.get("title")) else str(row.get("title"))
        docs.append({"id": str(row["id"]), "text": (title + ". " + str(content)).strip()})
    print(f"[load] {len(docs)} docs with content (of {len(df)} rows)", flush=True)
    return docs


def build_session(docs):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    print(f"[embed] model={common.EMB_MODEL} device={dev}", flush=True)

    def embed(texts):
        return em.encode(list(texts), normalize_embeddings=True, batch_size=128).tolist()

    session = sac.Session("memory", dim=common.DIM, embedder=embed,
                          generator=LLM().as_generator())
    session.reranker = sac.QwenReranker()   # Qwen3-Reranker (paper-caliber), full-primitive harness
    session.add(docs)
    print(f"[session] added {session.store.count()} docs", flush=True)
    return session


def generate(session):
    DATA.mkdir(parents=True, exist_ok=True)
    # full-text generator: generate_multihop's _gen reads out[0], so return the WHOLE reply as one item
    gllm = LLM()

    def full_gen(prompt):
        return [gllm.complete(prompt, system="You write rigorous multi-hop questions, or NONE.")]

    counts = {}
    for n in HOPS:
        out_path = DATA / f"su_multihop_{n}docs.jsonl"
        if out_path.exists() and sum(1 for _ in out_path.open()) >= PER_HOP:
            counts[n] = sum(1 for _ in out_path.open())
            print(f"[gen {n}d] reuse existing {counts[n]} queries", flush=True)
            continue
        print(f"[gen {n}d] generating (target={GEN_TARGET}, chunk={GEN_SAMPLE_CHUNK})...", flush=True)
        rows = generate_multihop(session, n_docs=n, target=GEN_TARGET, workers=GEN_WORKERS,
                                 sample_chunk=GEN_SAMPLE_CHUNK, generator=full_gen,
                                 out_path=str(out_path), progress_every=25)
        counts[n] = len(rows)
        print(f"[gen {n}d] wrote {len(rows)} queries -> {out_path}", flush=True)
    print(f"[gen] counts: {counts} (gen cost ${gllm.usage.cost_usd:.2f})", flush=True)
    return counts


def benchmark(session):
    chat = agents.lc_chat()
    arms = ["dense", "tool", "sac"]
    keys = ["recall", "all", "n", "searches", "steps", "in", "out"]
    lock = threading.Lock()
    out = {}
    records = []
    for ds in HOPS:
        path = DATA / f"su_multihop_{ds}docs.jsonl"
        rows = [json.loads(l) for l in path.open()][:PER_HOP]
        if not rows:
            print(f"[bench {ds}hop] no queries, skipping", flush=True)
            continue
        agg = {a: dict.fromkeys(keys, 0.0) for a in arms}

        def one(r):
            q, gold = r["query"], r["gold_ids"]
            m = {}
            dids = session.search(q, top_k=K, mode="dense").ids()
            m["dense"] = (recall(gold, dids), {"searches": 1, "steps": 0, "in": 0, "out": 0})
            tgen = LLM(); tt = Tools(session, tgen, BUDGET)
            tids, tm = tool_harness(chat, tt, q)
            m["tool"] = (recall(gold, tids), {"searches": tt.searches, "steps": tm["steps"],
                         "in": tm["lc_in"] + tgen.usage.input_tokens,
                         "out": tm["lc_out"] + tgen.usage.output_tokens})
            sgen = LLM(); st = Tools(session, sgen, BUDGET)
            sids, _sm = code_harness(sgen, st, q)
            m["sac"] = (recall(gold, sids), {"searches": st.searches, "steps": 1,
                        "in": sgen.usage.input_tokens, "out": sgen.usage.output_tokens})
            with lock:
                for a in arms:
                    (rc, al), meta = m[a]
                    agg[a]["recall"] += rc; agg[a]["all"] += al; agg[a]["n"] += 1
                    for kk in ("searches", "steps", "in", "out"):
                        agg[a][kk] += meta[kk]
                    records.append({"hop": ds, "arm": a, "recall": rc, "all": al,
                                    "searches": meta["searches"], "turns": meta["steps"],
                                    "in_tok": meta["in"], "out_tok": meta["out"]})
                n = int(agg["dense"]["n"])
                if n % 10 == 0:
                    print(f"[bench {ds}hop] {n}/{len(rows)} " +
                          " ".join(f"{a}=r{agg[a]['recall']/n:.2f}/all{agg[a]['all']/n:.2f}/"
                                   f"srch{agg[a]['searches']/n:.1f}" for a in arms), flush=True)

        with ThreadPoolExecutor(max_workers=BENCH_WORKERS) as ex:
            list(as_completed([ex.submit(one, r) for r in rows]))
        n = int(agg["dense"]["n"])
        out[f"{ds}hop"] = {"n": n, "arms": {a: {
            "recall@10": round(agg[a]["recall"] / n, 4),
            "all_golds@10": round(agg[a]["all"] / n, 4),
            "avg_searches": round(agg[a]["searches"] / n, 2),
            "avg_model_turns": round(agg[a]["steps"] / n, 2),
            "avg_in_tokens": int(agg[a]["in"] / n),
            "avg_out_tokens": int(agg[a]["out"] / n)} for a in arms}}
        print(f"\n===== {ds}-hop (n={n}, budget={BUDGET}) =====", flush=True)
        print(f"  {'arm':6s} {'recall@10':>9s} {'all@10':>7s} {'searches':>9s} "
              f"{'turns':>6s} {'in_tok':>7s} {'out_tok':>8s}", flush=True)
        for a in arms:
            rr = out[f"{ds}hop"]["arms"][a]
            print(f"  {a:6s} {rr['recall@10']:>9.3f} {rr['all_golds@10']:>7.3f} "
                  f"{rr['avg_searches']:>9.1f} {rr['avg_model_turns']:>6.1f} "
                  f"{rr['avg_in_tokens']:>7d} {rr['avg_out_tokens']:>8d}", flush=True)

    (HERE / "su_recall.json").write_text(json.dumps(out, indent=2))
    with (HERE / "su_recall_perquery.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n[done] saved su_recall.json + su_recall_perquery.jsonl ({len(records)} rows)", flush=True)
    return out


def main():
    docs = load_docs()
    session = build_session(docs)
    generate(session)
    benchmark(session)


if __name__ == "__main__":
    main()
