"""DEEP-MODE SAC benchmark — the COST of going deep, and what `explore` adds.

Two arms, measured on HotpotQA (2/3/4-hop) and SearchUnify docs (2/3/4-hop):

  1. sac_deep            = phase1.agents.run_sac(session, q, deep=True, max_retries=3)
                           the "write code -> JUDGE -> deepen on failure with prior sandbox
                           state persisting" agent. (BEFORE explore.)
  2. sac_deep + explore  = the SAME deep agent, but its hop-1 prompt is SEEDED with the
                           corpus profile that `explore`/`session.describe(llm=True)` learns
                           (content-type mix + recommended primitives), injected as an extra
                           guidance message via a chat wrapper. (AFTER explore.)

For each query/arm we record: recall@10, all_golds@10, hops (deepening rounds), retrieval
searches (search + fan-out sub-searches + hyde/prf/answerability), input+output tokens, cost.

Baselines (dense / tool / single-shot sac) are NOT recomputed — they come from
recall_fair.json / su_recall.json and are merged in the charts/RESULTS step.

    python -m experiments.deep_sac.run_deep_sac [per_hop=50] [workers=4]

INTERNAL — never pushed. SU uses internal customer docs.
"""
from __future__ import annotations

import json
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import agents, common
from phase1.llm import LLM

HERE = Path(__file__).parent
HOTPOT_DATA = Path(__file__).parents[1] / "multi_hop_synth_queries" / "data"
SU_DATA = Path(__file__).parents[1] / "su_multihop" / "data"
SU_CSV = Path.home() / "scripts" / "data" / "su_docs_2.csv"
K = 10
HOPS = (2, 3, 4)
MAX_RETRIES = 3


# --------------------------------------------------------------------------- #
# shared building blocks                                                       #
# --------------------------------------------------------------------------- #
def make_embedder():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)

    def embed(texts):
        return em.encode(list(texts), normalize_embeddings=True, batch_size=128,
                         show_progress_bar=False).tolist()

    return embed


def instrument(session):
    """Count underlying retrieval calls on THIS session (thread-safe; fan-out sub-searches
    go through self.search, so search_many/decompose count each sub-query)."""
    counter = {"n": 0}
    lock = threading.Lock()

    def wrap(orig):
        def w(*a, **k):
            with lock:
                counter["n"] += 1
            return orig(*a, **k)
        return w

    for name in ("search", "hyde_search", "prf_search", "answerability"):
        setattr(session, name, wrap(getattr(session, name)))
    return counter


class HintChat:
    """Wrap a ChatOpenAI so every codegen call gets the explore corpus-profile hint
    injected right after the system prompt. Judge uses a separate (plain) chat, so the
    judge is NOT biased — only the code-writing agent is seeded."""

    def __init__(self, inner, hint: str):
        self.inner = inner
        self.hint = hint

    def invoke(self, msgs):
        from langchain_core.messages import HumanMessage
        new = list(msgs)
        new.insert(1, HumanMessage(content=self.hint))
        return self.inner.invoke(new)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def build_hint(describe_session) -> str:
    """Run explore's corpus profiling ONCE (session.describe with the LLM characterization),
    turn it into a short 'what this data is + recommended primitives' hint."""
    prof = describe_session.describe(n_samples=8, llm=True)
    cts = prof.get("content_types", {})
    llm_txt = (prof.get("llm") or "").strip()
    fields = prof.get("fields") or prof.get("metadata_keys") or {}
    return (
        "CORPUS PROFILE (learned by an `explore` pass over THIS data — use it to pick the right "
        "hop-1 strategy so you succeed in FEWER deepening hops):\n"
        f"- content-type mix (sampled): {cts}\n"
        f"- fields: {fields}\n"
        f"- characterization & recommended primitives:\n{llm_txt}\n\n"
        "How to apply: lead HOP 1 with the recommended primitives for this data type (prose Q&A -> "
        "dense/hybrid + HyDE; exact tokens/IDs -> add keyword/regex). Multi-fact questions need "
        "decompose_search across the distinct sub-facts. Do not fan out blindly — spend the cheap "
        "hop-1 budget on the primitives this corpus rewards."
    )


def recall(gold, ids):
    g = set(gold)
    top = set(ids[:K])
    return (len(g & top) / len(g) if g else 0.0), int(g <= top)


ARMS = ["sac_deep", "sac_deep_explore", "sac_deep_fewshot"]


def build_fewshot(name) -> str:
    """The explore fewshot exemplar block for THIS corpus (per-winning-template example queries),
    injected as a hop-1 hint — the grounded alternative to the static describe() profile."""
    from pathlib import Path as _P
    from search_as_code.explore import ProfilePack, fewshot_exemplars, format_fewshot_block
    pack_map = {"hotpotqa": "pack_hotpotqa_multihop", "su": "pack_su_multihop"}
    pdir = _P(__file__).parents[1] / "primitive_selection" / pack_map.get(name, "")
    try:
        block = format_fewshot_block(fewshot_exemplars(ProfilePack.open(str(pdir)), per_template=3))
        return "LEARNED STRATEGY EXEMPLARS (from an explore labeling pass over THIS corpus):\n" + block
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] no fewshot pack ({e})", flush=True)
        return ""
_KEYS = ["recall", "all", "hops", "searches", "in", "out", "cost", "n", "errors"]


def _blank_agg():
    return {a: dict.fromkeys(_KEYS, 0.0) for a in ARMS}


def run_query(store, embedder, reranker, generator, chat, hint, fewshot_hint, q, gold):
    """Run all deep arms on one query with fresh, per-query, instrumented sessions."""
    out = {}
    for arm in ARMS:
        session = sac.Session(store, embedder=embedder, reranker=reranker, generator=generator)
        cnt = instrument(session)
        if arm == "sac_deep_explore":
            use_chat = HintChat(chat, hint)
        elif arm == "sac_deep_fewshot":
            use_chat = HintChat(chat, fewshot_hint) if fewshot_hint else chat
        else:
            use_chat = chat
        try:
            res = agents.run_sac(session, q, chat=use_chat, judge_chat=chat, k=K,
                                 max_retries=MAX_RETRIES, deep=True)
            rc, al = recall(gold, res["ids"])
            u = res["usage"]
            out[arm] = {"recall": rc, "all": al, "hops": res["hops"], "searches": cnt["n"],
                        "in": u["input_tokens"], "out": u["output_tokens"],
                        "cost": u["cost_usd"], "error": None}
        except Exception as e:  # noqa: BLE001
            out[arm] = {"recall": 0.0, "all": 0, "hops": 0, "searches": cnt["n"],
                        "in": 0, "out": 0, "cost": 0.0,
                        "error": f"{type(e).__name__}: {e}"}
            print(f"  [ERR {arm}] {type(e).__name__}: {e}", flush=True)
    return out


def bench_corpus(name, store, embedder, reranker, generator, datasets, per_hop, workers):
    """datasets: dict hop -> list of {query, gold_ids}. Returns per-hop arm aggregates."""
    chat = agents.lc_chat()
    # explore profiling ONCE per corpus
    prof_session = sac.Session(store, embedder=embedder, reranker=reranker, generator=generator)
    hint = build_hint(prof_session)
    fewshot_hint = build_fewshot(name)
    print(f"\n[{name}] static hint ({len(hint)} chars); fewshot hint ({len(fewshot_hint)} chars)\n", flush=True)

    lock = threading.Lock()
    result, records = {}, []
    for hop in HOPS:
        rows = datasets[hop][:per_hop]
        if not rows:
            print(f"[{name} {hop}hop] no queries, skip", flush=True)
            continue
        agg = _blank_agg()

        def one(r):
            q, gold = r["query"], r["gold_ids"]
            m = run_query(store, embedder, reranker, generator, chat, hint, fewshot_hint, q, gold)
            with lock:
                for a in ARMS:
                    d = m[a]
                    agg[a]["recall"] += d["recall"]; agg[a]["all"] += d["all"]
                    agg[a]["hops"] += d["hops"]; agg[a]["searches"] += d["searches"]
                    agg[a]["in"] += d["in"]; agg[a]["out"] += d["out"]
                    agg[a]["cost"] += d["cost"]; agg[a]["n"] += 1
                    agg[a]["errors"] += 1 if d["error"] else 0
                    records.append({"corpus": name, "hop": hop, "arm": a, **d})
                n = int(agg[ARMS[0]]["n"])
                if n % 5 == 0:
                    msg = " ".join(
                        f"{a}=r{agg[a]['recall']/n:.2f}/all{agg[a]['all']/n:.2f}/"
                        f"h{agg[a]['hops']/n:.1f}/in{int(agg[a]['in']/n)}" for a in ARMS)
                    print(f"[{name} {hop}hop] {n}/{len(rows)} {msg}", flush=True)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(as_completed([ex.submit(one, r) for r in rows]))

        n = int(agg[ARMS[0]]["n"])
        result[f"{hop}hop"] = {"n": n, **{a: {
            "recall@10": round(agg[a]["recall"] / n, 4),
            "all_golds@10": round(agg[a]["all"] / n, 4),
            "avg_hops": round(agg[a]["hops"] / n, 2),
            "avg_searches": round(agg[a]["searches"] / n, 2),
            "avg_in_tokens": int(agg[a]["in"] / n),
            "avg_out_tokens": int(agg[a]["out"] / n),
            "avg_cost_usd": round(agg[a]["cost"] / n, 5),
            "errors": int(agg[a]["errors"]),
        } for a in ARMS}}
        print(f"\n===== [{name}] {hop}-hop (n={n}) =====", flush=True)
        print(f"  {'arm':17s} {'recall':>7s} {'all':>6s} {'hops':>5s} {'srch':>5s} "
              f"{'in_tok':>7s} {'out_tok':>7s} {'$':>8s} err", flush=True)
        for a in ARMS:
            rr = result[f"{hop}hop"][a]
            print(f"  {a:17s} {rr['recall@10']:>7.3f} {rr['all_golds@10']:>6.3f} "
                  f"{rr['avg_hops']:>5.2f} {rr['avg_searches']:>5.1f} {rr['avg_in_tokens']:>7d} "
                  f"{rr['avg_out_tokens']:>7d} {rr['avg_cost_usd']:>8.5f} {rr['errors']}", flush=True)
    return result, records


# --------------------------------------------------------------------------- #
# corpora                                                                      #
# --------------------------------------------------------------------------- #
def hotpot_store(embedder):
    store = sac.connect("opensearch", index="hotpotqa", dim=common.DIM, hosts=[common.OS_HOST],
                        text_field="text", vector_field="vector")
    return store


def hotpot_datasets():
    ds = {}
    for hop in HOPS:
        path = HOTPOT_DATA / f"multihop_{hop}docs_queries.jsonl"
        ds[hop] = [json.loads(l) for l in path.open()]
    return ds


def su_store(embedder, generator):
    df = pd.read_csv(SU_CSV)
    docs = []
    for _, row in df.iterrows():
        content = row.get("content")
        if pd.isna(content) or not str(content).strip():
            continue
        title = "" if pd.isna(row.get("title")) else str(row.get("title"))
        docs.append({"id": str(row["id"]), "text": (title + ". " + str(content)).strip()})
    print(f"[su] {len(docs)} docs with content", flush=True)
    loader = sac.Session("memory", dim=common.DIM, embedder=embedder, generator=generator)
    loader.add(docs)
    print(f"[su] added {loader.store.count()} docs to memory store", flush=True)
    return loader.store


def su_datasets():
    ds = {}
    for hop in HOPS:
        path = SU_DATA / f"su_multihop_{hop}docs.jsonl"
        ds[hop] = [json.loads(l) for l in path.open()]
    return ds


def main():
    per_hop = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    print(f"[deep_sac] per_hop={per_hop} workers={workers} max_retries={MAX_RETRIES} "
          f"cuda={torch.cuda.is_available()}", flush=True)

    embedder = make_embedder()
    reranker = sac.CrossEncoderReranker()   # light, local; both deep arms use it identically
    generator = LLM().as_generator()

    out = {}

    # ---- HotpotQA ----
    try:
        hstore = hotpot_store(embedder)
        hres, hrec = bench_corpus("hotpotqa", hstore, embedder, reranker, generator,
                                  hotpot_datasets(), per_hop, workers)
        out["hotpotqa"] = hres
    except Exception:
        print("[hotpotqa] FAILED:\n" + traceback.format_exc(), flush=True)
        hrec = []

    # ---- SearchUnify ----
    try:
        sstore = su_store(embedder, generator)
        sres, srec = bench_corpus("su", sstore, embedder, reranker, generator,
                                  su_datasets(), per_hop, workers)
        out["su"] = sres
    except Exception:
        print("[su] FAILED:\n" + traceback.format_exc(), flush=True)
        srec = []

    (HERE / "deep_recall_fewshot.json").write_text(json.dumps(out, indent=2))
    with (HERE / "deep_recall_fewshot_perquery.jsonl").open("w") as f:
        for r in (hrec + srec):
            f.write(json.dumps(r) + "\n")
    print(f"\n[deep_sac] wrote deep_recall.json + deep_recall_perquery.jsonl "
          f"({len(hrec) + len(srec)} rows)", flush=True)


if __name__ == "__main__":
    main()
