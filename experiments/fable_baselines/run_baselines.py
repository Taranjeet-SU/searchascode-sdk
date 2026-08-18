"""fable.md WS6 — the no-LLM baseline arms, one runner, three corpora.

Arms (no LLM calls, so they run without an API key):
  dense          one dense search @10                       (the gate's floor)
  bm25           one keyword search @10                     (full-text where the index has it)
  hybrid         RRF(dense@50, bm25@50) -> @10
  dense_rerank   dense@50 -> MiniLM cross-encoder -> @10    (the P1-10 missing control)

Corpora, mirroring the published configs exactly:
  browsecomp   830 queries, precomputed gte-base corpus vectors (memory) + OS `browsecomp`
               full-text index for BM25; golds = re-fetched official qrels (BC-1 fix).
  hotpotqa     multihop_{2,3,4}docs_queries.jsonl over the OS `hotpotqa` index (gte-base),
               forge-disjoint slice rows[200:300] — same slice as RESULTS.md §4b.
  su           su_multihop_{2,3,4}docs.jsonl over an in-memory session built from
               ~/scripts/data/su_docs_2.csv (INTERNAL docs; only aggregates are written).

Metrics: recall@10 and all_golds@10, with paired-bootstrap deltas vs dense
(`sac.metrics.compare`). Output: experiments/fable_baselines/baselines_<corpus>.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import search_as_code as sac
import search_as_code.primitives as P
from phase1 import common

HERE = Path(__file__).parent
K, POOL = 10, 50


def _embedder():
    import torch
    from sentence_transformers import SentenceTransformer
    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    def embed(texts):
        return em.encode(list(texts), normalize_embeddings=True, batch_size=128).tolist()
    return embed


def _metrics(rows, arms):
    """rows: list of {arm: (recall, allg)} dicts -> aggregates + paired CIs vs dense."""
    out = {}
    for a in arms:
        rec = [r[a][0] for r in rows]
        allg = [r[a][1] for r in rows]
        out[a] = {"recall@10": round(float(np.mean(rec)), 4),
                  "all_golds@10": round(float(np.mean(allg)), 4), "n": len(rec)}
    for a in arms:
        if a == "dense":
            continue
        cmp_r = sac.compare([r[a][0] for r in rows], [r["dense"][0] for r in rows])
        out[a]["recall_delta_vs_dense"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in cmp_r.items()}
    return out


def _eval_rows(rows, dense_ids50, kw_ids50, texts_of, rerank, golds_of):
    """Shared arm logic. *_ids50: qidx -> list[str]; texts_of(ids)->list[str]."""
    arms = ["dense", "bm25", "hybrid", "dense_rerank"]
    out = []
    for i, r in enumerate(rows):
        gold = golds_of(r)
        d50, k50 = dense_ids50(i), kw_ids50(i)
        res = {"dense": d50[:K], "bm25": k50[:K]}
        # RRF over the two id lists (rank-based, same as P.fuse on id lists)
        agg = {}
        for lst in (d50, k50):
            for rank, did in enumerate(lst):
                agg[did] = agg.get(did, 0.0) + 1.0 / (60 + rank + 1)
        res["hybrid"] = [d for d, _ in sorted(agg.items(), key=lambda x: -x[1])][:K]
        if d50:
            texts = texts_of(d50)
            scores = rerank(r["query"], texts)
            order = list(np.argsort(scores)[::-1])
            res["dense_rerank"] = [d50[j] for j in order[:K]]
        else:
            res["dense_rerank"] = []
        out.append({a: (sac.recall_at_k(res[a], gold, K), sac.all_golds_at_k(res[a], gold, K))
                    for a in arms})
    return out, arms


def run_browsecomp():
    from experiments.browsecomp import bc_common
    golds = bc_common.load_golds()
    queries = bc_common.load_queries()
    qids = [q for q in queries if q in golds]
    rows = [{"qid": q, "query": queries[q], "gold": golds[q]} for q in qids]

    vecs = np.load(bc_common.VECS_NPY)
    ids = [str(i) for i in json.loads(bc_common.IDS_JSON.read_text())]
    id_arr = np.array(ids)
    V = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    embed = _embedder()
    qv = np.asarray(embed([r["query"] for r in rows]), dtype=np.float32)
    dense50 = [list(id_arr[np.argsort(-(V @ qv[i]))[:POOL]]) for i in range(len(rows))]

    os_sess = sac.Session("opensearch", index="browsecomp", dim=common.DIM,
                          hosts=[common.OS_HOST], text_field="text", vector_field="vector",
                          embedder=embed)
    kw50 = []
    for r in rows:
        try:
            kw50.append(os_sess.search(r["query"], top_k=POOL, mode="keyword").ids())
        except Exception:
            kw50.append([])

    def texts_of(idlist):
        docs = {d.id: (d.text or "") for d in os_sess.store.get(idlist)}
        return [docs.get(i, "")[:1500] for i in idlist]

    rr = sac.CrossEncoderReranker()
    per, arms = _eval_rows(rows, lambda i: dense50[i], lambda i: kw50[i], texts_of,
                           lambda q, ts: rr(q, ts), lambda r: r["gold"])
    return _metrics(per, arms)


def run_hotpot():
    embed = _embedder()
    sess = sac.Session("opensearch", index="hotpotqa", dim=common.DIM, hosts=[common.OS_HOST],
                       text_field="text", vector_field="vector", embedder=embed)
    rr = sac.CrossEncoderReranker()
    data_dir = Path(__file__).parents[1] / "multi_hop_synth_queries" / "data"
    out = {}
    for ds in (2, 3, 4):
        rows = [json.loads(l) for l in (data_dir / f"multihop_{ds}docs_queries.jsonl").open()][200:300]
        dense50 = [sess.search(r["query"], top_k=POOL, mode="dense").ids() for r in rows]
        kw50 = [sess.search(r["query"], top_k=POOL, mode="keyword").ids() for r in rows]

        def texts_of(idlist):
            docs = {d.id: (d.text or "") for d in sess.store.get(idlist)}
            return [docs.get(i, "")[:1500] for i in idlist]

        per, arms = _eval_rows(rows, lambda i: dense50[i], lambda i: kw50[i], texts_of,
                               lambda q, ts: rr(q, ts), lambda r: [str(g) for g in r["gold_ids"]])
        out[f"{ds}hop"] = _metrics(per, arms)
    return out


def run_su():
    embed = _embedder()
    # Mirror su_multihop.run_su_multihop.load_docs: doc id = the CSV `id` column (a URL —
    # gold_ids reference these), text = title + ". " + content, rows without content dropped.
    csvp = Path.home() / "scripts" / "data" / "su_docs_2.csv"
    docs = []
    with csvp.open() as f:
        for row in csv.DictReader(f):
            content = (row.get("content") or "").strip()
            if not content:
                continue
            title = row.get("title") or ""
            docs.append({"id": str(row["id"]), "text": f"{title}. {content}".strip()})
    sess = sac.Session("memory", dim=common.DIM, embedder=embed)
    B = 64
    for i in range(0, len(docs), B):
        sess.add(docs[i:i + B])
    rr = sac.CrossEncoderReranker()
    data_dir = Path(__file__).parents[1] / "su_multihop" / "data"
    out = {}
    for ds in (2, 3, 4):
        rows = [json.loads(l) for l in (data_dir / f"su_multihop_{ds}docs.jsonl").open()][:100]
        dense50 = [sess.search(r["query"], top_k=POOL, mode="dense").ids() for r in rows]
        kw50 = [sess.search(r["query"], top_k=POOL, mode="keyword").ids() for r in rows]
        text_by_id = {d["id"]: d["text"] for d in docs}

        def texts_of(idlist):
            return [text_by_id.get(i, "")[:1500] for i in idlist]

        per, arms = _eval_rows(rows, lambda i: dense50[i], lambda i: kw50[i], texts_of,
                               lambda q, ts: rr(q, ts), lambda r: [str(g) for g in r["gold_ids"]])
        out[f"{ds}hop"] = _metrics(per, arms)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    runners = {"browsecomp": run_browsecomp, "hotpotqa": run_hotpot, "su": run_su}
    for name, fn in runners.items():
        if which not in ("all", name):
            continue
        t0 = time.time()
        print(f"== {name} ==", flush=True)
        res = fn()
        res["_meta"] = {"embedder": common.EMB_MODEL, "reranker": "cross-encoder ms-marco MiniLM",
                        "pool": POOL, "k": K, "date": "2026-08-18",
                        "caveats": ["no-LLM baseline arms only; SAC/tool arms need an API key",
                                    "SU aggregates only; docs are internal (DS-2)"]}
        out = HERE / f"baselines_{name}.json"
        out.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2)[:1500], flush=True)
        print(f"[{name} done in {time.time()-t0:.0f}s -> {out}]", flush=True)
