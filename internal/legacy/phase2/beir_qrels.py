"""Label the primitive templates on REAL (query, relevant-doc) pairs from BEIR qrels — no
synthetic generation. One dataset per invocation, written to phase2/runs/<dataset>_pack/.

Then phase2/global_router.py pools all datasets into one GLOBAL template router.

    python -m phase2.beir_qrels <dataset[,dataset...]> [workers=6] [max_docs=200000]
"""
from __future__ import annotations

import json
import sys

import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from internal.legacy.phase2 import beir
from search_as_code.explore import (
    Explorer,
    ProfilePack,
    analyze_failures,
    unsolved,
    write_dataset_csv,
)

_EM = None


def embedder():
    global _EM
    if _EM is None:
        _EM = SentenceTransformer(common.EMB_MODEL,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
    return _EM


def build_items(dataset, max_docs):
    corpus, queries, qrels = beir.load(dataset)
    docs = [{"id": str(i), "text": (v.get("title", "") + ". " + v.get("text", "")).strip()}
            for i, v in corpus.items() if v.get("text")][:max_docs]
    have = {d["id"] for d in docs}
    items = []
    for qid, rels in qrels.items():
        golds = [str(d) for d, r in rels.items() if int(r) > 0 and str(d) in have]
        if qid in queries and golds:
            items.append({"query": queries[qid], "gold_ids": golds, "dataset": dataset})
    return docs, items


def run_one(dataset, workers, max_docs, gen):
    print(f"\n[beir-qrels] === {dataset} ===", flush=True)
    docs, items = build_items(dataset, max_docs)
    print(f"[beir-qrels] {len(docs)} docs, {len(items)} labeled queries (real qrels)", flush=True)
    em = embedder()

    def embed(texts):
        return em.encode(list(texts), normalize_embeddings=True, batch_size=128).tolist()

    s = sac.Session("memory", dim=common.DIM, embedder=embed,
                    reranker=sac.QwenReranker(), generator=gen)
    s.add(docs)
    out = f"phase2/runs/{dataset}_qrels_pack"   # distinct from any synthetic *_pack run
    ex = Explorer(s, ProfilePack.open(out))
    ds = ex.dataset(queries=items, k=10, P=20, label_llm=True, label_rerank=True,
                    workers=workers, batch_size=256, resume=True, progress_every=2)
    print(f"[beir-qrels] {dataset}: labeled {len(ds)} | oracle={ds.meta['oracle_coverage']} "
          f"| unsolved={ds.meta.get('unsolved')}", flush=True)
    print(f"[beir-qrels] {dataset}: label dist {ds.meta.get('label_distribution')}", flush=True)
    write_dataset_csv(ex.pack)

    uns = unsolved(ex.pack)
    if uns:
        fa = analyze_failures(s, uns, sample=300)
        print(f"[beir-qrels] {dataset}: UNSOLVED={len(uns)} failure buckets {fa['fractions']}", flush=True)
        (ex.pack.root / "failures.json").write_text(json.dumps(fa, indent=2))
    ex.set_model("hist_gb", max_iter=400, learning_rate=0.07)
    m = ex.train(cv=5)
    print(f"[beir-qrels] {dataset}: router cv_acc={m.get('cv_accuracy')} "
          f"best_single={m.get('best_single_template_acc')} lift={m.get('router_lift_over_fixed')}",
          flush=True)


def main():
    datasets = (sys.argv[1] if len(sys.argv) > 1 else "scifact,nfcorpus,arguana").split(",")
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    max_docs = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
    gen = LLM().as_generator()
    for d in datasets:
        try:
            run_one(d.strip(), workers, max_docs, gen)
        except Exception as e:
            print(f"[beir-qrels] {d}: FAILED {type(e).__name__}: {e}", flush=True)
    print("[beir-qrels] done", flush=True)


if __name__ == "__main__":
    main()
