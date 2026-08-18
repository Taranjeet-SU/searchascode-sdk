"""Template-router training on a PUBLIC BEIR corpus — the SAME pipeline as the Altera run,
different dataset (memory-backed, no SSH tunnel), so we can track both in parallel.

    explorer.dataset(n=5000, label_llm=True, label_rerank=True, workers=6)  # atomic/sharded/GPU
    explorer.set_model("hist_gb"); explorer.train(cv=5)

    python -m phase2.beir_train [dataset=scifact] [n=5000] [workers=6]
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
from search_as_code.explore import Explorer, ProfilePack, unsolved, write_dataset_csv


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    out = f"phase2/runs/{dataset}_pack"

    print(f"[beir-train] loading {dataset} corpus...", flush=True)
    corpus, _q, _qr = beir.load(dataset)
    docs = [{"id": i, "text": ((v.get("title", "") + ". " + v.get("text", "")).strip())}
            for i, v in corpus.items() if v.get("text")]
    print(f"[beir-train] {len(docs)} docs | embedding with {common.EMB_MODEL} on "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    def embed(texts):
        return em.encode(list(texts), normalize_embeddings=True, batch_size=128).tolist()

    s = sac.Session("memory", dim=common.DIM, embedder=embed,
                    reranker=sac.QwenReranker(), generator=LLM().as_generator())
    s.add(docs)                                  # embeds all docs (batched, GPU)
    print(f"[beir-train] corpus indexed: {s.store.count()} docs", flush=True)

    ex = Explorer(s, ProfilePack.open(out))
    print(f"[beir-train] dataset(n={n}, label_llm=ON, label_rerank=ON, workers={workers})",
          flush=True)
    ds = ex.dataset(n=n, rephrases=2, k=10, P=20, label_llm=True, label_rerank=True,
                    workers=workers, batch_size=256, resume=True, progress_every=1)
    print(f"[beir-train] labeled {len(ds)} | oracle={ds.meta['oracle_coverage']} "
          f"| unsolved={ds.meta.get('unsolved')}", flush=True)
    print(f"[beir-train] label distribution: {ds.meta.get('label_distribution')}", flush=True)

    paths = write_dataset_csv(ex.pack)
    print(f"[beir-train] CSV: {paths['labels']} ({paths['rows']} rows) + {paths['template_recall']}",
          flush=True)
    uns = unsolved(ex.pack)
    if uns:
        __import__("pathlib").Path(f"{out}/unsolved.jsonl").write_text(
            "\n".join(json.dumps(u) for u in uns) + "\n")
        print(f"[beir-train] UNSOLVED={len(uns)} -> {out}/unsolved.jsonl", flush=True)

    ex.set_model("hist_gb", max_iter=400, learning_rate=0.07)
    m = ex.train(cv=5)
    print(f"\n===== {dataset.upper()} TEMPLATE ROUTER =====")
    print(f"  queries {m['n']} | solved {m['solved']} | oracle {m['oracle_coverage']:.3f}")
    if m.get("cv_accuracy") is not None:
        print(f"  best single tmpl : {m['best_single_template_acc']:.3f}")
        print(f"  ROUTER CV ACC    : {m['cv_accuracy']:.3f} +/- {m['cv_std']:.3f}")
        print(f"  lift vs fixed    : {m['router_lift_over_fixed']:+.3f}")
    else:
        print(f"  (no router: {m.get('note')})")
    print(f"  label dist       : {m.get('solved_label_distribution')}")
    __import__("pathlib").Path(f"{out}/router_meta.json").write_text(json.dumps(m, indent=2))
    print(f"[beir-train] saved to {out}", flush=True)


if __name__ == "__main__":
    main()
