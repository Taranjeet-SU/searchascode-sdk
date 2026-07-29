"""Router fitting: collect/generate labeled queries -> label against the 20 templates ->
train the classifier -> metrics. Backs :meth:`Explorer.fit`.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from .router import TemplateRouter, featurize, label_from_pools, train_router
from .templates import TEMPLATE_NAMES, base_pools, rerank_cache


def _rephrase(session, query: str, n: int) -> list[str]:
    """Ask the generator for ``n`` paraphrases that keep the same information need."""
    if n <= 0 or session.generator is None:
        return []
    prompt = (f"Rewrite the search query below in {n} different ways with the same meaning "
              "(synonyms, word order, abbreviation vs expansion). One per line, no numbering."
              f"\n\nQUERY: {query}")
    try:
        out = session.generator(prompt)
        txt = out[0] if isinstance(out, list) else str(out)
    except Exception:
        return []
    lines = [ln.strip("-•* ").strip() for ln in txt.splitlines() if ln.strip()]
    return [ln for ln in lines if ln.lower() != query.lower()][:n]


def _collect_queries(explorer, queries, n, rephrases, gen_llm):
    session, pack, config = explorer.session, explorer.pack, explorer.config
    if queries is not None:
        out = []
        for it in queries:
            if isinstance(it, dict):
                out.append({"query": it["query"], "gold_id": it.get("gold_id") or it.get("gold")})
            else:
                out.append({"query": it[0], "gold_id": it[1]})
        return out[:n]

    # generate grounded synth queries (+ rephrases) from a fresh sample of the corpus
    from .engine import ExploreContext
    from .stages import _gen_queries
    ctx = ExploreContext(session=session, pack=pack, config=config)
    per_doc = int(config.get("synth_per_doc", 3))
    need_base = max(1, n // (1 + max(0, rephrases)))
    n_docs = max(1, need_base // per_doc + 1)
    try:
        sample = session.store.sample(n_docs)
    except Exception:
        sample = []
    out = []
    for di, d in enumerate(sample):
        text = getattr(d, "text", None) or ""
        for _diff, q in _gen_queries(ctx, text, per_doc):
            out.append({"query": q, "gold_id": d.id})
            for rp in _rephrase(session, q, rephrases):
                out.append({"query": rp, "gold_id": d.id})
            if len(out) >= n:
                return out[:n]
        if (di + 1) % 25 == 0:
            print(f"[fit] generated {len(out)}/{n} queries from {di + 1} docs", flush=True)
    return out[:n]


def _batch_embed(session, texts, bs=64):
    """Embed all query texts in batches (one forward per batch) — much cheaper than a
    per-query call when the backend embedder can batch."""
    out = []
    for i in range(0, len(texts), bs):
        out.extend(session.embedder.embed(texts[i:i + bs]))
        if len(texts) > bs and (i // bs) % 10 == 0:
            print(f"[fit] embedded {min(i + bs, len(texts))}/{len(texts)} queries", flush=True)
    return out


def fit_router(explorer, *, queries=None, n=5000, rephrases=2, k=10, P=25,
               label_llm=False, label_rerank=False, progress_every=100) -> dict:
    session, pack = explorer.session, explorer.pack
    t0 = time.time()
    data = _collect_queries(explorer, queries, n, rephrases, gen_llm=label_llm)
    if not data:
        raise RuntimeError("no queries to fit on (provide queries= or ensure the store samples)")

    # embed all queries up front in batches — far cheaper than one-at-a-time, and the
    # vector is reused for both the dense pool and the feature row.
    embs = _batch_embed(session, [it["query"] for it in data], bs=64)
    emb_dim = len(embs[0]) if embs else 0

    rows, X, y = [], [], []
    per_template = Counter()
    any_hit = Counter()          # how often each template retrieves gold (oracle view)
    solved = 0
    for i, item in enumerate(data):
        q, gold = item["query"], item["gold_id"]
        emb = embs[i]
        pools, docs = base_pools(session, q, P=P, use_llm=label_llm, query_vec=emb)
        rr = rerank_cache(session, q, pools, docs=docs) if label_rerank else None
        best, hits = label_from_pools(pools, rr, gold, k=k)
        for name, h in hits.items():
            any_hit[name] += h
        X.append(featurize(q, emb)); y.append(best)
        if best != "none":
            solved += 1; per_template[best] += 1
        rows.append({"query": q, "gold_id": gold, "best": best})
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[fit] labeled {i + 1}/{len(data)}  solved={solved}  "
                  f"({(i + 1) / (time.time() - t0):.1f} q/s)", flush=True)

    pack.write_jsonl("router_labels.jsonl", rows)

    # train on solved queries only (a query no template answers has no valid target)
    idx = [j for j, lab in enumerate(y) if lab != "none"]
    metrics = {
        "n_labeled": len(data), "solved": solved,
        "oracle_coverage": round(solved / len(data), 4),
        "n_templates": len(TEMPLATE_NAMES),
        "label_distribution": dict(per_template),
        "template_hit_rate@k": {t: round(any_hit[t] / len(data), 4) for t in TEMPLATE_NAMES},
        "seconds": round(time.time() - t0, 1),
    }
    if len(idx) >= 10 and len(per_template) >= 2:
        Xs = np.array([X[j] for j in idx], dtype=np.float32)
        ys = np.array([y[j] for j in idx])
        res = train_router(Xs, ys)
        best_single = max(per_template.values()) / solved      # "always pick the best fixed template"
        metrics.update({
            "cv_accuracy": res["cv_accuracy"], "cv_std": res["cv_std"],
            "cv_folds": res["cv_folds"], "train_accuracy": res["train_accuracy"],
            "best_single_template_acc": round(best_single, 4),
            "router_lift_over_fixed": (round(res["cv_accuracy"] - best_single, 4)
                                       if res["cv_accuracy"] is not None else None),
        })
        router = TemplateRouter(res["model"], res["classes"], emb_dim=emb_dim, metrics=metrics)
        router.save(pack.path("router.pkl"))
        explorer.router = router
    else:
        metrics["note"] = "too few solved/labeled queries or classes to train a router"

    pack.write_json("router_meta.json", metrics)
    pack.record_stage("router", "ok" if "cv_accuracy" in metrics else "rejected",
                      seconds=metrics["seconds"], summary={
                          "n": len(data), "solved": solved,
                          "cv_acc": metrics.get("cv_accuracy"),
                          "vs_fixed": metrics.get("router_lift_over_fixed")},
                      artifacts=["router.pkl", "router_labels.jsonl", "router_meta.json"])
    return metrics
