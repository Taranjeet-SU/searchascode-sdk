"""Standard training subsystem for the template router.

Turns the one-shot fit into three composable, **atomic** steps:

    explorer = sac.explore(session, out="pack/")
    explorer.dataset(n=5000, rephrases=2, label_rerank=True, workers=6)  # build/label -> disk
    explorer.set_model("hist_gb", max_iter=400, learning_rate=0.07)      # pick the estimator
    metrics = explorer.train(cv=5)                                       # train + evaluate

Design goals the user asked for:
- **atomic / resumable** — the dataset is written in per-batch shards with a checkpoint, so a
  crash resumes from the last completed batch; a finished dataset is loaded, not rebuilt.
- **GPU when available** — embeddings and the cross-encoder use the session's embedder/reranker,
  which auto-select cuda (see embeddings.py); labeling parallelizes IO across ``workers``.
- **batch storage** — features are ``.npy`` shards, labels are ``.jsonl`` shards under
  ``pack/dataset/shards/``; the full matrix is concatenated on load.
- **swappable model** — ``set_model`` takes a registry name, a factory, or any fitted-estimator
  class, so the router head can be tuned/replaced without touching the dataset.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fit import _batch_embed, _collect_queries
from .router import featurize, label_via_templates
from .templates import TEMPLATE_NAMES, StrategyContext


# --------------------------------------------------------------------------- #
# swappable model head                                                          #
# --------------------------------------------------------------------------- #
def _hist_gb(**p):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**{"max_iter": 400, "learning_rate": 0.07,
                                             "max_depth": 6, "random_state": 0, **p})


def _logreg(**p):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**{"max_iter": 1000, "C": 1.0, **p})


def _random_forest(**p):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**{"n_estimators": 300, "random_state": 0, "n_jobs": -1, **p})


def _mlp(**p):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(**{"hidden_layer_sizes": (256,), "max_iter": 300, "random_state": 0, **p})


MODEL_REGISTRY = {"hist_gb": _hist_gb, "logreg": _logreg,
                  "random_forest": _random_forest, "mlp": _mlp}


def make_model(spec="hist_gb", **params):
    """spec: a name in MODEL_REGISTRY, a factory callable, or an estimator instance."""
    if hasattr(spec, "fit"):                       # already an estimator
        return spec
    if callable(spec):                             # a factory
        return spec(**params)
    if spec not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{spec}'; options: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[spec](**params)


# --------------------------------------------------------------------------- #
# atomic sharded dataset                                                        #
# --------------------------------------------------------------------------- #
def _atomic_write_bytes(path: Path, write_fn):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        write_fn(f)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


@dataclass
class RouterDataset:
    """Features + labels for the template router (loaded from sharded storage)."""
    X: np.ndarray
    y: list
    queries: list
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.y)


def build_dataset(explorer, *, n=5000, rephrases=2, k=10, P=25, label_llm=False,
                  label_rerank=False, workers=1, batch_size=256, resume=True,
                  queries=None, progress_every=1) -> RouterDataset:
    """Generate/label queries into an atomic, resumable, sharded dataset on disk."""
    session, pack = explorer.session, explorer.pack
    ddir = pack.root / "dataset"
    sdir = ddir / "shards"
    sdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ddir / "checkpoint.json"
    qfile = ddir / "queries.jsonl"

    # ---- 1. queries (generate once, or take provided/persisted) ----------
    if queries is not None:
        data = _load_query_list(queries)
        _write_jsonl(qfile, data)
    elif resume and qfile.exists():
        data = _read_jsonl(qfile)
    else:
        data = _collect_queries(explorer, None, n, rephrases, gen_llm=label_llm)
        if not data:
            raise RuntimeError("no queries generated (check the store samples / generator)")
        _write_jsonl(qfile, data)
    N = len(data)
    n_batches = (N + batch_size - 1) // batch_size

    # ---- 2. resume checkpoint -------------------------------------------
    done = 0
    if resume and ckpt_path.exists():
        done = json.loads(ckpt_path.read_text()).get("batches_done", 0)

    rr_lock = threading.Lock()
    t0 = time.time()
    for bi in range(done, n_batches):
        batch = data[bi * batch_size:(bi + 1) * batch_size]
        embs = _batch_embed(session, [b["query"] for b in batch], bs=64)

        def _label(j):
            item, emb = batch[j], embs[j]
            ctx = StrategyContext(session, item["query"], P_pool=P, emb=emb, use_llm=label_llm,
                                  use_rerank=label_rerank, top_k=k, rerank_lock=rr_lock)
            best, hits = label_via_templates(ctx, item["gold_id"], k=k)
            return j, featurize(item["query"], emb).astype(np.float32), best, hits

        rows = [None] * len(batch)
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for fut in as_completed([ex.submit(_label, j) for j in range(len(batch))]):
                    j, feat, best, hits = fut.result()
                    rows[j] = (feat, best, hits)
        else:
            for j in range(len(batch)):
                _, feat, best, hits = _label(j)
                rows[j] = (feat, best, hits)

        feats = np.stack([r[0] for r in rows])
        labs = [{"query": batch[j]["query"], "gold_id": batch[j]["gold_id"],
                 "best": rows[j][1], "hits": rows[j][2]} for j in range(len(batch))]
        _atomic_write_bytes(sdir / f"feat_{bi:05d}.npy", lambda f: np.save(f, feats))
        _atomic_write_text(sdir / f"lab_{bi:05d}.jsonl",
                           "\n".join(json.dumps(x) for x in labs) + "\n")
        _atomic_write_text(ckpt_path, json.dumps(
            {"batches_done": bi + 1, "n_batches": n_batches, "N": N,
             "batch_size": batch_size, "emb_dim": int(feats.shape[1] - 8)}))
        if progress_every and (bi + 1) % progress_every == 0:
            n_lab = (bi + 1) * batch_size
            print(f"[dataset] batch {bi + 1}/{n_batches} "
                  f"(~{min(n_lab, N)}/{N} labeled, {min(n_lab, N)/(time.time()-t0):.1f} q/s)",
                  flush=True)

    ds = load_dataset(pack)
    _atomic_write_text(ddir / "meta.json", json.dumps(ds.meta, indent=2))
    return ds


def load_dataset(pack) -> RouterDataset:
    """Concatenate the on-disk shards into a RouterDataset."""
    ddir = pack.root / "dataset"
    sdir = ddir / "shards"
    feats, labs = [], []
    for f in sorted(sdir.glob("feat_*.npy")):
        feats.append(np.load(f))
    for f in sorted(sdir.glob("lab_*.jsonl")):
        labs.extend(_read_jsonl(f))
    X = np.concatenate(feats) if feats else np.zeros((0, 0), dtype=np.float32)
    y = [r["best"] for r in labs]
    from collections import Counter
    any_hit = Counter()
    for r in labs:
        for t, h in (r.get("hits") or {}).items():
            any_hit[t] += h
    solved = sum(1 for lab in y if lab != "none")
    meta = {"n": len(y), "solved": solved,
            "oracle_coverage": round(solved / len(y), 4) if y else 0.0,
            "n_templates": len(TEMPLATE_NAMES),
            "label_distribution": dict(Counter(lab for lab in y if lab != "none")),
            "template_hit_rate@k": {t: round(any_hit[t] / len(y), 4) for t in TEMPLATE_NAMES} if y else {}}
    return RouterDataset(X=X, y=y, queries=[r["query"] for r in labs], meta=meta)


# --------------------------------------------------------------------------- #
# training                                                                      #
# --------------------------------------------------------------------------- #
def train_router_model(dataset: RouterDataset, model_spec="hist_gb", cv=5, **model_params):
    from collections import Counter

    from sklearn.model_selection import cross_val_score

    y = dataset.y
    idx = [i for i, lab in enumerate(y) if lab != "none"]
    if len(idx) < 10 or len({y[i] for i in idx}) < 2:
        return None, {**dataset.meta, "note": "too few solved queries / classes to train"}
    Xs = dataset.X[idx]
    ys = np.array([y[i] for i in idx])
    counts = Counter(ys)
    per = counts  # label distribution among solved
    best_single = max(counts.values()) / len(ys)

    model = make_model(model_spec, **model_params)
    min_class = min(counts.values())
    folds = max(2, min(cv, min_class))
    cv_acc = cv_std = None
    if min_class >= 2:
        sc = cross_val_score(model, Xs, ys, cv=folds, scoring="accuracy")
        cv_acc, cv_std = float(sc.mean()), float(sc.std())
    model.fit(Xs, ys)
    metrics = {**dataset.meta,
               "model": model_spec if isinstance(model_spec, str) else type(model).__name__,
               "cv_folds": folds, "cv_accuracy": cv_acc, "cv_std": cv_std,
               "train_accuracy": float((model.predict(Xs) == ys).mean()),
               "best_single_template_acc": round(best_single, 4),
               "router_lift_over_fixed": (round(cv_acc - best_single, 4) if cv_acc else None),
               "solved_label_distribution": dict(per)}
    return model, metrics


# --------------------------------------------------------------------------- #
# small io helpers                                                              #
# --------------------------------------------------------------------------- #
def _write_jsonl(path: Path, rows):
    _atomic_write_text(path, "\n".join(json.dumps(r) for r in rows) + "\n")


def _read_jsonl(path: Path):
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def _load_query_list(queries):
    """Accept a list of {query,gold_id}/(q,gold), or a path to queries.jsonl."""
    if isinstance(queries, (str, Path)):
        rows = _read_jsonl(Path(queries))
    else:
        rows = list(queries)
    out = []
    for it in rows:
        if isinstance(it, dict):
            out.append({"query": it["query"], "gold_id": it.get("gold_id") or it.get("gold")})
        else:
            out.append({"query": it[0], "gold_id": it[1]})
    return out
