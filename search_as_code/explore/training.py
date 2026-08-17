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


def _xgb(**p):
    from xgboost import XGBClassifier
    return XGBClassifier(**{"n_estimators": 400, "learning_rate": 0.07, "max_depth": 6,
                            "subsample": 0.9, "colsample_bytree": 0.9, "tree_method": "hist",
                            "n_jobs": -1, "random_state": 0, **p})


MODEL_REGISTRY = {"hist_gb": _hist_gb, "logreg": _logreg,
                  "random_forest": _random_forest, "mlp": _mlp, "xgb": _xgb}


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
                  queries=None, progress_every=1, all_golds=True) -> RouterDataset:
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
            golds = item.get("gold_ids") or [item["gold_id"]]     # single or multi (qrels)
            ctx = StrategyContext(session, item["query"], P_pool=P, emb=emb, use_llm=label_llm,
                                  use_rerank=label_rerank, top_k=k, rerank_lock=rr_lock)
            best, hits = label_via_templates(ctx, set(golds), k=k, all_golds=all_golds)
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
        labs = [{"query": batch[j]["query"],
                 "gold_id": (batch[j].get("gold_ids") or [batch[j].get("gold_id")])[0],
                 "gold_ids": batch[j].get("gold_ids") or [batch[j]["gold_id"]],
                 "dataset": batch[j].get("dataset", ""),
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
    from collections import Counter

    from .router import best_from_hits

    X = np.concatenate(feats) if feats else np.zeros((0, 0), dtype=np.float32)
    # derive the label from the stored per-template recall@k hits with the current winner
    # policy (cheapest template that solves) — decoupled from the expensive labeling pass.
    y = [best_from_hits(r.get("hits") or {}) for r in labs]
    any_hit = Counter()
    for r in labs:
        for t, h in (r.get("hits") or {}).items():
            any_hit[t] += h
    solved = sum(1 for lab in y if lab != "none")
    meta = {"n": len(y), "solved": solved, "unsolved": len(y) - solved,
            "oracle_coverage": round(solved / len(y), 4) if y else 0.0,
            "n_templates": len(TEMPLATE_NAMES),
            "label_distribution": dict(Counter(lab for lab in y if lab != "none")),
            "template_hit_rate@k": {t: round(any_hit[t] / len(y), 4) for t in TEMPLATE_NAMES} if y else {}}
    return RouterDataset(X=X, y=y, queries=[r["query"] for r in labs], meta=meta)


def unsolved(pack) -> list[dict]:
    """Queries that NO template solved (recall@k miss for all 16) — candidates for a new
    primitive or a duplication issue (gold has near-dupes so an equivalent doc was retrieved)."""
    from .router import best_from_hits
    ddir = pack.root / "dataset" / "shards"
    out = []
    for f in sorted(ddir.glob("lab_*.jsonl")):
        for r in _read_jsonl(f):
            if best_from_hits(r.get("hits") or {}) == "none":
                out.append({"query": r["query"], "gold_id": r["gold_id"]})
    return out


def write_dataset_csv(pack, out_dir=None) -> dict[str, str]:
    """Persist the labeled dataset as CSV for reuse/inspection. Writes two files:

    - ``labels.csv``          one row per query: query, gold_id, winner (cheapest solver),
                              solved, and ``hit_<template>`` (recall@k, 0/1) for all 16 templates.
    - ``template_recall.csv`` one row per template: tier, cost, recall@k, times_winner.

    Reads the on-disk shards, so it works during/after labeling. Returns {name: path}.
    """
    import csv

    from .router import best_from_hits
    from .templates import TEMPLATE_COST, TEMPLATE_DOCS, TEMPLATE_NAMES

    out = Path(out_dir) if out_dir else pack.root
    out.mkdir(parents=True, exist_ok=True)
    sdir = pack.root / "dataset" / "shards"
    rows = []
    for f in sorted(sdir.glob("lab_*.jsonl")):
        rows.extend(_read_jsonl(f))

    from collections import Counter
    hit_count = Counter()
    win_count = Counter()
    n = len(rows)

    lpath = out / "labels.csv"
    with lpath.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["query", "gold_id", "winner", "solved"] + [f"hit_{t}" for t in TEMPLATE_NAMES])
        for r in rows:
            hits = r.get("hits") or {}
            winner = best_from_hits(hits)
            solved = int(winner != "none")
            if solved:
                win_count[winner] += 1
            for t in TEMPLATE_NAMES:
                hit_count[t] += int(hits.get(t, 0))
            w.writerow([r.get("query", ""), r.get("gold_id", ""), winner, solved]
                       + [int(hits.get(t, 0)) for t in TEMPLATE_NAMES])

    tpath = out / "template_recall.csv"
    with tpath.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["template", "tier", "cost", "recall@k", "times_winner", "win_frac"])
        for t in sorted(TEMPLATE_NAMES, key=lambda x: TEMPLATE_COST.get(x, 99)):
            recall = round(hit_count[t] / n, 4) if n else 0.0
            w.writerow([t, TEMPLATE_DOCS[t]["tier"], TEMPLATE_COST.get(t), recall,
                        win_count[t], round(win_count[t] / n, 4) if n else 0.0])
    return {"labels": str(lpath), "template_recall": str(tpath), "rows": n}


def fewshot_exemplars(pack, per_template: int = 3, max_query_chars: int = 160) -> dict:
    """Per-winning-template example queries mined from the labeling pass.

    For each labeled query we know the CHEAPEST template that actually retrieved its gold (the
    winner). Grouping queries by winner yields, per strategy, real *corpus-grounded* examples of
    the queries that strategy wins on — the empirical answer to "which primitive chain works for
    queries like this on THIS data." Returns, ordered by how often each strategy wins::

        {template: {"tier","does","differs","n_wins": int, "examples": [query, ...]}}

    Feed to an agent via :func:`format_fewshot_block` so it picks a strategy from evidence rather
    than a static blanket instruction (a corpus-wide "always decompose" hint measurably hurt — see
    experiments/multi_hop_synth_queries §11).
    """
    from collections import Counter, defaultdict

    from .router import best_from_hits
    from .templates import TEMPLATE_DOCS

    sdir = pack.root / "dataset" / "shards"
    groups: dict[str, list[str]] = defaultdict(list)
    counts: Counter = Counter()
    for f in sorted(sdir.glob("lab_*.jsonl")):
        for r in _read_jsonl(f):
            w = best_from_hits(r.get("hits") or {})
            if w == "none":
                continue
            counts[w] += 1
            q = (r.get("query") or "").strip()
            if q and len(groups[w]) < per_template * 6:      # keep a small pool to pick from
                groups[w].append(q)

    out: dict[str, dict] = {}
    for t, qs in groups.items():
        seen, picked = set(), []
        for q in sorted(qs, key=len):                        # shorter, distinct = more readable exemplars
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append(q[:max_query_chars])
            if len(picked) >= per_template:
                break
        d = TEMPLATE_DOCS.get(t, {})
        out[t] = {"tier": d.get("tier", ""), "does": d.get("does", ""),
                  "differs": d.get("differs", ""), "n_wins": counts[t], "examples": picked}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n_wins"]))


def format_fewshot_block(exemplars: dict, max_templates: int = 8) -> str:
    """Render :func:`fewshot_exemplars` as a prompt block an agent reads to pick a strategy."""
    if not exemplars:
        return ""
    lines = ["Learned strategy exemplars for THIS corpus (an `explore` labeling pass recorded which "
             "primitive chain actually retrieved the gold for queries like these). Match the "
             "incoming query to the closest exemplars to pick your first strategy:"]
    for t, d in list(exemplars.items())[:max_templates]:
        ex = "; ".join(f'"{q}"' for q in d["examples"]) or "(no example)"
        lines.append(f"- {t} — {d['does']} (wins {d['n_wins']}x). e.g. {ex}")
    lines.append("If the query matches no exemplar well, start cheap (dense/hybrid) and deepen only "
                 "if the result looks weak — do NOT fan out blindly.")
    return "\n".join(lines)


def _tok(s: str) -> set:
    import re
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def classify_failure(session, item, k=50, sem_lo=0.45, lex_lo=0.08,
                     collision=5, band=0.02) -> dict:
    """Diagnose why NO template solved a query, into one of four buckets:
      - ``low_similarity``   : gold is far in BOTH semantic and lexical space -> no current
                               signal (dense/keyword/hyde/regex) can find it => a NEW PRIMITIVE.
      - ``synonym_metadata`` : semantically close but lexically disjoint -> synonyms / expansion
                               / metadata concept needed.
      - ``rank_collision``   : gold is retrievable but buried among many similarly-scored docs
                               (poor rank due to collision).
      - ``unexplained``      : none of the above cleanly explains the miss.
    Uses cheap signals only (query/gold embeddings + token overlap + dense rank/score density).
    """
    import numpy as np

    q = item["query"]
    golds = {str(g) for g in (item.get("gold_ids") or [item["gold_id"]])}
    qv = np.asarray(session.embedder.embed([q])[0], dtype=np.float32)
    qv = qv / (np.linalg.norm(qv) + 1e-9)
    qtok = _tok(q)
    best_sem, best_lex = 0.0, 0.0
    for d in session.store.get(list(golds)):
        if getattr(d, "vector", None) is not None:
            gv = np.asarray(d.vector, dtype=np.float32)
            best_sem = max(best_sem, float(qv @ (gv / (np.linalg.norm(gv) + 1e-9))))
        if d.text:
            gt = _tok(d.text)
            best_lex = max(best_lex, len(qtok & gt) / (len(qtok) + 1e-9))
    res = session.store.query_vector(qv.tolist(), top_k=k)
    scores = [float(h.score) for h in res]
    gold_rank = next((i for i, h in enumerate(res) if h.id in golds), None)
    near = sum(1 for s in scores if abs(s - scores[gold_rank]) <= band) if gold_rank is not None else 0

    if best_sem < sem_lo and best_lex < lex_lo:
        cat = "low_similarity"
    elif best_lex < lex_lo:
        cat = "synonym_metadata"
    elif gold_rank is not None and near >= collision:
        cat = "rank_collision"
    else:
        cat = "unexplained"
    return {"category": cat, "sem": round(best_sem, 3), "lex": round(best_lex, 3),
            "gold_rank": gold_rank, "near_density": near, "query": q[:90]}


def analyze_failures(session, items, sample=300) -> dict:
    """Bucket the unsolved queries into the four failure categories (see classify_failure)."""
    from collections import Counter
    cats, ex = Counter(), {}
    for it in list(items)[:sample]:
        try:
            r = classify_failure(session, it)
        except Exception:
            continue
        cats[r["category"]] += 1
        ex.setdefault(r["category"], []).append(r)
    n = sum(cats.values())
    return {"checked": n, "categories": dict(cats),
            "fractions": {c: round(v / n, 3) for c, v in cats.items()} if n else {},
            "examples": {c: v[:4] for c, v in ex.items()}}


def duplication_scan(session, items, sample=80, k=5, sim_thresh=0.9) -> dict:
    """For unsolved gold docs: is there a DIFFERENT doc that's a near-duplicate of the gold and
    outranks it in dense search? If so, the 'miss' is likely a duplication artifact (an
    equivalent doc was retrieved), not a genuine retrieval-capability gap.

    Returns counts + examples; ``near_dup`` are dedup issues, the rest are new-primitive gaps.
    """
    items = list(items)[:sample]
    checked = near_dup = 0
    examples = []
    for it in items:
        gold = it["gold_id"]
        try:
            docs = session.store.get([gold])
            gtext = docs[0].text if docs else None
            if not gtext:
                continue
            gvec = session.embedder.embed([gtext])[0]
            res = session.store.query_vector(gvec, top_k=k)
        except Exception:
            continue
        checked += 1
        others = [h for h in res if h.id != gold]
        if others and float(others[0].score) >= sim_thresh:
            near_dup += 1
            if len(examples) < 10:
                examples.append({"gold_id": gold, "near_dup_id": others[0].id,
                                 "score": round(float(others[0].score), 3),
                                 "query": it["query"][:80]})
    return {"checked": checked, "near_dup": near_dup,
            "near_dup_frac": round(near_dup / checked, 3) if checked else None,
            "gap_frac": round(1 - near_dup / checked, 3) if checked else None,
            "examples": examples}


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
    """Accept a list of {query, gold_id | gold_ids | gold} / (q, gold), or a path to a jsonl.
    Preserves the full gold SET (qrels have several relevant docs) and the dataset tag."""
    if isinstance(queries, (str, Path)):
        rows = _read_jsonl(Path(queries))
    else:
        rows = list(queries)
    out = []
    for it in rows:
        if isinstance(it, dict):
            golds = it.get("gold_ids")
            single = it.get("gold_id") or it.get("gold")
            golds = golds or ([single] if single is not None else [])
            out.append({"query": it["query"],
                        "gold_id": golds[0] if golds else None,
                        "gold_ids": golds,
                        "dataset": it.get("dataset", "")})
        else:
            out.append({"query": it[0], "gold_id": it[1], "gold_ids": [it[1]], "dataset": ""})
    return out
