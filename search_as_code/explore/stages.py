"""Exploration stages.

Implemented: :class:`SampleStage` (stratified sample), :class:`ProfileStage` (content-type
mix + LLM characterization), :class:`SynthesizeStage` (grounded synthetic queries) and
:class:`ValidateStage` (held-out retrieval eval + keep-if-better report). The rest
(ontology/crossdoc/router/codegen) are typed stubs recorded as ``planned`` so the pipeline
runs end-to-end and the pack shows the roadmap.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np

from .._genutil import gen_text
from .engine import ExploreContext, Stage  # noqa: F401  (re-exported for convenience)


# --------------------------------------------------------------------------- #
# tiny, dependency-free k-means so sampling stays stratified without sklearn    #
# --------------------------------------------------------------------------- #
def _kmeans(x: np.ndarray, k: int, iters: int = 25, seed: int = 0):
    """Return (labels, centroids). k-means++ init, cosine-friendly on L2-normed x."""
    n = len(x)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    # k-means++ seeding
    centers = [int(rng.integers(n))]
    d2 = ((x - x[centers[0]]) ** 2).sum(1)
    for _ in range(1, k):
        probs = d2 / (d2.sum() or 1.0)
        nxt = int(rng.choice(n, p=probs))
        centers.append(nxt)
        d2 = np.minimum(d2, ((x - x[nxt]) ** 2).sum(1))
    cent = x[centers].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        # assign
        dists = ((x[:, None, :] - cent[None, :, :]) ** 2).sum(2)
        new = dists.argmin(1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        # update
        for j in range(k):
            m = labels == j
            if m.any():
                cent[j] = x[m].mean(0)
    return labels, cent


# --------------------------------------------------------------------------- #
# Stage 1 — stratified sample                                                   #
# --------------------------------------------------------------------------- #
class SampleStage(Stage):
    """Draw a *representative* working sample: pull a larger random pool, cluster it in
    embedding space, then keep a few docs nearest each centroid. This surfaces rare doc
    types that a flat random-n sample would miss."""

    name = "sample"
    produces = ["sample.jsonl", "sample_meta.json"]

    def run(self, ctx: ExploreContext) -> dict:
        pool_size = int(ctx.cfg("pool_size", 200))
        per_cluster = int(ctx.cfg("per_cluster", 3))
        pool = ctx.store.sample(pool_size)
        pool = [d for d in pool if (d.text or "").strip()]
        if not pool:
            ctx.pack.write_jsonl("sample.jsonl", [])
            ctx.pack.write_json("sample_meta.json", {"pool": 0, "clusters": 0, "sample": 0})
            return {"pool": 0, "clusters": 0, "sample": 0}

        vecs = np.asarray(ctx.embedder.embed([d.text or "" for d in pool]), dtype=np.float32)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        n_clusters = int(ctx.cfg("n_clusters", max(1, min(8, len(pool) // 5))))
        labels, cent = _kmeans(vecs, n_clusters, seed=int(ctx.cfg("seed", 0)))

        rows, sizes = [], {}
        for j in range(len(cent)):
            idx = np.where(labels == j)[0]
            if len(idx) == 0:
                continue
            sizes[str(j)] = int(len(idx))
            # closest to centroid first → most typical of the cluster
            order = idx[np.argsort(((vecs[idx] - cent[j]) ** 2).sum(1))]
            for i in order[:per_cluster]:
                d = pool[int(i)]
                rows.append({"id": d.id, "cluster": j, "text": d.text,
                             "metadata": d.metadata or {}})
        ctx.pack.write_jsonl("sample.jsonl", rows)
        ctx.pack.write_json("sample_meta.json",
                            {"pool": len(pool), "clusters": len(sizes),
                             "cluster_sizes": sizes, "sample": len(rows)})
        return {"pool": len(pool), "clusters": len(sizes), "sample": len(rows)}


# --------------------------------------------------------------------------- #
# Stage 2 — profile                                                             #
# --------------------------------------------------------------------------- #
class ProfileStage(Stage):
    """Characterize the corpus: schema + per-chunk content-type mix (heuristic) and,
    when a generator is present, an LLM-written profile of the data (data type, key
    entities, recommended primitives) — both overall and per cluster."""

    name = "profile"
    requires = ["sample"]
    produces = ["content_profile.json"]

    def run(self, ctx: ExploreContext) -> dict:
        from ..primitives import content_type

        sample = ctx.pack.read_jsonl("sample.jsonl")
        use_llm = bool(ctx.cfg("llm", ctx.generator is not None))

        types: dict[str, int] = {}
        for r in sample:
            ct = content_type(r.get("text") or "")
            types[ct] = types.get(ct, 0) + 1

        try:
            schema = ctx.store.describe_schema()
        except Exception:
            schema = {"backend": getattr(ctx.store, "backend", "?")}

        overall_llm = None
        cluster_profiles: dict[str, str] = {}
        if use_llm and ctx.generator is not None:
            overall_llm = _llm_profile(ctx, schema, [r["text"] for r in sample[:8]])
            by_cluster: dict[int, list[str]] = {}
            for r in sample:
                by_cluster.setdefault(r.get("cluster", 0), []).append(r.get("text") or "")
            for c, texts in sorted(by_cluster.items()):
                cluster_profiles[str(c)] = _llm_profile(ctx, schema, texts[:4])

        profile = {
            "schema": schema,
            "content_types": types,
            "n_sampled": len(sample),
            "llm_overall": overall_llm,
            "llm_by_cluster": cluster_profiles,
        }
        ctx.pack.write_json("content_profile.json", profile)
        return {"content_types": types, "clusters_profiled": len(cluster_profiles),
                "llm": bool(overall_llm)}


def _llm_profile(ctx: ExploreContext, schema: dict, texts: list[str]) -> str:
    fields = schema.get("fields") or schema.get("metadata_keys") or {}
    rows = "\n".join(f"- {(t or '')[:300]}" for t in texts if t) or "(no text)"
    prompt = (
        "You are profiling a search corpus before writing retrieval code.\n"
        f"Backend: {schema.get('backend') or schema.get('index')}  "
        f"Approx docs: {schema.get('count')}\nFields: {fields}\n\n"
        f"Sample documents:\n{rows}\n\n"
        "In 4-6 short lines: (1) what kind of data this is (prose, tables, curated "
        "fact-cards, code, mixed?), (2) key entities/fields a query targets, (3) which "
        "retrieval primitives fit best (keyword/exact & regex for part-numbers & fact-cards, "
        "dense/hyde for prose, fielded/phrase for structured fields)."
    )
    try:
        out = ctx.generator(prompt)
        return gen_text(out)  # whole profile, not line 1 (GEN-1)
    except Exception as e:  # pragma: no cover
        return f"(llm profile unavailable: {e})"


# --------------------------------------------------------------------------- #
# Stages 3-7 — typed stubs (recorded as ``planned`` until implemented)          #
# --------------------------------------------------------------------------- #
class _PlannedStage(Stage):
    _todo = "not implemented yet"

    def run(self, ctx: ExploreContext) -> dict:
        raise NotImplementedError(self._todo)


class OntologyStage(_PlannedStage):
    """LLM-induce a domain ontology (entities/relations/synonyms) from the sample; then
    best-effort enrich from the web and reconcile; surface for review."""
    name = "ontology"
    requires = ["profile"]
    produces = ["ontology.json"]
    _todo = "LLM-induced + web-enriched ontology (next pass)"


class CrossDocStage(_PlannedStage):
    """Link documents via the ontology (entity co-occurrence / KG edges) — the
    cross-document relation layer used for candidate expansion."""
    name = "crossdoc"
    requires = ["ontology"]
    produces = ["crossdoc.json"]
    _todo = "cross-document relation graph (next pass)"


class SynthesizeStage(Stage):
    """Generate stratified easy/medium/hard queries **grounded in the sampled docs** — each
    query's answer lives in a known document (its ``gold_id``). This is a leakage-free eval/
    training set (built from the corpus, never from the downstream test set) used by
    ``validate`` (and later the router/few-shot mining)."""

    name = "synthesize"
    requires = ["sample"]
    produces = ["synth_queries.jsonl"]

    def run(self, ctx: ExploreContext) -> dict:
        if ctx.generator is None:
            raise RuntimeError("synthesize needs a Session generator")
        sample = ctx.pack.read_jsonl("sample.jsonl")
        max_docs = int(ctx.cfg("synth_docs", 30))
        per_doc = int(ctx.cfg("synth_per_doc", 3))
        rows = []
        for r in sample[:max_docs]:
            for diff, q in _gen_queries(ctx, r.get("text") or "", per_doc):
                rows.append({"query": q, "gold_id": r["id"],
                             "cluster": r.get("cluster"), "difficulty": diff})
        ctx.pack.write_jsonl("synth_queries.jsonl", rows)
        return {"queries": len(rows), "from_docs": min(len(sample), max_docs),
                "by_difficulty": dict(Counter(x["difficulty"] for x in rows))}


def _gen_queries(ctx: ExploreContext, text: str, per_doc: int):
    if not text.strip():
        return []
    prompt = (
        f"From the technical document below, write {per_doc} distinct search questions whose "
        "answer is IN this document. Vary difficulty across: 'easy' (direct keyword/fact "
        "lookup), 'medium' (paraphrased/conceptual), 'hard' (indirect or multi-constraint). "
        'Return ONLY a JSON list of {"difficulty": "...", "query": "..."}. Keep each query '
        f"self-contained (no 'this document').\n\nDOCUMENT:\n{text[:1200]}"
    )
    try:
        out = ctx.generator(prompt)
        txt = "\n".join(out) if isinstance(out, list) else str(out)  # generator may line-split; rejoin for JSON
    except Exception:
        return []
    m = re.search(r"\[.*\]", txt, re.DOTALL)
    if not m:
        return []
    res = []
    try:
        for o in json.loads(m.group(0)):
            q = (o.get("query") or "").strip()
            if q:
                res.append((o.get("difficulty", "?"), q))
    except Exception:
        return []
    return res[:per_doc]


class RouterStage(_PlannedStage):
    """Run primitive combos over the synth queries, label which retrieves gold, train
    the XGBoost primitive router (query+profile features → best combo)."""
    name = "router"
    requires = ["synthesize"]
    produces = ["router.pkl", "router_meta.json"]
    _todo = "combo-exploration + XGB router (port from phase4)"


class CodegenStage(_PlannedStage):
    """Mine winning combos into (situation→chain) templates and (query→code) few-shots;
    optionally LLM-codegen new sandbox-validated primitives; write prompt overrides."""
    name = "codegen"
    requires = ["router"]
    produces = ["templates.json", "few_shots.json", "prompt_overrides.md"]
    _todo = "template/few-shot mining + sandbox-validated primitive codegen (next pass)"


class ValidateStage(Stage):
    """Measure retrieval on the grounded synth queries: recall@k for each base strategy
    (dense/keyword/hybrid), overall and per cluster. Establishes the baseline and the
    best strategy per data-shape — the signal later tunings must beat. Writes a JSON +
    a human-readable REPORT.md."""

    name = "validate"
    requires = ["synthesize"]
    produces = ["validation.json", "REPORT.md"]

    STRATEGIES = ["dense", "keyword", "hybrid"]

    def run(self, ctx: ExploreContext) -> dict:
        qs = ctx.pack.read_jsonl("synth_queries.jsonl")
        if not qs:
            raise RuntimeError("no synth queries to validate on")
        k = int(ctx.cfg("validate_k", 10))
        hit = {s: 0 for s in self.STRATEGIES}
        by_cluster: dict = {}     # cluster -> strategy -> [hits, n]
        by_diff: dict = {}        # difficulty -> strategy -> [hits, n]
        for row in qs:
            gold, c, d = row["gold_id"], row.get("cluster"), row.get("difficulty", "?")
            for s in self.STRATEGIES:
                ids = _search_ids(ctx, s, row["query"], k)
                h = int(gold in ids)
                hit[s] += h
                by_cluster.setdefault(c, {}).setdefault(s, [0, 0])
                by_cluster[c][s][0] += h
                by_cluster[c][s][1] += 1
                by_diff.setdefault(d, {}).setdefault(s, [0, 0])
                by_diff[d][s][0] += h
                by_diff[d][s][1] += 1

        n = len(qs)
        recall = {s: hit[s] / n for s in self.STRATEGIES}
        best = max(recall, key=lambda s: recall[s])
        cluster_best = {str(c): _best(d) for c, d in by_cluster.items()}
        result = {"n": n, "k": k, "recall_at_k": recall, "best_overall": best,
                  "cluster_best": cluster_best,
                  "by_difficulty": {d: _rates(v) for d, v in by_diff.items()}}
        ctx.pack.write_json("validation.json", result)
        ctx.pack.path("REPORT.md").write_text(_report_md(ctx, result))
        return {"n": n, "best": best,
                "recall_at_k": {s: round(recall[s], 3) for s in self.STRATEGIES}}

    def validate(self, ctx: ExploreContext, summary: dict) -> tuple[bool, str]:
        """Implement the keep-if-usable gate the engine defines and the docs advertise.

        ``Stage.validate`` and docs/EXPLORE.md's robustness table describe rejecting output
        that does not beat baseline as "the honesty rule", but no stage in default_pipeline()
        overrode it — so ``status="rejected"`` was unreachable and the table over-claimed
        (SDK-A5). A validation pass whose best strategy retrieves nothing is measuring noise
        (usually: the synth queries are not answerable from the corpus), so it is rejected
        rather than recorded as a baseline later stages would trust.
        """
        recall = summary.get("recall_at_k") or {}
        best = max(recall.values(), default=0.0)
        floor = float(ctx.cfg("validate_min_recall", 0.05))
        if best <= 0.0:
            return False, "no strategy retrieved any gold document — baseline is not usable"
        if best < floor:
            return False, (f"best recall@k {best:.3f} is below the usable floor {floor:.2f} "
                           f"(set config['validate_min_recall'] to override)")
        return True, f"baseline established: {summary.get('best')} @ {best:.3f}"


def _search_ids(ctx: ExploreContext, mode: str, query: str, k: int) -> set:
    try:
        return set(ctx.session.search(query, top_k=k, mode=mode).ids())
    except Exception:
        return set()


def _best(strat_counts: dict) -> str:
    return max(strat_counts, key=lambda s: (strat_counts[s][0] / strat_counts[s][1]
                                            if strat_counts[s][1] else 0.0))


def _rates(strat_counts: dict) -> dict:
    return {s: round(v[0] / v[1], 3) if v[1] else None for s, v in strat_counts.items()}


def _report_md(ctx: ExploreContext, res: dict) -> str:
    lines = ["# Exploration validation report", "",
             f"Grounded synthetic queries: **{res['n']}**  ·  recall@{res['k']}", "",
             "## Recall@k by strategy",
             "| strategy | recall |", "|---|---|"]
    for s, r in sorted(res["recall_at_k"].items(), key=lambda x: -x[1]):
        star = "  ⭐" if s == res["best_overall"] else ""
        lines.append(f"| {s} | {r:.3f}{star} |")
    lines += ["", "## Best strategy per cluster", "| cluster | best |", "|---|---|"]
    for c, s in sorted(res["cluster_best"].items()):
        lines.append(f"| {c} | {s} |")
    lines += ["", "## Recall@k by difficulty", "| difficulty | " +
              " | ".join(ValidateStage.STRATEGIES) + " |",
              "|---|" + "---|" * len(ValidateStage.STRATEGIES)]
    for d, rates in res["by_difficulty"].items():
        lines.append(f"| {d} | " + " | ".join(str(rates.get(s)) for s in
                     ValidateStage.STRATEGIES) + " |")
    lines += ["", "_Baseline for keep-if-better gating: later tunings (router, codegen) must "
              "beat these recall numbers on the same queries to be kept._"]
    return "\n".join(lines)
