"""Exploration stages.

Implemented now: :class:`SampleStage` (stratified sample via clustering) and
:class:`ProfileStage` (content-type mix + LLM characterization). The remaining five
are typed stubs that raise ``NotImplementedError`` so the pipeline runs end-to-end and
the pack shows the roadmap; each is filled in as its own pass.
"""

from __future__ import annotations

import numpy as np

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
        return out[0] if isinstance(out, list) else str(out)
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


class SynthesizeStage(_PlannedStage):
    """Generate stratified easy/medium/hard queries grounded in the sample + ontology
    (no test leakage) to train the router and mine few-shots."""
    name = "synthesize"
    requires = ["profile"]
    produces = ["synth_queries.jsonl"]
    _todo = "stratified synthetic-query generation (port from phase4)"


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


class ValidateStage(_PlannedStage):
    """Measure retrieval on a held-out synth slice with vs without the pack; keep only
    tunings that beat baseline; emit a report."""
    name = "validate"
    requires = ["profile"]
    produces = ["validation.json", "REPORT.md"]
    _todo = "held-out validation + keep-if-better gating (next pass)"
