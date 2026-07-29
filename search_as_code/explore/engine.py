"""The exploration engine — a resumable, validate-before-keep stage runner.

``explore(session, out=...)`` walks an ordered list of :class:`Stage` objects. Each
stage reads/writes artifacts on the :class:`ProfilePack`. The engine:

- **skips** stages already done (artifacts present) unless ``force`` or the corpus
  fingerprint changed (drift);
- runs a stage, then calls its ``validate`` gate — a stage whose output does not beat
  baseline is recorded as ``rejected`` and its artifacts are *not* trusted downstream
  (the honesty rule: keep only tunings that help);
- records status/timing/summary for every stage in the manifest, and never lets one
  stage's failure abort the rest.
"""

from __future__ import annotations

import abc
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .pack import ProfilePack


@dataclass
class ExploreContext:
    """Everything a stage needs: the live Session, the pack to write to, and config."""

    session: Any                     # search_as_code.Session
    pack: ProfilePack
    config: dict = field(default_factory=dict)

    @property
    def store(self):
        return self.session.store

    @property
    def embedder(self):
        return self.session.embedder

    @property
    def generator(self):
        return self.session.generator

    def cfg(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


class Stage(abc.ABC):
    """One unit of exploration. Subclasses declare what they produce/require."""

    name: str = "stage"
    produces: list[str] = []          # artifact filenames written on success
    requires: list[str] = []          # names of stages that must be ``ok`` first

    @abc.abstractmethod
    def run(self, ctx: ExploreContext) -> dict:
        """Do the work, write artifacts, return a small summary dict for the manifest."""

    def validate(self, ctx: ExploreContext, summary: dict) -> tuple[bool, str]:
        """Gate: return (keep, reason). Default keeps everything. Learning stages
        override this to reject output that does not beat baseline."""
        return True, ""


def corpus_fingerprint(store, k: int = 12) -> str:
    """Cheap, stable signature of the corpus so we can detect drift. Combines the
    doc count with a hash of a small deterministic-ish text sample."""
    try:
        count = store.count()
    except Exception:
        count = -1
    texts = []
    try:
        for d in store.sample(k):
            texts.append((d.text or "")[:120])
    except Exception:
        pass
    h = hashlib.sha1(("|".join(sorted(texts))).encode("utf-8", "ignore")).hexdigest()[:12]
    return f"n{count}-{h}"


def run_pipeline(session, pack: ProfilePack, stages: Optional[list[Stage]] = None, *,
                 force: bool = False, config: Optional[dict] = None) -> ProfilePack:
    """Run the stage pipeline into ``pack`` (resumable, drift-aware, validate-before-keep)."""
    ctx = ExploreContext(session=session, pack=pack, config=config or {})
    fp = corpus_fingerprint(session.store)
    drift = pack.fingerprint_changed(fp)
    pack.set_fingerprint(fp)
    if stages is None:
        stages = default_pipeline()

    done: set[str] = set()
    for stage in stages:
        missing = [r for r in stage.requires if r not in done]
        if missing:
            pack.record_stage(stage.name, "skipped",
                              note=f"unmet requires: {', '.join(missing)}")
            continue
        if pack.is_done(stage.name) and not force and not drift:
            done.add(stage.name)
            continue

        t0 = time.time()
        try:
            summary = stage.run(ctx)
            keep, reason = stage.validate(ctx, summary)
            secs = time.time() - t0
            if keep:
                pack.record_stage(stage.name, "ok", seconds=secs, summary=summary,
                                  artifacts=list(stage.produces), note=reason)
                done.add(stage.name)
            else:
                pack.record_stage(stage.name, "rejected", seconds=secs, summary=summary,
                                  note=reason or "did not beat baseline")
        except NotImplementedError as e:
            pack.record_stage(stage.name, "planned", seconds=time.time() - t0, note=str(e))
        except Exception as e:  # never let one stage abort the pipeline
            pack.record_stage(stage.name, "error", seconds=time.time() - t0,
                              note=f"{type(e).__name__}: {e}")
    return pack


class Explorer:
    """Handle returned by :func:`explore`. Wraps the ProfilePack (attribute access is
    delegated) and adds the training API for the template router:

        explorer = sac.explore(session, out="pack/")
        explorer.dataset(n=5000, rephrases=2, label_rerank=True, workers=6)  # atomic, sharded
        explorer.set_model("hist_gb", max_iter=400)                          # swappable head
        m = explorer.train(cv=5)                                             # -> accuracy
        tmpl = explorer.route("part number for AGFC019")                     # predicted template

    ``fit(...)`` is a convenience that runs ``dataset`` then ``train`` in one call.
    """

    def __init__(self, session, pack: ProfilePack, config: Optional[dict] = None):
        self.session = session
        self.pack = pack
        self.config = config or {}
        self.router = None
        self._model_spec = "hist_gb"
        self._model_params: dict = {}
        self._dataset = None

    def __getattr__(self, name):        # delegate is_done/read_json/report/... to the pack
        return getattr(self.pack, name)

    # ---- training API ----------------------------------------------------
    def set_model(self, model="hist_gb", **params) -> "Explorer":
        """Choose the router head: a MODEL_REGISTRY name ('hist_gb','logreg','random_forest',
        'mlp'), a factory callable, or an estimator instance. Extra kwargs go to the model."""
        self._model_spec = model
        self._model_params = params
        return self

    def dataset(self, *, n=5000, rephrases=2, k=10, P=25, label_llm=False, label_rerank=False,
                workers=1, batch_size=256, resume=True, queries=None, progress_every=1):
        """Build (or resume/load) the atomic sharded router dataset — generate/collect queries,
        embed on GPU, label against every template, persist per-batch shards. See training.py."""
        from .training import build_dataset
        self._dataset = build_dataset(
            self, n=n, rephrases=rephrases, k=k, P=P, label_llm=label_llm,
            label_rerank=label_rerank, workers=workers, batch_size=batch_size,
            resume=resume, queries=queries, progress_every=progress_every)
        return self._dataset

    def train(self, cv: int = 5, **model_params):
        """Train the chosen model on the built dataset; save router.pkl + router_meta.json."""
        from .training import load_dataset, train_router_model
        if self._dataset is None:
            self._dataset = load_dataset(self.pack)
        params = {**self._model_params, **model_params}
        model, metrics = train_router_model(self._dataset, self._model_spec, cv=cv, **params)
        if model is not None:
            from .router import TemplateRouter
            emb_dim = self._dataset.X.shape[1] - 8 if len(self._dataset) else 0
            router = TemplateRouter(model, classes=sorted(set(self._dataset.y) - {"none"}),
                                    emb_dim=emb_dim, metrics=metrics)
            router.save(self.pack.path("router.pkl"))
            self.router = router
        self.pack.write_json("router_meta.json", metrics)
        self.pack.record_stage("router", "ok" if metrics.get("cv_accuracy") is not None
                               else "rejected", summary={
                                   "n": metrics.get("n"), "solved": metrics.get("solved"),
                                   "cv_acc": metrics.get("cv_accuracy"),
                                   "vs_fixed": metrics.get("router_lift_over_fixed")},
                               artifacts=["router.pkl", "router_meta.json"])
        return metrics

    def fit(self, queries=None, n: int = 5000, rephrases: int = 2, k: int = 10,
            P: int = 25, label_llm: bool = False, label_rerank: bool = False,
            workers: int = 1, batch_size: int = 256, model=None, cv: int = 5,
            progress_every: int = 1):
        """Convenience: dataset(...) then train(...). Returns the training metrics."""
        if model is not None:
            self.set_model(model)
        self.dataset(n=n, rephrases=rephrases, k=k, P=P, label_llm=label_llm,
                     label_rerank=label_rerank, workers=workers, batch_size=batch_size,
                     queries=queries, progress_every=progress_every)
        return self.train(cv=cv)

    def route(self, query: str) -> str:
        if self.router is None:
            from .router import TemplateRouter
            p = self.pack.path("router.pkl")
            self.router = TemplateRouter.load(p) if p.exists() else TemplateRouter()
        emb = self.session.embedder.embed([query])[0]
        return self.router.predict(query, emb)


def explore(session, out: str, stages: Optional[list[Stage]] = None, *,
            force: bool = False, config: Optional[dict] = None) -> "Explorer":
    """Run the exploration pipeline over ``session``'s corpus into a ProfilePack at ``out``,
    and return an :class:`Explorer` (delegates to the pack; adds ``.fit()``/``.route()``)."""
    pack = ProfilePack.open(out)
    run_pipeline(session, pack, stages, force=force, config=config)
    return Explorer(session, pack, config=config)


def default_pipeline() -> list[Stage]:
    """The full seven-stage pipeline. Stages not yet implemented raise
    NotImplementedError and are recorded as ``planned`` — the pipeline still runs
    end-to-end and the pack shows the roadmap."""
    from .stages import (
        CodegenStage,
        CrossDocStage,
        OntologyStage,
        ProfileStage,
        RouterStage,
        SampleStage,
        SynthesizeStage,
        ValidateStage,
    )

    return [
        SampleStage(),
        ProfileStage(),
        OntologyStage(),
        CrossDocStage(),
        SynthesizeStage(),
        RouterStage(),
        CodegenStage(),
        ValidateStage(),
    ]
