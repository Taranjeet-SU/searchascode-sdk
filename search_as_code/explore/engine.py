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
    delegated) and adds :meth:`fit` — learn the template router from labeled queries.

        explorer = sac.explore(session, out="pack/")
        m = explorer.fit(n=5000)          # label queries -> train router -> metrics
        print(m["cv_accuracy"])
        tmpl = explorer.route("part number for AGFC019")   # predicted template
    """

    def __init__(self, session, pack: ProfilePack, config: Optional[dict] = None):
        self.session = session
        self.pack = pack
        self.config = config or {}
        self.router = None

    def __getattr__(self, name):        # delegate is_done/read_json/report/... to the pack
        return getattr(self.pack, name)

    def fit(self, queries=None, n: int = 5000, rephrases: int = 2, k: int = 10,
            P: int = 25, label_llm: bool = False, label_rerank: bool = False,
            workers: int = 1, progress_every: int = 100):
        """Label queries by which template retrieves their gold doc, then train the router.

        ``queries``: iterable of ``{"query","gold_id"}`` (or ``(query, gold_id)``). If None,
        generate ~``n`` grounded synthetic queries (each + ``rephrases`` paraphrases) from the
        corpus sample. ``label_llm``/``label_rerank`` toggle the hyde/decompose pools and the
        cross-encoder pass during labeling (off by default — those cost an LLM/GPU call per
        query at label time). Returns a metrics dict and writes router.* to the pack.
        """
        from .fit import fit_router
        return fit_router(self, queries=queries, n=n, rephrases=rephrases, k=k, P=P,
                          label_llm=label_llm, label_rerank=label_rerank,
                          workers=workers, progress_every=progress_every)

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
