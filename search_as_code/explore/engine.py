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


def explore(session, out: str, stages: Optional[list[Stage]] = None, *,
            force: bool = False, config: Optional[dict] = None) -> ProfilePack:
    """Run the exploration pipeline over ``session``'s corpus into a ProfilePack at ``out``.

    Parameters
    ----------
    session : Session   the bound corpus (store + embedder + optional generator).
    out     : str       directory for the ProfilePack (created if absent).
    stages  : list      ordered Stage instances; defaults to :func:`default_pipeline`.
    force   : bool       re-run every stage even if its artifacts exist.
    config  : dict       knobs passed through to stages (sample sizes, k, etc.).
    """
    pack = ProfilePack.open(out)
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
