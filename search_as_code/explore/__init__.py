"""``sac.explore`` — the corpus **exploration / onboarding phase**.

Run once when SAC is installed against a corpus. It samples the data, profiles it
(schema + content-type mix + LLM characterization), and — as later stages land —
induces a domain ontology, generates synthetic queries, trains a primitive router,
and mines templates/few-shots. Everything it learns is written to a versioned
:class:`ProfilePack` that a ``Session`` loads at query time.

    import search_as_code as sac
    s = sac.Session("opensearch", index="docs", ...)
    pack = sac.explore(s, out="docs_pack/")     # onboarding
    print(pack.report())

The pipeline is resumable (each stage writes its own artifact), drift-aware (re-runs
when the corpus fingerprint changes), and validate-before-keep (a tuning is kept only
if it beats baseline).
"""

from .engine import (
    ExploreContext,
    Explorer,
    Stage,
    corpus_fingerprint,
    default_pipeline,
    explore,
    run_pipeline,
)
from .pack import ProfilePack
from .report import write_csv_report
from .router import TemplateRouter
from .templates import TEMPLATE_NAMES

__all__ = [
    "explore",
    "Explorer",
    "ProfilePack",
    "Stage",
    "ExploreContext",
    "default_pipeline",
    "run_pipeline",
    "corpus_fingerprint",
    "write_csv_report",
    "TemplateRouter",
    "TEMPLATE_NAMES",
]
