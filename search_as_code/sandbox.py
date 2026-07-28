"""Execution layer for search-as-code.

Agent-generated Python runs *here*, with a :class:`Session` injected as ``sac``
and the primitives in scope.  Bulky intermediate results stay in this namespace;
only what the code ``print``s or assigns to ``evidence`` is returned to the
model.  That boundary is the whole efficiency argument of code-mode retrieval.

``LocalExecutor`` is an in-process runner for development and tests.  It is NOT a
security boundary — untrusted code needs a real isolate (Docker / e2b / Pyodide)
behind the same :class:`Sandbox` interface.
"""

from __future__ import annotations

import abc
import contextlib
import io
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from . import primitives as P
from .session import Session
from .types import Document, Hit, ResultSet


@dataclass
class ExecResult:
    ok: bool
    evidence: Any = None
    stdout: str = ""
    error: Optional[str] = None
    state_keys: list[str] = field(default_factory=list)

    def for_model(self) -> dict[str, Any]:
        """The compact payload to feed back into the model context."""
        payload: dict[str, Any] = {"ok": self.ok}
        if self.evidence is not None:
            payload["evidence"] = self.evidence
        if self.stdout.strip():
            payload["stdout"] = self.stdout.strip()[:4000]
        if self.error:
            payload["error"] = self.error
        if self.state_keys:
            payload["state_keys"] = self.state_keys
        return payload


class Sandbox(abc.ABC):
    @abc.abstractmethod
    def run(self, code: str) -> ExecResult: ...


class LocalExecutor(Sandbox):
    """In-process executor. State persists across ``run`` calls via the shared
    Session, enabling multi-turn workflows without re-fetching."""

    def __init__(self, session: Session):
        self.session = session
        self._globals = self._build_namespace()

    def _build_namespace(self) -> dict[str, Any]:
        return {
            "sac": self.session,          # the harness handle
            "session": self.session,      # alias
            "Document": Document,
            "Hit": Hit,
            "ResultSet": ResultSet,
            # primitives available as bare functions too
            "fan_out": P.fan_out,
            "fuse": P.fuse,
            "rerank": P.rerank,
            "dedup": P.dedup,
            "freshness": P.freshness,
            "mmr": P.mmr,
            "expand": P.expand,
            "decompose": P.decompose,
            # scoring/shaping + consensus helpers the SAC prompt references
            "consensus": P.consensus,
            "confidence": P.confidence,
            "abstain": P.abstain,
            "normalize_scores": P.normalize_scores,
            "relative_score_fusion": P.relative_score_fusion,
            "diversity_quota": P.diversity_quota,
            "score_cutoff": P.score_cutoff,
            "__builtins__": _safe_builtins(),
        }

    def run(self, code: str) -> ExecResult:
        self._globals.pop("evidence", None)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<agent-code>", "exec"), self._globals)  # noqa: S102
        except Exception:
            return ExecResult(
                ok=False,
                stdout=buf.getvalue(),
                error=traceback.format_exc(limit=3),
                state_keys=self.session.state_keys(),
            )
        return ExecResult(
            ok=True,
            evidence=self._globals.get("evidence"),
            stdout=buf.getvalue(),
            state_keys=self.session.state_keys(),
        )


def _safe_builtins() -> dict[str, Any]:
    """A conservative builtins allowlist for the local executor.

    Real isolation belongs in the sandbox backend; this only trims the sharpest
    edges so a stray ``open``/``__import__`` in dev doesn't touch the host.
    """
    import builtins

    allow = {
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "len", "list", "map", "max", "min", "print", "range", "reversed",
        "round", "set", "sorted", "str", "sum", "tuple", "zip", "isinstance",
        "getattr", "hasattr", "True", "False", "None", "Exception",
    }
    return {k: getattr(builtins, k) for k in allow if hasattr(builtins, k)}
