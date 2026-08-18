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
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from . import primitives as P
from .session import Session
from .types import Document, Hit, ResultSet


class SandboxTimeout(RuntimeError):
    """Agent code exceeded the executor's wall-clock budget."""


class _CappedIO(io.StringIO):
    """StringIO that stops accepting writes past ``cap`` chars (SDK-C19: stdout used
    to accumulate unbounded; truncation happened only cosmetically in ``for_model``)."""

    def __init__(self, cap: int):
        super().__init__()
        self._cap = cap
        self.truncated = False

    def write(self, s: str) -> int:  # type: ignore[override]
        room = self._cap - self.tell()
        if room <= 0:
            self.truncated = True
            return len(s)
        if len(s) > room:
            self.truncated = True
            super().write(s[:room])
            return len(s)
        return super().write(s)


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
    def run(self, code: str, query: Optional[str] = None) -> ExecResult: ...


class LocalExecutor(Sandbox):
    """In-process executor. State persists across ``run`` calls via the shared
    Session and any variables the agent code defines — but the *injected* names
    (primitives, ``sac``, types) are re-bound on every run, so hop-1 code doing
    ``fuse = None`` cannot poison hop 2 (SDK-C19).

    Robustness (not security — see module docstring): a wall-clock ``timeout_s``
    enforced via a per-thread trace hook (interrupts Python-level loops; a single
    long C call can still overrun), and a ``max_stdout`` cap on captured output.
    """

    def __init__(self, session: Session, timeout_s: float = 20.0, max_stdout: int = 100_000):
        self.session = session
        self.timeout_s = timeout_s
        self.max_stdout = max_stdout
        self._injected = self._build_namespace()
        self._globals: dict[str, Any] = dict(self._injected)

    def _build_namespace(self) -> dict[str, Any]:
        session = self.session

        # expand/decompose need a generate callable; bind the session's generator so the
        # natural call the model writes — expand(query) — works instead of TypeError-ing.
        def expand(query: str, n: int = 4) -> list[str]:
            return P.expand(query, session._require_generator(), n=n)

        def decompose(query: str) -> list[str]:
            return P.decompose(query, session._require_generator())

        def rephrase(query: str) -> str:
            return P.rephrase(query, session._require_generator())

        def topics(query: str, n: int = 5) -> list[str]:
            return P.topics(query, session._require_generator(), n=n)

        def auto_filter(query: str, fields=None) -> dict:
            return P.auto_filter(query, session._require_generator(), fields=fields)

        return {
            "sac": self.session,          # the harness handle
            "session": self.session,      # alias
            "Document": Document,
            "Hit": Hit,
            "ResultSet": ResultSet,
            # primitives available as bare functions too — the FULL public set
            # (nine were missing; agent code calling an advertised name got NameError)
            "fan_out": P.fan_out,
            "fuse": P.fuse,
            "rrf": P.rrf,
            "rerank": P.rerank,
            "dedup": P.dedup,
            "freshness": P.freshness,
            "mmr": P.mmr,
            "expand": expand,
            "decompose": decompose,
            "rephrase": rephrase,
            "topics": topics,
            "auto_filter": auto_filter,
            "normalize_query": P.normalize_query,
            "rare_terms": P.rare_terms,
            "extract": P.extract,
            "content_type": P.content_type,
            "quality_filter": P.quality_filter,
            # scoring/shaping + consensus helpers the SAC prompt references
            "consensus": P.consensus,
            "confidence": P.confidence,
            "abstain": P.abstain,
            "score_cliff": P.score_cliff,
            "result_diversity": P.result_diversity,
            "max_similarity": P.max_similarity,
            "normalize_scores": P.normalize_scores,
            "relative_score_fusion": P.relative_score_fusion,
            "diversity_quota": P.diversity_quota,
            "score_cutoff": P.score_cutoff,
            "__builtins__": _safe_builtins(),
        }

    def run(self, code: str, query: Optional[str] = None) -> ExecResult:
        self._globals.update(self._injected)          # undo any poisoning of injected names
        if query is not None:
            self._globals["query"] = query            # the documented contract (surface.py)
        self._globals.pop("evidence", None)
        buf = _CappedIO(self.max_stdout)
        deadline = time.monotonic() + self.timeout_s

        def _trace(frame, event, arg):                # per-thread, unlike SIGALRM
            if time.monotonic() > deadline:
                raise SandboxTimeout(f"agent code exceeded {self.timeout_s:.0f}s wall clock")
            return _trace

        sys.settrace(_trace)
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
        finally:
            sys.settrace(None)
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
