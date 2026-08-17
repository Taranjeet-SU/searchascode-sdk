"""Core data model shared across every adapter and primitive.

The whole point of search-as-code is that agent-generated code manipulates a
*single* set of types regardless of which vector database is underneath.  These
types are that lingua franca.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator, Optional


@dataclass(slots=True)
class Document:
    """A stored item. ``vector`` and ``text`` are both optional so the same type
    works for dense, sparse, and metadata-only records."""

    id: str
    text: Optional[str] = None
    vector: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_metadata(self, **kw: Any) -> "Document":
        md = {**self.metadata, **kw}
        return replace(self, metadata=md)


@dataclass(slots=True)
class Hit:
    """A single retrieved result.

    ``score`` is normalized so that *larger is better* for every adapter, even
    those that natively return distances.  ``query`` records which fan-out query
    produced the hit (used by fusion/dedup); ``store`` records the source
    backend (used when searching across multiple stores).
    """

    id: str
    score: float
    document: Optional[Document] = None
    query: Optional[str] = None
    store: Optional[str] = None

    @property
    def text(self) -> Optional[str]:
        return self.document.text if self.document else None

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata if self.document else {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)


class ResultSet(list):
    """A list of :class:`Hit` with chainable, model-free helpers.

    Every primitive returns a ``ResultSet`` so agent code can fluently compose
    ``store.search(...).dedup().top(5).to_evidence()`` without dragging the raw
    payloads back through the model context.

    ``.info`` carries per-primitive side signals (``consensus``'s agreement/votes,
    ``degraded`` fallback reasons). It is **propagated across chained calls**: signals used
    to be attached as ad-hoc attributes, and top/dedup/where construct a NEW ResultSet, so
    they vanished the moment agent code chained anything (SDK-C13).
    """

    #: side signals; see class docstring. Chained calls carry this forward.
    info: dict

    def __init__(self, *args, info: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.info = dict(info or {})

    def _derive(self, hits) -> "ResultSet":
        """A new ResultSet from ``hits`` that keeps this one's side signals."""
        return ResultSet(hits, info=self.info)

    # Documented consensus() signals, read from .info so they survive chaining while the
    # attribute spelling the prompt surface teaches (`cons.agreement`) keeps working.
    @property
    def agreement(self) -> float:
        return float(self.info.get("agreement", 0.0))

    @property
    def votes(self) -> dict:
        return dict(self.info.get("votes", {}))

    @property
    def n_lists(self) -> int:
        return int(self.info.get("n_lists", 0))

    @property
    def degraded(self) -> dict:
        """``{reason: count}`` of silent fallbacks taken to produce this result (LEG-5)."""
        return dict(self.info.get("degraded", {}))

    def mark_degraded(self, reason: str, n: int = 1) -> "ResultSet":
        """Record that a strategy fell back instead of running as intended.

        The audit's finding was that measurement could not distinguish "this strategy lost"
        from "this strategy crashed", because ~49 bare `except Exception` blocks in the SDK
        swallow and continue. Keep the fallback, but COUNT it (LEG-5).
        """
        d = dict(self.info.get("degraded", {}))
        d[reason] = d.get(reason, 0) + n
        self.info["degraded"] = d
        return self

    def top(self, k: int) -> "ResultSet":
        return self._derive(sorted(self, key=lambda h: h.score, reverse=True)[:k])

    def ids(self) -> list[str]:
        return [h.id for h in self]

    def texts(self) -> list[str]:
        return [h.text or "" for h in self]

    def where(self, predicate: Callable[[Hit], bool]) -> "ResultSet":
        return self._derive(h for h in self if predicate(h))

    def dedup(self, key: Optional[Callable[[Hit], Any]] = None) -> "ResultSet":
        """Keep the highest-scoring hit per key (defaults to hit id)."""
        keyfn = key or (lambda h: h.id)
        best: dict[Any, Hit] = {}
        for h in self:
            k = keyfn(h)
            if k not in best or h.score > best[k].score:
                best[k] = h
        return self._derive(best.values()).top(len(best))

    def to_evidence(
        self,
        fields: Optional[Iterable[str]] = None,
        max_chars: int = 500,
    ) -> list[dict[str, Any]]:
        """Compact, context-friendly representation to hand back to the model.

        This is the boundary between "bulky state kept in the sandbox" and "the
        few structured facts the model actually needs to see."
        """
        out: list[dict[str, Any]] = []
        for h in self:
            row: dict[str, Any] = {"id": h.id, "score": round(h.score, 4)}
            if h.text is not None:
                row["text"] = h.text[:max_chars]
            if fields is None:
                if h.metadata:
                    row["metadata"] = h.metadata
            else:
                for f in fields:
                    if f in h.metadata:
                        row[f] = h.metadata[f]
            out.append(row)
        return out

    def __iter__(self) -> Iterator[Hit]:  # typing aid
        return super().__iter__()


@dataclass(slots=True)
class Capabilities:
    """What a backend can do natively.  The primitive layer reads this to decide
    when to emulate a feature in-SDK so agent code stays portable."""

    dense: bool = True
    keyword: bool = False          # BM25 / full-text
    hybrid: bool = False           # server-side dense+sparse fusion
    regex: bool = False            # exact / pattern / operator search (code-friendly)
    multi_vector: bool = False     # late-interaction (ColBERT/ColPali) MaxSim
    server_side_embedding: bool = False
    native_rerank: bool = False
    metadata_filter: bool = True
    max_top_k: Optional[int] = None
