"""Chroma adapter. ``pip install 'search-as-code[chroma]'``.

Chroma returns L2/cosine *distances* (smaller is better); we convert to a
larger-is-better similarity so scores are comparable across every backend.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, cast

from .._resilience import DEFAULT_BATCH_SIZE, chunked
from ..errors import InvalidFilterError, MissingDependencyError
from ..filters import normalize
from ..types import Capabilities, Document, Hit, ResultSet
from .base import VectorStore

_OP_MAP = {"$eq": "$eq", "$ne": "$ne", "$gt": "$gt", "$gte": "$gte",
           "$lt": "$lt", "$lte": "$lte", "$in": "$in", "$nin": "$nin"}


class ChromaStore(VectorStore):
    backend = "chroma"

    def __init__(self, collection: str = "sac", persist_path: Optional[str] = None,
                 batch_size: int = DEFAULT_BATCH_SIZE, **_: Any):
        try:
            import chromadb
        except ImportError as e:  # pragma: no cover - optional dep
            raise MissingDependencyError("chromadb", extra="search-as-code[chroma]") from e
        self._client = chromadb.PersistentClient(path=persist_path) if persist_path else chromadb.Client()
        self._col = self._client.get_or_create_collection(collection)
        self.batch_size = batch_size

    def capabilities(self) -> Capabilities:
        return Capabilities(dense=True, keyword=False, hybrid=False, metadata_filter=True)

    def upsert(self, docs: Sequence[Document]) -> None:
        docs = [d for d in docs if d.vector is not None]
        for batch in chunked(docs, self.batch_size):
            self._col.upsert(
                ids=[d.id for d in batch],
                # `docs` is filtered to vector-bearing documents above, but the comprehension
                # still types as list[list[float] | None] — make the narrowing explicit.
                # cast: chromadb's stub types `embeddings` invariantly, but a list of
                # per-document vectors is the documented input.
                embeddings=cast(Any, [list(d.vector or []) for d in batch]),
                documents=[d.text or "" for d in batch],
                metadatas=[d.metadata or {"_": ""} for d in batch],
            )

    def _to_where(self, flt: Optional[dict]) -> Optional[dict]:
        if not flt:
            return None
        clauses: list[dict[str, Any]] = []
        for field_name, cond in normalize(flt).items():
            if field_name == "$and":
                for sub in cond:
                    w = self._to_where(sub)
                    if w:
                        clauses.append(w)
                continue
            if field_name == "$or":
                subs = [w for w in (self._to_where(x) for x in cond) if w]
                if subs:
                    clauses.append(cast(dict, {"$or": subs}))
                continue
            if field_name.startswith("$"):
                # Never skip silently: an unsupported operator used to be dropped, so the
                # query ran UNFILTERED and over-returned — the same fail-open shape as
                # SDK-C2 on OpenSearch (issues.md ADP-3).
                raise InvalidFilterError(
                    "unsupported filter operator for the chroma backend",
                    op=field_name, backend="chroma",
                )
            clauses.append({field_name: {_OP_MAP[op]: v for op, v in cond.items() if op in _OP_MAP}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else cast(dict, {"$and": clauses})

    @staticmethod
    def _rows(res: Any, key: str, outer: bool) -> list:
        """One result column from a Chroma response, as a plain list.

        Chroma types every column as Optional and query() nests each one inside a
        per-query list, so direct indexing is not type-safe (and is a real IndexError
        when a column is omitted from `include`).
        """
        col = (res or {}).get(key)
        if not col:
            return []
        return list(col[0] or []) if outer else list(col)

    def query_vector(self, vector, top_k=10, flt=None) -> ResultSet:
        res = self._col.query(
            query_embeddings=cast(Any, [list(vector)]),
            n_results=top_k,
            where=self._to_where(flt),
            include=["documents", "metadatas", "distances"],
        )
        ids = self._rows(res, "ids", True)
        dists = self._rows(res, "distances", True)
        texts = self._rows(res, "documents", True)
        metas = self._rows(res, "metadatas", True)
        hits = []
        for i, _id in enumerate(ids):
            dist = dists[i] if i < len(dists) else 0.0
            hits.append(
                Hit(
                    id=str(_id),
                    score=1.0 / (1.0 + float(dist)),  # distance -> larger-is-better
                    document=Document(
                        id=str(_id),
                        text=texts[i] if i < len(texts) else None,
                        metadata=dict(metas[i]) if i < len(metas) and metas[i] else {},
                    ),
                    store=self.backend,
                )
            )
        return ResultSet(hits)

    def get(self, ids: Sequence[str]) -> list[Document]:
        res = self._col.get(ids=list(ids), include=["documents", "metadatas"])
        got = self._rows(res, "ids", False)
        texts = self._rows(res, "documents", False)
        metas = self._rows(res, "metadatas", False)
        return [
            Document(id=str(_id),
                     text=texts[i] if i < len(texts) else None,
                     metadata=dict(metas[i]) if i < len(metas) and metas[i] else {})
            for i, _id in enumerate(got)
        ]

    def delete(self, ids: Sequence[str]) -> None:
        self._col.delete(ids=list(ids))

    def count(self) -> int:
        return self._col.count()
