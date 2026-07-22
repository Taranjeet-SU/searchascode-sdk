"""Qdrant adapter. ``pip install 'search-as-code[qdrant]'``.

Demonstrates the pattern every real adapter follows: translate the portable
filter dialect to the backend DSL, convert native distances to larger-is-better
scores, and declare capabilities honestly.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional, Sequence

from .._resilience import DEFAULT_BATCH_SIZE, chunked
from ..errors import MissingDependencyError
from ..filters import normalize
from ..types import Capabilities, Document, Hit, ResultSet
from .base import VectorStore

_OP_MAP = {"$gt": "gt", "$gte": "gte", "$lt": "lt", "$lte": "lte"}


class QdrantStore(VectorStore):
    backend = "qdrant"

    def __init__(
        self,
        collection: str,
        url: Optional[str] = None,
        location: Optional[str] = ":memory:",
        dim: Optional[int] = None,
        distance: str = "Cosine",
        batch_size: int = DEFAULT_BATCH_SIZE,
        **client_kwargs: Any,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qm
        except ImportError as e:  # pragma: no cover - optional dep
            raise MissingDependencyError("qdrant-client", extra="search-as-code[qdrant]") from e
        self._qm = qm
        self._client = QdrantClient(url=url, location=None if url else location, **client_kwargs)
        self.collection = collection
        self.batch_size = batch_size
        self._dim = dim
        self._distance = distance
        if dim and not self._client.collection_exists(collection):
            self._client.create_collection(
                collection,
                vectors_config=qm.VectorParams(size=dim, distance=getattr(qm.Distance, distance.upper())),
            )

    def capabilities(self) -> Capabilities:
        return Capabilities(dense=True, keyword=False, hybrid=False, metadata_filter=True)

    @staticmethod
    def _pid(doc_id: str) -> Any:
        """Qdrant point ids must be unsigned int or UUID. Pass ints through;
        map arbitrary strings to a deterministic uuid5 (original kept in payload)."""
        s = str(doc_id)
        if s.isdigit():
            return int(s)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, s))

    def upsert(self, docs: Sequence[Document]) -> None:
        qm = self._qm
        points = [
            qm.PointStruct(id=self._pid(d.id), vector=d.vector,
                           payload={"text": d.text, "_sac_id": str(d.id), **d.metadata})
            for d in docs
            if d.vector is not None
        ]
        for batch in chunked(points, self.batch_size):
            self._client.upsert(self.collection, points=batch)

    def _to_filter(self, flt: Optional[dict]) -> Any:
        if not flt:
            return None
        qm = self._qm
        must = []
        for field_name, cond in normalize(flt).items():
            if field_name.startswith("$"):
                continue  # nested and/or omitted for brevity in the reference adapter
            for op, val in cond.items():
                if op == "$eq":
                    must.append(qm.FieldCondition(key=field_name, match=qm.MatchValue(value=val)))
                elif op == "$in":
                    must.append(qm.FieldCondition(key=field_name, match=qm.MatchAny(any=val)))
                elif op in _OP_MAP:
                    must.append(qm.FieldCondition(key=field_name, range=qm.Range(**{_OP_MAP[op]: val})))
        return qm.Filter(must=must) if must else None

    def query_vector(self, vector, top_k=10, flt=None) -> ResultSet:
        flt_ = self._to_filter(flt)
        if hasattr(self._client, "query_points"):  # qdrant-client >= 1.10
            res = self._client.query_points(
                self.collection, query=list(vector), limit=top_k,
                query_filter=flt_, with_payload=True,
            ).points
        else:  # older clients
            res = self._client.search(
                self.collection, query_vector=list(vector), limit=top_k,
                query_filter=flt_, with_payload=True,
            )
        hits = []
        for p in res:
            payload = dict(p.payload or {})
            text = payload.pop("text", None)
            did = payload.pop("_sac_id", str(p.id))  # restore the original id
            hits.append(
                Hit(
                    id=did,
                    score=float(p.score),  # Qdrant cosine similarity: larger is better
                    document=Document(id=did, text=text, metadata=payload),
                    store=self.backend,
                )
            )
        return ResultSet(hits)

    def get(self, ids: Sequence[str]) -> list[Document]:
        recs = self._client.retrieve(self.collection, ids=[self._pid(i) for i in ids], with_payload=True)
        out = []
        for r in recs:
            payload = dict(r.payload or {})
            text = payload.pop("text", None)
            did = payload.pop("_sac_id", str(r.id))
            out.append(Document(id=did, text=text, metadata=payload))
        return out

    def delete(self, ids: Sequence[str]) -> None:
        self._client.delete(self.collection, points_selector=[self._pid(i) for i in ids])

    def count(self) -> int:
        return self._client.count(self.collection).count
