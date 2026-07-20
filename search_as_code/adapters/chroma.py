"""Chroma adapter. ``pip install 'search-as-code[chroma]'``.

Chroma returns L2/cosine *distances* (smaller is better); we convert to a
larger-is-better similarity so scores are comparable across every backend.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..filters import normalize
from ..types import Capabilities, Document, Hit, ResultSet
from .base import VectorStore

_OP_MAP = {"$eq": "$eq", "$ne": "$ne", "$gt": "$gt", "$gte": "$gte",
           "$lt": "$lt", "$lte": "$lte", "$in": "$in", "$nin": "$nin"}


class ChromaStore(VectorStore):
    backend = "chroma"

    def __init__(self, collection: str = "sac", persist_path: Optional[str] = None, **_: Any):
        try:
            import chromadb
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError("pip install 'search-as-code[chroma]'") from e
        self._client = chromadb.PersistentClient(path=persist_path) if persist_path else chromadb.Client()
        self._col = self._client.get_or_create_collection(collection)

    def capabilities(self) -> Capabilities:
        return Capabilities(dense=True, keyword=False, hybrid=False, metadata_filter=True)

    def upsert(self, docs: Sequence[Document]) -> None:
        docs = [d for d in docs if d.vector is not None]
        if not docs:
            return
        self._col.upsert(
            ids=[d.id for d in docs],
            embeddings=[d.vector for d in docs],
            documents=[d.text or "" for d in docs],
            metadatas=[d.metadata or {"_": ""} for d in docs],
        )

    def _to_where(self, flt: Optional[dict]) -> Optional[dict]:
        if not flt:
            return None
        clauses = []
        for field_name, cond in normalize(flt).items():
            if field_name.startswith("$"):
                continue
            clauses.append({field_name: {_OP_MAP[op]: v for op, v in cond.items() if op in _OP_MAP}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def query_vector(self, vector, top_k=10, flt=None) -> ResultSet:
        res = self._col.query(
            query_embeddings=[list(vector)],
            n_results=top_k,
            where=self._to_where(flt),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        ids = res["ids"][0]
        for i, _id in enumerate(ids):
            dist = res["distances"][0][i]
            hits.append(
                Hit(
                    id=_id,
                    score=1.0 / (1.0 + float(dist)),  # distance -> larger-is-better
                    document=Document(
                        id=_id,
                        text=res["documents"][0][i],
                        metadata=res["metadatas"][0][i] or {},
                    ),
                    store=self.backend,
                )
            )
        return ResultSet(hits)

    def get(self, ids: Sequence[str]) -> list[Document]:
        res = self._col.get(ids=list(ids), include=["documents", "metadatas"])
        return [
            Document(id=_id, text=res["documents"][i], metadata=res["metadatas"][i] or {})
            for i, _id in enumerate(res["ids"])
        ]

    def delete(self, ids: Sequence[str]) -> None:
        self._col.delete(ids=list(ids))

    def count(self) -> int:
        return self._col.count()
