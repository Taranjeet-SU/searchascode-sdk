"""Milvus-lite adapter — in-process (local file), no server.

Uses the embedded Milvus-lite engine via ``MilvusClient(uri=<file>)``. Dense is
native (COSINE); keyword/regex/hybrid are served by a composed MemoryStore. String
doc ids are stored as the VARCHAR primary key.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Optional, Sequence

from ..errors import MissingDependencyError
from ..types import Capabilities, Document, Hit, ResultSet
from .base import VectorStore
from .memory import MemoryStore


class MilvusStore(VectorStore):
    backend = "milvus"

    def __init__(self, dim: int, collection: str = "sac", uri: Optional[str] = None, **_: Any):
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as e:  # pragma: no cover - optional dep
            raise MissingDependencyError("pymilvus", extra="search-as-code[milvus]") from e
        self.dim = int(dim)
        self.collection = collection
        # milvus-lite treats the uri as a data dir it will create -> must NOT pre-exist
        self._uri = uri or os.path.join(tempfile.mkdtemp(prefix="sac_milvus_"), "milvus.db")
        self._client = MilvusClient(uri=self._uri)
        self._mem = MemoryStore()
        if self._client.has_collection(collection):
            self._client.drop_collection(collection)
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        idx = self._client.prepare_index_params()
        idx.add_index(field_name="vector", metric_type="COSINE", index_type="AUTOINDEX")
        self._client.create_collection(collection, schema=schema, index_params=idx)

    def capabilities(self) -> Capabilities:
        return Capabilities(dense=True, keyword=True, hybrid=True, regex=True,
                            server_side_embedding=False, native_rerank=False,
                            metadata_filter=True)

    def upsert(self, docs: Sequence[Document]) -> None:
        rows = [{"id": str(d.id), "vector": list(d.vector)} for d in docs if d.vector is not None]
        B = 2000
        for i in range(0, len(rows), B):
            self._client.insert(self.collection, data=rows[i:i + B])
        self._mem.upsert(docs)

    def query_vector(self, vector, top_k=10, flt=None) -> ResultSet:
        from ..filters import matches
        res = self._client.search(self.collection, data=[list(vector)], anns_field="vector",
                                  limit=top_k * (8 if flt else 1), output_fields=["id"])
        hits = []
        for row in res[0]:
            did = row["id"] if isinstance(row, dict) else row.id
            score = row["distance"] if isinstance(row, dict) else row.distance
            doc = self._mem._docs.get(str(did))
            if doc is None or (flt and not matches(doc.metadata, flt)):
                continue
            hits.append(Hit(id=str(did), score=float(score), document=doc, store=self.backend))
            if len(hits) >= top_k:
                break
        return ResultSet(hits)

    def query_keyword(self, text, top_k=10, flt=None) -> ResultSet:
        return self._mem.query_keyword(text, top_k=top_k, flt=flt)

    def query_regex(self, pattern, top_k=10, flt=None) -> ResultSet:
        return self._mem.query_regex(pattern, top_k=top_k, flt=flt)

    def query_hybrid(self, vector, text, top_k=10, flt=None, alpha=0.5) -> ResultSet:
        dense = self.query_vector(vector, top_k=top_k * 4, flt=flt)
        kw = self.query_keyword(text, top_k=top_k * 4, flt=flt)
        from ..primitives import fuse
        return fuse([dense, kw], weights=[alpha, 1 - alpha]).top(top_k)

    def get(self, ids: Sequence[str]) -> list[Document]:
        return self._mem.get(ids)

    def count(self) -> int:
        return self._mem.count()
