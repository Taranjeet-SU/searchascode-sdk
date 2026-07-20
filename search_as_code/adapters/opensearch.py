"""OpenSearch adapter. ``pip install 'search-as-code[opensearch]'`` (opensearch-py).

Implements the full retrieval surface OpenSearch does natively:
* dense kNN (HNSW ``knn_vector``) → ``query_vector``
* BM25 full-text (``match``) → ``query_keyword``
* hybrid → dense + BM25 fused with RRF (portable; no server pipeline needed)
* ``regexp`` exact/pattern search → ``query_regex``
* metadata filtering via the portable filter dialect → OpenSearch ``bool`` filter
* aggregations (``aggregate``) for the analysis-class primitives

Scores are returned larger-is-better (OpenSearch already does this for both
_score and kNN similarity), so no conversion is needed.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..filters import normalize
from ..types import Capabilities, Document, Hit, ResultSet
from .base import VectorStore

_RANGE_OPS = {"$gt": "gt", "$gte": "gte", "$lt": "lt", "$lte": "lte"}


class OpenSearchStore(VectorStore):
    backend = "opensearch"

    def __init__(
        self,
        index: str,
        dim: Optional[int] = None,
        hosts: Optional[list] = None,
        host: str = "localhost",
        port: int = 9200,
        use_ssl: bool = False,
        http_auth: Optional[tuple] = None,
        space_type: str = "cosinesimil",
        text_field: str = "text",
        vector_field: str = "vector",
        **client_kwargs: Any,
    ):
        try:
            from opensearchpy import OpenSearch
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError("pip install opensearch-py") from e
        self.index = index
        self.dim = dim
        self.text_field = text_field
        self.vector_field = vector_field
        self._space = space_type
        self.client = OpenSearch(
            hosts=hosts or [{"host": host, "port": port}],
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_show_warn=False,
            **client_kwargs,
        )
        if dim is not None:
            self.ensure_index(dim)

    # ---- index lifecycle -------------------------------------------------
    def ensure_index(self, dim: int) -> None:
        if self.client.indices.exists(index=self.index):
            return
        body = {
            "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {
                "properties": {
                    self.vector_field: {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": "hnsw",
                            "engine": "lucene",
                            "space_type": self._space,
                        },
                    },
                    self.text_field: {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 32766}},
                    },
                }
            },
        }
        self.client.indices.create(index=self.index, body=body)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            dense=True, keyword=True, hybrid=True, regex=True,
            metadata_filter=True, native_rerank=False, server_side_embedding=False,
        )

    # ---- ingestion -------------------------------------------------------
    def upsert(self, docs: Sequence[Document]) -> None:
        from opensearchpy.helpers import bulk

        actions = []
        for d in docs:
            src: dict[str, Any] = {**(d.metadata or {})}
            if d.text is not None:
                src[self.text_field] = d.text
            if d.vector is not None:
                src[self.vector_field] = list(d.vector)
            actions.append({"_index": self.index, "_id": d.id, "_source": src})
        if actions:
            bulk(self.client, actions, refresh=True)

    # ---- filter translation ---------------------------------------------
    def _to_filter(self, flt: Optional[dict]) -> list[dict]:
        if not flt:
            return []
        clauses: list[dict] = []
        for field, cond in normalize(flt).items():
            if field.startswith("$"):
                continue  # nested and/or omitted in reference adapter
            for op, val in cond.items():
                if op == "$eq":
                    clauses.append({"term": {field: val}})
                elif op == "$ne":
                    clauses.append({"bool": {"must_not": {"term": {field: val}}}})
                elif op == "$in":
                    clauses.append({"terms": {field: list(val)}})
                elif op == "$nin":
                    clauses.append({"bool": {"must_not": {"terms": {field: list(val)}}}})
                elif op in _RANGE_OPS:
                    clauses.append({"range": {field: {_RANGE_OPS[op]: val}}})
        return clauses

    def _hits(self, resp: dict) -> ResultSet:
        out = []
        for h in resp["hits"]["hits"]:
            src = dict(h.get("_source", {}))
            text = src.pop(self.text_field, None)
            src.pop(self.vector_field, None)
            out.append(
                Hit(
                    id=str(h["_id"]),
                    score=float(h["_score"]) if h.get("_score") is not None else 0.0,
                    document=Document(id=str(h["_id"]), text=text, metadata=src),
                    store=self.backend,
                )
            )
        return ResultSet(out)

    # ---- retrieval primitives -------------------------------------------
    def query_vector(self, vector, top_k=10, flt=None) -> ResultSet:
        knn = {"vector": list(vector), "k": top_k}
        fclauses = self._to_filter(flt)
        if fclauses:
            knn["filter"] = {"bool": {"must": fclauses}}
        body = {"size": top_k, "query": {"knn": {self.vector_field: knn}}}
        return self._hits(self.client.search(index=self.index, body=body))

    def query_keyword(self, text, top_k=10, flt=None) -> ResultSet:
        must = [{"match": {self.text_field: text}}]
        body = {"size": top_k, "query": {"bool": {"must": must, "filter": self._to_filter(flt)}}}
        return self._hits(self.client.search(index=self.index, body=body))

    def query_hybrid(self, vector, text, top_k=10, flt=None, alpha=0.5) -> ResultSet:
        # Portable hybrid: run both, fuse with RRF in-SDK (no server pipeline needed).
        dense = self.query_vector(vector, top_k=top_k * 4, flt=flt)
        kw = self.query_keyword(text, top_k=top_k * 4, flt=flt)
        from ..primitives import fuse

        return fuse([dense, kw], weights=[alpha, 1 - alpha]).top(top_k)

    def query_regex(self, pattern, top_k=10, flt=None) -> ResultSet:
        # regexp on the whole-text keyword subfield; OpenSearch anchors to the full
        # value, so callers wrap with .* for substring (e.g. r".*def search.*").
        body = {
            "size": top_k,
            "query": {"bool": {
                "must": [{"regexp": {f"{self.text_field}.keyword": {"value": pattern, "flags": "ALL"}}}],
                "filter": self._to_filter(flt),
            }},
        }
        return self._hits(self.client.search(index=self.index, body=body))

    # ---- analysis-class primitive ---------------------------------------
    def aggregate(self, aggs: dict, flt: Optional[dict] = None, size: int = 0) -> dict:
        """Run a native OpenSearch aggregation (facets, stats, group-by)."""
        body: dict[str, Any] = {"size": size, "aggs": aggs}
        fclauses = self._to_filter(flt)
        if fclauses:
            body["query"] = {"bool": {"filter": fclauses}}
        return self.client.search(index=self.index, body=body).get("aggregations", {})

    # ---- fetch / admin ---------------------------------------------------
    def get(self, ids: Sequence[str]) -> list[Document]:
        if not ids:
            return []
        resp = self.client.mget(index=self.index, body={"ids": list(ids)})
        out = []
        for d in resp["docs"]:
            if not d.get("found"):
                continue
            src = dict(d.get("_source", {}))
            text = src.pop(self.text_field, None)
            src.pop(self.vector_field, None)
            out.append(Document(id=str(d["_id"]), text=text, metadata=src))
        return out

    def delete(self, ids: Sequence[str]) -> None:
        for i in ids:
            try:
                self.client.delete(index=self.index, id=i, refresh=True)
            except Exception:
                pass

    def count(self) -> int:
        return int(self.client.count(index=self.index)["count"])
