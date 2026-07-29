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

from .._resilience import DEFAULT_BATCH_SIZE, chunked, with_retry
from ..errors import BackendError, InvalidArgumentError, MissingDependencyError
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
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **client_kwargs: Any,
    ):
        try:
            from opensearchpy import OpenSearch
        except ImportError as e:  # pragma: no cover - optional dep
            raise MissingDependencyError("opensearch-py", extra="search-as-code[opensearch]") from e
        self.index = index
        self.dim = dim
        self.text_field = text_field
        self.vector_field = vector_field
        self._space = space_type
        self.batch_size = batch_size
        # Lean on the client's native resilience for transient connection/timeout
        # failures (retry_on_timeout), and wrap our own calls in BackendError so
        # callers get one typed failure regardless of backend.
        self.client = OpenSearch(
            hosts=hosts or [{"host": host, "port": port}],
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=timeout,
            max_retries=max_retries,
            retry_on_timeout=True,
            **client_kwargs,
        )
        if dim is not None:
            self.ensure_index(dim)

    def _search(self, body: dict) -> dict:
        """One choke point for reads: wrap transport errors as BackendError."""
        try:
            return self.client.search(index=self.index, body=body)
        except Exception as e:  # opensearchpy transport/connection errors
            raise BackendError("opensearch search failed", index=self.index,
                               cause=type(e).__name__) from e

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
        if not actions:
            return
        # Batch large ingests so a single request can't blow past body-size limits,
        # and retry each batch with backoff on transient failures.
        batches = list(chunked(actions, self.batch_size))
        for i, batch in enumerate(batches):
            refresh = i == len(batches) - 1  # refresh once, on the final batch
            with_retry(bulk, self.client, batch, refresh=refresh,
                       backend="opensearch", op="bulk")

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
        return self._hits(self._search(body))

    def query_keyword(self, text, top_k=10, flt=None) -> ResultSet:
        must = [{"match": {self.text_field: text}}]
        body = {"size": top_k, "query": {"bool": {"must": must, "filter": self._to_filter(flt)}}}
        return self._hits(self._search(body))

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
        return self._hits(self._search(body))

    # ---- extended lexical primitives (OpenSearch-native) -----------------
    def query_phrase(self, text, top_k=10, flt=None, slop=0, field=None) -> ResultSet:
        """Ordered-phrase match with optional ``slop`` (proximity). Taxonomy:
        ``phrase_search`` / ``proximity_search`` / ``score_phrase_or_proximity``."""
        f = field or self.text_field
        body = {"size": top_k, "query": {"bool": {
            "must": [{"match_phrase": {f: {"query": text, "slop": slop}}}],
            "filter": self._to_filter(flt)}}}
        return self._hits(self._search(body))

    def query_fielded(self, text, fields, top_k=10, flt=None, type="best_fields") -> ResultSet:
        """Multi-field search with per-field boosts (``fielded_search`` /
        ``field_boost``). ``fields`` e.g. ``["title^2", "text"]``."""
        body = {"size": top_k, "query": {"bool": {
            "must": [{"multi_match": {"query": text, "fields": list(fields), "type": type}}],
            "filter": self._to_filter(flt)}}}
        return self._hits(self._search(body))

    def query_prefix(self, prefix, top_k=10, flt=None, field=None) -> ResultSet:
        """Match docs containing an indexed *term* beginning with ``prefix``
        (``prefix_search``). Defaults to the analyzed text field (term-level);
        pass ``field="<f>.keyword"`` for whole-value prefixing."""
        f = field or self.text_field
        body = {"size": top_k, "query": {"bool": {
            "must": [{"prefix": {f: prefix.lower()}}], "filter": self._to_filter(flt)}}}
        return self._hits(self._search(body))

    def query_wildcard(self, pattern, top_k=10, flt=None, field=None) -> ResultSet:
        """Glob-style wildcard match (``*``/``?``) on the keyword subfield
        (``wildcard_search``)."""
        f = field or f"{self.text_field}.keyword"
        body = {"size": top_k, "query": {"bool": {
            "must": [{"wildcard": {f: {"value": pattern}}}], "filter": self._to_filter(flt)}}}
        return self._hits(self._search(body))

    def query_fuzzy(self, text, top_k=10, flt=None, fuzziness="AUTO", field=None) -> ResultSet:
        """Edit-distance tolerant match (``fuzzy_search``); ``fuzziness`` is an int
        or ``"AUTO"``."""
        f = field or self.text_field
        body = {"size": top_k, "query": {"bool": {
            "must": [{"match": {f: {"query": text, "fuzziness": fuzziness}}}],
            "filter": self._to_filter(flt)}}}
        return self._hits(self._search(body))

    def more_like_this(self, text=None, ids=None, top_k=10, flt=None,
                       min_term_freq=1, min_doc_freq=1, fields=None) -> ResultSet:
        """Find-similar / recommendation via MLT from free ``text`` and/or seed
        ``ids`` already in the index (``recommendation_search``)."""
        mlt: dict[str, Any] = {"fields": fields or [self.text_field],
                               "min_term_freq": min_term_freq, "min_doc_freq": min_doc_freq}
        like: list[Any] = []
        if text:
            like.append(text)
        if ids:
            like.extend({"_index": self.index, "_id": str(i)} for i in ids)
        if not like:
            raise InvalidArgumentError("more_like_this needs text and/or ids")
        mlt["like"] = like
        body = {"size": top_k, "query": {"bool": {
            "must": [{"more_like_this": mlt}], "filter": self._to_filter(flt)}}}
        return self._hits(self._search(body))

    def random_sample(self, size=10, flt=None, seed=None) -> ResultSet:
        """Random (optionally seeded, reproducible) sample of the filtered corpus
        (``random_sample``)."""
        rnd: dict[str, Any] = {}
        if seed is not None:
            rnd = {"seed": seed, "field": "_seq_no"}
        body = {"size": size, "query": {"function_score": {
            "query": {"bool": {"filter": self._to_filter(flt)}},
            "random_score": rnd}}}
        return self._hits(self._search(body))

    def browse(self, top_k=10, flt=None, sort_field=None, after=None) -> ResultSet:
        """Query-less enumeration of the (filtered) corpus with cursor paging via
        ``search_after`` (``browse_or_scroll``). Sorts by ``sort_field`` or ``_id``."""
        sort = [{sort_field: "asc"}] if sort_field else [{"_seq_no": "asc"}]
        body: dict[str, Any] = {"size": top_k, "sort": sort,
                                "query": {"bool": {"filter": self._to_filter(flt)}}}
        if after is not None:
            body["search_after"] = after
        return self._hits(self._search(body))

    # ---- analysis-class primitive ---------------------------------------
    def aggregate(self, aggs: dict, flt: Optional[dict] = None, size: int = 0) -> dict:
        """Run a native OpenSearch aggregation (facets, stats, group-by)."""
        body: dict[str, Any] = {"size": size, "aggs": aggs}
        fclauses = self._to_filter(flt)
        if fclauses:
            body["query"] = {"bool": {"filter": fclauses}}
        return self._search(body).get("aggregations", {})

    def facet(self, field, size=20, flt=None) -> dict[str, int]:
        """Categorical value counts for ``field`` (``facet``). Returns {value: count}."""
        agg = self.aggregate({"f": {"terms": {"field": field, "size": size}}}, flt=flt)
        return {b["key"]: b["doc_count"] for b in agg.get("f", {}).get("buckets", [])}

    def count_distinct(self, field, flt=None) -> int:
        """Approximate distinct-value count for ``field`` (``count_distinct``)."""
        agg = self.aggregate({"c": {"cardinality": {"field": field}}}, flt=flt)
        return int(agg.get("c", {}).get("value", 0))

    def stats(self, field, flt=None) -> dict[str, float]:
        """min/max/avg/sum/count for a numeric ``field`` (``aggregate_statistics``)."""
        agg = self.aggregate({"s": {"stats": {"field": field}}}, flt=flt)
        return agg.get("s", {})

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

    def sample(self, n: int = 5, fields: Optional[Sequence[str]] = None) -> list[Document]:
        # Only pull the text field (+ any requested extras) — never the whole _source: on
        # wide docs the full source is ~15 KB/doc, so a large sample can be many MB and time
        # out. The vector is always excluded.
        want = list(fields) if fields else [self.text_field]
        body = {"size": n, "_source": want,
                "query": {"function_score": {"query": {"match_all": {}}, "random_score": {}}}}
        out = []
        for h in self.client.search(index=self.index, body=body)["hits"]["hits"]:
            src = dict(h.get("_source", {}))
            text = src.pop(self.text_field, None)
            src.pop(self.vector_field, None)
            out.append(Document(id=str(h["_id"]), text=text, metadata=src))
        return out

    def describe_schema(self) -> dict:
        """Fields -> ES types (+ a couple sample docs) so the agent knows the shape first."""
        m = self.client.indices.get_mapping(index=self.index)
        props = next(iter(m.values()))["mappings"].get("properties", {})

        def flatten(p, pre=""):
            out = {}
            for k, v in p.items():
                if v.get("type"):
                    out[pre + k] = v["type"]
                if "properties" in v:
                    out.update(flatten(v["properties"], pre + k + "."))
            return out

        fields = flatten(props)
        samples = self.sample(2)
        return {"index": self.index, "count": self.count(),
                "text_field": self.text_field, "vector_field": self.vector_field,
                "n_fields": len(fields), "fields": fields,
                "sample_text": [(d.text or "")[:200] for d in samples]}
