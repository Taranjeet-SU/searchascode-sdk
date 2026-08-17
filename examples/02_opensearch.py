"""OpenSearch backend quickstart for search-as-code.

Shows the whole flow against a local OpenSearch: connect, ingest, and run every
retrieval mode through the SAME unified API you'd use for any other backend.

Prereqs:
    pip install -e '.[opensearch]'          # opensearch-py
    # a local OpenSearch on :9200 (tarball, security off, single node), e.g.
    #   cd opensearch-2.17.1 && OPENSEARCH_JAVA_OPTS="-Xms2g -Xmx2g" bin/opensearch

Run:
    python examples/opensearch_quickstart.py
"""

import time

import search_as_code as sac

INDEX = "sac_quickstart"

CORPUS = [
    {"id": "1", "text": "def search(query): return db.query(query)", "metadata": {"kind": "code", "year": 2024}},
    {"id": "2", "text": "reciprocal rank fusion merges ranked lists by rank position", "metadata": {"kind": "doc", "year": 2023}},
    {"id": "3", "text": "the QueryEngine class exposes a search() method for agents", "metadata": {"kind": "code", "year": 2024}},
    {"id": "4", "text": "vector databases power semantic retrieval for LLM agents", "metadata": {"kind": "doc", "year": 2025}},
]


def main() -> None:
    # Same Session API as every other backend — only the connect string changes.
    # HashEmbedder (default) keeps this dependency-free; swap in a real embedder
    # (e.g. sentence-transformers) for quality. dim must match the embedder.
    s = sac.Session("opensearch", index=INDEX, dim=256, hosts=[{"host": "127.0.0.1", "port": 9200}])
    s.store.client.indices.delete(index=INDEX, ignore=[404])
    s.store.ensure_index(256)

    s.add(CORPUS)
    time.sleep(1)  # let the refresh settle
    print("indexed:", s.store.count(), "docs\n")

    q = "how do agents search"
    print("dense   :", s.search(q, top_k=3).ids())
    print("keyword :", s.search("reciprocal rank fusion", top_k=3, mode="keyword").ids())
    print("hybrid  :", s.search("search method", top_k=3, mode="hybrid").ids())
    print("regex   :", s.search(r".*def search.*", top_k=3, mode="regex").ids())
    print("filtered:", s.search("search", top_k=5, mode="keyword", filter={"kind": "code"}).ids())

    # native aggregation (analysis-class primitive)
    agg = s.store.aggregate({"by_kind": {"terms": {"field": "kind.keyword"}}})
    print("aggregate:", [(b["key"], b["doc_count"]) for b in agg["by_kind"]["buckets"]])

    # compact, context-friendly evidence to hand back to a model
    print("\nevidence:", s.search(q, top_k=2).to_evidence(fields=["kind", "year"], max_chars=60))

    s.store.client.indices.delete(index=INDEX, ignore=[404])
    print("\ncleaned up.")


if __name__ == "__main__":
    main()
