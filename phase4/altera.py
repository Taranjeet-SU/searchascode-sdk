"""Altera sandbox connection helpers (INTERNAL — not for GitHub).

- OpenSearch reachable via SSH tunnel on http://localhost:8055 (-> remote :8050).
- Dense retrieval uses the fine-tuned embedder gaggi009/gte-alt-v1 (matches the
  `gte_altera` vectors in ft_document). Token via env HF_TOKEN.
- BM25 over ft_document content + altera_kg_v2 knowledge cards.
"""
from __future__ import annotations

import os
import threading
import time

import requests

OS_URL = os.environ.get("ALTERA_OS", "http://localhost:8055")
FT_DOC = "1_27_fluid_topics_clone__ft_document"
FT_TOPIC = "1_27_fluid_topics_clone__ft_topic"
KG = "altera_kg_v2"
FT_VECTOR = "text_vector_498724ac-55bd-11f1-9a70-0242ac12000agte_altera"
FTP = "1_27_fluid_topics_clone___ft_document___"  # field prefix in ft_document
# prefer the locally-curled copy (the HF hub download hangs on this network)
_LOCAL = os.path.join(os.path.dirname(__file__), "models", "gte-alt-v1")
EMB_MODEL = _LOCAL if os.path.exists(os.path.join(_LOCAL, "model.safetensors")) else "gaggi009/gte-alt-v1"



_sdk_embedder = None
_embed_lock = threading.Lock()


def embedder():
    """Use the STANDARD SDK robust custom-transformer embedder (handles gte-alt-v1's
    meta-buffer/GPU quirks). Keeps this script on the standard SDK, not a fork."""
    global _sdk_embedder
    if _sdk_embedder is None:
        from search_as_code.embeddings import get_embedder
        _sdk_embedder = get_embedder("transformers", model=EMB_MODEL, pooling="cls",
                                     token=os.environ.get("HF_TOKEN"))
    return _sdk_embedder


def embed(text: str):
    with _embed_lock:                              # one forward at a time (thread-safe)
        return embedder().embed([text])[0]


def _search(index: str, body: dict, retries: int = 4, timeout: int = 25):
    """Resilient OpenSearch query over the (sometimes flaky) SSH tunnel: retry with
    backoff, and degrade to [] rather than crashing the whole run on a tunnel blip."""
    for attempt in range(retries):
        try:
            r = requests.post(f"{OS_URL}/{index}/_search", json=body, timeout=timeout)
            r.raise_for_status()
            return r.json()["hits"]["hits"]
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return []


def dense(query: str, k: int = 10):
    """kNN over ft_document with the gte-alt-v1 query vector."""
    body = {"size": k, "query": {"knn": {FT_VECTOR: {"vector": embed(query), "k": k}}},
            "_source": [FTP + "ft_title", FTP + "content", FTP + "readerUrl", FTP + "documentId"]}
    return [_doc(h) for h in _search(FT_DOC, body)]


def bm25_doc(query: str, k: int = 10):
    body = {"size": k, "query": {"match": {FTP + "content": query}},
            "_source": [FTP + "ft_title", FTP + "content", FTP + "readerUrl", FTP + "documentId"]}
    return [_doc(h) for h in _search(FT_DOC, body)]


def bm25_kg(query: str, k: int = 10):
    """BM25 over the knowledge-graph cards (answer/evidence/content fields)."""
    body = {"size": k, "query": {"multi_match": {"query": query,
            "fields": ["answer", "evidence", "content", "doc_title", "facet"]}},
            "_source": ["answer", "evidence", "content", "doc_title", "docid", "family", "applies_to"]}
    out = []
    for h in _search(KG, body):
        s = h["_source"]
        txt = s.get("answer") or s.get("evidence") or s.get("content") or ""
        out.append({"id": h["_id"], "score": h["_score"], "title": s.get("doc_title", ""),
                    "text": txt, "url": f"docid:{s.get('docid')}", "meta": s})
    return out


def _doc(h):
    s = h["_source"]
    return {"id": h["_id"], "score": h["_score"], "title": s.get(FTP + "ft_title", ""),
            "text": s.get(FTP + "content", ""), "url": s.get(FTP + "readerUrl", ""),
            "docid": s.get(FTP + "documentId", "")}


def ping():
    r = requests.get(f"{OS_URL}/_cat/indices?h=index,docs.count", timeout=15)
    return r.text
