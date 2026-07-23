"""Altera sandbox connection helpers (INTERNAL — not for GitHub).

- OpenSearch reachable via SSH tunnel on http://localhost:8055 (-> remote :8050).
- Dense retrieval uses the fine-tuned embedder gaggi009/gte-alt-v1 (matches the
  `gte_altera` vectors in ft_document). Token via env HF_TOKEN.
- BM25 over ft_document content + altera_kg_v2 knowledge cards.
"""
from __future__ import annotations

import os
import threading

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

_model = None
_tok = None


def embedder():
    """Load gte-alt-v1 via transformers directly on CPU (queries only).

    Avoids two failures of the SentenceTransformer path with this custom GTE-v1.5:
    (1) the memory-efficient "unpad" branch computes bad position_ids -> disable via
    config flags; (2) meta-device init leaves the non-persistent position_ids buffer
    uninitialized -> load with low_cpu_mem_usage=False and re-materialize buffers.
    Pooling = CLS token + L2 normalize (per 1_Pooling/config.json).
    """
    global _model, _tok
    if _model is None:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        conf = AutoConfig.from_pretrained(EMB_MODEL, trust_remote_code=True,
                                          token=os.environ.get("HF_TOKEN"))
        conf.unpad_inputs = False
        conf.use_memory_efficient_attention = False
        # force CPU: the custom GTE-v1.5 kernels device-assert on GPU here, and we only
        # ever embed short queries -> CPU is plenty; leaves the GPU free for the reranker.
        _tok = AutoTokenizer.from_pretrained(EMB_MODEL, token=os.environ.get("HF_TOKEN"))
        _model = AutoModel.from_pretrained(EMB_MODEL, config=conf, trust_remote_code=True,
                                           low_cpu_mem_usage=False,
                                           token=os.environ.get("HF_TOKEN")).to("cpu").eval()
        # meta-device init leaves non-persistent buffers corrupted:
        #  - position_ids -> re-materialize as arange
        #  - rotary cos_cached/sin_cached -> NaN; recompute from inv_freq (finite)
        maxp = getattr(conf, "max_position_embeddings", 512)
        for mod in _model.modules():
            if hasattr(mod, "position_ids"):
                mod.register_buffer("position_ids", torch.arange(maxp), persistent=False)
            if hasattr(mod, "cos_cached") and hasattr(mod, "inv_freq"):
                L = int(mod.cos_cached.shape[0])
                t = torch.arange(L, dtype=torch.float32)
                freqs = torch.outer(t, mod.inv_freq.float())
                emb = torch.cat((freqs, freqs), dim=-1)
                mod.register_buffer("cos_cached", emb.cos(), persistent=False)
                mod.register_buffer("sin_cached", emb.sin(), persistent=False)
    return _model, _tok


_embed_lock = threading.Lock()


def embed(text: str):
    import torch
    model, tok = embedder()
    with _embed_lock:                              # CPU model: no concurrent forward
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model(**enc)
        cls = out.last_hidden_state[:, 0]          # CLS pooling
        cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
        return cls[0].tolist()


def dense(query: str, k: int = 10):
    """kNN over ft_document with the gte-alt-v1 query vector."""
    body = {"size": k, "query": {"knn": {FT_VECTOR: {"vector": embed(query), "k": k}}},
            "_source": [FTP + "ft_title", FTP + "content", FTP + "readerUrl", FTP + "documentId"]}
    hits = requests.post(f"{OS_URL}/{FT_DOC}/_search", json=body, timeout=60).json()["hits"]["hits"]
    return [_doc(h) for h in hits]


def bm25_doc(query: str, k: int = 10):
    body = {"size": k, "query": {"match": {FTP + "content": query}},
            "_source": [FTP + "ft_title", FTP + "content", FTP + "readerUrl", FTP + "documentId"]}
    hits = requests.post(f"{OS_URL}/{FT_DOC}/_search", json=body, timeout=60).json()["hits"]["hits"]
    return [_doc(h) for h in hits]


def bm25_kg(query: str, k: int = 10):
    """BM25 over the knowledge-graph cards (answer/evidence/content fields)."""
    body = {"size": k, "query": {"multi_match": {"query": query,
            "fields": ["answer", "evidence", "content", "doc_title", "facet"]}},
            "_source": ["answer", "evidence", "content", "doc_title", "docid", "family", "applies_to"]}
    hits = requests.post(f"{OS_URL}/{KG}/_search", json=body, timeout=60).json()["hits"]["hits"]
    out = []
    for h in hits:
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
