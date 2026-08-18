"""Embedder configs for the SOTA-embedder comparison.

bge/e5 need DIFFERENT prefixes for queries vs passages — getting this wrong tanks
recall, so we handle it explicitly. Each config yields (query_embed, passage_embed).
"""

from __future__ import annotations

EMBEDDERS = {
    "gte-base":  {"kind": "st", "model": "thenlper/gte-base",      "dim": 768,
                  "q_prefix": "", "p_prefix": ""},
    "bge-large": {"kind": "st", "model": "BAAI/bge-large-en-v1.5", "dim": 1024,
                  "q_prefix": "Represent this sentence for searching relevant passages: ", "p_prefix": ""},
    "e5-large":  {"kind": "st", "model": "intfloat/e5-large-v2",   "dim": 1024,
                  "q_prefix": "query: ", "p_prefix": "passage: "},
    # OpenAI API embedders — no local download; symmetric (no query/passage prefix).
    "openai-large": {"kind": "openai", "model": "text-embedding-3-large", "dim": 1024},
    "openai-small": {"kind": "openai", "model": "text-embedding-3-small", "dim": 1536},
}


def index_name(key: str) -> str:
    return "fiqa" if key == "gte-base" else f"fiqa_{key.replace('-', '_')}"


def _openai_encoder(model: str, dim: int):
    from openai import OpenAI
    client = OpenAI()

    def enc(texts):
        out = []
        xs = [t if t.strip() else " " for t in texts]
        for i in range(0, len(xs), 256):
            for attempt in range(5):
                try:
                    r = client.embeddings.create(model=model, input=xs[i:i + 256], dimensions=dim)
                    out.extend([d.embedding for d in r.data]); break
                except Exception:
                    import time
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))
        return out
    return enc


def build(key: str, device: str = "cuda"):
    """Return (query_embed, passage_embed, dim, index_name)."""
    cfg = EMBEDDERS[key]
    if cfg["kind"] == "openai":
        from phase1 import common
        common.load_env()
        enc = _openai_encoder(cfg["model"], cfg["dim"])
        return enc, enc, cfg["dim"], index_name(key)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["model"], device=device)

    def _enc(prefix):
        def enc(texts):
            xs = [prefix + t for t in texts] if prefix else list(texts)
            return model.encode(xs, normalize_embeddings=True, convert_to_numpy=True,
                                batch_size=128, show_progress_bar=False).tolist()
        return enc

    return _enc(cfg["q_prefix"]), _enc(cfg["p_prefix"]), cfg["dim"], index_name(key)
