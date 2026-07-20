"""Concrete reranker implementations (optional; local, no API).

A reranker is any ``callable(query, texts) -> list[float]``. ``CrossEncoderReranker``
wraps a sentence-transformers cross-encoder — the standard two-stage
retrieve-then-rerank primitive — and plugs straight into ``Session(reranker=...)``.
"""

from __future__ import annotations

from typing import Sequence


class CrossEncoderReranker:
    """Cross-encoder relevance scorer (e.g. BAAI/bge-reranker, ms-marco-MiniLM).

    Lazy-loads the model on first call so importing the SDK stays cheap.
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None):
        self.model_name = model
        self._device = device
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            import torch

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = CrossEncoder(self.model_name, device=device)
        return self._model

    def __call__(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        model = self._ensure()
        pairs = [(query, t or "") for t in texts]
        return [float(s) for s in model.predict(pairs, show_progress_bar=False)]
