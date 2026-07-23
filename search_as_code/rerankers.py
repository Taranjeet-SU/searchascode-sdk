"""Concrete reranker implementations (optional; local, no API).

A reranker is any ``callable(query, texts) -> list[float]``. ``CrossEncoderReranker``
wraps a sentence-transformers cross-encoder — the standard two-stage
retrieve-then-rerank primitive — and plugs straight into ``Session(reranker=...)``.
"""

from __future__ import annotations

from typing import Any, Sequence


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
            import torch
            from sentence_transformers import CrossEncoder

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = CrossEncoder(self.model_name, device=device)
        return self._model

    def __call__(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        model = self._ensure()
        pairs = [(query, t or "") for t in texts]
        return [float(s) for s in model.predict(pairs, show_progress_bar=False)]


class QwenReranker:
    """Qwen3-Reranker — an LLM reranker that scores relevance via the yes/no logit.
    Much stronger than ms-marco cross-encoders on out-of-domain corpora (FiQA etc.).
    Same ``callable(query, texts) -> list[float]`` interface. Lazy-loaded.
    """

    _PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the '
               'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
               '<|im_end|>\n<|im_start|>user\n')
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model: str = "Qwen/Qwen3-Reranker-0.6B", device: str | None = None,
                 max_length: int = 512,
                 instruction: str = "Given a query, retrieve passages that answer it"):
        self.model_name = model
        self._device = device
        self.max_length = max_length
        self.instruction = instruction
        self._model: Any = None

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tok = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
            ).to(dev).eval()  # type: ignore[arg-type]
            self.dev = dev
            self.tid_yes = self.tok.convert_tokens_to_ids("yes")
            self.tid_no = self.tok.convert_tokens_to_ids("no")
        return self._model

    def __call__(self, query: str, texts: Sequence[str], batch_size: int = 16) -> list[float]:
        if not texts:
            return []
        import torch

        self._ensure()
        prompts = [f"{self._PREFIX}<Instruct>: {self.instruction}\n<Query>: {query}\n"
                   f"<Document>: {t or ''}{self._SUFFIX}" for t in texts]
        scores: list[float] = []
        for i in range(0, len(prompts), batch_size):
            enc = self.tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_length).to(self.dev)
            with torch.no_grad():
                logits = self._model(**enc).logits[:, -1, :]
            yn = torch.stack([logits[:, self.tid_no], logits[:, self.tid_yes]], dim=1)
            p = torch.log_softmax(yn.float(), dim=1)[:, 1].exp()
            scores.extend(p.cpu().tolist())
        return scores
