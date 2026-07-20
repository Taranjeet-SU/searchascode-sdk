"""Shared config, env, and helpers for the Phase 1 harness.

Central place for: loading the OpenAI key, the embedding model, the OpenSearch
index, and factory helpers used by ingest, benchmark, agent, and UI.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

# --- paths ---------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "phase1" / "data"
RUNS_DIR = REPO / "phase1" / "runs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# --- dataset / model config ---------------------------------------------
DATASET = "fiqa"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
SPLIT = "test"
EMB_MODEL = "thenlper/gte-base"   # 768-d, matches the taxonomy stack
DIM = 768
INDEX = "fiqa"
OS_HOST = {"host": "127.0.0.1", "port": 9200}
LLM_MODEL = "gpt-4.1-mini"

# Fairness knob: both SAC and tool-calling reformulate the query into exactly this
# many formulations (original + N-1 rephrasings), so query expansion is controlled.
N_QUERY_VARIANTS = 4

# gpt-4.1-mini pricing (USD per 1M tokens) — for cost accounting in the benchmark
LLM_PRICE = {"input": 0.40, "cached_input": 0.10, "output": 1.60}


def load_env() -> None:
    """Load OPENAI_API_KEY from the taxonomy project's .env if not already set."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    env_path = Path.home() / "taxonomy" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# --- BEIR loading (self-contained; mirrors taxonomy/src/data.py) ---------
def download_beir() -> Path:
    out_dir = DATA_DIR / DATASET
    if (out_dir / "corpus.jsonl").exists():
        return out_dir
    import requests
    zip_path = DATA_DIR / f"{DATASET}.zip"
    if not zip_path.exists():
        print(f"[data] downloading {BEIR_URL}")
        r = requests.get(BEIR_URL, stream=True, timeout=300)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    print(f"[data] extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    return out_dir


def load_corpus(beir_dir: Path) -> dict[str, dict]:
    corpus = {}
    with open(beir_dir / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d.get("title", ""), "text": d.get("text", "")}
    return corpus


def load_queries(beir_dir: Path) -> dict[str, str]:
    q = {}
    with open(beir_dir / "queries.jsonl") as f:
        for line in f:
            d = json.loads(line)
            q[d["_id"]] = d.get("text", "")
    return q


def load_qrels(beir_dir: Path, split: str = SPLIT) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(beir_dir / "qrels" / f"{split}.tsv") as f:
        f.readline()  # header
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[cid] = int(score)
    return qrels


# --- factories -----------------------------------------------------------
_embedder = None


def get_embedder():
    """Cached sentence-transformer embedder wrapped for the SDK (GPU if available)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(EMB_MODEL, device=device)

        def embed(texts):
            return model.encode(
                list(texts), batch_size=128, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ).tolist()

        _embedder = embed
    return _embedder


def get_session():
    import search_as_code as sac

    return sac.Session("opensearch", index=INDEX, dim=DIM, hosts=[OS_HOST],
                       embedder=get_embedder())
