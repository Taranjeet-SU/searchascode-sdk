"""Generic BEIR harness: load any small BEIR dataset, ingest into OpenSearch, and
run dense/hybrid/SAC/tool eval — so the multi-dataset campaign is uniform.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from phase1 import common

DATA = Path(common.REPO) / "phase2" / "data"
DIM = 768
BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

# dataset -> qrels split
DATASETS = {"scifact": "test", "nfcorpus": "test", "arguana": "test",
            "scidocs": "test", "trec-covid": "test"}


def ensure(name: str) -> Path:
    d = DATA / name
    if (d / "corpus.jsonl").exists():
        return d
    zp = DATA / f"{name}.zip"
    if not zp.exists():
        import requests
        r = requests.get(f"{BASE_URL}/{name}.zip", stream=True, timeout=300); r.raise_for_status()
        with open(zp, "wb") as f:
            for c in r.iter_content(1 << 16):
                f.write(c)
    with zipfile.ZipFile(zp) as z:
        z.extractall(DATA)
    return d


def eval_data(dataset: str):
    """Dispatch (queries, qrels, opensearch_index) for any dataset in the project,
    so the learning pipeline works uniformly (fiqa / hotpotqa / any BEIR set)."""
    if dataset == "fiqa":
        q = json.loads((common.DATA_DIR / "queries.json").read_text())
        qr = json.loads((common.DATA_DIR / "qrels.json").read_text())
        return q, qr, common.INDEX
    if dataset == "hotpotqa":
        q = json.loads((DATA / "hotpot_queries.json").read_text())
        qr = json.loads((DATA / "hotpot_qrels.json").read_text())
        return q, qr, "hotpotqa"
    _, queries, qrels = load(dataset)
    return queries, qrels, dataset


def load(name: str):
    d = ensure(name)
    corpus = {}
    with open(d / "corpus.jsonl") as f:
        for line in f:
            j = json.loads(line); corpus[j["_id"]] = {"title": j.get("title", ""), "text": j.get("text", "")}
    queries = {}
    with open(d / "queries.jsonl") as f:
        for line in f:
            j = json.loads(line); queries[j["_id"]] = j.get("text", "")
    qrels = {}
    with open(d / "qrels" / f"{DATASETS[name]}.tsv") as f:
        f.readline()
        for line in f:
            qid, cid, sc = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[cid] = int(sc)
    return corpus, queries, qrels
