"""Generate multi-hop synthetic queries over HotpotQA — each requires N docs to answer.

See README.md. Strategy: build a CHAIN of N related docs (seed -> BM25 neighbor -> neighbor of
that -> ...), each consecutive pair sharing keywords; then the LLM writes ONE question that needs
ALL N docs (or NONE, skipped — don't force it). Stops at target count. n_docs picks the dataset.

    python -m experiments.multi_hop_synth_queries.generate [target=1000] [workers=8] [n_docs=2]
"""
from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from phase1.llm import LLM

OS = "http://localhost:9200"
IDX = "hotpotqa"
DATA = Path(__file__).parent / "data"


def outfile(n_docs):
    return DATA / f"multihop_{n_docs}docs_queries.jsonl"


PROMPT = """You are given {n} documents. Write ONE question that can be answered ONLY by using \
ALL {n} documents together — no subset of them should be sufficient. Chain the facts across the \
documents (bridge / comparison / aggregation).

Rules:
- The question must genuinely require a distinct fact from EACH of the {n} documents.
- Ask a natural, self-contained question. Do NOT mention "the documents".
- If the documents don't share enough common ground to support such a question, output exactly NONE.

Return ONLY JSON: {{"question": "...", "facts": ["fact from doc 1", ... {n} items]}}  or  {{"question": "NONE"}}

{blocks}"""


def _search(body):
    r = requests.post(f"{OS}/{IDX}/_search", json=body, timeout=30)
    r.raise_for_status()
    return r.json()["hits"]["hits"]


def sample_seeds(n, seed):
    return _search({"size": n, "_source": ["title", "text"],
                    "query": {"function_score": {"query": {"match_all": {}},
                                                 "random_score": {"seed": seed, "field": "_seq_no"}}}})


def neighbors(doc, used, k=8):
    q = (doc["_source"].get("title", "") + " " + doc["_source"].get("text", "")[:300]).strip()
    hits = _search({"size": k + len(used) + 1, "_source": ["title", "text"],
                    "query": {"multi_match": {"query": q, "fields": ["title^2", "text"]}}})
    out = []
    for h in hits:
        if h["_id"] in used:
            continue
        if h["_source"].get("title", "").lower() == doc["_source"].get("title", "").lower():
            continue
        out.append(h)
    return out


def build_chain(seed, n_docs):
    """Chain seed -> neighbor -> neighbor-of-neighbor ... of length n_docs (or None if it can't extend)."""
    group, used, cur = [seed], {seed["_id"]}, seed
    for _ in range(n_docs - 1):
        nbrs = neighbors(cur, used, k=6)
        if not nbrs:
            return None
        nxt = nbrs[0]
        group.append(nxt); used.add(nxt["_id"]); cur = nxt
    return group


def gen_query(llm, group):
    blocks = "\n\n".join(
        f"DOCUMENT {i+1} (title: {d['_source'].get('title','')}):\n{d['_source'].get('text','')[:800]}"
        for i, d in enumerate(group))
    p = PROMPT.format(n=len(group), blocks=blocks)
    try:
        r = llm.complete(p, system="You write rigorous multi-hop questions, or NONE.")
    except Exception:
        return None
    m = re.search(r"\{.*\}", r, re.DOTALL)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
    except Exception:
        return None
    q = (o.get("question") or "").strip()
    if not q or q.upper() == "NONE":
        return None
    return {"query": q, "gold_ids": [d["_id"] for d in group],
            "titles": [d["_source"].get("title", "") for d in group],
            "facts": o.get("facts", []), "n_docs": len(group)}


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n_docs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    llm = LLM()
    DATA.mkdir(parents=True, exist_ok=True)
    out_path = outfile(n_docs)
    lock = threading.Lock()
    seen = set()
    written, tried = [0], [0]
    fh = out_path.open("w")

    def worker(group):
        q = gen_query(llm, group)
        with lock:
            tried[0] += 1
            if q and written[0] < target:
                fh.write(json.dumps(q) + "\n"); fh.flush()
                written[0] += 1
                if written[0] % 25 == 0:
                    print(f"[multihop-{n_docs}d] {written[0]}/{target} "
                          f"({tried[0]} chains, {written[0]/max(1,tried[0]):.0%} yield, "
                          f"${llm.usage.cost_usd:.2f})", flush=True)
        return bool(q)

    batch = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while written[0] < target:
            groups = []
            for s in sample_seeds(80, seed=batch):
                g = build_chain(s, n_docs)
                if g:
                    key = tuple(sorted(d["_id"] for d in g))
                    if key not in seen:
                        seen.add(key); groups.append(g)
            batch += 1
            if not groups:
                continue
            futs = [ex.submit(worker, g) for g in groups]
            for _ in as_completed(futs):
                if written[0] >= target:
                    break
    fh.close()
    print(f"[multihop-{n_docs}d] DONE {written[0]} queries ({tried[0]} chains) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
