"""Multi-document synthetic query generation — a standard, backend-agnostic capability.

Produces queries that require **N specific documents together** to answer (multi-hop), grounded
in a corpus. A query's success criterion downstream is **all N gold docs in recall@k** — the
regime where multi-strategy / code-mode retrieval beats a single dense pass.

Method (chain-of-related-docs):
1. sample a seed document from the store,
2. extend a CHAIN of length N: seed -> keyword-neighbor -> neighbor-of-that -> …  (each
   consecutive pair shares terms but is a different doc; skip near-identical titles),
3. ask the generator for ONE question answerable only by using ALL N docs — or `NONE` (skip;
   never force a question when the docs lack common ground).

Works over any :class:`~search_as_code.Session` (store.sample + keyword search + generator), so it
runs on memory, OpenSearch, or any adapter. See ``experiments/multi_hop_synth_queries`` for a
runnable driver and the HotpotQA/SearchUnify datasets built with it.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

_PROMPT = """You are given {n} documents. Write ONE question that can be answered ONLY by using \
ALL {n} documents together — no subset should be sufficient. Chain the facts across them \
(bridge / comparison / aggregation).

Rules:
- The question must genuinely require a distinct fact from EACH of the {n} documents.
- Ask a natural, self-contained question. Do NOT mention "the documents".
- If the documents don't share enough common ground for such a question, output exactly NONE.

Return ONLY JSON: {{"question": "...", "facts": ["fact from doc 1", ... {n} items]}}  or  {{"question": "NONE"}}

{blocks}"""


def _title(doc) -> str:
    return (getattr(doc, "metadata", None) or {}).get("title", "") or ""


def _neighbors(session, doc, used: set, k: int = 6):
    """Keyword neighbors of ``doc`` (share terms), excluding used ids and near-identical titles."""
    q = (_title(doc) + " " + (doc.text or "")[:300]).strip()
    try:
        hits = session.search(q, top_k=k + len(used) + 1, mode="keyword")
    except Exception:
        return []
    out = []
    t0 = _title(doc)
    for h in hits:
        d = h.document
        if h.id in used:
            continue
        if t0 and _title(d).lower() == t0.lower():   # skip near-identical titles (only if titled)
            continue
        out.append(d)
    return out


def _build_chain(session, seed, n_docs: int):
    chain, used, cur = [seed], {seed.id}, seed
    for _ in range(n_docs - 1):
        nbrs = _neighbors(session, cur, used)
        if not nbrs:
            return None
        nxt = nbrs[0]
        chain.append(nxt); used.add(nxt.id); cur = nxt
    return chain


def _gen(generator, chain):
    blocks = "\n\n".join(f"DOCUMENT {i + 1} (title: {_title(d)}):\n{(d.text or '')[:800]}"
                         for i, d in enumerate(chain))
    prompt = _PROMPT.format(n=len(chain), blocks=blocks)
    try:
        out = generator(prompt)
        txt = out[0] if isinstance(out, list) else str(out)
    except Exception:
        return None
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
    except Exception:
        return None
    q = (o.get("question") or "").strip()
    if not q or q.upper() == "NONE":
        return None
    return {"query": q, "gold_ids": [d.id for d in chain], "titles": [_title(d) for d in chain],
            "facts": o.get("facts", []), "n_docs": len(chain)}


def generate_multihop(session, n_docs: int = 2, target: int = 1000, *, workers: int = 8,
                      sample_chunk: int = 80, generator=None, out_path: Optional[str] = None,
                      progress_every: int = 25) -> list[dict]:
    """Generate ``target`` multi-hop queries (each needing ``n_docs`` docs) over the Session corpus.

    Returns a list of ``{query, gold_ids:[…N…], titles, facts, n_docs}``. If ``out_path`` is given,
    also streams them to that jsonl. ``generator`` defaults to the Session's generator.
    """
    gen = generator or session.generator
    if gen is None:
        raise RuntimeError("generate_multihop needs a generator (pass generator= or set it on the Session)")

    lock = threading.Lock()
    out: list[dict] = []
    seen = set()
    fh = open(out_path, "w") if out_path else None

    def worker(chain):
        q = _gen(gen, chain)
        if q:
            with lock:
                if len(out) < target:
                    out.append(q)
                    if fh:
                        fh.write(json.dumps(q) + "\n"); fh.flush()
                    if progress_every and len(out) % progress_every == 0:
                        print(f"[multihop {n_docs}d] {len(out)}/{target}", flush=True)
        return bool(q)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        empty_rounds = 0
        while len(out) < target and empty_rounds < 3:
            try:
                seeds = session.store.sample(sample_chunk)
            except Exception:
                break
            chains = []
            for s in seeds:
                c = _build_chain(session, s, n_docs)
                if not c:
                    continue
                key = tuple(sorted(d.id for d in c))
                if key not in seen:
                    seen.add(key); chains.append(c)
            if not chains:
                empty_rounds += 1
                continue
            empty_rounds = 0
            for fut in as_completed([ex.submit(worker, c) for c in chains]):
                if len(out) >= target:
                    break
    if fh:
        fh.close()
    return out[:target]
