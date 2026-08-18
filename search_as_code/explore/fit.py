"""Query collection and batch embedding for the router dataset.

Despite the module name, the labeling/training pipeline lives in ``training.py``; this module
holds the shared helpers it imports (``_collect_queries``, ``_batch_embed``, ``_rephrase``).
See the note at the bottom for what was removed and why (issues.md SDK-R1).
"""

from __future__ import annotations

from .._genutil import gen_lines


def _rephrase(session, query: str, n: int) -> list[str]:
    """Ask the generator for ``n`` paraphrases that keep the same information need."""
    if n <= 0 or session.generator is None:
        return []
    prompt = (f"Rewrite the search query below in {n} different ways with the same meaning "
              "(synonyms, word order, abbreviation vs expansion). One per line, no numbering."
              f"\n\nQUERY: {query}")
    try:
        out = session.generator(prompt)
    except Exception:
        return []
    # gen_lines, not out[0]: indexing a line-splitting generator yielded ONE paraphrase
    # regardless of n, which shrank and de-diversified the router training set (GEN-1).
    lines = gen_lines(out)
    return [ln for ln in lines if ln.lower() != query.lower()][:n]


def _collect_queries(explorer, queries, n, rephrases, gen_llm):
    session, pack, config = explorer.session, explorer.pack, explorer.config
    if queries is not None:
        out = []
        for it in queries:
            if isinstance(it, dict):
                out.append({"query": it["query"], "gold_id": it.get("gold_id") or it.get("gold")})
            else:
                out.append({"query": it[0], "gold_id": it[1]})
        return out[:n]

    # generate grounded synth queries (+ rephrases) from a fresh sample of the corpus
    from .engine import ExploreContext
    from .stages import _gen_queries
    ctx = ExploreContext(session=session, pack=pack, config=config)
    per_doc = int(config.get("synth_per_doc", 3))
    # Sample docs in CHUNKS (a single huge sample can be many MB / time out on wide docs),
    # deduping by id, generating as we go and stopping once we have n queries.
    out, seen, di = [], set(), 0
    chunk = int(config.get("sample_chunk", 250))
    max_batches = max(1, (n // 2) // chunk + 4)
    for _b in range(max_batches):
        try:
            batch = session.store.sample(chunk)
        except Exception:
            break
        fresh = [d for d in batch if d.id not in seen]
        if not fresh:
            break
        for d in fresh:
            seen.add(d.id)
            di += 1
            text = getattr(d, "text", None) or ""
            for _diff, q in _gen_queries(ctx, text, per_doc):
                out.append({"query": q, "gold_id": d.id})
                for rp in _rephrase(session, q, rephrases):
                    out.append({"query": rp, "gold_id": d.id})
                if len(out) >= n:
                    return out[:n]
            if di % 25 == 0:
                print(f"[fit] generated {len(out)}/{n} queries from {di} docs", flush=True)
    return out[:n]


def _batch_embed(session, texts, bs=64):
    """Embed all query texts in batches (one forward per batch) — much cheaper than a
    per-query call when the backend embedder can batch."""
    out = []
    for i in range(0, len(texts), bs):
        out.extend(session.embedder.embed(texts[i:i + bs]))
        if len(texts) > bs and (i // bs) % 10 == 0:
            print(f"[fit] embedded {min(i + bs, len(texts))}/{len(texts)} queries", flush=True)
    return out


# NOTE: ``fit_router()`` used to live here — a second, complete copy of the labeling +
# training pipeline that duplicated ``training.build_dataset`` + ``training.train_router_model``
# and wrote the same artifacts, but was **referenced nowhere**: not by ``Explorer.fit()`` (which
# calls dataset() + train()), not by ``explore/__init__.py``, not by any test or experiment. It
# also trained with DIFFERENT hyperparameters than the live path (max_iter=300, lr=0.08 vs
# 400/0.07), so the repo carried two divergent trainers, one of them unreachable. Deleted along
# with its only consumer, ``router.train_router`` (issues.md SDK-R1). The helpers above are the
# part that is genuinely shared — ``training.py`` imports them.
