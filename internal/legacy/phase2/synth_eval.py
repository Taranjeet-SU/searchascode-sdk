"""Primitive-sufficiency probe on synthetic constraint / cross-document queries.

Mirrors the SU archetype: a corpus of release notes (feature -> introduced-in-version),
and constraint queries whose answer requires RETRIEVAL + COMPUTATION:

    "latest release of <product> where I can use <A> and <B>"  ->  max(intro[A], intro[B])

Dense RAG returns passages (can't compute the max). SAC writes code that retrieves both
facts and computes the answer. This probe measures ANSWER accuracy and surfaces whether the
current primitives suffice.

Run: python -m phase2.synth_eval
"""

from __future__ import annotations

import random
import re

import search_as_code as sac
from phase1 import agents, common
from phase1.llm import LLM
from search_as_code.sandbox import LocalExecutor

common.load_env()

PRODUCTS = ["Nimbus", "Orion", "Vertex", "Atlas", "Cobalt"]
FEATURES = ["embedding models", "fine-tuning of embedding models", "hybrid search",
            "role-based access control", "audit logging", "multilingual search",
            "SSO integration", "vector compression", "reranking", "knowledge graphs"]


def build_corpus(seed=0):
    random.seed(seed)
    docs, facts = [], {}          # facts[(product, feature)] = version introduced
    for p in PRODUCTS:
        for f in FEATURES:
            v = f"{random.randint(1,6)}.{random.randint(0,9)}"
            facts[(p, f)] = v
            docs.append({
                "id": f"{p}-{f}".replace(" ", "_"),
                "text": f"{p} Release {v} adds support for {f}. This release note documents "
                        f"that {f} became available in {p} version {v}.",
                "metadata": {"product": p, "feature": f, "version": v},
            })
    return docs, facts


def vkey(v):  # version sort key
    a, b = v.split("."); return (int(a), int(b))


def build_queries(facts, n=12, seed=1):
    random.seed(seed)
    qs = []
    for _ in range(n):
        p = random.choice(PRODUCTS)
        a, b = random.sample(FEATURES, 2)
        va, vb = facts[(p, a)], facts[(p, b)]
        answer = va if vkey(va) >= vkey(vb) else vb   # earliest release where BOTH hold = max
        qs.append({"q": f"What is the latest release of {p} where I can use {a} and {b}?",
                   "answer": answer, "product": p, "features": [a, b]})
    return qs


SAC_CONSTRAINT_SYSTEM = """You answer constraint queries over a release-notes corpus by writing Python
against the `sac` SDK. Each document's metadata has {product, feature, version}. `sac` and `query` are
in scope. To answer "latest release where I can use A and B": retrieve the docs for each feature of the
product, read their `version` from metadata, and COMPUTE the answer (the max of the two versions, since
both must be available). Output ONE ```python block that assigns the answer version string to `answer`.
Helpers: sac.search(query, top_k, mode, filter={"product": ...}); hit.get("version"), hit.get("feature").
Example:
```python
a = sac.search("embedding models", top_k=3, filter={"product": "Nimbus"})
b = sac.search("audit logging",   top_k=3, filter={"product": "Nimbus"})
va = a.top(1)[0].get("version"); vb = b.top(1)[0].get("version")
answer = max([va, vb], key=lambda v: tuple(int(x) for x in v.split(".")))
```"""


def run_sac_answer(session, q, chat):
    from langchain_core.messages import SystemMessage, HumanMessage
    resp = chat.invoke([SystemMessage(content=SAC_CONSTRAINT_SYSTEM), HumanMessage(content=f"Query: {q}")])
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    code = (m.group(1) if m else raw).strip()
    box = LocalExecutor(session); box._globals["query"] = q
    r = box.run(code)
    ans = box._globals.get("answer")
    return (str(ans) if ans is not None else ""), code, r.ok


def dense_answer(session, q):
    # Plain dense RAG: take the top passage and extract a version if present.
    hits = session.search(q, top_k=3, mode="dense")
    for h in hits:
        m = re.search(r"version (\d+\.\d+)", h.text or "")
        if m:
            return m.group(1)
    return ""


def main():
    docs, facts = build_corpus()
    queries = build_queries(facts)
    from sentence_transformers import SentenceTransformer
    import torch
    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).tolist()
    session = sac.Session("memory", embedder=embed)        # portable: in-memory backend
    session.add(docs)
    chat = agents.lc_chat()

    sac_ok = dense_ok = 0
    print(f"{'answer':>8} {'dense':>6} {'SAC':>6}  query")
    for item in queries:
        d = dense_answer(session, item["q"])
        s, code, ok = run_sac_answer(session, item["q"], chat)
        sac_ok += (s == item["answer"]); dense_ok += (d == item["answer"])
        print(f"{item['answer']:>8} {d or '—':>6} {s or '—':>6}  {item['q'][:60]}")
    n = len(queries)
    print(f"\nANSWER ACCURACY over {n} constraint queries:")
    print(f"  dense RAG : {dense_ok}/{n} = {dense_ok/n:.2f}")
    print(f"  SAC       : {sac_ok}/{n} = {sac_ok/n:.2f}")
    print("\nsample SAC code:\n", code[:400])


if __name__ == "__main__":
    main()
