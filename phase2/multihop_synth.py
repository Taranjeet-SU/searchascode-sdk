"""Synthetic multi-hop BRIDGE benchmark (HotpotQA-style) — no download needed.

Corpus of two fact types:
  * "<Product> is developed by <Company>."          (hop-1 doc)
  * "<Company> is headquartered in <City>."          (hop-2 doc)
Bridge query: "In which city is the developer of <Product> headquartered?"
  -> answer = City;  gold supporting docs = [product-dev doc, company-hq doc].

A single dense query finds the product-dev doc (it names the product) but CANNOT
know which company's HQ doc to fetch (the query never names the company) -> it
misses hop-2. Decompose/bridge retrieval (find company from hop-1, then search its
HQ) retrieves BOTH. This is the structural multi-hop win.

Metrics: recall@10 of the 2 gold docs, and all_found@10 (both retrieved).

    python -m phase2.multihop_synth
"""
from __future__ import annotations

import random
import re

import search_as_code as sac
from phase1 import agents, common
from phase1.llm import LLM
from search_as_code.sandbox import LocalExecutor

COMPANIES = ["Nimbus", "Orion", "Vertex", "Atlas", "Cobalt", "Zephyr", "Quasar", "Helix",
             "Aster", "Borealis", "Cinder", "Delta", "Ember", "Fjord", "Gale", "Hollow"]
CITIES = ["Portland", "Austin", "Denver", "Boston", "Seattle", "Chicago", "Atlanta", "Miami",
          "Dallas", "Phoenix", "Newark", "Tucson", "Reno", "Fresno", "Tampa", "Buffalo"]
PRODUCTS = ["Falcon", "Comet", "Pulse", "Nova", "Beacon", "Quartz", "Lumen", "Onyx", "Prism",
            "Raven", "Slate", "Talon", "Umbra", "Vine", "Willow", "Yonder", "Zenith", "Arc",
            "Bolt", "Cove", "Dune", "Echo", "Flint", "Grove"]


def build():
    random.seed(7)
    company_city = {c: random.choice(CITIES) for c in COMPANIES}
    product_company = {p: random.choice(COMPANIES) for p in PRODUCTS}
    docs, hq_id, dev_id = [], {}, {}
    for c, city in company_city.items():
        i = f"hq_{c}"; hq_id[c] = i
        docs.append({"id": i, "text": f"{c} is a technology company headquartered in {city}.",
                     "metadata": {"kind": "hq", "company": c, "city": city}})
    for p, c in product_company.items():
        i = f"dev_{p}"; dev_id[p] = i
        docs.append({"id": i, "text": f"The {p} platform is developed and maintained by {c}.",
                     "metadata": {"kind": "dev", "product": p, "company": c}})
    queries = []
    for p, c in product_company.items():
        queries.append({"q": f"In which city is the developer of the {p} platform headquartered?",
                        "answer": company_city[c], "gold": [dev_id[p], hq_id[c]], "bridge": c})
    return docs, queries


BRIDGE_SYS = """You answer 2-hop bridge questions by writing Python against the `sac` search SDK.
`sac` and `query` are in scope. Strategy: (1) search for the FIRST fact, (2) READ the bridge entity
from the top result's text/metadata, (3) search for the SECOND fact using that entity. Collect BOTH
supporting doc ids. Output ONE ```python block assigning the ~10 best doc ids to `evidence` (put the
two supporting docs first). Hits: h.id, h.text, h.get('company'), h.get('city').
Example:
```python
hop1 = sac.search("developer of the Falcon platform", top_k=5)
company = hop1.top(1)[0].get("company")            # bridge entity
hop2 = sac.search(company + " headquartered city", top_k=5)
evidence = [hop1.top(1)[0].id, hop2.top(1)[0].id] + hop1.ids() + hop2.ids()
```"""


def run_bridge_sac(session, q, chat):
    from langchain_core.messages import SystemMessage, HumanMessage
    r = chat.invoke([SystemMessage(content=BRIDGE_SYS), HumanMessage(content=f"Query: {q}")])
    raw = r.content if isinstance(r.content, str) else str(r.content)
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    code = (m.group(1) if m else raw).strip()
    box = LocalExecutor(session); box._globals["query"] = q
    res = box.run(code)
    ids = [str(x) for x in (res.evidence or [])] if isinstance(res.evidence, list) else []
    return ids, code, res.ok


def main():
    docs, queries = build()
    from sentence_transformers import SentenceTransformer
    import torch
    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).tolist()
    s = sac.Session("memory", embedder=embed, generator=LLM().as_generator())
    s.add(docs)
    chat = agents.lc_chat()
    print(f"corpus={len(docs)} docs, {len(queries)} bridge queries\n")

    def rec(ids, gold): return len(set(ids[:10]) & set(gold)) / len(gold)
    def allf(ids, gold): return 1.0 if set(gold) <= set(ids[:10]) else 0.0
    import numpy as np
    D = {"dense_r": [], "dense_a": [], "sac_r": [], "sac_a": []}
    for i, item in enumerate(queries):
        q, gold = item["q"], item["gold"]
        d = s.search(q, 10, mode="dense").ids()
        sids, code, ok = run_bridge_sac(s, q, chat)
        D["dense_r"].append(rec(d, gold)); D["dense_a"].append(allf(d, gold))
        D["sac_r"].append(rec(sids, gold)); D["sac_a"].append(allf(sids, gold))
        if i < 3:
            print(f"Q: {q}\n  dense found {int(rec(d,gold)*2)}/2  SAC found {int(rec(sids,gold)*2)}/2")
    print(f"\n===== multi-hop bridge ({len(queries)} queries) =====")
    print(f"  dense   recall@10={np.mean(D['dense_r']):.3f}  all_found@10={np.mean(D['dense_a']):.3f}")
    print(f"  SAC     recall@10={np.mean(D['sac_r']):.3f}  all_found@10={np.mean(D['sac_a']):.3f}")


if __name__ == "__main__":
    main()
