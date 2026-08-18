"""Learning phase: mine reusable rules from a dataset's retrieval failures by asking
the LLM one case at a time, aggregate into a LEARNED PROFILE, and store it in a DB
(OpenSearch index `sac_learned`) so the runtime can pull it.

Profile = {aliases (spelling), glossary (acronym->expansion), synonyms (euphemism/
related), routes (condition->strategy)}. Runtime feeds these into normalize_query,
expand, and the decision-rule prompt: standard code + per-dataset learned code.

    python -m phase2.learn_rules --dataset fiqa --n 120 --max-cases 40
"""
from __future__ import annotations

import argparse
import json

from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM

LEARN_INDEX = "sac_learned"

SYS = """You improve a retrieval system. Given a search query and a known-relevant document it FAILED
to retrieve, propose ONE reusable, generalizable rule that would help find such documents in future.
Output ONLY compact JSON, one of:
{"type":"alias","from":"cheque","to":"check"}                # spelling / variant normalization
{"type":"glossary","from":"CD","to":"certificate of deposit"} # acronym / abbreviation expansion
{"type":"synonym","from":"disappear","to":["die","pass away"]}# euphemism / related-term expansion
{"type":"route","when":"query has an acronym or number","use":"keyword_boost"}  # strategy hint
Choose the SINGLE most useful, generalizable rule. Prefer alias/glossary/synonym over route."""


def mine(dataset: str, n: int, max_cases: int, split: str = "train"):
    from internal.legacy.phase2 import beir
    q, qr, index = beir.eval_data(dataset)
    # Mine on the TRAIN split only. This used to take the first n of the same ordered
    # dict that impact_eval evaluates on, leaking 80% of the eval set (P2-1).
    from internal.legacy.phase2.splits import pick
    qids = pick(qr, split, n=None)[:n] if split != "all" else \
        [x for x in qr if any(v > 0 for v in qr[x].values())][:n]
    print(f"[mine] dataset={dataset} split={split} n_qids={len(qids)}", flush=True)
    em = SentenceTransformer(common.EMB_MODEL, device="cuda")
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).tolist()
    store = sac.connect("opensearch", index=index, dim=common.DIM, hosts=[common.OS_HOST])
    llm = LLM()

    # collect miss cases (gold not in dense top-100)
    cases = []
    for x in qids:
        gold = {d for d, v in qr[x].items() if v > 0}
        top = set(store.query_vector(embed([q[x]])[0], top_k=100).ids())
        for g in gold - top:
            gt = (store.get([g]) or [None])[0]
            if gt and gt.text:
                cases.append((q[x], gt.text))
        if len(cases) >= max_cases:
            break

    profile = {"dataset": dataset, "aliases": {}, "glossary": {}, "synonyms": {}, "routes": []}
    import re
    for qtext, gtext in cases[:max_cases]:
        raw = llm.complete(f"Query: {qtext}\nRelevant document: {gtext[:400]}\nThey share few words.", system=SYS)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            continue
        try:
            r = json.loads(m.group(0))
        except Exception:
            continue
        t = r.get("type")
        if t == "alias" and r.get("from") and r.get("to"):
            profile["aliases"][str(r["from"]).lower()] = str(r["to"])
        elif t == "glossary" and r.get("from") and r.get("to"):
            profile["glossary"][str(r["from"]).lower()] = str(r["to"])
        elif t == "synonym" and r.get("from"):
            to = r["to"] if isinstance(r["to"], list) else [r["to"]]
            profile["synonyms"].setdefault(str(r["from"]).lower(), [])
            profile["synonyms"][str(r["from"]).lower()] += [str(x) for x in to]
        elif t == "route" and r.get("when"):
            profile["routes"].append({"when": r["when"], "use": r.get("use", "")})

    # ---- store in DB (OpenSearch) so runtime can pull ----
    store.client.index(index=LEARN_INDEX, id=dataset, body=profile, refresh=True)
    (common.REPO / "phase2" / "runs").mkdir(exist_ok=True)
    (common.REPO / "phase2" / "runs" / f"learned_{dataset}.json").write_text(json.dumps(profile, indent=2))

    print(f"[learn] mined {len(cases[:max_cases])} cases  (llm ${llm.usage.cost_usd:.4f})")
    print(f"[learn] aliases={len(profile['aliases'])} glossary={len(profile['glossary'])} "
          f"synonyms={len(profile['synonyms'])} routes={len(profile['routes'])}")
    print(f"[learn] stored in OpenSearch index '{LEARN_INDEX}' id='{dataset}'\n")
    print("sample aliases  :", dict(list(profile["aliases"].items())[:6]))
    print("sample glossary :", dict(list(profile["glossary"].items())[:6]))
    print("sample synonyms :", dict(list(profile["synonyms"].items())[:6]))
    print("sample routes   :", profile["routes"][:4])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiqa")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max-cases", type=int, default=40)
    ap.add_argument("--split", default="train", choices=["train", "test", "all"],
                    help="mine on this split; evaluate on the OTHER one (P2-1)")
    a = ap.parse_args()
    mine(a.dataset, a.n, a.max_cases, a.split)
