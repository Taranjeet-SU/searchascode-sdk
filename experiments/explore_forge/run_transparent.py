"""Transparent explore → capture → store → distill, with a full readable trace.

For each synthetic multi-hop query (gold known, oracle for scoring only):
  - assemble the DYNAMIC prompt = FIXED base + cross-query MEMORY (wins from earlier queries) +
    IN-SESSION FINDINGS (this query's earlier hops) + SKILLS catalog;
  - the LLM writes the next APPROACH as structured OpenSearch queries {mode, query} (hybrid / hyde /
    fielded / keyword / dense) per sub-fact — these are CAPTURED verbatim;
  - execute, and ATTRIBUTE each gold to the exact (mode, query) that surfaced it;
  - write the hop's finding to MEMORY (in-hop) so the next hop sees it; iterate up to max_hops;
  - on solve, STORE the winning (mode, query) patterns to cross-query memory.

After all queries, an LLM DISTILLER reads the captured winning patterns and PROPOSES new
primitives / subagents (JSON), which are forged + persisted. Everything is written to transcript.md.

    python -m experiments.explore_forge.run_transparent [n=10] [max_hops=3]
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from search_as_code import primitives as P
from phase1 import common
from phase1.llm import LLM
from search_as_code.harness import AgentMemory, HarnessForge, HarnessStore, SkillRegistry

DATA = Path(__file__).parents[1] / "multi_hop_synth_queries" / "data"
HERE = Path(__file__).parent
FIXED_BASE = ("You are a retrieval agent solving a MULTI-HOP question that needs SEVERAL different "
              "documents. Decompose it into sub-facts and, for EACH, pick the best OpenSearch query: "
              "mode 'hybrid' (balanced), 'hyde' (when the entity is only DESCRIBED, not named — "
              "hallucinate the answer doc), 'fielded' (a named entity → title+text match), 'keyword', "
              "or 'dense'. Return ONLY JSON: {\"reason\":\"...\",\"plan\":[{\"mode\":\"hyde\",\"query\":\"...\"}, ...]}.")

MODES = {
    "hybrid":  lambda s, q, k: s.search(q, top_k=k, mode="hybrid"),
    "dense":   lambda s, q, k: s.search(q, top_k=k, mode="dense"),
    "keyword": lambda s, q, k: s.search(q, top_k=k, mode="keyword"),
    "hyde":    lambda s, q, k: s.hyde_search(q, top_k=k),
    "fielded": lambda s, q, k: (s.store.query_fielded(q, ["title", "text"], top_k=k)
                                if getattr(s.store, "query_fielded", None) else s.search(q, k, mode="keyword")),
}


def _json(txt):
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class T:
    """Transcript writer — prints and buffers."""
    def __init__(self): self.buf = []
    def __call__(self, *a):
        line = " ".join(str(x) for x in a); print(line, flush=True); self.buf.append(line)
    def save(self, path): Path(path).write_text("\n".join(self.buf))


def explore_query(session, gen, q, gold, titles, memory, skills, max_hops, t):
    gold = [str(g) for g in gold]
    t(f"\nSTARTING QUERY: {q}")
    t(f"GOLD (oracle, scoring only): {[f'{g}({ti})' for g, ti in zip(gold, titles)]}")

    recalled = memory.recall(q, k=4)
    t("\n--- LEARNING FROM PREVIOUS QUERIES (recalled from long-term memory) ---")
    t("\n".join(f"  - {m.content}" for m in recalled) or "  (none yet — first queries)")

    pooled, attribution, winning = [], {}, []
    for hop in range(1, max_hops + 1):
        findings = memory.working_context(max_chars=500, kinds={"finding"})
        dynamic = (FIXED_BASE
                   + "\n\nMEMORY (wins from earlier queries):\n" + ("\n".join(f"- {m.content}" for m in recalled) or "(none)")
                   + "\n\nIN-SESSION FINDINGS (earlier hops of THIS query):\n" + (findings or "(none)")
                   + "\n\nSKILLS:\n" + skills.summaries())
        t(f"\n--- HOP {hop} ---")
        if hop == 1:
            t("DYNAMIC PROMPT (fixed base + memory + findings + skills):")
            t("  " + dynamic.replace("\n", "\n  ")[:900] + " …")
        else:
            t("DYNAMIC PROMPT now also carries IN-SESSION FINDINGS:\n  " + (findings or "(none)"))
        try:
            plan = _json(gen.complete(dynamic + f"\n\nQUESTION: {q}"))
        except Exception as e:
            plan = None; t(f"  [llm error: {e}]")
        steps = (plan or {}).get("plan", []) if isinstance(plan, dict) else []
        t(f"LLM APPROACH: reason={ (plan or {}).get('reason','')[:120] }")
        for st in steps:
            t(f"   query: ({st.get('mode')}) \"{st.get('query','')[:70]}\"")

        newly = []
        for st in steps:
            mode, sub = st.get("mode", "hybrid"), st.get("query", q)
            fn = MODES.get(mode, MODES["hybrid"])
            try:
                rs = fn(session, sub, 30)
            except Exception:
                continue
            ids = rs.ids()
            pooled.append(ids)
            for g in gold:
                if g not in attribution and g in ids[:10]:
                    attribution[g] = f"({mode}) \"{sub[:50]}\" @rank{ids.index(g)+1}"
                    winning.append({"mode": mode, "query": sub})
                    newly.append(g)
        fused_ids = _fuse_ids(pooled)[:10]
        got = [g for g in gold if g in fused_ids]
        t("EXECUTED → per-gold attribution:")
        for g, ti in zip(gold, titles):
            t(f"   gold {g} ({ti[:34]}): {attribution.get(g, 'NOT found yet')}")
        memory.observe(f"hop{hop}: found {sorted(set(newly))}; fused all-golds={len(got)}/{len(gold)}", kind="finding")
        t(f"IN-HOP MEMORY WRITE: hop{hop} found {sorted(set(newly))}; fused {len(got)}/{len(gold)} golds in top-10")
        if len(got) == len(gold):
            t(f"RESULT: SOLVED in {hop} hop(s).")
            break
    solved = len(got) == len(gold)
    if not solved:
        t(f"RESULT: partial — {len(got)}/{len(gold)} golds after {max_hops} hops.")

    if winning:
        wp = "; ".join(f"{w['mode']}:'{w['query'][:40]}'" for w in winning)
        memory.remember(f"query \"{q[:70]}\" ({len(gold)}-hop) solved={solved}: winning queries = {wp}",
                        kind="skill_win", winning=winning, solved=solved)
        t(f"\n--- CROSS-QUERY MEMORY WRITE ---\n  remembered winning queries: {wp}")
    return solved, winning


def _rs(ids):
    from search_as_code.types import Hit, ResultSet
    return ResultSet([Hit(id=i, score=1.0 / (r + 1)) for r, i in enumerate(ids)])


def _fuse_ids(lists, k=60):
    s = {}
    for lst in lists:
        for r, i in enumerate(lst):
            s[i] = s.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(s, key=lambda i: -s[i])


def distill(gen, wins, forge, t):
    t("\n\n################  DISTILL — LLM creates new primitives / subagents  ################")
    patterns = [w.content for w in wins]
    t(f"Fed the LLM {len(patterns)} captured winning patterns, e.g.:")
    for p in patterns[:4]:
        t("  - " + p[:130])
    prompt = ("You are improving a retrieval agent. Below are winning query patterns captured from "
              "solved multi-hop questions (each lists the mode:query that surfaced a gold doc). Propose "
              "REUSABLE modules. Return ONLY JSON: {\"skills\":[{\"name\":\"..\",\"when_to_use\":\"..\","
              "\"retrievers\":[\"hyde\",\"fielded\",\"hybrid\"],\"combine\":\"fuse\"}], "
              "\"subagents\":[{\"name\":\"..\",\"when_to_use\":\"..\",\"plan\":[\"arsenal_single\"]}], "
              "\"rules\":[\"..\"]}\n\nWINNING PATTERNS:\n" + "\n".join(patterns[:20]))
    prop = _json(gen.complete(prompt)) or {}
    t("\nLLM PROPOSED:")
    t("  " + json.dumps(prop, indent=2).replace("\n", "\n  ")[:1200])
    created = []
    for sk in prop.get("skills", [])[:4]:
        try:
            forge.create_skill(sk["name"], sk.get("when_to_use", ""), sk.get("retrievers", ["hybrid"]),
                               combine=sk.get("combine", "fuse"), origin="llm")
            created.append(("skill", sk["name"]))
        except Exception as e:
            t(f"  [skill {sk.get('name')} skipped: {e}]")
    for sa in prop.get("subagents", [])[:4]:
        try:
            forge.create_subagent(sa["name"], sa.get("when_to_use", ""), sa.get("plan", ["arsenal_single"]))
            created.append(("subagent", sa["name"]))
        except Exception:
            pass
    for rule in prop.get("rules", [])[:6]:
        forge.refine_prompt(rule)
    t(f"\nFORGED + PERSISTED: {created} + {len(prop.get('rules', []))} rule(s)")
    return created


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_hops = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    t = T()
    rows = []
    for hop in (2, 3, 4):
        rs = [json.loads(l) for l in (DATA / f"multihop_{hop}docs_queries.jsonl").open()][:n]
        for r in rs:
            r["n_docs"] = hop
        rows += rs
    random.seed(1); random.shuffle(rows); rows = rows[:n]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda x: em.encode(list(x), normalize_embeddings=True, batch_size=128).tolist()  # noqa: E731
    gen = LLM()
    session = sac.Session("opensearch", index="hotpotqa", dim=common.DIM, hosts=[common.OS_HOST],
                          text_field="text", vector_field="vector", embedder=embed, generator=gen.as_generator())
    memory = AgentMemory(path=str(HERE / "transparent_memory.jsonl"), embedder=session.embedder)
    skills = SkillRegistry(embedder=session.embedder)
    store = HarnessStore(path=str(HERE / "transparent_store"))
    forge = HarnessForge(store, skills, memory)

    t("################  FIXED (base) PROMPT — same for every query  ################")
    t(FIXED_BASE)
    t(f"\n################  SKILLS available at start  ################\n{skills.summaries()}")

    solved = 0
    for i, r in enumerate(rows, 1):
        t(f"\n\n########################  QUERY {i}/{len(rows)}  ({r['n_docs']}-hop)  ########################")
        ok, _ = explore_query(session, gen, r["query"], r["gold_ids"], r.get("titles", [""] * len(r["gold_ids"])),
                              memory, skills, max_hops, t)
        solved += int(ok)
    t(f"\n\n################  EXPLORATION SUMMARY  ################\nsolved {solved}/{len(rows)} ; "
      f"memory: {memory.stats()}")

    wins = [m for m in memory.longterm if m.kind == "skill_win"]
    distill(gen, wins, forge, t)

    t("\n################  MEMORY (what it retained)  ################")
    t(f"  cross-query long-term ({len(memory.longterm)}): e.g.")
    for m in memory.longterm[:3]:
        t("   - " + m.content[:120])
    t(f"  in-session working ({len(memory.working)}): e.g.")
    for m in memory.working[-3:]:
        t("   - [" + m.kind + "] " + m.content[:100])
    t("\n################  FINAL SKILLS (incl. LLM-forged)  ################\n" + skills.summaries())
    t("\n################  LEARNED RULES (self-modifiable prompt)  ################\n" + store.learnings_block())

    t.save(HERE / "transcript.md")
    print(f"\n[saved transcript.md + transparent_store/ + transparent_memory.jsonl]")


if __name__ == "__main__":
    main()
