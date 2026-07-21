# Phase 2 probe — constraint / cross-document queries (the SU archetype)

Synthetic release-notes corpus (products × features → introduced-in-version), 12 constraint
queries: *"latest release of P where I can use A and B"* → answer = `max(intro[A], intro[B])`.

| approach | answer accuracy |
|---|---|
| dense RAG | **5/12 = 0.42** |
| **SAC (retrieve + compute)** | **12/12 = 1.00** |

**Why:** dense returns a passage's version but cannot compute an intersection/max across two
documents. SAC writes code that retrieves both facts and computes the answer:

```python
a = sac.search("<A>", top_k=3, filter={"product": P})
b = sac.search("<B>", top_k=3, filter={"product": P})
va = a.top(1)[0].get("version"); vb = b.top(1)[0].get("version")
answer = max([va, vb], key=lambda v: tuple(int(x) for x in v.split(".")))
```

## Takeaways
1. **Code-as-search wins structurally on constraint/cross-doc queries** (1.00 vs 0.42) — the
   opposite of FiQA (simple semantic), where it only ties dense. The value is *retrieval +
   computation*.
2. **Primitive sufficiency:** for max-version constraints the existing primitives suffice
   (`search` + metadata `filter` + `.get` + code). The sandbox (arbitrary Python) IS the
   primitive dense/tool-calling lack. Reproduce: `python -m phase2.synth_eval`.
3. **Next:** harder constraints (ranges/limits, multi-entity joins, tabular facts) to find
   where a **KG + constraint primitives** become necessary — the Phase 2 roadmap.
