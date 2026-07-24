"""Thin CLI over the Altera primitives so a human/LLM agent can drive retrieval by hand.

    python -m phase4.altera_call <dense|keyword|kb> "<query>" [k]

Prints ranked results (rank, id/url, title, text snippet) so the caller can inspect the
state and decide the next call. (dense loads gte-alt-v1 on first use; kb/keyword don't.)
"""
from __future__ import annotations

import sys

from phase4 import altera

FN = {"dense": altera.dense, "keyword": altera.bm25_doc, "kb": altera.bm25_kg}


def main():
    prim = sys.argv[1]
    query = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    docs = FN[prim](query, k)
    print(f"# {prim}('{query}', k={k}) -> {len(docs)} hits")
    for i, d in enumerate(docs):
        loc = d.get("url") or d.get("id")
        print(f"{i+1}. [{loc}]  {str(d.get('title') or '')[:85]}")
        print(f"    {str(d.get('text') or '').strip()[:320]}".replace("\n", " "))


if __name__ == "__main__":
    main()
