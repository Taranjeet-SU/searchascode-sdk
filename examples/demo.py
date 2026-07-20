"""Search-as-code end-to-end demo — runs with zero external services.

    python examples/demo.py

Shows the whole loop: connect to a backend, ingest, then run an *agent-style*
Python program in the sandbox that fans out queries, fuses, reranks, and returns
only compact evidence — the bulky candidate set never leaves the sandbox.

Swap `Session("memory")` for `Session("qdrant", collection=...)` (or chroma /
pgvector) and NOTHING else in the agent code changes. That portability is the
whole point.
"""

from search_as_code import LocalExecutor, Session

CORPUS = [
    {"id": "d1", "text": "CVE-2024-1234 affects OpenSSL 3.0 heap overflow", "metadata": {"vendor": "openssl", "year": 2024}},
    {"id": "d2", "text": "Mozilla security advisory MFSA 2024-01 use-after-free in Firefox", "metadata": {"vendor": "mozilla", "year": 2024}},
    {"id": "d3", "text": "Jenkins advisory: arbitrary file read via CLI", "metadata": {"vendor": "jenkins", "year": 2023}},
    {"id": "d4", "text": "vector databases enable semantic retrieval for agents", "metadata": {"vendor": "none", "year": 2024}},
    {"id": "d5", "text": "reciprocal rank fusion combines lexical and dense signals", "metadata": {"vendor": "none", "year": 2025}},
]

# The program an agent would generate — portable across every backend.
AGENT_CODE = """
# 1. Fan out over query variants (concurrent, no serial model turns)
queries = ["security advisory vulnerability", "CVE heap overflow", "Firefox use-after-free"]
candidates = sac.search_many(queries, top_k=5, mode="hybrid")

# 2. Keep bulky candidates in the sandbox, out of the model context
sac.remember("candidates", candidates)

# 3. Filter + rerank down to the few facts worth returning
recent = candidates.where(lambda h: h.get("year", 0) >= 2024)
best = sac.rerank("security vulnerability advisory", recent, top_k=3)

print(f"scanned {len(candidates)} candidates, kept {len(best)}")

# 4. Only this compact evidence returns to the model
evidence = best.to_evidence(fields=["vendor", "year"], max_chars=120)
"""


def main() -> None:
    session = Session("memory")           # <-- change backend here only
    session.add(CORPUS)

    box = LocalExecutor(session)
    result = box.run(AGENT_CODE)

    print("=" * 60)
    print("sandbox stdout:", result.stdout.strip())
    print("ok:", result.ok, "| state kept in sandbox:", result.state_keys)
    print("-" * 60)
    print("evidence returned to model:")
    for row in result.evidence or []:
        print("  ", row)
    print("=" * 60)
    print("payload the model actually sees:")
    print(result.for_model())


if __name__ == "__main__":
    main()
