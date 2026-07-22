# Phase 3 (planned) — multi-backend primitive integration + cross-DB relevance

Prove the core thesis ("one primitive API, any vector DB") by implementing the remaining
adapters, spinning each backend up, running the primitive conformance suite, and running
**the same HotpotQA multi-hop eval on every backend** to compare relevance.

## Backends to integrate
| backend | how it runs here | status |
|---|---|---|
| memory | in-process | ✅ shipped |
| OpenSearch | tarball, :9200 | ✅ shipped |
| Qdrant | embedded/local | ✅ adapter (untested live) |
| Chroma | in-process/embedded | ✅ adapter |
| pgvector (Postgres) | local Postgres | ✅ adapter |
| **FAISS** | in-process (no server) | ⬜ new adapter |
| **nmslib** | in-process (no server) | ⬜ new adapter |
| **Elasticsearch** | tarball/local :9200-ish | ⬜ new adapter (≈ OpenSearch) |
| **Milvus** | milvus-lite (in-process) or server | ⬜ new adapter |
| **SQLite / SQL** | in-process (brute-force / sqlite-vss) | ⬜ new adapter |
| **MongoDB** | local mongod + Atlas-vector (or brute-force) | ⬜ new adapter |
| **Pinecone** | cloud API (needs key) | ⬜ new adapter |

In-process ones (FAISS, nmslib, SQLite, Milvus-lite, Chroma) need **no server** — safest here
given the Docker socket is permission-denied. Server ones (Elasticsearch, MongoDB, Milvus-full)
run from tarball/binary as our user; Pinecone needs a cloud key.

## Steps
1. **Implement each adapter** against the `VectorStore` contract (`adapters/base.py`): `query_vector`,
   metadata filter → native DSL, capabilities honest (many are dense-only → keyword/hybrid emulated).
2. **Spin up** each backend (in-process where possible; tarball/binary for servers).
3. **Primitive conformance suite** — run `tests/test_opensearch.py`-style tests per backend (dense,
   filter, get, count; keyword/hybrid/regex where native, else emulated). Produces a backend×primitive
   pass matrix (the "capability emulation keeps agent code portable" proof).
4. **Cross-DB relevance** — ingest the same HotpotQA reduced corpus into each backend, run dense +
   SAC on N queries, report recall@10 / all_found@10 **per backend**. Expect near-parity on dense
   (same vectors), differences only where a backend adds native hybrid/rerank.
5. **Report** — conformance matrix + relevance-per-backend table + notes (setup cost, native features).

## Deliverables
- `search_as_code/adapters/{faiss,nmslib,elasticsearch,milvus,sqlite,mongo,pinecone}.py`
- `tests/test_<backend>.py` (skip if backend absent)
- `phase3/spin_backends.py` (bring up), `phase3/conformance.py` (matrix), `phase3/cross_db_relevance.py`
- Report: backend × primitive conformance + HotpotQA relevance per backend.

## Risks / notes
- Docker socket is permission-denied → prefer in-process/tarball; Pinecone needs a key.
- ES ≈ OpenSearch adapter (fork with client swap). Milvus-lite avoids a server.
- SQLite/SQL adapter doubles as the "no vector-DB needed" brute-force reference.
