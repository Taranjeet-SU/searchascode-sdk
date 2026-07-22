"""Performance benchmark harness for search-as-code.

Subcommands (each prints a table and writes raw JSON to benchmarks/results/):

    python -m benchmarks.bench scalability   # A: ingest throughput, corpus-size latency, fan-out
    python -m benchmarks.bench throughput     # B: single + concurrent QPS on live OpenSearch
    python -m benchmarks.bench micro          # E2: primitive ops/sec
    python -m benchmarks.bench resilience     # E3: retry wrapper + chunking overhead
    python -m benchmarks.bench embedding      # E4: gte-base embedding throughput (GPU)
    python -m benchmarks.bench all            # everything that needs no OpenAI key

Design: measurements isolate the component under test. Throughput uses a random
768-d query vector (latency doesn't depend on vector *meaning*), so no embedder
load is needed except for the dedicated embedding benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as stats
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

import search_as_code as sac
from search_as_code import primitives as P
from search_as_code._resilience import chunked, with_retry
from search_as_code.types import Document, Hit, ResultSet

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
OS_HOST = {"host": "127.0.0.1", "port": 9200}
WORDS = ("bank account cheque deposit money order invest stock fund tax return loan "
         "credit debit interest rate mortgage dividend broker portfolio balance "
         "transfer wire savings retirement pension bond yield currency exchange").split()


def _pctile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


def _summary(times_ms: list[float]) -> dict[str, float]:
    return {
        "n": len(times_ms),
        "mean_ms": round(stats.fmean(times_ms), 3),
        "p50_ms": round(_pctile(times_ms, 50), 3),
        "p95_ms": round(_pctile(times_ms, 95), 3),
        "p99_ms": round(_pctile(times_ms, 99), 3),
        "qps": round(1000.0 / stats.fmean(times_ms), 1) if times_ms else 0.0,
    }


def _rand_text(rng: random.Random, n: int = 12) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def _save(name: str, payload: dict[str, Any]) -> None:
    payload["_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[saved] benchmarks/results/{name}.json")


# ============================================================ A. SCALABILITY
def bench_scalability() -> dict[str, Any]:
    rng = random.Random(0)
    out: dict[str, Any] = {}

    # --- A2/A3: memory backend, query latency + build vs corpus size ---
    print("\n=== A2/A3 memory backend: build + query latency vs corpus size ===")
    dim = 128
    a2, a3 = {}, {}
    for size in (1_000, 10_000, 50_000):
        docs = [Document(id=str(i), text=_rand_text(rng),
                         vector=[rng.random() for _ in range(dim)]) for i in range(size)]
        store = sac.connect("memory")
        t0 = time.perf_counter()
        store.upsert(docs)
        qv = [rng.random() for _ in range(dim)]
        store.query_vector(qv, top_k=10)  # triggers matrix build
        build_s = time.perf_counter() - t0
        matrix_mb = round(getattr(store, "_matrix").nbytes / 1e6, 1)
        times = []
        for _ in range(200):
            qv = [rng.random() for _ in range(dim)]
            t = time.perf_counter()
            store.query_vector(qv, top_k=10)
            times.append((time.perf_counter() - t) * 1000)
        a2[size] = _summary(times)
        a3[size] = {"build_s": round(build_s, 3), "matrix_mb": matrix_mb}
        print(f"  {size:>6} docs: build {build_s:6.3f}s  matrix {matrix_mb:6.1f}MB  "
              f"query p50 {a2[size]['p50_ms']:.3f}ms  p95 {a2[size]['p95_ms']:.3f}ms  "
              f"{a2[size]['qps']:.0f} qps")
    out["A2_query_latency_vs_corpus"] = a2
    out["A3_build_and_footprint"] = a3

    # --- A4: fan-out (search_many) scaling ---
    print("\n=== A4 fan-out (search_many) scaling ===")
    s = sac.Session("memory")
    s.add([{"id": str(i), "text": _rand_text(rng)} for i in range(10_000)])
    variants = [_rand_text(rng, 6) for _ in range(16)]
    a4 = {}
    for nq in (1, 4, 8, 16):
        qs = variants[:nq]
        t = time.perf_counter()
        s.search_many(qs, top_k=10, concurrency=nq)
        dt = time.perf_counter() - t
        a4[nq] = {"total_s": round(dt, 4), "ms_per_query": round(dt * 1000 / nq, 3)}
        print(f"  {nq:>2} queries: {dt*1000:7.1f}ms total  ({dt*1000/nq:.2f} ms/query)")
    out["A4_fanout_scaling"] = a4

    # --- A1: OpenSearch ingest throughput vs batch size ---
    try:
        print("\n=== A1 OpenSearch ingest throughput vs batch size ===")
        idim, ndocs = 64, 5_000
        idocs = [Document(id=str(i), text=_rand_text(rng),
                          vector=[rng.random() for _ in range(idim)]) for i in range(ndocs)]
        a1 = {}
        for bs in (100, 500, 1000):
            store = sac.connect("opensearch", index="sac_bench_ingest", dim=None,
                                hosts=[OS_HOST], batch_size=bs)
            store.client.indices.delete(index="sac_bench_ingest", ignore=[404])
            store.ensure_index(idim)
            t = time.perf_counter()
            store.upsert(idocs)
            dt = time.perf_counter() - t
            a1[bs] = {"docs": ndocs, "total_s": round(dt, 3), "docs_per_sec": round(ndocs / dt, 1)}
            print(f"  batch={bs:>4}: {dt:6.2f}s  {ndocs/dt:8.1f} docs/sec")
        store.client.indices.delete(index="sac_bench_ingest", ignore=[404])
        out["A1_ingest_throughput"] = a1
    except Exception as e:
        out["A1_ingest_throughput"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  [skipped A1: {type(e).__name__}: {e}]")

    _save("scalability", out)
    return out


# ============================================================ B. THROUGHPUT
def bench_throughput(index: str = "fiqa", dim: int = 768, n: int = 300) -> dict[str, Any]:
    rng = random.Random(1)
    out: dict[str, Any] = {"index": index}
    try:
        store = sac.connect("opensearch", index=index, dim=None, hosts=[OS_HOST])
        count = store.count()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        print(f"[throughput skipped: {e}]")
        _save("throughput", out)
        return out
    out["doc_count"] = count
    print(f"\n=== B1 single-thread QPS per mode ({index}, {count} docs) ===")

    def rvec():
        v = np.asarray([rng.random() for _ in range(dim)], dtype=np.float32)
        return (v / (np.linalg.norm(v) or 1.0)).tolist()

    def rquery():
        return _rand_text(rng, 5)

    modes: dict[str, Callable[[], Any]] = {
        "dense": lambda: store.query_vector(rvec(), top_k=10),
        "keyword": lambda: store.query_keyword(rquery(), top_k=10),
        "hybrid": lambda: store.query_hybrid(rvec(), rquery(), top_k=10),
        "regex": lambda: store.query_regex(".*bank.*", top_k=10),
    }
    b1 = {}
    for mode, fn in modes.items():
        for _ in range(10):  # warm up
            fn()
        times = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t) * 1000)
        b1[mode] = _summary(times)
        print(f"  {mode:>8}: {b1[mode]['qps']:7.1f} qps  p50 {b1[mode]['p50_ms']:6.2f}ms  "
              f"p95 {b1[mode]['p95_ms']:6.2f}ms")
    out["B1_single_thread_qps"] = b1

    print("\n=== B2 concurrent QPS vs workers (dense) ===")
    b2 = {}
    for workers in (1, 2, 4, 8, 16):
        qvs = [rvec() for _ in range(n)]
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda v: store.query_vector(v, top_k=10), qvs))
        dt = time.perf_counter() - t
        b2[workers] = {"qps": round(n / dt, 1), "total_s": round(dt, 3)}
        print(f"  {workers:>2} workers: {n/dt:8.1f} qps  ({dt:.2f}s for {n})")
    out["B2_concurrent_qps"] = b2
    _save("throughput", out)
    return out


# ============================================================ E2. MICRO
def bench_micro(pool: int = 200, iters: int = 300) -> dict[str, Any]:
    rng = random.Random(2)
    dim = 64

    def mk_rs(k: int) -> ResultSet:
        hits = []
        for i in range(k):
            v = [rng.random() for _ in range(dim)]
            hits.append(Hit(id=str(i), score=rng.random(),
                            document=Document(id=str(i), text=_rand_text(rng), vector=v)))
        return ResultSet(hits)

    rs_a, rs_b = mk_rs(pool), mk_rs(pool)
    qv = [rng.random() for _ in range(dim)]
    sess = sac.Session("memory")  # for semantic_dedup (uses embedder)

    ops: dict[str, Callable[[], Any]] = {
        "fuse(2xN)": lambda: P.fuse([rs_a, rs_b]),
        "relative_score_fusion": lambda: P.relative_score_fusion([rs_a, rs_b]),
        "dedup": lambda: rs_a.dedup(),
        "score_cutoff(band)": lambda: P.score_cutoff(rs_a, method="band"),
        "diversity_quota": lambda: P.diversity_quota(rs_a, key=lambda h: h.id[0], max_per_group=2),
        "mmr(top10)": lambda: P.mmr(qv, rs_a, top_k=10),
        "rerank(lexical)": lambda: P.rerank("bank account", rs_a, top_k=10),
        "semantic_dedup": lambda: sess.semantic_dedup(rs_a, threshold=0.9),
        "confidence": lambda: P.confidence(rs_a),
    }
    out: dict[str, Any] = {"pool_size": pool}
    print(f"\n=== E2 primitive micro-throughput (pool={pool}) ===")
    res = {}
    for name, fn in ops.items():
        for _ in range(5):
            fn()
        times = []
        for _ in range(iters):
            t = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t) * 1000)
        m = _summary(times)
        res[name] = {"ms_per_call": m["mean_ms"], "ops_per_sec": round(1000 / m["mean_ms"], 1),
                     "p95_ms": m["p95_ms"]}
        print(f"  {name:>22}: {res[name]['ops_per_sec']:9.1f} ops/s  "
              f"({res[name]['ms_per_call']:.4f} ms/call)")
    out["E2_primitive_ops"] = res
    _save("micro", out)
    return out


# ============================================================ E3. RESILIENCE
def bench_resilience(iters: int = 50_000) -> dict[str, Any]:
    out: dict[str, Any] = {}
    print("\n=== E3 resilience overhead ===")

    def noop():
        return 1

    t = time.perf_counter()
    for _ in range(iters):
        noop()
    direct = (time.perf_counter() - t) / iters * 1e6

    t = time.perf_counter()
    for _ in range(iters):
        with_retry(noop, attempts=3, backoff=0)
    wrapped = (time.perf_counter() - t) / iters * 1e6
    out["retry_wrapper"] = {"direct_us": round(direct, 4), "wrapped_us": round(wrapped, 4),
                            "overhead_us": round(wrapped - direct, 4)}
    print(f"  with_retry overhead: {wrapped-direct:.3f} µs/call "
          f"(direct {direct:.3f} → wrapped {wrapped:.3f})")

    # chunked() throughput
    data = list(range(1_000_000))
    t = time.perf_counter()
    n = sum(len(c) for c in chunked(data, 500))
    dt = time.perf_counter() - t
    out["chunked"] = {"items": n, "s": round(dt, 4), "items_per_sec": round(n / dt, 0)}
    print(f"  chunked(): {n/dt/1e6:.1f}M items/sec")
    _save("resilience", out)
    return out


# ============================================================ E4. EMBEDDING
def bench_embedding(n: int = 2_000) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except Exception as e:
        out["error"] = f"missing dep: {e}"
        print(f"[embedding skipped: {e}]")
        _save("embedding", out)
        return out
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("thenlper/gte-base", device=dev)
    rng = random.Random(3)
    texts = [_rand_text(rng, 40) for _ in range(n)]
    model.encode(texts[:64], show_progress_bar=False)  # warm up
    print(f"\n=== E4 embedding throughput (gte-base, {dev}) ===")
    res = {}
    for bs in (32, 128, 256):
        t = time.perf_counter()
        model.encode(texts, batch_size=bs, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
        dt = time.perf_counter() - t
        res[bs] = {"texts": n, "s": round(dt, 3), "texts_per_sec": round(n / dt, 1)}
        print(f"  batch={bs:>3}: {n/dt:8.1f} texts/sec  ({dt:.2f}s for {n})")
    out["E4_embedding_throughput"] = {"device": dev, "by_batch_size": res}
    _save("embedding", out)
    return out


SUITES = {
    "scalability": bench_scalability,
    "throughput": bench_throughput,
    "micro": bench_micro,
    "resilience": bench_resilience,
    "embedding": bench_embedding,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="search-as-code benchmark harness")
    ap.add_argument("suite", choices=[*SUITES, "all"])
    args = ap.parse_args()
    if args.suite == "all":
        for name, fn in SUITES.items():
            print(f"\n{'#'*70}\n# {name}\n{'#'*70}")
            fn()
    else:
        SUITES[args.suite]()


if __name__ == "__main__":
    main()
