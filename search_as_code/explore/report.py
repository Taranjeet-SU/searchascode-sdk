"""CSV export for a ProfilePack — flat tables for spreadsheets / quick review.

Writes up to three CSVs next to the pack (or a chosen dir):
- ``stages.csv``     one row per pipeline stage (status/timing/summary)
- ``clusters.csv``   one row per sample cluster (size, content-type mix, LLM profile)
- ``documents.csv``  one row per sampled document (cluster, content_type, snippet)
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from ..primitives import content_type
from .pack import ProfilePack


def write_csv_report(pack: ProfilePack, out_dir: str | None = None) -> dict[str, str]:
    """Write the CSV tables; return {name: path}. Missing artifacts are skipped."""
    out = Path(out_dir) if out_dir else pack.root
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    # ---- stages.csv ------------------------------------------------------
    spath = out / "stages.csv"
    with spath.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "status", "seconds", "summary", "artifacts", "note"])
        for name, st in pack.manifest.get("stages", {}).items():
            summary = "; ".join(f"{k}={v}" for k, v in (st.get("summary") or {}).items())
            w.writerow([name, st.get("status", ""), st.get("seconds", ""),
                        summary, " ".join(st.get("artifacts", [])), st.get("note", "")])
    written["stages"] = str(spath)

    sample = pack.read_jsonl("sample.jsonl")
    profile = pack.read_json("content_profile.json") or {}
    llm_by_cluster = profile.get("llm_by_cluster") or {}
    sizes = (pack.read_json("sample_meta.json") or {}).get("cluster_sizes", {})

    if sample:
        # ---- documents.csv ----------------------------------------------
        dpath = out / "documents.csv"
        with dpath.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "cluster", "content_type", "chars", "snippet"])
            for r in sample:
                text = r.get("text") or ""
                snippet = " ".join(text.split())[:200]
                w.writerow([r.get("id"), r.get("cluster"),
                            content_type(text), len(text), snippet])
        written["documents"] = str(dpath)

        # ---- clusters.csv -----------------------------------------------
        by_cluster: dict = {}
        for r in sample:
            by_cluster.setdefault(r.get("cluster"), []).append(r)
        cpath = out / "clusters.csv"
        with cpath.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cluster", "pool_size", "n_sampled", "content_types",
                        "llm_profile", "example_snippet"])
            for c in sorted(by_cluster, key=lambda x: (x is None, x)):
                rows = by_cluster[c]
                cts = Counter(content_type(r.get("text") or "") for r in rows)
                ct_str = ", ".join(f"{k}:{v}" for k, v in cts.most_common())
                example = " ".join((rows[0].get("text") or "").split())[:160]
                w.writerow([c, sizes.get(str(c), ""), len(rows), ct_str,
                            (llm_by_cluster.get(str(c)) or "").strip(), example])
        written["clusters"] = str(cpath)

    return written
