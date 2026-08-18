#!/usr/bin/env python3
"""Check that every relative link in the repo's markdown resolves.

`docs/` is 13 loose .md files with no build step and no link checker (STR-6), which is why
DOC-6 survived: the public README links to `benchmark_changelog.md`, a file that is not on
`origin/main`. A link checker is a five-line CI job.

    python3 scripts/check_doc_links.py [--public]

``--public`` additionally checks that every link target reachable from the published files is
itself tracked in git — the check that would have caught DOC-6 specifically.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#", "tel:")


def _tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    return set(out.split())


def main(argv: list[str]) -> int:
    public_only = "--public" in argv
    tracked = _tracked()
    md_files = sorted(p for p in ROOT.rglob("*.md")
                      if ".venv" not in p.parts and "node_modules" not in p.parts
                      and str(p.relative_to(ROOT)) in tracked)

    broken: list[str] = []
    untracked: list[str] = []
    for md in md_files:
        rel_md = md.relative_to(ROOT)
        for m in LINK_RE.finditer(md.read_text(errors="ignore")):
            target = m.group(1).split("#", 1)[0]
            if not target or target.startswith(SKIP_PREFIXES):
                continue
            resolved = (md.parent / target).resolve()
            try:
                rel = resolved.relative_to(ROOT)
            except ValueError:
                broken.append(f"{rel_md}: '{target}' escapes the repository")
                continue
            if not resolved.exists():
                broken.append(f"{rel_md}: '{target}' does not exist")
            elif public_only and resolved.is_file() and str(rel) not in tracked:
                untracked.append(f"{rel_md}: '{target}' exists locally but is NOT tracked "
                                 f"(broken for anyone who clones — see DOC-6)")

    for line in broken + untracked:
        print(f"  - {line}")
    if broken or untracked:
        print(f"\n{len(broken)} broken link(s), {len(untracked)} link(s) to untracked files "
              f"across {len(md_files)} markdown files.")
        return 1
    print(f"All relative links resolve across {len(md_files)} markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
