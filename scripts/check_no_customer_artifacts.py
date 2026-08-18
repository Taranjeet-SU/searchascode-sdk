#!/usr/bin/env python3
"""Block customer/internal artifacts from being staged or pushed.

`soul.md` rule 1 says customer- and corpus-specific work never reaches the public repo, and
`CLAUDE.md` restates it as "never `git add -A`". That discipline has already failed twice:
`origin/main` carries `experiments/browsecomp/` and `experiments/su_multihop/` files (GOV-1),
51 `phase4/altera*` files are gitignored-but-tracked (GOV-3), and a customer SSH public key sits
in the repo root matching no ignore rule (GOV-2).

A rule an agent must remember is not a control. This is the control: run it from pre-commit
(staged files) or from CI (`--check-tree`, over everything git tracks).

    python3 scripts/check_no_customer_artifacts.py <paths...>
    python3 scripts/check_no_customer_artifacts.py --check-tree
"""
from __future__ import annotations

import re
import subprocess
import sys

# Path patterns that must never be tracked in the public repository.
BLOCKED_PATHS = [
    (re.compile(r"(^|/)phase4/.*altera", re.I), "phase4 customer (Altera) work — GOV-3/LEG-3"),
    (re.compile(r"altera", re.I), "customer name in path — soul.md rule 1"),
    (re.compile(r"(^|/)experiments/(browsecomp|su_multihop)/", re.I),
     "internal benchmark data marked 'do not push' — GOV-1"),
    (re.compile(r"searchunify", re.I), "customer name in path — soul.md rule 1"),
    (re.compile(r"\.pub$"), "SSH public key — GOV-2"),
    (re.compile(r"(^|/)id_(rsa|ed25519|ecdsa)"), "SSH private key"),
    (re.compile(r"\.(pem|key|p12|pfx)$"), "credential material"),
    (re.compile(r"(^|/)\.secrets$|(^|/)\.env$"), "secrets file"),
]

# Content patterns worth refusing even in an otherwise-innocent file.
BLOCKED_CONTENT = [
    (re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key material"),
    (re.compile(rb"sk-[A-Za-z0-9]{20,}"), "an OpenAI-style API key"),
]

TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".sh", ".toml", ".cfg", ".ini"}


def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def check(paths: list[str]) -> list[str]:
    problems = []
    for path in paths:
        # This script names the very patterns it blocks, so exempt itself and the audit log.
        if path in ("scripts/check_no_customer_artifacts.py", "issues.md",
                    ".pre-commit-config.yaml", "CLAUDE.md", "soul.md", "STRUCTURE.md"):
            continue
        for pat, why in BLOCKED_PATHS:
            if pat.search(path):
                problems.append(f"{path}: {why}")
                break
        if any(path.endswith(s) for s in TEXT_SUFFIXES):
            try:
                with open(path, "rb") as fh:
                    blob = fh.read(200_000)
            except OSError:
                continue
            for pat, why in BLOCKED_CONTENT:
                if pat.search(blob):
                    problems.append(f"{path}: contains {why}")
    return problems


def main(argv: list[str]) -> int:
    paths = _tracked_files() if "--check-tree" in argv else [a for a in argv if not a.startswith("-")]
    problems = check(paths)
    if problems:
        sys.stderr.write("Refusing: customer/internal artifacts must not be tracked "
                         "(soul.md rule 1, issues.md GOV-1/2/3).\n")
        for p in problems:
            sys.stderr.write(f"  - {p}\n")
        sys.stderr.write("\nMove the file out of the repo, or add it to .gitignore AND "
                         "`git rm --cached` it if it is already tracked.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
