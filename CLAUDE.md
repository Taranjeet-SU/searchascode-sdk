# CLAUDE.md — always-loaded instructions for this repo

Read this first, then [`soul.md`](soul.md) (the docs constitution) and [`STRUCTURE.md`](STRUCTURE.md)
(the repo map). `soul.md` describes itself as "our `CLAUDE.md`-style always-loaded file" — it is not
auto-loaded by anything; **this** file is. Keep both in sync.

## 1. Issue logging is a standing routine (not a one-off task)

[`issues.md`](issues.md) at the repo root is the **canonical repo-wide audit log**. Whenever you
find a problem while working here — no matter what the task was — **append it to `issues.md` before
you finish the turn**. This includes problems you hit incidentally, worked around, or decided not to
fix.

Log three kinds of issue:

| category | tag | what it means |
|---|---|---|
| **approach** | `[A]` | the design, the claim, or the experiment methodology is wrong |
| **code** | `[C]` | a bug, a fragility, a silent failure, an efficiency problem |
| **redundancy** | `[R]` | duplicate implementations, dead code, unreachable paths |

Format — one entry per issue, grouped in a section per area, newest section appended at the bottom:

```markdown
#### 🟥 AREA-N `[C]` One-line title stating the defect
`path/to/file.py:123` — what is wrong, why it matters, and what the fix is. Cite file:line for
every claim. Include the reproduction/verification command when there is one.
```

Severity: 🟥 blocker or wrong results · 🟧 friction or risk · 🟨 minor or cosmetic.

**Rules:**
- **Append only.** Never rewrite, reorder, or delete existing entries — the log is a history. To
  mark something fixed, add `**FIXED** <date> <commit>` under the entry, keeping the original text.
- **Evidence over opinion.** Every issue needs a `file:line` reference and, where the claim is
  testable, the command that shows it.
- **Report negative findings too** (`soul.md` rule 5 — honesty). If a documented property does not
  hold, that is an issue even if the code "works".
- Per-experiment friction logs (e.g. `experiments/qwen8b_sac/issues.md`) **stay where they are** —
  they are local to a run. Cross-reference them from `issues.md`; do not fold them in.

A `SessionStart` hook in `.claude/settings.json` re-states this rule and reports the current
issue count at the start of every session.

## 2. Standard vs internal (the one rule that governs placement)

From `soul.md`: **generalizable** code/learnings → the standard SDK (`search_as_code/`) and the
public repo (`main` → GitHub). **Customer/corpus-specific** work (Altera, SearchUnify, BrowseComp
data, tokens, bespoke scripts) stays internal and gitignored — **never pushed**.

⚠️ Known violations are logged in `issues.md` §2 (GOV-1/2/3): some paths marked "INTERNAL, do not
push" are already on `origin/main`, and `.gitignore` does not untrack already-committed files.
**Before any push, check what is actually staged** — `git ls-tree -r --name-only origin/main | grep -iE 'altera|searchunify|browsecomp'`.

## 3. Working conventions

- **Pull before you start and before you push.** Multiple agents share this checkout
  concurrently — a run may be in flight (check `ps -ef | grep run_explore`).
- **Never `git add -A`.** Untracked customer artifacts and an SSH `.pub` key sit in the repo root.
  Stage explicit paths.
- **The GPU is shared** with another user. Kill only your own PIDs; serialize GPU jobs.
- **Keep it green:** `make check` (lint + typecheck + tests + repo guard + doc links) — one
  definition, the same target CI runs. Do not re-spell the command inline; it drifted before.
- Capture findings the same session they happen (`soul.md` rule 3): `issues.md` for defects,
  `CHANGELOG.md` for work done, `learnings_standard.md` if generalizable, `research.md` if it came
  from a paper.
