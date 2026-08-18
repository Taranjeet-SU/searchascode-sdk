"""Curated test queries tagged by difficulty (easy/medium/hard) and hop type
(single-hop vs multi-hop) for exercising the agents in the arena / evals.

- single-hop  : answerable from ONE passage → SAC should one-shot it (hop 1).
- multi-hop   : needs 2+ facts combined / decomposition → should trigger SAC hop 2 (wide + consensus)
                and multiple tool-calling rounds.
- the last two are likely NOT in the FiQA corpus → they test the answerability/abstain signal.
Domain matches the cached corpus (BEIR-FiQA: personal-finance Q&A).
"""

from __future__ import annotations

TEST_QUERIES: list[dict] = [
    # ---- easy · single-hop (one obvious passage) ----
    {"q": "Can I deposit a cheque made out to my business into my business account?", "level": "easy", "hops": "single"},
    {"q": "Can I send a money order from USPS as a business?", "level": "easy", "hops": "single"},
    {"q": "What is a 401(k)?", "level": "easy", "hops": "single"},

    # ---- medium · single-hop (one topic, some nuance) ----
    {"q": "What are the tax implications of selling stock I received as RSUs?", "level": "medium", "hops": "single"},
    {"q": "What is the difference between a Roth IRA and a Traditional IRA?", "level": "medium", "hops": "single"},
    {"q": "How does dollar-cost averaging work when buying index funds?", "level": "medium", "hops": "single"},

    # ---- medium · multi-hop (needs 2 facts combined) ----
    {"q": "If I roll over my 401(k) to an IRA, are there tax penalties, and does it affect my annual IRA contribution limit?", "level": "medium", "hops": "multi"},
    {"q": "Given a 5% student-loan rate, should I pay the loan off early or invest in an index fund instead?", "level": "medium", "hops": "multi"},
    {"q": "How do short-term vs long-term capital gains rates differ, and how does holding period decide which applies?", "level": "medium", "hops": "multi"},

    # ---- hard · multi-hop (comparison / chained reasoning) ----
    {"q": "For a self-employed person, compare a SEP-IRA and a Solo 401(k) for maximizing retirement contributions and tax deduction.", "level": "hard", "hops": "multi"},
    {"q": "If I sell shares at a loss and rebuy within 30 days, how does the wash-sale rule affect my cost basis and future capital gains?", "level": "hard", "hops": "multi"},
    {"q": "Compare the tax treatment of qualified vs non-qualified dividends and how each affects someone in the 24% bracket.", "level": "hard", "hops": "multi"},

    # ---- hard · likely NOT in corpus (tests answerability / abstain) ----
    {"q": "What was the exact closing price of Grazitti Interactive stock on 3 March 2026?", "level": "hard", "hops": "absent"},
    {"q": "Summarize the 2027 IRS contribution limits for HSAs.", "level": "hard", "hops": "absent"},
]


def labeled() -> list[str]:
    """Display strings like '[easy·single] Can I deposit ...' for a dropdown."""
    return [f"[{d['level']}·{d['hops']}] {d['q']}" for d in TEST_QUERIES]


def strip_label(text: str) -> str:
    """Return the raw query from a '[level·hops] query' label (pass-through if unlabeled)."""
    if text.startswith("[") and "] " in text:
        return text.split("] ", 1)[1]
    return text
