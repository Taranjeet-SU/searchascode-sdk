"""AGT-1 regression: the strategist prompt must be backend-aware.

The static prompt commanded `session.store._search(body)` on every backend; on MemoryStore every
authored program crashed with AttributeError — SU exploration scored 0.075 vs dense 0.800."""
from __future__ import annotations

import search_as_code as sac
from search_as_code.harness.agentic import AUTHOR_SYSTEM, build_author_system


def test_memory_backend_prompt_has_no_raw_dsl():
    s = sac.Session("memory")
    s.add([{"id": "1", "text": "x"}])
    sys_prompt = build_author_system(s)
    assert "RAW OpenSearch DSL. Returns a dict" not in sys_prompt   # the call is not OFFERED
    assert "match_phrase" not in sys_prompt                          # no raw-DSL worked example
    assert "do NOT call session.store._search" in sys_prompt        # and it is explicitly warned off
    assert "mode='keyword'" in sys_prompt            # the portable escalation replaces it
    assert "fuse_ids" in sys_prompt                  # helpers still documented


def test_dsl_backend_prompt_keeps_raw_escalation():
    class FakeOS:
        def _search(self, body):                     # duck-typed: only hasattr matters
            return {"hits": {"hits": []}}

    class FakeSession:
        store = FakeOS()

    sys_prompt = build_author_system(FakeSession())
    assert "session.store._search" in sys_prompt
    assert "match_phrase" in sys_prompt


def test_backcompat_constant_is_the_dsl_flavor():
    assert "session.store._search" in AUTHOR_SYSTEM
