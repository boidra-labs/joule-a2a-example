"""In-process short-term memory: per context_id turn history, no backend service.

When the SAP Agent Memory client is unavailable (no service bound), the agent
keeps the last N turns per context_id in process so multi-turn conversations
still have history. Survives across requests while the app runs; reset on
restart. Isolated per context_id.
"""
import app.agent as a
from langchain_core.messages import AIMessage, HumanMessage
from app.agent import (
    _short_term_remember,
    _short_term_history,
    _short_term_clear,
    _SHORT_TERM_MAX_TURNS,
)


def setup_function(_):
    _short_term_clear()


def test_remember_then_load_as_messages():
    _short_term_remember("ctx1", "show errors for 2025", "Here are 3 errors...")
    msgs = _short_term_history("ctx1")
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage]
    assert msgs[0].content == "show errors for 2025"
    assert msgs[1].content == "Here are 3 errors..."


def test_isolated_per_context():
    _short_term_remember("ctxA", "qA", "aA")
    _short_term_remember("ctxB", "qB", "aB")
    assert _short_term_history("ctxA")[0].content == "qA"
    assert _short_term_history("ctxB")[0].content == "qB"
    assert len(_short_term_history("ctxA")) == 2


def test_unknown_context_is_empty():
    assert _short_term_history("nope") == []


def test_capped_to_max_turns():
    for i in range(_SHORT_TERM_MAX_TURNS + 5):
        _short_term_remember("ctx1", f"q{i}", f"a{i}")
    msgs = _short_term_history("ctx1")
    # 2 messages per turn, capped at MAX_TURNS turns
    assert len(msgs) == _SHORT_TERM_MAX_TURNS * 2
    # oldest dropped, newest kept
    assert msgs[-1].content == f"a{_SHORT_TERM_MAX_TURNS + 4}"
    assert "q0" not in [m.content for m in msgs]


def test_assemble_history_uses_short_term_when_no_client():
    # No persistent client, no A2A history -> short-term fills the gap.
    _short_term_remember("ctx1", "previous question", "previous answer")
    out = a._assemble_history(memory_client=None, context_id="ctx1",
                              a2a_history=None, query="follow up")
    texts = [m.content for m in out]
    assert "previous question" in texts and "previous answer" in texts
