"""Per-context_id memory assembly: always load history AND always vector-search.

The bug being fixed: vector search used to run ONLY when there was no history,
so semantic recall was effectively dead in multi-turn sessions. _assemble_history
must always include the context's message history AND a vector-memory lookup
scoped to the same context_id.

Written to run under a plain runner (no pytest dependency): patches module
attrs directly and restores them.
"""
import contextlib

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import app.agent as a
from app.agent import _assemble_history


@contextlib.contextmanager
def _patch(**attrs):
    saved = {k: getattr(a, k) for k in attrs}
    for k, v in attrs.items():
        setattr(a, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(a, k, v)


def test_vector_search_always_runs_even_with_history():
    with _patch(
        _load_history=lambda mc, ctx: [HumanMessage(content="prior q"),
                                       AIMessage(content="prior a")],
        _search_relevant_memories=lambda mc, ctx, q:
            "Relevant context from past conversations:\n- old FI/042 report",
    ):
        out = _assemble_history(memory_client=object(), context_id="ctx1",
                                a2a_history=None, query="what about FI/042?")
    texts = [m.content for m in out]
    assert "prior q" in texts and "prior a" in texts          # history retained
    assert any(isinstance(m, SystemMessage) and "past conversations" in m.content
               for m in out)                                  # vector ALSO injected


def test_vector_search_runs_when_no_history():
    with _patch(
        _load_history=lambda mc, ctx: [],
        _search_relevant_memories=lambda mc, ctx, q:
            "Relevant context from past conversations:\n- x",
    ):
        out = _assemble_history(memory_client=object(), context_id="ctx1",
                                a2a_history=None, query="q")
    assert any(isinstance(m, SystemMessage) for m in out)


def test_search_scoped_to_context_id():
    seen = {}

    def fake_search(mc, ctx, q):
        seen["ctx"] = ctx
        return ""

    with _patch(_load_history=lambda mc, ctx: [], _search_relevant_memories=fake_search):
        _assemble_history(memory_client=object(), context_id="CTX-42",
                          a2a_history=None, query="q")
    assert seen["ctx"] == "CTX-42"


def test_no_memory_no_a2a_returns_empty():
    with _patch(_load_history=lambda mc, ctx: [],
                _search_relevant_memories=lambda mc, ctx, q: ""):
        out = _assemble_history(memory_client=None, context_id="ctx1",
                                a2a_history=None, query="q")
    assert out == []
