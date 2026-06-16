"""Analysis report must land in card.message (Joule reads parts[1].data.message).

The finalizer must not re-summarise or relocate the verbatim report. These tests
lock the two contracts the fix relies on:
  1. CardEnvelope accepts data=None (analysis carries the report in `message`).
  2. The analysis override sets message=<draft> verbatim and data=None.
"""
from app.schemas import CardResponse


# The exact transform applied in AIFAgent._finalize_structured for analysis.
from app.agent import _ascii


def _apply_analysis_override(parsed: dict, draft: str) -> dict:
    if isinstance(parsed, dict) and parsed.get("intent") == "analysis":
        parsed["message"] = draft
        parsed["data"] = None
    if isinstance(parsed.get("message"), str):
        parsed["message"] = _ascii(parsed["message"])
    return CardResponse(**parsed).model_dump()


def test_finalizer_message_forced_to_ascii_for_all_intents():
    # Non-analysis intent whose LLM message has em dash + emoji must be ASCII.
    card = {"intent": "statistics", "message": "Health \U0001F7E2 OK — all good · done",
            "data": {"interfaceCount": 1, "interfaces": []}}
    out = _apply_analysis_override(dict(card), "draft")
    out["message"].encode("ascii")  # must not raise
    assert "—" not in out["message"] and "\U0001F7E2" not in out["message"]


# The real builder draft is already plain ASCII (no emoji / em dash) so Joule
# can render it.
_REPORT = (
    "# AIF Interface Monitoring Report\n"
    "## Active Interfaces (traffic in period)\n"
    "| Interface | Version | Total | Errors | Error % | Warnings | Success | Health |\n"
    "| FINEXTBANK | 1 | 320 | 41 | 65.1% | 5 | 274 | Critical |\n"
)


def test_envelope_accepts_null_data():
    out = CardResponse(intent="analysis", message="# R", data=None).model_dump()
    assert out["data"] is None
    assert out["message"] == "# R"


def test_analysis_message_is_verbatim_draft():
    # Finalizer tried to summarise into the wrong place; override must win.
    bad = {"intent": "analysis", "message": "Here is a short summary.",
           "data": {"summary": _REPORT}}
    out = _apply_analysis_override(bad, _REPORT)
    assert out["message"] == _REPORT      # full report, verbatim, in `message`
    assert out["data"] is None            # no stray report in data
    assert "65.1%" in out["message"]


def test_non_analysis_intent_untouched():
    card = {"intent": "statistics", "message": "Health summary",
            "data": {"interfaceCount": 2, "interfaces": []}}
    out = _apply_analysis_override(dict(card), "irrelevant draft")
    assert out["message"] == "Health summary"     # not overwritten
    assert out["data"]["interfaceCount"] == 2     # data preserved
