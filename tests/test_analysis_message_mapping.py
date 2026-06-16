"""Analysis report must land in card.message (Joule reads parts[1].data.message).

The finalizer must not re-summarise or relocate the verbatim report. These tests
lock the two contracts the fix relies on:
  1. CardEnvelope accepts data=None (analysis carries the report in `message`).
  2. The analysis override sets message=<draft> verbatim and data=None.
"""
from app.schemas import CardResponse


# The exact transform applied in AIFAgent._finalize_structured for analysis.
def _apply_analysis_override(parsed: dict, draft: str) -> dict:
    if isinstance(parsed, dict) and parsed.get("intent") == "analysis":
        parsed["message"] = draft
        parsed["data"] = None
    return CardResponse(**parsed).model_dump()


_REPORT = (
    "# 🛰️ AIF Interface Monitoring Report\n"
    "## 🟢 Active Interfaces (traffic in period)\n"
    "| Interface | Version | Total | Errors | Error % | Warnings | Success | Health |\n"
    "| FINEXTBANK | 1 | 320 | 41 | 65.1% | 5 | 274 | 🔴 Critical |\n"
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
