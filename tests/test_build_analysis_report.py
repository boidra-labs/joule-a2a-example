"""Tests for _build_analysis_report — the deterministic analysis-report builder.

Covers the pure computation rules (grade, bars, trend, priority ordering) and the
zero-error path. Grounding text must be placed verbatim, never altered.
"""
from app.agent import _build_analysis_report

_STATS = [
    {"namespace": "NS1", "interfaceName": "FINEXTBANK", "interfaceVersion": "1",
     "total": 320, "errors": 41, "warnings": 5, "success": 274, "health": "Critical"},
    {"namespace": "NS1", "interfaceName": "ORDERS", "interfaceVersion": "1",
     "total": 200, "errors": 22, "warnings": 2, "success": 176, "health": "Critical"},
    {"namespace": "NS1", "interfaceName": "PAYMENTS", "interfaceVersion": "2",
     "total": 100, "errors": 0, "warnings": 0, "success": 100, "health": "Healthy"},
    {"namespace": "NS1", "interfaceName": "IDLE", "interfaceVersion": "1",
     "total": 0, "errors": 0, "warnings": 0, "success": 0, "health": "Healthy"},
]
_ERR_ROWS = [
    {"interfaceName": "FINEXTBANK", "logMessage": "Posting period closed",
     "processDate": "/Date(1686300000000)/", "status": "E"},
    {"interfaceName": "FINEXTBANK", "logMessage": "Posting period closed",
     "processDate": "/Date(1686300000000)/", "status": "E"},
    {"interfaceName": "ORDERS", "logMessage": "Customer not found",
     "processDate": "/Date(1686400000000)/", "status": "E"},
]
_RES = [
    {"msgId": "FI", "msgNo": "042", "messageText": "Posting period closed",
     "affectedInterfaces": ["FINEXTBANK", "ORDERS"], "occurrences": 42,
     "restartSafe": True, "rootCause": "The fiscal period is not open.",
     "resolutionSteps": ["Open period in OB52", "Reprocess message"],
     "sourceText": "[AIF Guide](https://x/doc)"},
    {"msgId": "SD", "msgNo": "007", "messageText": "Customer not found",
     "affectedInterfaces": ["ORDERS"], "occurrences": 5,
     "restartSafe": False, "rootCause": "Master data missing.",
     "resolutionSteps": ["Create customer XD01"],
     "sourceText": "SAP standard documentation for SD/007"},
]


def _report(**over):
    kw = dict(period="2023 to today", date_from="2023-01-01T00:00:00Z",
              date_to="2026-06-16T23:59:59Z", statistics=_STATS,
              error_rows=_ERR_ROWS, resolutions=_RES)
    kw.update(over)
    return _build_analysis_report(**kw)["report"]


def test_returns_markdown_with_all_sections():
    r = _report()
    assert "AIF Interface Monitoring Report" in r
    assert "Health Scorecard" in r
    assert "Active Interfaces" in r
    assert "Prioritized Action Plan" in r
    assert "Error Resolutions" in r
    assert "2023 to today" in r  # period echoed


def test_active_interfaces_excludes_zero_traffic():
    r = _report()
    assert "FINEXTBANK" in r and "ORDERS" in r and "PAYMENTS" in r
    assert "IDLE" not in r  # total == 0 -> not active
    # table header present (not just rows), with the Error % column
    assert "| Interface | Version | Total | Errors | Error % | Warnings | Success | Health |" in r


def test_no_ascii_charts_for_joule():
    """Joule mangles Unicode bar charts / code fences — none must appear."""
    r = _report()
    assert "```" not in r          # no fenced chart block
    assert "█" not in r            # no bar glyphs
    assert "trend" not in r.lower()  # no sparkline section
    for blk in "▁▂▃▄▅▆▇":
        assert blk not in r


def test_report_is_pure_ascii_for_joule():
    """Joule errors on non-ASCII (em-dash, middle dot, arrows, emoji). The whole
    report — including verbatim grounding/interface text — must be plain ASCII."""
    r = _report(grounded_total=8)
    offenders = sorted({ch for ch in r if ord(ch) > 127})
    assert not offenders, f"non-ASCII chars leaked into report: {offenders}"
    # encodes cleanly as ASCII (the operation Joule's transport effectively does)
    r.encode("ascii")


def test_non_ascii_grounding_text_is_normalised():
    """Verbatim grounding/interface content with non-ASCII must not break ASCII."""
    res = [dict(_RES[0], rootCause="Période non ouverte — fiscal", messageText="naïve—text")]
    stats = [dict(_STATS[0], interfaceName="FÍNEXTBÄNK")]
    r = _build_analysis_report(
        period="2023", date_from="2023-01-01T00:00:00Z", date_to="2023-12-31T23:59:59Z",
        statistics=stats, error_rows=[], resolutions=res, grounded_total=1,
    )["report"]
    r.encode("ascii")  # must not raise
    # accented letters are transliterated, not dropped: "Période" -> "Periode"
    assert "Periode non ouverte" in r
    assert "FINEXTBANK" in r


def test_error_share_percent_column():
    # FINEXTBANK: 41 errors / 63 total errors ≈ 65.1%
    r = _report()
    assert "65.1%" in r


def test_grade_reflects_error_percentage():
    # 63 errors / 620 total ≈ 10.2% -> grade D (<15%)
    r = _report()
    assert "grade: D" in r


def test_priority_plan_orders_by_score():
    # FI/042: 42 * (1+2) * 1.0 = 126 ; SD/007: 5 * (1+1) * 1.5 = 15 -> FI first.
    r = _report()
    fi = r.index("FI/042")
    sd = r.index("SD/007")
    assert fi < sd


def test_resolution_text_is_verbatim():
    r = _report()
    assert "The fiscal period is not open." in r
    assert "Open period in OB52" in r
    assert "[AIF Guide](https://x/doc)" in r


def test_truncation_note_when_more_distinct_errors():
    r = _report(grounded_total=8)  # 8 distinct existed, 2 grounded
    assert "more distinct" in r.lower()


def test_zero_error_path():
    healthy = [dict(s, errors=0, health="Healthy") for s in _STATS]
    r = _report(statistics=healthy, error_rows=[], resolutions=[], grounded_total=0)
    assert "Healthy" in r
    assert "Error Resolutions" not in r
    assert "Prioritized Action Plan" not in r
