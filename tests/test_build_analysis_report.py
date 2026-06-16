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


# Interface catalog (from list_interfaces_tool) — supplies the business-friendly
# description joined on namespace+name+version.
_CATALOG = [
    {"namespace": "NS1", "interfaceName": "FINEXTBANK", "interfaceVersion": "1",
     "about": "External bank statement import"},
    {"namespace": "NS1", "interfaceName": "ORDERS", "interfaceVersion": "1",
     "about": "Sales order replication"},
    {"namespace": "NS1", "interfaceName": "PAYMENTS", "interfaceVersion": "2",
     "about": "Outgoing payment posting"},
]


def _report(**over):
    kw = dict(period="2023 to today", date_from="2023-01-01T00:00:00Z",
              date_to="2026-06-16T23:59:59Z", statistics=_STATS,
              error_rows=_ERR_ROWS, resolutions=_RES, interfaces_catalog=_CATALOG)
    kw.update(over)
    return _build_analysis_report(**kw)["report"]


def test_returns_markdown_with_all_sections():
    r = _report()
    assert "AIF Interface Monitoring Report" in r
    assert "Health Scorecard" in r
    assert "Active Interfaces" in r
    assert "Top" in r and "Common Errors" in r          # top-errors list (replaces action plan)
    assert "Error Resolutions" in r
    assert "2023 to today" in r  # period echoed


def test_period_header_has_no_central_finance_label():
    # (a) the hardcoded "Central Finance" label must be gone.
    r = _report()
    header = r.splitlines()[1]
    assert "Central Finance" not in header
    assert "2023 to today" in header


def test_no_prioritized_action_plan_section():
    # (d) the action plan is removed entirely.
    assert "Prioritized Action Plan" not in _report()
    assert "Restart-safe" not in _report().split("Error Resolutions")[0]  # not in the list either


def test_active_interfaces_shows_description_not_namespace_version():
    # (c) business-friendly: description column, no Version column.
    r = _report()
    # description from the catalog join
    assert "External bank statement import" in r
    assert "Sales order replication" in r
    # the Version column header is gone; a Description column is present
    table_header = next(l for l in r.splitlines() if l.startswith("| Interface"))
    assert "Version" not in table_header
    assert "Description" in table_header
    assert "Total" in table_header
    # all-type totals still present
    assert "Warnings" in table_header and "Success" in table_header


def test_active_interfaces_excludes_zero_traffic():
    r = _report()
    assert "FINEXTBANK" in r and "ORDERS" in r and "PAYMENTS" in r
    assert "IDLE" not in r  # total == 0 -> not active


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


def test_scorecard_breaks_down_all_message_types():
    """Scorecard must report success/warning/error counts AND percentages,
    computed across all message types (not just errors)."""
    # totals: 620 msgs; errors 63; warnings 5+2+0+0=7; success 274+176+100+0=550
    r = _report()
    assert "620 messages" in r
    assert "550 success" in r or "Success: 550" in r
    # warnings surfaced
    assert "7 warning" in r.lower() or "warning: 7" in r.lower()
    # success rate computed (550/620 = 88.7%)
    assert "88.7%" in r


def test_status_percentages_sum_consistently():
    # 1 interface: 100 total = 10 err + 20 warn + 70 success
    stats = [{"namespace": "N", "interfaceName": "X", "interfaceVersion": "1",
              "total": 100, "errors": 10, "warnings": 20, "success": 70, "health": "Critical"}]
    res = [dict(_RES[0], occurrences=10, affectedInterfaces=["X"])]
    r = _build_analysis_report(period="2024", date_from="2024-01-01T00:00:00Z",
                               date_to="2024-12-31T23:59:59Z", statistics=stats,
                               error_rows=[], resolutions=res, grounded_total=1)["report"]
    assert "100 messages" in r
    assert "70 success (70.0%)" in r
    assert "20 warning (20.0%)" in r
    assert "10 error (10.0%)" in r


def test_scorecard_shows_inprocess_aborted_canceled():
    # Real payload shape: 132 = 70 success + 58 error + 2 warn + 1 inProcess
    #                         + 0 abort + 1 canceled
    stats = [{"namespace": "N", "interfaceName": "X", "interfaceVersion": "1",
              "total": 132, "errors": 58, "warnings": 2, "success": 70,
              "inProcess": 1, "aborted": 0, "canceled": 1, "health": "Critical"}]
    res = [dict(_RES[0], occurrences=58, affectedInterfaces=["X"])]
    r = _build_analysis_report(period="2024", date_from="2024-01-01T00:00:00Z",
                               date_to="2024-12-31T23:59:59Z", statistics=stats,
                               error_rows=[], resolutions=res, grounded_total=1)["report"]
    assert "132 messages" in r
    # every non-zero status is itemised in the breakdown
    assert "1 in-process" in r.lower()
    assert "1 canceled" in r.lower()
    # zero-count statuses (aborted) are omitted to keep it readable
    assert "0 aborted" not in r.lower()
    # traffic table shows in-process per interface
    table_header = next(l for l in r.splitlines() if l.startswith("| Interface"))
    assert "In-Process" in table_header


def test_grade_reflects_error_percentage():
    # 63 errors / 620 total ≈ 10.2% -> grade D (<15%)
    r = _report()
    assert "grade: D" in r


def test_top_errors_list_ordered_with_cumulative():
    # (e) most-common-errors list ordered by occurrences, with a running cumulative.
    r = _report()
    top_section = r.split("Common Errors")[1].split("Error Resolutions")[0]
    # ordered: "Posting period closed" (42) before "Customer not found" (5)
    assert top_section.index("Posting period closed") < top_section.index("Customer not found")
    header = next(l for l in top_section.splitlines() if l.startswith("| #"))
    # only Message, Occurrences, Cumulative — dropped Error/Criticality/Affected
    assert "Message" in header and "Occurrences" in header and "Cumulative" in header
    assert "Error" not in header and "Criticality" not in header and "Affected" not in header
    # cumulative accumulates: 42 then 42+5=47
    rows = [l for l in top_section.splitlines() if l.startswith("| 1 ") or l.startswith("| 2 ")]
    assert rows[0].rstrip().endswith("42 |")     # cumulative after row 1
    assert rows[1].rstrip().endswith("47 |")     # cumulative after row 2


def test_biggest_risk_no_empty_msgid_slash():
    # (b) empty msgId/msgNo must not render as "-/-"; defensive fallback.
    res = [dict(_RES[0], msgId="", msgNo="", occurrences=9)]
    r = _build_analysis_report(period="2024", date_from="2024-01-01T00:00:00Z",
                               date_to="2024-12-31T23:59:59Z", statistics=_STATS,
                               error_rows=[], resolutions=res, grounded_total=1,
                               interfaces_catalog=_CATALOG)["report"]
    assert "-/-" not in r


def test_resolution_text_is_verbatim_and_capped_to_3_bullets():
    # (f) resolutions show at most 3 short bullets.
    long_steps = dict(_RES[0], resolutionSteps=[f"step {i}" for i in range(1, 8)])
    r = _report(resolutions=[long_steps, _RES[1]], grounded_total=2)
    assert "The fiscal period is not open." in r        # root cause still verbatim
    res_section = r.split("Error Resolutions")[1]
    # only 3 numbered bullets for the first resolution
    assert "1. step 1" in res_section
    assert "3. step 3" in res_section
    assert "4. step 4" not in res_section


def test_resolutions_capped_at_20():
    many = [dict(_RES[0], msgNo=f"{i:03d}", occurrences=100 - i) for i in range(25)]
    r = _report(resolutions=many, grounded_total=25)
    res_section = r.split("Error Resolutions")[1]
    assert res_section.count("### ") <= 20


def test_truncation_note_when_more_distinct_errors():
    r = _report(grounded_total=8)  # 8 distinct existed, 2 grounded
    assert "more distinct" in r.lower()


def test_zero_error_path():
    healthy = [dict(s, errors=0, health="Healthy") for s in _STATS]
    r = _report(statistics=healthy, error_rows=[], resolutions=[], grounded_total=0)
    assert "Healthy" in r
    assert "Error Resolutions" not in r
    assert "Common Errors" not in r
