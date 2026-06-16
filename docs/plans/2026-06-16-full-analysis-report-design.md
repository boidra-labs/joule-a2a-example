# Full Analysis Report — Design (2026-06-16)

## Goal
Expand the AIF **analysis** intent (rules 1-10) into a full-coverage monitoring
report for a specific period, adding:
- an **active interface list** (interfaces with traffic in the period),
- **all error resolutions in a table, per MSGID/MSGNO** (top 5 distinct, grounded),
- four value-add visuals: **health scorecard + error-share bars**, **error-trend
  sparkline**, **prioritized action plan**, **risk callout box**,
- a redesigned Markdown response.

## Key decision: deterministic Python builder
LLMs miscompute bar lengths, daily buckets, and weighted ranking from raw rows.
A new tool **`build_analysis_report_tool`** assembles the finished Markdown in
Python (pure helper `_build_analysis_report`, mirrors `_resolve_date_range`).
The LLM gathers data + grounds errors, then returns the builder output verbatim.
Grounding text is only *placed* by Python — never edited or invented (strict
grounding preserved).

## Flow (revised rules 1-10)
1. `calculate_date_range_tool(period)` → from/to.
2. `get_interface_statistics_tool` → per-interface activity/health.
3. `list_error_messages_tool(status='E')` → error rows.
4. Group errors by distinct MSGID/MSGNO; take **top 5** by occurrence.
5. For each top-5: `run_doc_error_catalog_tool` once → grounded root cause + steps.
6. `build_analysis_report_tool(period, from, to, statistics, error_rows, resolutions[])`.
7. Return its Markdown verbatim; finalizer puts it in `message`, intent='analysis'.

Cost: up to 5 grounding calls (was 1). Builder notes when >5 distinct errors existed.

## Builder signature
`build_analysis_report_tool(period, date_from, date_to, statistics, error_rows,
resolutions) -> {"report": "<markdown>"}`

- `statistics`: interfaces[] from get_interface_statistics_tool.
- `error_rows`: messages[] from list_error_messages_tool (logMessage, processDate,
  interface fields). Worklist rows do NOT carry msgId/msgNo — group counts/trend
  by logMessage + interface.
- `resolutions`: [{msgId, msgNo, messageText, affectedInterfaces[], occurrences,
  restartSafe, rootCause, resolutionSteps[], sourceText}] — built by the LLM from
  its per-error grounding (verbatim).

## Computation rules (pure Python)
- Grade: err% → A <1%, B <3%, C <7%, D <15%, F ≥15%.
- Bars: `█ * round(20*val/maxval)` + `░` pad; max interface error count = full bar.
- Trend: bucket error processDate by day over [from,to]; counts → `▁▂▃▄▅▆▇█` by
  relative height; cap ~30 buckets, note if collapsed.
- Priority score: occurrences × (1 + affectedInterfaceCount) × (1.0 restart-safe
  else 1.5); sort desc; tercile → High/Med/Low.
- Risk callout: #1 priority row.
- Active interfaces: statistics rows with total > 0.

## Report layout
Title · Risk callout (blockquote) · Health Scorecard (grade + bars) · Error trend
sparkline · Active Interfaces table · Prioritized Action Plan table · Error
Resolutions (per distinct MSGID/MSGNO) · "+N more not grounded" note.
Zero-error path: Scorecard + Active Interfaces + "✅ Healthy" note only.

## Testing
Unit tests on `_build_analysis_report`: grade thresholds, bar math, daily
bucketing, priority ordering, zero-error path, grounding-missing row.
