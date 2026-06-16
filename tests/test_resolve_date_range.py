"""Tests for _resolve_date_range — the pure period-resolution helper.

Focus: the three reported bugs plus regression coverage for the cases that
already worked, so the fix doesn't break them.
"""
from datetime import datetime, timezone

import pytest

from app.agent import _resolve_date_range

_TODAY = datetime.now(timezone.utc).replace(microsecond=0)
_YEAR = _TODAY.strftime("%Y")
_TODAY_END = _TODAY.strftime("%Y-%m-%dT23:59:59Z")


# --- Bug 1: relative keyword on the END of an explicit range -----------------
def test_range_year_to_today_resolves_today_to_real_date():
    """'2023 to today' must end at today's real date, not 'today-12-31...'."""
    r = _resolve_date_range("2023 to today")
    assert r["from"] == "2023-01-01T00:00:00Z"
    assert r["to"] == _TODAY_END
    assert r["resolved"] is True


def test_range_year_to_now_resolves_now():
    r = _resolve_date_range("2023 to now")
    assert r["from"] == "2023-01-01T00:00:00Z"
    assert r["to"].startswith(_YEAR + "-")  # current instant, not a literal 'now'
    assert "now" not in r["to"]


# --- Bug 2: open-ended start defaults 'to' to today --------------------------
def test_from_year_defaults_to_today():
    """'from 2023' (open-ended) must run up to today, not end-of-2023."""
    r = _resolve_date_range("from 2023")
    assert r["from"] == "2023-01-01T00:00:00Z"
    # Open-ended end resolves to the current instant ("today"), not end-of-2023.
    assert r["to"].startswith(_TODAY.strftime("%Y-%m-%d"))


# --- Bug 3: no date restriction ----------------------------------------------
@pytest.mark.parametrize("period", ["all", "all time", "any", "no date restriction"])
def test_no_restriction_yields_wide_window(period):
    """'all'/'any'/'no date restriction' -> wide window ending today."""
    r = _resolve_date_range(period)
    assert r["resolved"] is True
    assert r["from"] == "2000-01-01T00:00:00Z"
    assert r["to"] == _TODAY_END


# --- Regression: bare-year range stays end-of-year on a real year ------------
def test_explicit_year_range_unchanged():
    r = _resolve_date_range("2022 to 2023")
    assert r["from"] == "2022-01-01T00:00:00Z"
    assert r["to"] == "2023-12-31T23:59:59Z"


def test_bare_year_unchanged():
    r = _resolve_date_range("2023")
    assert r["from"] == "2023-01-01T00:00:00Z"
    assert r["to"] == "2023-12-31T23:59:59Z"


def test_between_unchanged():
    r = _resolve_date_range("between 2025-01-01 and 2025-06-30")
    assert r["from"] == "2025-01-01T00:00:00Z"
    assert r["to"] == "2025-06-30T23:59:59Z"
