"""
Trading-calendar helpers - deliberately dependency-free (stdlib datetime
only, no torch/numpy/pandas, no other kronos submodules).

kronos/orchestrator.py imports these for its own phase gating, and
.github/workflows/train-runpod.yml runs this module directly on a plain
GitHub Actions runner (via `python -c "from kronos.calendar_utils import
is_nse_trading_day; ..."`) to decide whether tonight's scheduled
training run should happen at all - no pip install of the full
dependency stack needed just to check a weekday/holiday calendar, and no
risk of that check drifting out of sync with what Kronos itself uses,
since it's the same function either way.

Two calendars live here:
  - NYSE (nyse_holidays/is_trading_day) - every US holiday below is a
    fixed formula (nth weekday, Easter offset, etc.) that computes
    correctly for any year.
  - NSE (nse_holidays/is_nse_trading_day) - India's exchange holidays
    are mostly lunar/festival-calendar dates (Holi, Diwali, Eid, ...)
    that do NOT reduce to a formula and shift every year. These are
    hardcoded per calendar year in NSE_HOLIDAYS_BY_YEAR below, sourced
    from NSE's published annual holiday list. A year with no entry
    falls back to weekday-only (Mon-Fri) checking - logged once as a
    warning, since NSE also adds occasional ad-hoc holidays via
    late-notice circular that no calendar can predict in advance. This
    table needs a manual refresh each year; treat it as a cache, not a
    permanent constant.
"""

from __future__ import annotations

import logging
from datetime import date as ddate, timedelta
from typing import Dict, Optional, Set

logger = logging.getLogger("kronos.calendar_utils")


def _easter(year: int) -> ddate:
    """Anonymous Gregorian computus."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return ddate(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> ddate:
    """n-th (1-based) given weekday of a month; n=-1 for the last."""
    if n > 0:
        d = ddate(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + timedelta(days=offset + 7 * (n - 1))
    d = ddate(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def _observed(d: ddate) -> ddate:
    """Sat -> Fri, Sun -> Mon observance."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> Set[ddate]:
    return {
        _observed(ddate(year, 1, 1)),                 # New Year's Day
        _nth_weekday(year, 1, 0, 3),                  # MLK Day
        _nth_weekday(year, 2, 0, 3),                  # Presidents' Day
        _easter(year) - timedelta(days=2),            # Good Friday
        _nth_weekday(year, 5, 0, -1),                 # Memorial Day
        _observed(ddate(year, 6, 19)),                # Juneteenth
        _observed(ddate(year, 7, 4)),                 # Independence Day
        _nth_weekday(year, 9, 0, 1),                  # Labor Day
        _nth_weekday(year, 11, 3, 4),                 # Thanksgiving
        _observed(ddate(year, 12, 25)),               # Christmas
    }


def is_trading_day(d: ddate) -> bool:
    if d.weekday() >= 5:
        return False
    return d not in nyse_holidays(d.year)


def next_trading_day(d: ddate) -> ddate:
    nxt = d + timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt


# ---------------------------------------------------------------------------
# NSE (India) trading calendar
# ---------------------------------------------------------------------------

# NSE equity-segment holidays. Hardcoded per year - see module docstring
# for why these can't be formula-derived like NYSE's. Cross-checked
# against NSE's published 2026 annual holiday list (smallcase, Groww,
# Upstox); two 2026 dates are worth calling out:
#   - Nov 8 2026 (Diwali Laxmi Pujan) already falls on a Sunday, so it
#     isn't listed as a separate closed weekday - NSE runs a special
#     one-hour "Muhurat Trading" session that evening instead, which
#     this weekday-based calendar has no reason to model.
#   - NSE occasionally adds an ad-hoc holiday via late circular (days'
#     notice, not months) that no calendar published in advance can
#     capture - this table is a best-effort cache, not a guarantee.
NSE_HOLIDAYS_BY_YEAR: Dict[int, Set[ddate]] = {
    2026: {
        ddate(2026, 1, 26),    # Republic Day
        ddate(2026, 3, 3),     # Holi
        ddate(2026, 3, 26),    # Ram Navami
        ddate(2026, 3, 31),    # Mahavir Jayanti
        ddate(2026, 4, 3),     # Good Friday
        ddate(2026, 4, 14),    # Dr. Baba Saheb Ambedkar Jayanti
        ddate(2026, 5, 1),     # Maharashtra Day
        ddate(2026, 5, 28),    # Bakri Id (Eid al-Adha)
        ddate(2026, 6, 26),    # Muharram
        ddate(2026, 9, 14),    # Ganesh Chaturthi
        ddate(2026, 10, 2),    # Mahatma Gandhi Jayanti
        ddate(2026, 10, 20),   # Dussehra
        ddate(2026, 11, 10),   # Diwali - Balipratipada
        ddate(2026, 11, 24),   # Guru Nanak Jayanti
        ddate(2026, 12, 25),   # Christmas
    },
    # 2027. Dates that follow the Gregorian calendar (Republic Day,
    # Ambedkar Jayanti, Maharashtra Day, Gandhi Jayanti, Christmas) are
    # exact. The lunar/luni-solar festivals are the published-in-advance
    # estimates and NSE confirms them by circular each December, so treat
    # them as a best-effort cache and re-check against NSE's official
    # 2027 list when it publishes. Weekend dates are omitted - they are
    # already non-trading days.
    2027: {
        ddate(2027, 1, 26),    # Republic Day (Tue)
        ddate(2027, 3, 22),    # Holi (Mon)
        ddate(2027, 3, 26),    # Good Friday (Fri)
        ddate(2027, 4, 14),    # Dr. Baba Saheb Ambedkar Jayanti (Wed)
        ddate(2027, 4, 15),    # Ram Navami (Thu)
        ddate(2027, 4, 19),    # Mahavir Jayanti (Mon)
        ddate(2027, 5, 17),    # Bakri Id (Mon)
        ddate(2027, 6, 15),    # Muharram (Tue)
        ddate(2027, 9, 3),     # Ganesh Chaturthi (Fri)
        ddate(2027, 10, 8),    # Dussehra (Fri)
        ddate(2027, 10, 29),   # Diwali - Balipratipada (Fri)
        ddate(2027, 11, 12),   # Guru Nanak Jayanti (Fri)
    },
}

# How far ahead the table must remain populated. The failure this guards
# against: the table held 2026 ONLY, so from 2027-01-01 every festival
# holiday would have been treated as a trading day - the scheduler would
# have provisioned RunPod pods on closed days and the orchestrator would
# have expected market data that did not exist. It warned, but only into
# a log nobody reads daily.
NSE_CALENDAR_WARN_DAYS = 90

_warned_missing_nse_years: Set[int] = set()


def nse_holidays(year: int) -> Set[ddate]:
    holidays = NSE_HOLIDAYS_BY_YEAR.get(year)
    if holidays is None:
        if year not in _warned_missing_nse_years:
            logger.warning(
                "[calendar_utils] no NSE holiday list for %d - falling back to "
                "weekday-only trading-day checks (festival holidays will be "
                "missed). Update NSE_HOLIDAYS_BY_YEAR in kronos/calendar_utils.py.",
                year,
            )
            _warned_missing_nse_years.add(year)
        return set()
    return holidays


def nse_calendar_coverage_gap(today: Optional[ddate] = None,
                              horizon_days: int = NSE_CALENDAR_WARN_DAYS
                              ) -> Optional[str]:
    """Returns a message if the holiday table runs out within
    `horizon_days`, else None.

    Exists so calendar expiry is an ACTIONABLE signal - surfaced in the
    daily report and assertable in CI - rather than a warning logged once
    per process into a file nobody tails. Call it from the daily report
    path and from a test.
    """
    today = today or ddate.today()
    horizon = today + timedelta(days=horizon_days)
    missing = sorted({y for y in (today.year, horizon.year)
                      if y not in NSE_HOLIDAYS_BY_YEAR})
    if not missing:
        return None
    return (f"NSE holiday table has no entry for {', '.join(map(str, missing))} "
            f"(checked {today} .. {horizon}). Without it every festival "
            f"holiday in those years is treated as a trading day. Update "
            f"NSE_HOLIDAYS_BY_YEAR in kronos/calendar_utils.py.")


def is_nse_trading_day(d: ddate) -> bool:
    if d.weekday() >= 5:
        return False
    return d not in nse_holidays(d.year)


def next_nse_trading_day(d: ddate) -> ddate:
    nxt = d + timedelta(days=1)
    while not is_nse_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt
