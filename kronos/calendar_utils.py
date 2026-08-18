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
from typing import Dict, Set

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
}

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


def is_nse_trading_day(d: ddate) -> bool:
    if d.weekday() >= 5:
        return False
    return d not in nse_holidays(d.year)


def next_nse_trading_day(d: ddate) -> ddate:
    nxt = d + timedelta(days=1)
    while not is_nse_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt
