"""
NYSE trading-calendar helpers - deliberately dependency-free (stdlib
datetime only, no torch/numpy/pandas, no other kronos submodules).

kronos/orchestrator.py imports these for its own phase gating, and
.github/workflows/train-runpod.yml runs this module directly on a plain
GitHub Actions runner (via `python -c "from kronos.calendar_utils import
is_trading_day; ..."`) to decide whether tonight's scheduled training run
should happen at all - no pip install of the full dependency stack
needed just to check a weekday/holiday calendar, and no risk of that
check drifting out of sync with what Kronos itself uses, since it's the
same function either way.
"""

from __future__ import annotations

from datetime import date as ddate, timedelta
from typing import Set


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
