"""
Resumable archive backfill: equity and F&O bhavcopies, 2019 onwards.

WHY THIS IS A SCRIPT AND NOT AN AD-HOC COMMAND. The first backfill was
a one-line nohup job. It ran for an hour, cached 1,615 equity sessions,
and died on http.client.IncompleteRead - an exception the fetchers'
retry loops could not catch (see nightevolver/nethttp.py). Everything it
had learned about WHERE IT HAD GOT TO lived in that process. Restarting
meant re-deriving the gap by hand.

A backfill against a throttling public archive will be interrupted. The
design assumption here is that it fails, repeatedly, and that the only
state that matters is what is already on disk:

  * The CACHE IS THE PROGRESS RECORD. A session is done when its zip is
    cached, so a rerun skips it with no bookkeeping file to go stale.
  * Days are attempted OLDEST FIRST, so an interrupted run leaves a
    contiguous prefix rather than a scatter.
  * A day that 404s in BOTH formats is a genuine non-trading day and is
    recorded in a holidays file, so the next run does not re-attempt
    every weekday the exchange was shut.

WHY 2019 AND NOT FURTHER BACK. The one live result in this project is
atm_iv -> vol_5d at p = 0.065 over ~16 windows. Nothing about it is a
modelling problem - the effect size is what it is and the p-value is
limited by the number of independent windows. Roughly 7 years of daily
history is what turns ~16 windows into ~30, which is the difference
between "close" and a result that can clear its own null cloud. That is
the entire reason this backfill exists.

WHAT THIS DELIBERATELY DOES NOT DO. It does not parse, it does not build
features, and it does not decide anything. It fills a cache. Every
statistical decision stays in the walk-forward scripts, where the null
cloud is.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from nightevolver import derivatives as D          # noqa: E402
from nightevolver import nse_prices as P           # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill")

STATE_DIR = Path(__file__).parent.parent / "data" / "cache"


def holiday_file(kind: str) -> Path:
    return STATE_DIR / f"{kind}_nontrading.json"


def load_holidays(kind: str) -> Set[str]:
    f = holiday_file(kind)
    if not f.exists():
        return set()
    try:
        return set(json.loads(f.read_text()))
    except (json.JSONDecodeError, OSError):
        return set()


def save_holidays(kind: str, days: Set[str]) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        holiday_file(kind).write_text(json.dumps(sorted(days), indent=0))
    except OSError:
        pass


def sessions(start: str, end: str) -> List[pd.Timestamp]:
    """Weekdays in range. Holidays are discovered, not hardcoded - an
    exchange calendar that drifts out of date silently drops real
    sessions, which is worse than a few wasted 404s."""
    return list(pd.bdate_range(start, end))


def backfill(kind: str, start: str, end: str, pause: float,
             limit: int | None) -> int:
    """Returns the number of sessions newly cached."""
    if kind == "equity":
        cached_of = lambda d: P._cache_path(d)                    # noqa: E731
        fetch = lambda d: P.fetch_bhav_day(d, ["RELIANCE"], use_cache=True)[1]
    else:
        cached_of = lambda d: D._cache_path(d)                    # noqa: E731
        fetch = lambda d: D.fetch_fo_raw(d, use_cache=True)[1]    # noqa: E731

    known_holidays = load_holidays(kind)
    days = sessions(start, end)
    todo = [d for d in days
            if not cached_of(d).exists()
            and f"{d:%Y-%m-%d}" not in known_holidays]

    logger.info("[%s] %d weekdays in range, %d already cached, "
                "%d known non-trading, %d to attempt",
                kind, len(days), sum(cached_of(d).exists() for d in days),
                len(known_holidays), len(todo))
    if limit:
        todo = todo[:limit]
        logger.info("[%s] limited to %d this run", kind, len(todo))

    got = absent = failed = 0
    for i, d in enumerate(todo, 1):
        try:
            reason = fetch(d)
        except Exception as e:                      # noqa: BLE001
            # A backfill must not die on one bad day. The whole point of
            # this script is that an hour of progress is never lost to a
            # single unhandled exception - which is exactly how the
            # previous run ended.
            logger.warning("[%s] %s raised %s: %s",
                           kind, f"{d:%Y-%m-%d}", type(e).__name__, e)
            failed += 1
            continue

        if reason in ("ok", "cache", "cached"):
            got += 1
        elif reason == "absent":
            absent += 1
            known_holidays.add(f"{d:%Y-%m-%d}")
        else:
            failed += 1

        if i % 25 == 0:
            logger.info("[%s] %d/%d  cached=%d non-trading=%d failed=%d "
                        "(at %s)", kind, i, len(todo), got, absent, failed,
                        f"{d:%Y-%m-%d}")
            save_holidays(kind, known_holidays)
        if pause:
            time.sleep(pause)

    save_holidays(kind, known_holidays)
    logger.info("[%s] DONE  newly cached=%d  non-trading=%d  failed=%d",
                kind, got, absent, failed)
    if failed:
        logger.warning("[%s] %d days failed after their retries - rerun to "
                       "pick them up; failures are almost always 403 "
                       "throttling and clear on a later attempt", kind, failed)
    return got


def report(start: str, end: str) -> None:
    """What the cache actually holds. Printed before and after so an
    interrupted run still leaves the operator knowing where it got to."""
    days = sessions(start, end)
    print("\n" + "=" * 66)
    print(f"CACHE COVERAGE  {start} .. {end}   ({len(days)} weekdays)")
    print("=" * 66)
    for kind, path_of in (("equity", P._cache_path), ("f&o", D._cache_path)):
        have = [d for d in days if path_of(d).exists()]
        by_year: dict[int, int] = {}
        for d in have:
            by_year[d.year] = by_year.get(d.year, 0) + 1
        pct = 100.0 * len(have) / max(len(days), 1)
        print(f"\n  {kind:<8} {len(have):>5} / {len(days)} sessions "
              f"({pct:5.1f}%)")
        for y in sorted(by_year):
            n_wd = sum(1 for d in days if d.year == y)
            print(f"      {y}  {by_year[y]:>3} / {n_wd:>3}")
    print("=" * 66 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--kinds", nargs="+", default=["equity", "fo"],
                   choices=["equity", "fo"])
    p.add_argument("--pause", type=float, default=0.15,
                   help="seconds between requests; the archive 403s under "
                        "load and a pause costs less than the retries it "
                        "prevents")
    p.add_argument("--limit", type=int, default=None,
                   help="attempt at most N sessions per kind this run")
    p.add_argument("--report-only", action="store_true")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    end = a.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    report(a.start, end)
    if a.report_only:
        return 0

    for kind in a.kinds:
        backfill(kind, a.start, end, a.pause, a.limit)

    report(a.start, end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
