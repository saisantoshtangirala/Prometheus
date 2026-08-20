"""
NSE participant-wise positioning (FII / DII / Client / Pro) as features.

WHY THIS SOURCE, and not the one that gets recommended first
-------------------------------------------------------------
The obvious FII/DII source is `nseindia.com/api/fiidiiTradeReact`. It is
useless for research and it is worth saying why, because it looks fine
until you try to backtest it: **it ignores its date parameters.**
Measured directly - `?date=01-Aug-2026` and `?from=...&to=...` both
returned byte-identical 216-byte payloads containing only the latest
session. There is no history behind it. You cannot backtest a feature
you can only observe once, and accumulating it forward yields ~250
points a year.

This module uses the NSCCL archive instead:

    archives.nseindia.com/content/nsccl/fao_participant_vol_DDMMYYYY.csv

which IS date-addressable and IS backfillable - verified reachable for
2019-04-01, 2022-01-03, 2024-06-03 and 2026-01-02. It reports, per
session, long and short contract counts in index futures, stock futures
and options, broken out by Client / DII / FII / Pro.

That is *positioning*, not just flow, which is the more useful quantity:
it is a stock rather than a difference, so it does not need a baseline
to interpret.

LOOK-AHEAD, WHICH IS THE WHOLE BALLGAME HERE
--------------------------------------------
The report for session T is published AFTER the close on T. A strategy
standing at bar T deciding what to hold into T+1 cannot have seen it.
Using it unshifted would be a one-day look-ahead into a dataset that
directly encodes what large players did - which is exactly the kind of
leak that manufactures a spectacular, entirely fake backtest.

So `PUBLISH_LAG_BARS = 1` is applied unconditionally and there is a test
that fails if it is removed. Every feature at row t is built from the
report of session t-1 or earlier.

These features are MARKET-WIDE, not per-asset: one number per session,
broadcast across assets. That is an honest representation of what the
data is, and it caps how much they can explain about the cross-section.
"""

from __future__ import annotations

import csv
import io
import logging
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("nightevolver.flows")

ARCHIVE_URL = ("https://archives.nseindia.com/content/nsccl/"
               "fao_participant_vol_{ddmmyyyy}.csv")

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "nse_flows"

# The report for session T lands after T's close. Shift by at least one
# bar or the feature is a look-ahead. See module docstring.
PUBLISH_LAG_BARS = 1

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
       "Accept": "text/csv,*/*"}

# Names are stable across the archive's history but carry stray trailing
# whitespace in the header row ("Future Stock Short       "), so every
# lookup goes through _norm().
PARTICIPANTS = ("Client", "DII", "FII", "Pro")

FLOW_FEATURE_NAMES: Tuple[str, ...] = (
    "fii_idx_fut_net",
    "fii_idx_fut_net_chg",
    "fii_stk_fut_net",
    "dii_idx_fut_net",
    "client_idx_fut_net",
    "fii_opt_directional",
)
N_FLOW_FEATURES = len(FLOW_FEATURE_NAMES)


def _norm(s: str) -> str:
    return " ".join(s.split()).strip().lower()


@dataclass(frozen=True)
class ParticipantDay:
    """One session's participant table: {participant: {field: value}}."""
    date: pd.Timestamp
    table: Dict[str, Dict[str, float]]

    def get(self, participant: str, field: str) -> float:
        return self.table.get(participant, {}).get(_norm(field), float("nan"))


def parse_participant_csv(text: str, date: pd.Timestamp) -> Optional[ParticipantDay]:
    """Parse the NSCCL participant-volume CSV.

    Returns None rather than raising on a malformed/absent body: a
    nightly job that dies because one 2021 session is missing from the
    archive is worse than one that carries on with a gap.
    """
    rows = list(csv.reader(io.StringIO(text)))
    header: Optional[List[str]] = None
    table: Dict[str, Dict[str, float]] = {}

    for row in rows:
        if not row:
            continue
        first = row[0].strip().strip('"')
        if _norm(first) == "client type":
            header = [_norm(c) for c in row]
            continue
        if header is None or first not in PARTICIPANTS:
            continue
        vals: Dict[str, float] = {}
        for name, raw in zip(header[1:], row[1:]):
            raw = raw.strip().replace(",", "")
            try:
                vals[name] = float(raw)
            except ValueError:
                continue
        if vals:
            table[first] = vals

    if not table:
        return None
    return ParticipantDay(date=date, table=table)


def _cache_path(date: pd.Timestamp) -> Path:
    return CACHE_DIR / f"{date:%Y%m%d}.csv"


def fetch_participant_day(date: pd.Timestamp,
                          use_cache: bool = True,
                          timeout: int = 20,
                          max_attempts: int = 6) -> Tuple[Optional[ParticipantDay], str]:
    """Fetch one session. Returns (day, reason) where reason is one of
    "cache", "ok", "absent" (HTTP 404 - a genuine holiday) or
    "throttled"/"error" (we failed to get data that probably exists).

    THE REASON MATTERS, and this signature exists because of a measured
    bug. The archive answers HTTP **403** under load, not 404 - and it
    does so *intermittently*: probing 2023-01-10 serially returned 403
    five times and then `OK 955` on the sixth attempt, same URL.

    The first version of this function collapsed every failure to None.
    That silently converted throttle responses into phantom holidays and
    lost ~20% of all sessions (190/year retrieved against NSE's ~245),
    which forward-fill then papered over by presenting stale positioning
    as current. A dataset that is quietly 20% stale is worse than one
    that fails, because it still produces a number.
    """
    cp = _cache_path(date)
    if use_cache and cp.exists():
        try:
            day = parse_participant_csv(
                cp.read_text(encoding="utf-8", errors="replace"), date)
            if day is not None:
                return day, "cache"
        except OSError:
            pass

    url = ARCHIVE_URL.format(ddmmyyyy=f"{date:%d%m%Y}")
    text: Optional[str] = None
    reason = "error"
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as f:
                text = f.read().decode("utf-8", "replace")
            reason = "ok"
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Genuinely not published: weekend or exchange holiday.
                # Retrying cannot help and only adds load.
                return None, "absent"
            reason = "throttled" if e.code in (403, 429) else "error"
        except (urllib.error.URLError, OSError, TimeoutError):
            reason = "error"
        # Exponential backoff with jitter. Without the jitter, threads
        # that collide once tend to collide again in lockstep.
        if attempt < max_attempts - 1:
            time.sleep(min(8.0, 0.5 * (2 ** attempt)) * (0.5 + random.random()))

    if text is None:
        return None, reason

    day = parse_participant_csv(text, date)
    if day is not None and use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cp.write_text(text, encoding="utf-8")
        except OSError:
            pass
    return day, (reason if day is not None else "error")


def fetch_participant_range(dates: pd.DatetimeIndex,
                            max_workers: int = 6,
                            use_cache: bool = True,
                            max_unresolved_frac: float = 0.05,
                            ) -> Dict[pd.Timestamp, ParticipantDay]:
    """Fetch a set of sessions concurrently, distinguishing real
    holidays from failures we could not resolve.

    Raises if more than `max_unresolved_frac` of sessions ended in
    throttle/error rather than a clean 404. Proceeding with a silently
    incomplete flow series is how you get a confident number built on
    stale data - see fetch_participant_day's docstring.

    Concurrency is deliberately modest: this hits a public exchange
    archive, and pushing harder is both rude and self-defeating, since
    403s are exactly what over-driving it produces.
    """
    out: Dict[pd.Timestamp, ParticipantDay] = {}
    reasons: Dict[str, int] = {}

    def one(d):
        return d, fetch_participant_day(d, use_cache=use_cache)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for date, (day, reason) in ex.map(one, dates):
            reasons[reason] = reasons.get(reason, 0) + 1
            if day is not None:
                out[date] = day

    unresolved = reasons.get("throttled", 0) + reasons.get("error", 0)
    logger.info("[flows] %d/%d sessions | holidays(404)=%d unresolved=%d %s",
                len(out), len(dates), reasons.get("absent", 0), unresolved,
                {k: v for k, v in sorted(reasons.items())})

    if dates.size and unresolved / len(dates) > max_unresolved_frac:
        raise RuntimeError(
            f"{unresolved}/{len(dates)} sessions ({unresolved/len(dates):.1%}) "
            f"could not be fetched (throttled/error, NOT 404). Refusing to "
            f"build flow features from a silently incomplete series - "
            f"forward-fill would present stale positioning as current. "
            f"Re-run to extend the on-disk cache, or raise "
            f"max_unresolved_frac deliberately if a gap is acceptable."
        )
    return out


def _net_ratio(long_v: float, short_v: float) -> float:
    """(L - S) / (L + S), in [-1, 1]. NaN when the participant did not
    trade that product at all, which is common for DII in options."""
    tot = long_v + short_v
    if not np.isfinite(tot) or tot <= 0:
        return float("nan")
    return (long_v - short_v) / tot


def build_flow_frame(days: Dict[pd.Timestamp, ParticipantDay]) -> pd.DataFrame:
    """Raw (unshifted, unnormalised) flow features indexed by SESSION date.

    The publish lag is NOT applied here - it is applied in
    align_flow_features, so that the shift happens exactly once and
    against the price calendar rather than this one.
    """
    recs = []
    for date in sorted(days):
        d = days[date]
        rec = {
            "fii_idx_fut_net": _net_ratio(d.get("FII", "Future Index Long"),
                                          d.get("FII", "Future Index Short")),
            "fii_stk_fut_net": _net_ratio(d.get("FII", "Future Stock Long"),
                                          d.get("FII", "Future Stock Short")),
            "dii_idx_fut_net": _net_ratio(d.get("DII", "Future Index Long"),
                                          d.get("DII", "Future Index Short")),
            "client_idx_fut_net": _net_ratio(d.get("Client", "Future Index Long"),
                                             d.get("Client", "Future Index Short")),
        }
        # Directional options positioning: long calls and short puts are
        # both bullish expressions; long puts and short calls both
        # bearish. Netting them gives one signed exposure number.
        bull = (d.get("FII", "Option Index Call Long")
                + d.get("FII", "Option Index Put Short"))
        bear = (d.get("FII", "Option Index Put Long")
                + d.get("FII", "Option Index Call Short"))
        rec["fii_opt_directional"] = _net_ratio(bull, bear)
        recs.append(rec)

    df = pd.DataFrame(recs, index=pd.DatetimeIndex(sorted(days)))
    # Change in FII index-future positioning. Computed on the session
    # calendar (consecutive sessions), before any reindex to the price
    # calendar, so a market holiday does not read as a flat day.
    df["fii_idx_fut_net_chg"] = df["fii_idx_fut_net"].diff()
    return df[list(FLOW_FEATURE_NAMES)]


def _causal_zscore(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """Expanding z-score. Full-sample standardisation would leak the
    future into every row - the same trap data_loader guards against."""
    mean = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std().replace(0.0, np.nan)
    return (s - mean) / std


def align_flow_features(flow: pd.DataFrame,
                        price_dates: pd.DatetimeIndex,
                        lag_bars: int = PUBLISH_LAG_BARS) -> np.ndarray:
    """Align raw flow features onto the price calendar, lagged and squashed.

    Returns [T, N_FLOW_FEATURES] on `price_dates`, causally normalised
    and tanh-squashed to ~[-1, 1] so these sit on the same scale as the
    technical indicators the genome already votes on.

    Order of operations is deliberate:
      1. reindex to the price calendar and forward-fill (a stale reading
         is what a trader actually has on a day with no new report),
      2. THEN shift by lag_bars (so the shift is in trading bars, which
         is the unit the lag actually means),
      3. THEN z-score causally, and squash.
    """
    if lag_bars < 1:
        raise ValueError(
            f"lag_bars={lag_bars} would use a report published after the "
            "close of the bar it is applied to - that is look-ahead. "
            "See nightevolver/flows.py docstring."
        )

    aligned = flow.reindex(flow.index.union(price_dates)).sort_index()
    aligned = aligned.ffill().reindex(price_dates)
    aligned = aligned.shift(lag_bars)

    cols = []
    for name in FLOW_FEATURE_NAMES:
        z = _causal_zscore(aligned[name].astype(float))
        cols.append(np.tanh(z.to_numpy(dtype=np.float64) / 2.0))
    out = np.stack(cols, axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def load_flow_features(price_dates: pd.DatetimeIndex,
                       use_cache: bool = True,
                       max_workers: int = 8) -> np.ndarray:
    """End-to-end: fetch the sessions covering `price_dates` and return
    aligned [T, N_FLOW_FEATURES].

    Fetches a small pad before the first price date so the lag and the
    diff have something to stand on.
    """
    if len(price_dates) == 0:
        return np.zeros((0, N_FLOW_FEATURES))
    start = price_dates[0] - pd.Timedelta(days=10)
    sessions = pd.bdate_range(start, price_dates[-1])
    days = fetch_participant_range(sessions, max_workers=max_workers,
                                   use_cache=use_cache)
    if not days:
        logger.warning("[flows] no participant data retrieved - returning zeros. "
                       "Flow features will carry NO information; do not read a "
                       "null result on them as evidence about the data source.")
        return np.zeros((len(price_dates), N_FLOW_FEATURES))
    return align_flow_features(build_flow_frame(days), price_dates)
