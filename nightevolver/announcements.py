"""
NSE corporate announcements: event intensity per name.

WHAT THIS IS, AND WHAT IT IS NOT. This is not the announcement's
CONTENT - no model reads the PDF. It is the fact and timing of a
disclosure: how many filings a company made today, and how long it has
been since the last one. That is deliberately modest, and it is the part
that is measurable without an unvalidated language model sitting inside
an unvalidated feature.

The mechanism is plausible without being clever: disclosures cluster
around events, events move prices, and an unusual burst of filings for a
name that normally files twice a month is information available before
the daily bar closes. Whether that survives a null control is exactly
what the audit is for - this module's job is to make the question
askable.

ACCESS REQUIRES A SESSION COOKIE. www.nseindia.com/api/* answers 403 to
a bare request and 200 to the same request carrying cookies the
homepage sets. Measured here: a cold request returned 403; after one
GET of the homepage the cookie jar held AKA_A2, _abck, ak_bmsc and
bm_sz, and the identical API call returned real data. Anything that
treats the 403 as "endpoint gone" is wrong, and anything that retries
without warming up will retry forever.

WHAT THIS CANNOT DO. The endpoint serves a recent window, and the
date-range parameters cover a bounded span. Deep history is not
available in one call, so like the depth tape and RSS, the long series
has to be accumulated forward. `record_daily` is that entry point. The
difference from RSS is that a bounded backfill IS possible, so a few
months can be reconstructed rather than waited for.
"""

from __future__ import annotations

import http.cookiejar
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from .nethttp import TRANSIENT_NET_ERRORS

logger = logging.getLogger("nightevolver.announcements")

NSE_HOME = "https://www.nseindia.com"
ANN_API = "https://www.nseindia.com/api/corporate-announcements"
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "nse_announcements"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/html,application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

FEATURE_NAMES = (
    "ann_count",          # filings that day
    "ann_count_z",        # vs the name's own trailing 60-day mean
    "days_since_ann",     # bars since the last filing
)

ZSCORE_WINDOW = 60
_MIN_PERIODS = 20


def _make_opener() -> urllib.request.OpenerDirector:
    """An opener whose jar has been warmed on the NSE homepage.

    Without this every /api/ call is 403. With it the same call is 200.
    The cookies are set by the homepage response, so the warm-up GET is
    load-bearing rather than politeness.
    """
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    try:
        op.open(urllib.request.Request(NSE_HOME, headers=_HEADERS),
                timeout=25).read(2048)
    except TRANSIENT_NET_ERRORS as e:
        logger.warning("[announcements] homepage warm-up failed: %s", e)
    return op


def fetch_announcements(from_date: Optional[str] = None,
                        to_date: Optional[str] = None,
                        index: str = "equities",
                        max_attempts: int = 5,
                        use_cache: bool = True) -> List[Dict]:
    """Raw announcement records. Dates are DD-MM-YYYY, as NSE expects.

    Returns [] rather than raising: a missing window must degrade to
    "no events recorded" for those days, not take down a feature build
    that has every other day in hand.
    """
    params = {"index": index}
    if from_date:
        params["from_date"] = from_date
    if to_date:
        params["to_date"] = to_date

    key = urllib.parse.urlencode(sorted(params.items())).replace("&", "_")
    key = "".join(c if c.isalnum() or c in "-_=." else "_" for c in key)
    p = CACHE_DIR / f"{key}.json"
    if use_cache and p.exists():
        try:
            return json.loads(p.read_text())
        except ValueError:
            pass

    url = f"{ANN_API}?{urllib.parse.urlencode(params)}"
    for attempt in range(max_attempts):
        op = _make_opener()          # fresh jar each attempt; cookies expire
        try:
            with op.open(urllib.request.Request(url, headers=_HEADERS),
                         timeout=30) as f:
                payload = json.loads(f.read() or b"[]")
            rows = payload if isinstance(payload, list) else payload.get("data", [])
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(rows))
            return rows
        except urllib.error.HTTPError as e:
            if e.code not in (401, 403, 429, 503):
                logger.warning("[announcements] HTTP %s", e.code)
                return []
        except (*TRANSIENT_NET_ERRORS, ValueError):
            pass
        if attempt < max_attempts - 1:
            time.sleep(2.0 * (attempt + 1))
    logger.warning("[announcements] gave up after %d attempts", max_attempts)
    return []


def _record_date(rec: Dict) -> Optional[pd.Timestamp]:
    """The announcement's timestamp. None when unparseable.

    `an_dt` looks like '21-Aug-2026 15:57:43'. A record whose date
    cannot be read is DROPPED rather than dated to today: defaulting
    would concentrate every malformed row onto the current bar, which is
    the one bar a live feature is read from.
    """
    for field in ("an_dt", "sort_date", "exchdisstime", "dt"):
        raw = rec.get(field)
        if not raw:
            continue
        for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S",
                    "%d-%m-%Y %H:%M:%S"):
            try:
                return pd.Timestamp(datetime.strptime(str(raw).strip(), fmt).date())
            except ValueError:
                continue
    return None


def to_daily_counts(records: Sequence[Dict],
                    symbols: Sequence[str]) -> pd.DataFrame:
    """[date x symbol] filing counts from raw records."""
    syms = {str(s).upper().replace(".NS", "") for s in symbols}
    rows: List[tuple] = []
    for rec in records:
        sym = str(rec.get("symbol", "")).strip().upper()
        if sym not in syms:
            continue
        d = _record_date(rec)
        if d is not None:
            rows.append((d, sym))
    if not rows:
        return pd.DataFrame(columns=sorted(syms))
    df = pd.DataFrame(rows, columns=["date", "symbol"])
    return (df.groupby(["date", "symbol"]).size().unstack(fill_value=0)
            .reindex(columns=sorted(syms), fill_value=0).astype(float))


def _causal_z(frame: pd.DataFrame, window: int = ZSCORE_WINDOW) -> pd.DataFrame:
    """Trailing-only z-score - shift(1) before rolling. See delivery.py."""
    prior = frame.shift(1)
    mu = prior.rolling(window, min_periods=_MIN_PERIODS).mean()
    sd = prior.rolling(window, min_periods=_MIN_PERIODS).std(ddof=1)
    return (frame - mu) / sd.replace(0.0, np.nan)


def days_since_last(counts: pd.DataFrame) -> pd.DataFrame:
    """Bars since each name's most recent filing.

    Counted in BARS, not calendar days, because the panel this joins is
    indexed by trading session. A weekend is not two days of silence in
    a market that was closed.
    """
    out = pd.DataFrame(index=counts.index, columns=counts.columns, dtype=float)
    for col in counts.columns:
        since, vals = np.nan, []
        for v in counts[col].to_numpy():
            # Order matters: today's own filing makes this zero, and it
            # is known at the close, so it is causal.
            since = 0.0 if v > 0 else (since + 1.0 if np.isfinite(since) else np.nan)
            vals.append(since)
        out[col] = vals
    return out


def build_features(records: Sequence[Dict], symbols: Sequence[str],
                   dates: Optional[Sequence] = None) -> Dict[str, pd.DataFrame]:
    """{feature_name: DataFrame[date x symbol]}."""
    counts = to_daily_counts(records, symbols)
    if dates is not None:
        counts = counts.reindex(pd.DatetimeIndex(dates), fill_value=0.0)
    # A day with no filing is a real zero, not missing data.
    counts = counts.fillna(0.0)
    return {
        "ann_count": counts,
        "ann_count_z": _causal_z(counts),
        "days_since_ann": days_since_last(counts),
    }


def record_daily(directory: Path, symbols: Optional[Sequence[str]] = None,
                 now: Optional[datetime] = None) -> Path:
    """Append today's announcements to a dated JSONL file.

    Forward-recording, like the depth tape and the RSS poller: the
    endpoint's deep history is bounded, so the long series accumulates
    from here. Append rather than overwrite - running twice in a day
    must not discard the first run's records, since the API window
    moves.
    """
    now = now or datetime.utcnow()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"announcements_{now:%Y%m%d}.jsonl"
    recs = fetch_announcements(use_cache=False)
    keep = ({str(s).upper() for s in symbols} if symbols else None)
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for r in recs:
            if keep and str(r.get("symbol", "")).upper() not in keep:
                continue
            fh.write(json.dumps({
                "symbol": r.get("symbol"),
                "an_dt": r.get("an_dt"),
                "desc": r.get("desc") or r.get("subject"),
                "recorded": now.isoformat(),
            }) + "\n")
            n += 1
    logger.info("[announcements] recorded %d of %d records -> %s",
                n, len(recs), path)
    return path
