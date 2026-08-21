"""
NSE delivery data: what fraction of traded volume was actually taken.

WHAT THIS ADDS THAT VOLUME DOES NOT. Traded volume counts every
transaction, including intraday round trips that net to nothing by the
close. Delivery quantity counts only shares that changed hands for
settlement - someone paid in full and carried the position overnight.
The ratio separates conviction from churn, and two days with identical
OHLCV and identical volume can have completely different delivery
percentages. That difference is invisible to all 20 price channels the
GA currently searches.

Average trade size is the second axis. Turnover split across 160,000
trades and the same turnover across 16,000 trades are different markets:
the first is retail-shaped, the second institutional. Neither is
recoverable from the bhavcopy's OHLCV.

CAUSALITY. `sec_bhavdata_full_DDMMYYYY.csv` is published after day T's
close and describes it, so its features are known at T and predict
T -> T+1 - the same convention as prices and derivatives, and unlike
flows.py, which needs a lag because NSCCL publishes a day late.

The z-scored variants are computed with a strictly TRAILING window
(shift(1) before rolling) so a value at t never sees its own day. That
is the mechanical look-ahead this codebase has already been bitten by
once in the regime target, and a rolling mean that includes the current
bar is the easiest way to reintroduce it.
"""

from __future__ import annotations

import io
import logging
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("nightevolver.delivery")

DELIV_URL = ("https://nsearchives.nseindia.com/products/content/"
             "sec_bhavdata_full_{ddmmyyyy}.csv")

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "nse_delivery"

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "*/*"}

FEATURE_NAMES = (
    "deliv_pct",           # delivered / traded, as a fraction
    "deliv_pct_z",         # vs its own trailing 60-day mean
    "avg_trade_size_log",  # log(traded quantity / number of trades)
    "avg_trade_size_z",    # vs its own trailing 60-day mean
    "deliv_qty_z",         # delivered quantity vs trailing mean
)

ZSCORE_WINDOW = 60
_MIN_PERIODS = 20


def _cache_path(date: pd.Timestamp) -> Path:
    return CACHE_DIR / f"deliv_{date:%Y%m%d}.csv"


def fetch_delivery_raw(date: pd.Timestamp, timeout: int = 25,
                       max_attempts: int = 8,
                       use_cache: bool = True) -> Tuple[Optional[bytes], str]:
    """One day's file. reason in {ok, cached, absent, throttled, error}.

    Measured on this host: the current day returned 403 while the prior
    day returned 200. 403 here means "throttled or not published yet",
    NOT "no such trading day" - collapsing the two is what manufactured
    phantom holidays in flows.py and cost ~20% of sessions.
    """
    p = _cache_path(date)
    if use_cache and p.exists() and p.stat().st_size > 0:
        return p.read_bytes(), "cached"

    url = DELIV_URL.format(ddmmyyyy=f"{date:%d%m%Y}")
    reason = "error"
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as f:
                raw = f.read()
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                p.write_bytes(raw)
            return raw, "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, "absent"
            reason = "throttled" if e.code in (403, 429) else "error"
        except (urllib.error.URLError, OSError, TimeoutError):
            reason = "error"
        if attempt < max_attempts - 1:
            time.sleep(min(6.0, 0.5 * (2 ** attempt)) * (0.5 + random.random()))
    return None, reason


def parse_delivery(raw: bytes, symbols: Sequence[str]) -> Optional[pd.DataFrame]:
    """CSV bytes -> [symbol x raw columns] for the EQ series only.

    The file's headers carry leading spaces (' SERIES', ' DELIV_PER'),
    which is the kind of detail that silently yields an empty frame and
    an all-NaN feature column rather than an error.
    """
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except (ValueError, OSError):
        return None
    df.columns = [str(c).strip() for c in df.columns]
    need = {"SYMBOL", "SERIES", "TTL_TRD_QNTY", "NO_OF_TRADES", "DELIV_QTY"}
    if not need.issubset(df.columns):
        return None

    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df = df[df["SERIES"] == "EQ"]
    df = df[df["SYMBOL"].isin(set(symbols))]
    if df.empty:
        return None

    for c in ("TTL_TRD_QNTY", "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER"):
        if c in df.columns:
            # Suspended/illiquid rows carry '-' in the delivery columns.
            df[c] = pd.to_numeric(
                df[c].astype(str).str.strip().replace({"-": None}),
                errors="coerce")
    return df.set_index("SYMBOL")


def _causal_z(frame: pd.DataFrame, window: int = ZSCORE_WINDOW) -> pd.DataFrame:
    """Z-score against a STRICTLY trailing window.

    shift(1) before rolling, so the statistic at t is built from bars
    strictly before t. Without the shift the current bar contributes to
    its own mean and standard deviation, which is a small but real
    look-ahead - the same mechanism that made a random walk score
    rho = -0.39 against the regime target before it was corrected.
    """
    prior = frame.shift(1)
    mu = prior.rolling(window, min_periods=_MIN_PERIODS).mean()
    sd = prior.rolling(window, min_periods=_MIN_PERIODS).std(ddof=1)
    return (frame - mu) / sd.replace(0.0, np.nan)


def fetch_delivery_features(symbols: Sequence[str], start: str,
                            end: Optional[str] = None,
                            use_cache: bool = True,
                            max_workers: int = 6) -> Dict[str, pd.DataFrame]:
    """{feature_name: DataFrame[date x symbol]} over a date range."""
    from concurrent.futures import ThreadPoolExecutor

    syms = [str(s).upper().replace(".NS", "") for s in symbols]
    dates = pd.bdate_range(start, end or pd.Timestamp.today().normalize())

    def one(d):
        raw, reason = fetch_delivery_raw(d, use_cache=use_cache)
        if raw is None:
            return d, None, reason
        return d, parse_delivery(raw, syms), reason

    rows: Dict[pd.Timestamp, pd.DataFrame] = {}
    stats: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for d, df, reason in ex.map(one, dates):
            stats[reason] = stats.get(reason, 0) + 1
            if df is not None:
                rows[d] = df

    if not rows:
        logger.warning("[delivery] no days fetched: %s", stats)
        return {f: pd.DataFrame() for f in FEATURE_NAMES}
    logger.info("[delivery] %d/%d sessions (%s)", len(rows), len(dates), stats)

    idx = sorted(rows)

    def col(name: str) -> pd.DataFrame:
        return pd.DataFrame(
            {d: rows[d][name] if name in rows[d].columns else pd.Series(dtype=float)
             for d in idx}).T.reindex(columns=syms)

    traded, trades = col("TTL_TRD_QNTY"), col("NO_OF_TRADES")
    deliv = col("DELIV_QTY")

    # Derive the percentage rather than trusting DELIV_PER: the column is
    # absent or '-' for some rows, and a ratio computed from two columns
    # that ARE present is available more often than one that must be.
    deliv_pct = (deliv / traded.replace(0.0, np.nan)).clip(0.0, 1.0)
    avg_size = np.log((traded / trades.replace(0.0, np.nan)).replace(0.0, np.nan))

    return {
        "deliv_pct": deliv_pct,
        "deliv_pct_z": _causal_z(deliv_pct),
        "avg_trade_size_log": avg_size,
        "avg_trade_size_z": _causal_z(avg_size),
        "deliv_qty_z": _causal_z(np.log(deliv.replace(0.0, np.nan))),
    }
