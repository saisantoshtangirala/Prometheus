"""
BSE bhavcopy, for CROSS-EXCHANGE divergence - not for more tickers.

WHY NOT UNIVERSE EXPANSION. The obvious reason to add a second exchange
is more names, and here that reason is mostly wrong:

  * India's large and mid caps are DUAL-LISTED. Adding BSE's row for
    RELIANCE next to NSE's adds a near-duplicate of a series already in
    the panel - correlated ~1.0, contributing no independent
    observations while inflating every count that assumes independence.
  * The names that are BSE-ONLY are, almost by construction, the ones
    NSE did not attract: small, thin, wide-spread. The 22bp round-trip
    this project costs everything at is an NSE large-cap number. On a
    thin BSE-only name, realistic slippage alone can exceed it, so
    "expanding the universe" there quietly relaxes the cost assumption
    that every conclusion rests on.

WHAT IS ACTUALLY NEW. The same instrument trading in two venues at once
produces information neither venue produces alone:

  NSE-BSE SPREAD. Two prices for one claim on one company. The gap is
  small, mean-reverting, and moves with fragmentation, arbitrage
  capacity and one-sided flow. It cannot be computed from either
  exchange's data alone - which is exactly the property the existing 26
  price-transform channels lack, being functions of one series.

  VENUE SHARE. What fraction of the day's volume printed on BSE rather
  than NSE. Where trading migrates is a fact about participants, not
  about price.

Both are cheap: one file per session, same UDiFF schema NSE uses, so the
parsing is the same shape as nse_prices.py.

SERIES CODES DIFFER, and getting this wrong silently changes the
universe. NSE marks rolling-settlement equity 'EQ'. BSE uses group
letters - A and B are the main equity groups, T is trade-to-trade
(delivery-only, no intraday netting, a different microstructure), and
M/MS/MT are the SME platform. Filtering for 'EQ' against BSE returns an
empty frame, which downstream looks like a quiet day rather than a bug.
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

logger = logging.getLogger("nightevolver.bse")

BSE_URL = ("https://www.bseindia.com/download/BhavCopy/Equity/"
           "BhavCopy_BSE_CM_0_0_0_{yyyymmdd}_F_0000.CSV")

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "bse_bhav"

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "*/*"}

# A and B are BSE's main equity groups. T is trade-to-trade: delivery is
# compulsory and intraday netting is not allowed, so its microstructure
# is genuinely different and mixing it in would blend two regimes.
# M/MS/MT are the SME platform - different listing standards entirely.
BSE_EQUITY_GROUPS = ("A", "B")

FEATURE_NAMES = (
    "nse_bse_spread",      # nse RETURN - bse RETURN, in basis points
    "nse_bse_spread_abs",  # |spread|, dislocation size regardless of sign
    "bse_volume_share",    # bse / (nse + bse) traded quantity
)

# LEVELS CANNOT BE COMPARED ACROSS THESE TWO SOURCES. Measured, and it
# is the reason this feature is a return difference rather than the
# obvious price difference:
#
#     RELIANCE 2026-08-20   NSE panel 107.56   BSE bhavcopy 1307.50
#     naive level spread = -16,977 bps
#
# nse_prices.py returns a series BACK-ADJUSTED for corporate actions -
# divided down by cumulative factors so returns are continuous through
# splits and bonuses. The BSE bhavcopy carries the actual traded price.
# Neither is wrong; they are different conventions, and differencing
# them produces a huge, stable, entirely fictitious number that looks
# like a precise measurement.
#
# Returns are immune: an adjustment factor is constant within a day, so
# it cancels. The exception is the ex-date itself, where the adjusted
# and unadjusted series legitimately diverge by the whole action - those
# days are masked rather than reported as a 3000bp arbitrage.
EX_DATE_MASK_BPS = 500.0


def _returns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.pct_change()


def _cache_path(date: pd.Timestamp) -> Path:
    return CACHE_DIR / f"bse_{date:%Y%m%d}.csv"


def fetch_bse_raw(date: pd.Timestamp, timeout: int = 30,
                  max_attempts: int = 6,
                  use_cache: bool = True) -> Tuple[Optional[bytes], str]:
    """One day's BSE bhavcopy. reason in {ok, cached, absent, throttled, error}.

    Same 403-is-not-404 discipline as every other archive here: 404 means
    a genuine non-trading day, 403 means try again.
    """
    p = _cache_path(date)
    if use_cache and p.exists() and p.stat().st_size > 0:
        return p.read_bytes(), "cached"

    url = BSE_URL.format(yyyymmdd=f"{date:%Y%m%d}")
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


def parse_bse(raw: bytes, symbols: Sequence[str]) -> Optional[pd.DataFrame]:
    """CSV bytes -> [symbol x close, volume] for BSE equity groups."""
    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except (ValueError, OSError):
        return None
    df.columns = [str(c).strip() for c in df.columns]
    if not {"TckrSymb", "SctySrs", "ClsPric"}.issubset(df.columns):
        return None

    df["TckrSymb"] = df["TckrSymb"].astype(str).str.strip().str.upper()
    df["SctySrs"] = df["SctySrs"].astype(str).str.strip().str.upper()
    df = df[df["SctySrs"].isin(BSE_EQUITY_GROUPS)]
    df = df[df["TckrSymb"].isin({str(s).upper() for s in symbols})]
    if df.empty:
        return None

    for c in ("ClsPric", "TtlTradgVol"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # A symbol can appear once per group; keep the most-traded row rather
    # than an arbitrary first, so the price quoted is the liquid one.
    if "TtlTradgVol" in df.columns:
        df = df.sort_values("TtlTradgVol", ascending=False)
    return df.drop_duplicates("TckrSymb").set_index("TckrSymb")


def cross_exchange_features(nse_close: pd.DataFrame,
                            nse_volume: Optional[pd.DataFrame],
                            symbols: Sequence[str],
                            use_cache: bool = True,
                            max_workers: int = 6) -> Dict[str, pd.DataFrame]:
    """{feature_name: DataFrame[date x symbol]} from NSE vs BSE.

    `nse_close` supplies both the dates and the NSE leg, so the result is
    aligned to the existing panel by construction rather than by a join
    between two calendars that almost agree.

    A name absent from BSE on a given day yields NaN, not 0. Zero spread
    means "the two venues agreed exactly", which is a real and
    informative reading; a missing listing must not impersonate it.
    """
    from concurrent.futures import ThreadPoolExecutor

    syms = [str(s).upper().replace(".NS", "") for s in symbols]
    dates = list(pd.DatetimeIndex(nse_close.index))

    def one(d):
        raw, reason = fetch_bse_raw(pd.Timestamp(d), use_cache=use_cache)
        if raw is None:
            return d, None, reason
        return d, parse_bse(raw, syms), reason

    rows: Dict[pd.Timestamp, pd.DataFrame] = {}
    stats: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for d, df, reason in ex.map(one, dates):
            stats[reason] = stats.get(reason, 0) + 1
            if df is not None:
                rows[pd.Timestamp(d)] = df

    idx = pd.DatetimeIndex(dates)
    empty = pd.DataFrame(np.nan, index=idx, columns=syms)
    if not rows:
        logger.warning("[bse] no sessions fetched: %s", stats)
        return {f: empty.copy() for f in FEATURE_NAMES}
    logger.info("[bse] %d/%d sessions (%s)", len(rows), len(dates), stats)

    bse_close = empty.copy()
    bse_vol = empty.copy()
    for d, df in rows.items():
        for s in syms:
            if s in df.index:
                bse_close.at[d, s] = df.at[s, "ClsPric"]
                if "TtlTradgVol" in df.columns:
                    bse_vol.at[d, s] = df.at[s, "TtlTradgVol"]

    nse = nse_close.copy()
    nse.columns = [str(c).upper().replace(".NS", "") for c in nse.columns]
    nse = nse.reindex(columns=syms)

    # RETURN difference, not level difference - see EX_DATE_MASK_BPS.
    # Requires bse_close to be a contiguous series, so it is forward
    # filled by at most one bar: a name that did not print on BSE for a
    # single session should not silently produce a two-day return
    # masquerading as a one-day divergence.
    bse_ff = bse_close.ffill(limit=1)
    spread = (_returns(nse) - _returns(bse_ff)) * 1e4              # bps

    # Ex-dates: the adjusted and unadjusted series diverge by the whole
    # corporate action on exactly one bar. Masked rather than reported,
    # because a 3000bp "arbitrage" that is really a 1:1 bonus is the kind
    # of number that survives into a result.
    extreme = spread.abs() > EX_DATE_MASK_BPS
    if bool(extreme.any().any()):
        logger.info("[bse] masking %d bar(s) with |spread| > %.0f bps "
                    "(corporate-action ex-dates)",
                    int(extreme.to_numpy().sum()), EX_DATE_MASK_BPS)
    spread = spread.where(~extreme)

    if nse_volume is not None and not nse_volume.empty:
        nv = nse_volume.copy()
        nv.columns = [str(c).upper().replace(".NS", "") for c in nv.columns]
        nv = nv.reindex(index=idx, columns=syms)
        total = (nv + bse_vol).replace(0.0, np.nan)
        share = bse_vol / total
    else:
        share = empty.copy()

    return {
        "nse_bse_spread": spread,
        "nse_bse_spread_abs": spread.abs(),
        "bse_volume_share": share,
    }
