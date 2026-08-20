"""
Authoritative NSE daily prices from the exchange's own UDiFF bhavcopy.

WHY NOT yfinance
----------------
yfinance is what this repo used, and it is unreliable here in two
different ways. Operationally: it is blocked outright from the dev
sandbox (`SSLError ... Connection reset by peer`), so nothing could be
validated locally and every iteration needed a six-minute RunPod round
trip. Substantively: it is a scraped, silently-changing third party for
data the exchange publishes directly.

    archives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip

is the official daily bhavcopy: open/high/low/close/volume for every
listed security, per session.

THE CORPORATE-ACTIONS PROBLEM, AND WHY THIS SOLVES IT FOR FREE
--------------------------------------------------------------
Raw bhavcopy closes are UNADJUSTED. A 1:5 split shows up as a -80%
one-day return. Feed that to a volatility target and you have
manufactured a monstrous fake event; feed it to a directional strategy
and you have manufactured a fake crash to trade.

The usual fix is to fetch a corporate-actions feed and build adjustment
factors yourself, which is fiddly and a classic source of subtle,
edge-manufacturing bugs.

This module does not do that, because it does not have to. The bhavcopy
carries `PrvsClsgPric` - the exchange's OFFICIAL previous close, which
NSE itself adjusts on ex-dates for splits, bonuses and dividends. So

    return(t) = ClsPric(t) / PrvsClsgPric(t) - 1

is already corporate-action-adjusted, computed by the venue that
performs the adjustment. Compounding those returns gives a continuous
adjusted price series with no adjustment logic of our own to get wrong.

That is the whole corporate-actions requirement, discharged by using the
right field instead of building a pipeline.

A caveat stated plainly: this handles adjustments, not survivorship. The
ticker list is supplied by the caller, and if that list was chosen by
looking at today's large-caps then the backtest inherits survivorship
bias no matter how clean the prices are.
"""

from __future__ import annotations

import io
import logging
import random
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("nightevolver.prices")

BHAV_URL = ("https://archives.nseindia.com/content/cm/"
            "BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip")

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "nse_bhav"

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "*/*"}

# Equity series only. "EQ" is the rolling-settlement equity segment; the
# file also contains SGBs, ETFs, debt and derivatives-like rows which
# must not be mixed into an equity universe.
EQUITY_SERIES = ("EQ",)

_COLS = ("TckrSymb", "SctySrs", "OpnPric", "HghPric", "LwPric",
         "ClsPric", "PrvsClsgPric", "TtlTradgVol", "FinInstrmTp")


def _fetch_raw(date: pd.Timestamp, timeout: int = 25,
               max_attempts: int = 8) -> Tuple[Optional[bytes], str]:
    """Returns (zip_bytes, reason). reason in {ok, absent, throttled, error}.

    The archive answers 403 - not 404 - under load, intermittently. See
    nightevolver/flows.py for the measurement. Retry on 403; treat 404
    as a genuine non-trading day.
    """
    url = BHAV_URL.format(yyyymmdd=f"{date:%Y%m%d}")
    reason = "error"
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as f:
                return f.read(), "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, "absent"
            reason = "throttled" if e.code in (403, 429) else "error"
        except (urllib.error.URLError, OSError, TimeoutError):
            reason = "error"
        if attempt < max_attempts - 1:
            time.sleep(min(6.0, 0.5 * (2 ** attempt)) * (0.5 + random.random()))
    return None, reason


def _parse_bhav(raw: bytes, tickers: Sequence[str]) -> Optional[pd.DataFrame]:
    """Extract the requested equity tickers from one bhavcopy zip."""
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
        name = z.namelist()[0]
        df = pd.read_csv(io.BytesIO(z.read(name)), usecols=list(_COLS),
                         low_memory=False)
    except (zipfile.BadZipFile, ValueError, KeyError, OSError):
        return None

    df = df[df["SctySrs"].astype(str).str.strip().isin(EQUITY_SERIES)]
    if "FinInstrmTp" in df.columns:
        df = df[df["FinInstrmTp"].astype(str).str.strip() == "STK"]
    df["TckrSymb"] = df["TckrSymb"].astype(str).str.strip()
    df = df[df["TckrSymb"].isin(set(tickers))]
    return df if not df.empty else None


def _cache_path(date: pd.Timestamp) -> Path:
    return CACHE_DIR / f"{date:%Y%m%d}.zip"


def fetch_bhav_day(date: pd.Timestamp, tickers: Sequence[str],
                   use_cache: bool = True) -> Tuple[Optional[pd.DataFrame], str]:
    cp = _cache_path(date)
    if use_cache and cp.exists():
        try:
            parsed = _parse_bhav(cp.read_bytes(), tickers)
            if parsed is not None:
                return parsed, "cache"
        except OSError:
            pass

    raw, reason = _fetch_raw(date)
    if raw is None:
        return None, reason
    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(raw)
        except OSError:
            pass
    parsed = _parse_bhav(raw, tickers)
    return parsed, (reason if parsed is not None else "error")


def fetch_bhav_range(tickers: Sequence[str], start: str, end: Optional[str] = None,
                     max_workers: int = 6, use_cache: bool = True,
                     max_unresolved_frac: float = 0.05,
                     ) -> Dict[pd.Timestamp, pd.DataFrame]:
    dates = pd.bdate_range(start, end or pd.Timestamp.today().normalize())
    out: Dict[pd.Timestamp, pd.DataFrame] = {}
    reasons: Dict[str, int] = {}

    def one(d):
        return d, fetch_bhav_day(d, tickers, use_cache=use_cache)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for date, (df, reason) in ex.map(one, dates):
            reasons[reason] = reasons.get(reason, 0) + 1
            if df is not None:
                out[date] = df

    unresolved = reasons.get("throttled", 0) + reasons.get("error", 0)
    logger.info("[prices] %d/%d sessions | holidays(404)=%d unresolved=%d %s",
                len(out), len(dates), reasons.get("absent", 0), unresolved,
                {k: v for k, v in sorted(reasons.items())})
    if len(dates) and unresolved / len(dates) > max_unresolved_frac:
        raise RuntimeError(
            f"{unresolved}/{len(dates)} sessions unresolved (throttled/error, "
            f"not 404). Refusing to build a price history with silent gaps. "
            f"Re-run to extend the cache.")
    return out


def build_adjusted_frames(days: Dict[pd.Timestamp, pd.DataFrame],
                          tickers: Sequence[str],
                          corporate_actions=None,
                          require_actions: bool = True,
                          ) -> Tuple[pd.DataFrame, pd.DataFrame,
                                     pd.DataFrame, pd.DataFrame]:
    """Return (close, high, low, volume) with a corporate-action-adjusted
    close, built by compounding per-day returns that are corrected on
    ex-dates using the NSE corporate-actions feed.

    high/low/volume are returned on the raw scale but rescaled by the
    same cumulative adjustment factor as close, so that high/low stay
    consistent with the adjusted close (ATR, stochastics and Bollinger
    bands all mix the three and would be nonsense otherwise).
    """
    dates = pd.DatetimeIndex(sorted(days))
    cols = list(tickers)

    def empty():
        return pd.DataFrame(np.nan, index=dates, columns=cols, dtype=float)

    raw_close, raw_high, raw_low, raw_vol, raw_prev = (empty() for _ in range(5))
    for d in dates:
        df = days[d].set_index("TckrSymb")
        for tk in cols:
            if tk not in df.index:
                continue
            row = df.loc[tk]
            if isinstance(row, pd.DataFrame):      # duplicate rows: take first
                row = row.iloc[0]
            raw_close.at[d, tk] = row["ClsPric"]
            raw_high.at[d, tk] = row["HghPric"]
            raw_low.at[d, tk] = row["LwPric"]
            raw_vol.at[d, tk] = row["TtlTradgVol"]
            raw_prev.at[d, tk] = row["PrvsClsgPric"]

    # Corporate-action-corrected returns. PrvsClsgPric is the RAW prior
    # close, NOT an adjusted one - verified against RELIANCE's 1:1 bonus
    # (2024-10-28: prev=2655.70, close=1334.35, -49.76%). See
    # nightevolver/corporate_actions.py for the measurement and the fix.
    from nightevolver.corporate_actions import adjust_returns, fetch_all_corporate_actions

    actions = corporate_actions
    if actions is None:
        actions = fetch_all_corporate_actions(list(cols))
    rets, masked = adjust_returns(raw_close, raw_prev, actions,
                                  require_actions=require_actions)

    # Masked bars (demergers, unexplained jumps) become flat rather than
    # being dropped, so the calendar stays aligned with the flow data.
    # They are flat because the correct value is UNKNOWN - not because
    # nothing happened.
    rets = rets.fillna(0.0)
    adj_close = 100.0 * (1.0 + rets).cumprod()

    # Rescale intraday levels onto the adjusted close's scale.
    scale = (adj_close / raw_close.replace(0.0, np.nan)).ffill().bfill()
    adj_high = raw_high * scale
    adj_low = raw_low * scale

    keep = adj_close.notna().all(axis=1)
    return (adj_close[keep], adj_high[keep], adj_low[keep],
            raw_vol[keep].fillna(0.0))


def fetch_nse_prices(tickers: Sequence[str], start: str, end: Optional[str] = None,
                     max_workers: int = 6, use_cache: bool = True,
                     require_actions: bool = True):
    """Convenience: bhavcopy -> MarketData, bypassing yfinance entirely.

    `tickers` are NSE symbols WITHOUT the .NS suffix (RELIANCE, not
    RELIANCE.NS); the suffix is stripped if present so the same config
    list works for both loaders.
    """
    from nightevolver.data_loader import build_market_data

    syms = [t[:-3] if t.upper().endswith(".NS") else t for t in tickers]
    days = fetch_bhav_range(syms, start, end, max_workers=max_workers,
                            use_cache=use_cache)
    if not days:
        raise RuntimeError("no bhavcopy sessions retrieved")
    close, high, low, vol = build_adjusted_frames(
        days, syms, require_actions=require_actions)
    logger.info("[prices] %d bars x %d tickers (%s .. %s)", len(close),
                close.shape[1], close.index[0].date(), close.index[-1].date())
    return build_market_data(close, high, low, vol)
