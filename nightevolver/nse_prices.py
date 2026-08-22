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

THE CORPORATE-ACTIONS PROBLEM, AND WHY THE FIELD DOES NOT SOLVE IT
------------------------------------------------------------------
Raw bhavcopy closes are UNADJUSTED. A 1:5 split shows up as a -80%
one-day return. Feed that to a volatility target and you have
manufactured a monstrous fake event; feed it to a directional strategy
and you have manufactured a fake crash to trade.

THIS MODULE USED TO CLAIM THE PROBLEM WAS FREE, and it was wrong. The
claim was that `PrvsClsgPric` is the exchange's officially ADJUSTED
previous close, so that

    return(t) = ClsPric(t) / PrvsClsgPric(t) - 1

needs no adjustment logic of our own. It was stated as the reason not to
build a corporate-actions pipeline at all. Measured against the archive,
on two clean splits:

    IRCTC     ex 2021-10-28, 1:5    close 913.50 vs prev 4130.15  -77.88%
    NESTLEIND ex 2024-01-05, 1:10   close 2666.40 vs prev 27116.40 -90.17%

Those are the raw split ratios, undiminished. PrvsClsgPric carries the
PREVIOUS SESSION'S TRADED CLOSE, not an adjusted one.

So the pipeline is required after all, and corporate_actions.py is it:
adjust_returns divides the ex-date's previous close by the action's
price ratio, adds back dividends, masks demergers, and masks any
residual move still exceeding 25% as an action it does not know about.
Because PrvsClsgPric is unadjusted, that correction is applied once and
is not a double-adjustment.

This is also why `require_actions=True` is a hard error and not a
warning. A symbol with no action list is indistinguishable from a failed
fetch, and the cost of guessing wrong is an -80% bar. The residual mask
would catch a move that large, but masking silently discards a real
session; the guard makes the caller decide knowingly.

A second caveat stated plainly: this handles adjustments, not
survivorship. The ticker list is supplied by the caller, and if that
list was chosen by looking at today's large-caps then the backtest
inherits survivorship bias no matter how clean the prices are. Use
top_liquid_symbols(as_of=...) to rank the universe at a point in time.
"""

from __future__ import annotations

import http.client
import io
import logging
import random
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from .nethttp import TRANSIENT_NET_ERRORS

logger = logging.getLogger("nightevolver.prices")

BHAV_URL = ("https://archives.nseindia.com/content/cm/"
            "BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip")

# UDiFF begins in 2024; before that the archive uses the legacy layout.
# Supporting both is what makes a ~7-year panel reachable instead of
# ~2.5, and history is the binding constraint on the one live result in
# this project (atm_iv -> vol_5d at p=0.065, limited by power).
LEGACY_BHAV_URL = ("https://archives.nseindia.com/content/historical/"
                   "EQUITIES/{yyyy}/{MON}/cm{dd}{MON}{yyyy}bhav.csv.zip")
UDIFF_START = pd.Timestamp("2024-01-01")

# Legacy -> UDiFF names for the columns this module reads.
_LEGACY_RENAME = {
    "SYMBOL": "TckrSymb", "SERIES": "SctySrs", "OPEN": "OpnPric",
    "HIGH": "HghPric", "LOW": "LwPric", "CLOSE": "ClsPric",
    "PREVCLOSE": "PrvsClsgPric", "TOTTRDQTY": "TtlTradgVol",
    "TOTTRDVAL": "TtlTrfVal", "TIMESTAMP": "TradDt",
}

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
    mon = f"{date:%b}".upper()
    urls = [BHAV_URL.format(yyyymmdd=f"{date:%Y%m%d}"),
            LEGACY_BHAV_URL.format(yyyy=f"{date:%Y}", MON=mon, dd=f"{date:%d}")]
    if date < UDIFF_START:
        urls.reverse()

    reason = "error"
    for attempt in range(max_attempts):
        absent = 0
        for url in urls:
            try:
                req = urllib.request.Request(url, headers=_UA)
                with urllib.request.urlopen(req, timeout=timeout) as f:
                    return f.read(), "ok"
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    absent += 1
                    continue
                reason = "throttled" if e.code in (403, 429) else "error"
            except TRANSIENT_NET_ERRORS:
                reason = "error"
        # Both formats 404 -> a real non-trading day. Anything else is
        # throttling, which measured up to FOUR consecutive 403s on the
        # legacy path before succeeding - collapsing that into "absent"
        # is what manufactured phantom holidays in flows.py.
        if absent == len(urls):
            return None, "absent"
        if attempt < max_attempts - 1:
            time.sleep(min(6.0, 0.5 * (2 ** attempt)) * (0.5 + random.random()))
    return None, reason


# Indian ISINs encode the instrument class in the first three characters,
# and it is the ONLY field that separates company equity from the other
# things trading in the EQ series. Both `SctySrs` and `FinInstrmTp` read
# EQ/STK for all of them, in both file formats:
#
#   INE  company equity                    RELIANCE  INE002A01018
#   INF  mutual fund / ETF units           LIQUIDBEES INF732E01037
#   IN9  differential-voting-rights class  TATAMTRDVR IN9155A01020
#
# LIQUIDBEES - a money-market ETF pinned at a 1000 NAV - ranked into a
# 2019 top-100 by turnover and sat in the panel with 27 distinct closes
# across 1,825 bars and 0.01% annualised volatility. Nothing rejected it:
# it is EQ, it is STK, it is enormously traded. In a volatility study a
# constant series is not merely useless, it is selectable - the lowest-vol
# "stock" in the universe by a factor of 2,000.
#
# DVR lines are excluded for a different reason: TATAMTRDVR is the same
# company as TATAMOTORS, so keeping both counts one issuer twice and
# inflates every cross-sectional statistic with a near-duplicate.
EQUITY_ISIN_PREFIX = "INE"


def _equity_rows(df: pd.DataFrame) -> pd.DataFrame:
    """EQ-series company equity only - no ETFs, funds or DVR classes."""
    out = df[df["SctySrs"].astype(str).str.strip().isin(EQUITY_SERIES)]
    if "FinInstrmTp" in out.columns:
        out = out[out["FinInstrmTp"].astype(str).str.strip() == "STK"]
    if "ISIN" in out.columns:
        isin = out["ISIN"].astype(str).str.strip()
        # Only filter when the column is actually populated; a file that
        # omits ISINs must not silently yield an empty universe.
        if isin.str.startswith("IN").any():
            out = out[isin.str.startswith(EQUITY_ISIN_PREFIX)]
    return out


def _read_bhav_csv(raw: bytes) -> Optional[pd.DataFrame]:
    """Bhavcopy zip -> frame with UDiFF column names, whatever the era.

    THE NORMALISATION LIVES HERE AND NOWHERE ELSE. It used to be inlined
    in _parse_bhav, and top_liquid_symbols - which opens the same zip for
    a different purpose - carried its own copy that predated legacy
    support. Resolving a 2019 universe therefore died on

        KeyError: 'SctySrs'

    which is the loud version of this bug. The quiet version is a second
    reader that renames SOME columns and silently drops the rest.
    """
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), low_memory=False)
    except (zipfile.BadZipFile, ValueError, KeyError, OSError, IndexError):
        return None

    df.columns = [str(c).strip() for c in df.columns]
    if "SYMBOL" in df.columns and "TckrSymb" not in df.columns:
        df = df.rename(columns=_LEGACY_RENAME)
        df["FinInstrmTp"] = "STK"      # legacy equity files are all stock
    return df


def _parse_bhav(raw: bytes, tickers: Sequence[str]) -> Optional[pd.DataFrame]:
    """Extract the requested equity tickers from one bhavcopy zip."""
    df = _read_bhav_csv(raw)
    if df is None:
        return None

    missing = [c for c in _COLS if c not in df.columns]
    if missing:
        return None

    # FILTER BEFORE SUBSETTING. _COLS does not carry ISIN, so narrowing
    # the frame first would drop the only column that distinguishes an
    # ETF from a stock - and _equity_rows would then quietly become a
    # no-op that still looked like it was filtering.
    df = _equity_rows(df)
    df = df[[c for c in _COLS if c in df.columns]]
    df["TckrSymb"] = df["TckrSymb"].astype(str).str.strip()
    df = df[df["TckrSymb"].isin(set(tickers))]
    return df if not df.empty else None


def _zip_is_intact(raw: bytes) -> bool:
    """True if every member's CRC matches - i.e. the download completed."""
    try:
        return zipfile.ZipFile(io.BytesIO(raw)).testzip() is None
    except (zipfile.BadZipFile, ValueError, OSError):
        return False


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
    parsed = _parse_bhav(raw, tickers)
    # Only a STRUCTURALLY INTACT zip gets persisted. A short read that
    # does not raise IncompleteRead - chunked transfer can end cleanly on
    # a truncated body - would otherwise write a corrupt file that every
    # later run reads from cache and treats as a missing session, with no
    # fetch to correct it. Validating costs one CRC pass and turns a
    # permanent hole into one retry.
    #
    # The test is the zip, NOT `parsed is not None`: parsing also returns
    # None when the file is perfect but carries none of the requested
    # tickers, and refusing to cache those would re-download a good
    # bhavcopy on every run.
    if use_cache and _zip_is_intact(raw):
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(raw)
        except OSError:
            pass
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
                          min_coverage: float = 0.90,
                          min_breadth: float = 0.60,
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

    # NEGATIVE VOLUME IS A PARSE OR FEED DEFECT, NEVER A REAL SESSION - a
    # trade count cannot be negative. Nothing checked this before; audit
    # found the gap. Masked to NaN here (not zero, which would read as a
    # real halted-with-no-trades session) so it flows through the same
    # span/breadth logic as every other kind of missing data, rather than
    # silently entering volume-weighted features (avg_trade_size, Amihud
    # illiquidity, Kyle's lambda) as a negative denominator.
    bad_vol = raw_vol < 0
    if bad_vol.to_numpy().any():
        n_bad = int(bad_vol.to_numpy().sum())
        logger.warning("[prices] %d negative-volume cell(s) masked to NaN "
                       "- a trade count cannot be negative", n_bad)
        raw_vol = raw_vol.mask(bad_vol)

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

    # THE LIVE SPAN: first real trade to last real trade, per name.
    #
    # `rets.fillna(0.0)` used to be applied to the whole frame, on the
    # argument that a masked bar should stay flat to keep the calendar
    # aligned. That is right for a gap INSIDE a live series and wrong
    # everywhere else, because a name that delists simply stops having
    # rows - and filling those with zero returns manufactures a flat,
    # perfectly tradeable asset that persists to the end of the panel.
    #
    # Measured on the 2019 top-100 before the fix:
    #
    #   DHFL        1,824 finite returns, 71.0% exactly zero, flat at 7.56
    #   RELCAPITAL  1,824 finite returns, 67.4% exactly zero, flat at 9.37
    #   HDFC        1,824 finite returns, 42.0% exactly zero (merged 2023)
    #   RELIANCE    1,824 finite returns,  0.2% exactly zero
    #
    # DHFL went into liquidation. It was in the panel as a zero-volatility
    # instrument for three quarters of the history - the same defect as
    # the LIQUIDBEES ETF above, arrived at from the opposite direction,
    # and far more dangerous because a vol study will select it.
    #
    # The 0.90 coverage filter hid this rather than fixing it: it dropped
    # the dead names, which is survivorship bias - exactly what
    # top_liquid_symbols(as_of=...) exists to avoid.
    alive = raw_close.notna()
    started = alive.cummax()                          # from first trade on
    ending = alive[::-1].cummax()[::-1]                # until last trade
    span = started & ending

    # Inside the span a gap is "unknown, assume flat". Outside it the
    # instrument does not exist and must stay absent.
    adj_close = 100.0 * (1.0 + rets.where(span).fillna(0.0)).cumprod()
    adj_close = adj_close.where(span)

    # Rescale intraday levels onto the adjusted close's scale. ffill/bfill
    # is confined to the span for the same reason.
    scale = (adj_close / raw_close.replace(0.0, np.nan)).ffill().bfill()
    adj_high = (raw_high * scale).where(span)
    adj_low = (raw_low * scale).where(span)
    raw_vol = raw_vol.where(span)

    # Drop THIN SYMBOLS BEFORE dropping dates. The row filter below
    # requires every column to be present, so with a large universe a
    # single late-listed or long-suspended name would delete that date
    # for all the others - at 50 names one bad symbol can cost most of
    # the panel. Dropping the symbol costs one column; dropping the
    # dates costs the whole study.
    coverage = raw_close.notna().mean()
    thin = coverage[coverage < min_coverage].index.tolist()
    if thin:
        logger.warning("[prices] dropping %d symbol(s) below %.0f%% coverage: %s",
                       len(thin), min_coverage * 100,
                       ", ".join(f"{s}({coverage[s]:.0%})" for s in thin[:10]))
        adj_close = adj_close.drop(columns=thin)
        adj_high = adj_high.drop(columns=thin)
        adj_low = adj_low.drop(columns=thin)
        raw_vol = raw_vol.drop(columns=thin)
    if adj_close.shape[1] == 0:
        raise RuntimeError("every symbol fell below the coverage threshold")

    # A DATE SURVIVES ON BREADTH, NOT ON COMPLETENESS. Requiring every
    # column to be present was viable only because the zero-fill made
    # every column always present. On an honestly ragged panel it would
    # delete every date on which any name had delisted - which, with one
    # early failure like JETAIRWAYS, is the entire history.
    breadth = adj_close.notna().sum(axis=1)
    keep = breadth >= max(2, int(np.ceil(min_breadth * adj_close.shape[1])))
    n_alive = adj_close.notna().sum(axis=1)
    logger.info("[prices] %d/%d symbols kept, %d/%d dates pass breadth "
                "(names live per date: min %d, median %d, max %d)",
                adj_close.shape[1], len(cols), int(keep.sum()), len(keep),
                int(n_alive.min()), int(n_alive.median()), int(n_alive.max()))
    if not keep.any():
        raise RuntimeError(
            f"no date has at least {min_breadth:.0%} of symbols live")
    return (adj_close[keep], adj_high[keep], adj_low[keep], raw_vol[keep])


def top_liquid_symbols(as_of: str, n: int = 50, use_cache: bool = True,
                       min_price: float = 20.0) -> List[str]:
    """The `n` most-traded NSE equities by turnover on `as_of`.

    WHY POINT-IN-TIME. Picking "today's large caps" and backtesting them
    over the past two years is survivorship bias with extra steps: every
    name in the list is one that survived and stayed liquid, which is
    information the strategy would not have had. Selecting on a date at
    or before the training window starts does not eliminate the problem
    - names that later delisted still vanish from the price panel - but
    it removes the part that is purely an artefact of choosing the list
    after seeing the outcome.

    This is nearly free. The bhavcopy is ONE file per session containing
    every listed security, and it is already being downloaded and cached
    for the ten-name universe. Going to fifty names costs no extra
    network, and trade count scales linearly with the universe - which
    is the binding constraint on validating long-hold strategies here
    (see ga_engine.required_validation_bars).
    """
    date = pd.Timestamp(as_of)
    # Walk back to the most recent session actually present.
    for back in range(0, 10):
        raw, reason = _fetch_raw(date - pd.Timedelta(days=back))
        if raw is not None:
            break
    else:
        raise RuntimeError(f"no bhavcopy session within 10 days before {as_of}")

    df = _read_bhav_csv(raw)
    if df is None or "SctySrs" not in df.columns:
        raise RuntimeError(
            f"bhavcopy for {date.date()} is unreadable or has no series "
            "column - cannot rank a universe from it")
    df = _equity_rows(df)
    df = df[pd.to_numeric(df["ClsPric"], errors="coerce") >= min_price]

    turnover_col = "TtlTrfVal" if "TtlTrfVal" in df.columns else None
    if turnover_col is None:
        df["_t"] = (pd.to_numeric(df["ClsPric"], errors="coerce")
                    * pd.to_numeric(df["TtlTradgVol"], errors="coerce"))
        turnover_col = "_t"
    df[turnover_col] = pd.to_numeric(df[turnover_col], errors="coerce")
    df = df.dropna(subset=[turnover_col]).sort_values(turnover_col, ascending=False)

    syms = [str(s).strip() for s in df["TckrSymb"].head(n).tolist()]
    logger.info("[prices] top %d by turnover as of %s: %s%s", len(syms),
                date.date(), ", ".join(syms[:8]), " ..." if len(syms) > 8 else "")
    return syms


def fetch_nse_prices(tickers: Sequence[str], start: str, end: Optional[str] = None,
                     max_workers: int = 6, use_cache: bool = True,
                     require_actions: bool = True, with_flows: bool = False,
                     min_coverage: float = 0.90,
                     min_breadth: float = 0.60):
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
        days, syms, require_actions=require_actions,
        min_coverage=min_coverage, min_breadth=min_breadth)
    logger.info("[prices] %d bars x %d tickers (%s .. %s)", len(close),
                close.shape[1], close.index[0].date(), close.index[-1].date())

    flows = None
    if with_flows:
        from nightevolver.flows import load_flow_features
        flows = load_flow_features(close.index)
    return build_market_data(close, high, low, vol, flows=flows)
