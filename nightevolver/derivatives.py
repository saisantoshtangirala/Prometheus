"""
NSE F&O bhavcopy: open interest, put-call ratios, basis, implied vol.

WHY THIS IS THE FIRST NEW DATA CLASS WORTH ADDING. The 26 channels the
GA searches are 20 transforms of the same daily OHLCV plus 6 participant
flow series. A walk-forward with a null cloud showed that set carries no
directional edge, and no rearrangement of price transforms will change
that - they are all functions of one thin input.

Derivatives data is genuinely different information, not a different
view of the same numbers:

  * OPEN INTEREST is positioning, not price. It says how many contracts
    are still open and whether they are being built or unwound - a fact
    about what participants have committed to, which no OHLCV transform
    can express.
  * IMPLIED VOLATILITY is the market's forward-looking distribution.
    Realised vol looks backwards; IV is a price paid today for a claim
    on tomorrow.
  * SKEW is the asymmetry of that distribution - what downside insurance
    costs relative to upside. It has no realised-return analogue at all.
  * BASIS is the cost of carry and, when it dislocates, a direct read on
    leveraged positioning pressure.

And it is free, historical and complete back to the archive's start,
which the order-book tape is not - the tape must be recorded forward and
is worth nothing for months. This can be audited tonight.

CAUSALITY. Day T's file is published after day T's close and describes
that close. Features from it are therefore known at T and legitimately
predict T -> T+1, exactly like the equity bhavcopy this mirrors. No
extra lag is applied, and none is needed. `flows.py` DOES need one
because NSCCL participant data lands a day late; this does not, and
copying that lag would silently throw away a day of information.

IMPLIED VOLATILITY IS COMPUTED, NOT SUPPLIED. The bhavcopy carries no IV
column, so it is inverted from the option's own settlement price by
bisection on Black-Scholes. Bisection rather than Newton deliberately:
vega collapses for deep out-of-the-money and near-expiry options, and
Newton diverges there exactly where the data is thinnest and the wrong
answer is least visible. Bisection is slower and cannot diverge.
"""

from __future__ import annotations

import io
import logging
import math
import random
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from .nethttp import TRANSIENT_NET_ERRORS

logger = logging.getLogger("nightevolver.derivatives")

FO_URL = ("https://archives.nseindia.com/content/fo/"
          "BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip")

# The UDiFF format begins in 2024. Everything before that is the legacy
# layout, which is what makes a ~7-year history reachable instead of
# ~2.5 - and history is the binding constraint on the one live result
# here (atm_iv -> vol_5d at p=0.065, limited by power, not effect size).
LEGACY_FO_URL = ("https://archives.nseindia.com/content/historical/"
                 "DERIVATIVES/{yyyy}/{MON}/fo{dd}{MON}{yyyy}bhav.csv.zip")
UDIFF_START = pd.Timestamp("2024-01-01")

# Legacy -> UDiFF column names. The two files describe the same market
# with different vocabularies; normalising at the edge means nothing
# downstream has to know which era a row came from.
_LEGACY_RENAME = {
    "SYMBOL": "TckrSymb", "EXPIRY_DT": "XpryDt", "STRIKE_PR": "StrkPric",
    "OPTION_TYP": "OptnTp", "CLOSE": "ClsPric", "SETTLE_PR": "SttlmPric",
    "OPEN_INT": "OpnIntrst", "CHG_IN_OI": "ChngInOpnIntrst",
    "CONTRACTS": "TtlTradgVol", "TIMESTAMP": "TradDt",
}
_LEGACY_INSTRUMENT = {
    "FUTSTK": "STF", "OPTSTK": "STO", "FUTIDX": "IDF", "OPTIDX": "IDO",
}

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "nse_fo"

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "*/*"}

# FinInstrmTp values in the UDiFF F&O file.
STOCK_FUT, STOCK_OPT = "STF", "STO"
INDEX_FUT, INDEX_OPT = "IDF", "IDO"

# India risk-free proxy. The IV level is mildly sensitive to this and the
# CROSS-SECTIONAL and TIME-SERIES comparisons this module produces are
# not, since the same rate is applied everywhere. Kept explicit rather
# than buried so it can be replaced with an RBI series later.
RISK_FREE = 0.065

_USECOLS = ["TradDt", "FinInstrmTp", "TckrSymb", "XpryDt", "StrkPric",
            "OptnTp", "ClsPric", "SttlmPric", "UndrlygPric", "OpnIntrst",
            "ChngInOpnIntrst", "TtlTradgVol"]


# ---------------------------------------------------------------------
# Black-Scholes and its inverse
# ---------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot: float, strike: float, t: float, vol: float,
             is_call: bool, r: float = RISK_FREE) -> float:
    """Black-Scholes European price. No dividend term.

    NSE stock options are American, so this is an approximation. It is
    the right one here: the systematic error is small for short-dated
    near-the-money options, it is in the SAME DIRECTION for every name
    and day, and everything downstream compares IVs to each other rather
    than trading against them.
    """
    if t <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, (spot - strike) if is_call else (strike - spot))
    sq = vol * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t) / sq
    d2 = d1 - sq
    disc = math.exp(-r * t)
    if is_call:
        return spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    return strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def implied_vol(price: float, spot: float, strike: float, t: float,
                is_call: bool, r: float = RISK_FREE,
                lo: float = 1e-4, hi: float = 5.0,
                tol: float = 1e-4, max_iter: int = 60) -> float:
    """Invert Black-Scholes by bisection. NaN when there is no solution.

    Returns NaN rather than a clipped bound when the observed price is
    below intrinsic or above the no-arbitrage cap - those are stale or
    illiquid quotes, and a fabricated 0.0001 or 5.0 IV would be treated
    downstream as a real extreme observation. Absent is honest; a bound
    is a lie with a number attached.
    """
    if not (np.isfinite(price) and np.isfinite(spot) and np.isfinite(strike)):
        return float("nan")
    if t <= 0 or price <= 0 or spot <= 0 or strike <= 0:
        return float("nan")

    intrinsic = max(0.0, (spot - strike * math.exp(-r * t)) if is_call
                    else (strike * math.exp(-r * t) - spot))
    if price < intrinsic - 1e-6:
        return float("nan")
    cap = spot if is_call else strike * math.exp(-r * t)
    if price >= cap:
        return float("nan")

    f_lo = bs_price(spot, strike, t, lo, is_call, r) - price
    f_hi = bs_price(spot, strike, t, hi, is_call, r) - price
    if f_lo * f_hi > 0:
        return float("nan")          # not bracketed; no root in [lo, hi]

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price(spot, strike, t, mid, is_call, r) - price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------

def _zip_is_intact(raw: bytes) -> bool:
    """True if every member's CRC matches - i.e. the download completed."""
    try:
        return zipfile.ZipFile(io.BytesIO(raw)).testzip() is None
    except (zipfile.BadZipFile, ValueError, OSError):
        return False


def _cache_path(date: pd.Timestamp) -> Path:
    return CACHE_DIR / f"fo_{date:%Y%m%d}.zip"


def fetch_fo_raw(date: pd.Timestamp, timeout: int = 30,
                 max_attempts: int = 8,
                 use_cache: bool = True) -> Tuple[Optional[bytes], str]:
    """One day's F&O zip. reason in {ok, cached, absent, throttled, error}.

    403-not-404 under load is the measured behaviour of this archive -
    see flows.py, where collapsing both to None manufactured phantom
    holidays and silently lost ~20% of sessions. 404 means a genuine
    non-trading day; 403 and 429 mean try again.
    """
    p = _cache_path(date)
    if use_cache and p.exists() and p.stat().st_size > 0:
        return p.read_bytes(), "cached"

    # Try the format that matches the date first, then the other one.
    # The changeover is documented as 2024 but a hard cutoff would lose
    # any straddling week, and a 404 from the wrong format is cheap.
    mon = f"{date:%b}".upper()
    urls = [FO_URL.format(yyyymmdd=f"{date:%Y%m%d}"),
            LEGACY_FO_URL.format(yyyy=f"{date:%Y}", MON=mon, dd=f"{date:%d}")]
    if date < UDIFF_START:
        urls.reverse()

    reason = "error"
    for attempt in range(max_attempts):
        absent = 0
        for url in urls:
            try:
                req = urllib.request.Request(url, headers=_UA)
                with urllib.request.urlopen(req, timeout=timeout) as f:
                    raw = f.read()
                # See nse_prices._zip_is_intact: only a zip whose CRCs
                # check out is persisted, so a truncated body costs a
                # retry rather than a permanent cached hole.
                if use_cache and _zip_is_intact(raw):
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(raw)
                return raw, "ok"
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    absent += 1
                    continue
                reason = "throttled" if e.code in (403, 429) else "error"
            except TRANSIENT_NET_ERRORS:
                reason = "error"
        if absent == len(urls):
            return None, "absent"          # genuine non-trading day
        if attempt < max_attempts - 1:
            time.sleep(min(6.0, 0.5 * (2 ** attempt)) * (0.5 + random.random()))
    return None, reason


def parse_fo(raw: bytes) -> Optional[pd.DataFrame]:
    """Zip bytes -> tidy frame, or None if unreadable."""
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),
                         low_memory=False)
    except (zipfile.BadZipFile, ValueError, KeyError, OSError):
        return None
    # LEGACY SCHEMA NORMALISATION. Pre-2024 files use a different
    # vocabulary for the same market. Renaming at the edge means nothing
    # downstream needs to know which era a row came from - the
    # alternative is an era check at every use site, and the one that
    # gets forgotten produces silently empty features for a whole epoch.
    df.columns = [str(c).strip() for c in df.columns]
    if "INSTRUMENT" in df.columns and "FinInstrmTp" not in df.columns:
        df = df.rename(columns=_LEGACY_RENAME)
        df["FinInstrmTp"] = (df["INSTRUMENT"].astype(str).str.strip()
                             .map(_LEGACY_INSTRUMENT))
        # Legacy has NO underlying-price column. It is filled from the
        # equity bhavcopy by the caller; left absent here so a missing
        # spot is visible as NaN rather than silently defaulted.
        if "UndrlygPric" not in df.columns:
            df["UndrlygPric"] = np.nan
        # Futures rows carry OPTION_TYP 'XX' rather than an empty string.
        if "OptnTp" in df.columns:
            df["OptnTp"] = df["OptnTp"].astype(str).str.strip().replace(
                {"XX": ""})

    keep = [c for c in _USECOLS if c in df.columns]
    if "FinInstrmTp" not in keep or "TckrSymb" not in keep:
        return None
    df = df[keep].copy()
    df["TckrSymb"] = df["TckrSymb"].astype(str).str.strip()
    df["FinInstrmTp"] = df["FinInstrmTp"].astype(str).str.strip()
    for c in ("TradDt", "XpryDt"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("StrkPric", "ClsPric", "SttlmPric", "UndrlygPric",
              "OpnIntrst", "ChngInOpnIntrst", "TtlTradgVol"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "OptnTp" in df.columns:
        df["OptnTp"] = df["OptnTp"].astype(str).str.strip().str.upper()
    return df


# ---------------------------------------------------------------------
# Per-underlying daily features
# ---------------------------------------------------------------------

FEATURE_NAMES = (
    "pcr_oi",            # put OI / call OI, log
    "pcr_volume",        # put volume / call volume, log
    "oi_change_norm",    # futures OI change / futures OI
    "basis_annualised",  # (future - spot)/spot, annualised by days to expiry
    "atm_iv",            # near-expiry at-the-money implied vol
    "iv_skew",           # OTM put IV - OTM call IV, at ~+-5% moneyness
    "iv_term",           # next-expiry ATM IV - near-expiry ATM IV
    "opt_volume_ratio",  # option volume / futures volume, log
)


def _atm_iv_and_skew(opts: pd.DataFrame, spot: float,
                     t_years: float) -> Tuple[float, float]:
    """(ATM IV, skew) for one underlying and one expiry.

    Skew is IV(~5% OTM put) - IV(~5% OTM call). A fixed moneyness rather
    than a fixed delta: delta needs an IV to compute, which is the thing
    being measured, and the circularity costs more than the precision is
    worth on a daily bar.
    """
    if opts.empty or not np.isfinite(spot) or spot <= 0 or t_years <= 0:
        return float("nan"), float("nan")

    px = opts["SttlmPric"].where(opts["SttlmPric"] > 0, opts["ClsPric"])
    work = opts.assign(_px=px).dropna(subset=["StrkPric", "_px"])
    work = work[work["_px"] > 0]
    if work.empty:
        return float("nan"), float("nan")

    def iv_at(target: float, is_call: bool) -> float:
        side = work[work["OptnTp"] == ("CE" if is_call else "PE")]
        if side.empty:
            return float("nan")
        row = side.iloc[(side["StrkPric"] - target).abs().argsort().iloc[0]]
        return implied_vol(float(row["_px"]), spot, float(row["StrkPric"]),
                           t_years, is_call)

    civ, piv = iv_at(spot, True), iv_at(spot, False)
    # Average the two ATM legs, but guard the all-NaN case explicitly.
    # np.nanmean of [nan, nan] returns nan AND emits "Mean of empty
    # slice" per call - which on a 600-day fetch is thousands of lines
    # of warning for a case that is both expected and correctly handled
    # (a name with no liquid ATM options simply has no ATM IV). Warning
    # noise that is always benign trains the reader to ignore warnings.
    legs = [v for v in (civ, piv) if np.isfinite(v)]
    atm = float(sum(legs) / len(legs)) if legs else float("nan")
    otm_p = iv_at(spot * 0.95, False)
    otm_c = iv_at(spot * 1.05, True)
    skew = (otm_p - otm_c) if (np.isfinite(otm_p) and np.isfinite(otm_c)) \
        else float("nan")
    return atm, skew


def day_features(df: pd.DataFrame,
                 symbols: Sequence[str],
                 spot: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """One day's F&O frame -> [symbol x FEATURE_NAMES].

    Ratios are returned as LOGS. A raw put/call ratio is bounded below by
    0 and unbounded above, so its distribution is skewed and a rank
    statistic on it is dominated by the upper tail; log makes it roughly
    symmetric around 0 = balanced, which is also the value a missing
    side should fall back to.
    """
    want = set(symbols)
    out = pd.DataFrame(index=sorted(want), columns=list(FEATURE_NAMES),
                       dtype=float)
    if df is None or df.empty:
        return out

    trad = df["TradDt"].dropna()
    trad_dt = trad.iloc[0] if not trad.empty else None

    for sym, g in df[df["TckrSymb"].isin(want)].groupby("TckrSymb"):
        futs = g[g["FinInstrmTp"].isin((STOCK_FUT, INDEX_FUT))]
        opts = g[g["FinInstrmTp"].isin((STOCK_OPT, INDEX_OPT))]

        calls = opts[opts["OptnTp"] == "CE"]
        puts = opts[opts["OptnTp"] == "PE"]
        c_oi, p_oi = calls["OpnIntrst"].sum(), puts["OpnIntrst"].sum()
        if c_oi > 0 and p_oi > 0:
            out.at[sym, "pcr_oi"] = math.log(p_oi / c_oi)
        c_v, p_v = calls["TtlTradgVol"].sum(), puts["TtlTradgVol"].sum()
        if c_v > 0 and p_v > 0:
            out.at[sym, "pcr_volume"] = math.log(p_v / c_v)

        # SPOT. UDiFF supplies UndrlygPric; the legacy file does not, so
        # the caller passes the equity close instead. Without a spot
        # there is no moneyness, hence no ATM strike, hence no IV, skew
        # or basis - the four features that make this data worth having.
        # Falling back to the future's own price was considered and
        # rejected: basis would become identically zero by construction,
        # a fabricated reading rather than a missing one.
        sym_spot = float("nan")
        if spot is not None and sym in spot:
            sym_spot = float(spot[sym])

        if not futs.empty:
            near = futs.sort_values("XpryDt").iloc[0]
            f_oi = float(near.get("OpnIntrst", np.nan))
            f_doi = float(near.get("ChngInOpnIntrst", np.nan))
            if np.isfinite(f_oi) and f_oi > 0 and np.isfinite(f_doi):
                out.at[sym, "oi_change_norm"] = f_doi / f_oi

            spot_ = float(near.get("UndrlygPric", np.nan))
            if not np.isfinite(spot_):
                spot_ = sym_spot
            fpx = float(near.get("ClsPric", np.nan))
            xp = near.get("XpryDt")
            if (np.isfinite(spot_) and spot_ > 0 and np.isfinite(fpx)
                    and trad_dt is not None and pd.notna(xp)):
                days = max((xp - trad_dt).days, 1)
                out.at[sym, "basis_annualised"] = \
                    (fpx / spot_ - 1.0) * (365.0 / days)

            f_v = futs["TtlTradgVol"].sum()
            o_v = opts["TtlTradgVol"].sum()
            if f_v > 0 and o_v > 0:
                out.at[sym, "opt_volume_ratio"] = math.log(o_v / f_v)

            if not opts.empty and trad_dt is not None:
                expiries = sorted(x for x in opts["XpryDt"].dropna().unique())
                spot_u = spot_ if (np.isfinite(spot_) and spot_ > 0) else float("nan")
                ivs = []
                for xd in expiries[:2]:
                    t = max((pd.Timestamp(xd) - trad_dt).days, 1) / 365.0
                    atm, skew = _atm_iv_and_skew(
                        opts[opts["XpryDt"] == xd], spot_u, t)
                    ivs.append(atm)
                    if len(ivs) == 1:
                        out.at[sym, "atm_iv"] = atm
                        out.at[sym, "iv_skew"] = skew
                if len(ivs) == 2 and np.isfinite(ivs[0]) and np.isfinite(ivs[1]):
                    out.at[sym, "iv_term"] = ivs[1] - ivs[0]
    return out


def fetch_derivative_features(symbols: Sequence[str], start: str,
                              end: Optional[str] = None,
                              use_cache: bool = True,
                              max_workers: int = 6) -> Dict[str, pd.DataFrame]:
    """Daily derivative features per symbol, over a date range.

    Returns {feature_name: DataFrame[date x symbol]} so the result drops
    straight into the same [T, A] layout build_market_data uses for
    price-derived channels.

    SPOT IS SUPPLIED FROM THE EQUITY BHAVCOPY, and it has to be. The
    UDiFF F&O file (2024+) carries UndrlygPric; the legacy file does not.
    Without a spot there is no moneyness, hence no ATM strike, hence no
    atm_iv, iv_skew, iv_term or basis_annualised - so calling
    day_features without one returns FOUR SILENTLY EMPTY CHANNELS for
    every pre-2024 session while the other four look perfectly healthy.

    That is the precise failure this would have caused: the 2019-2023
    backfill exists to extend atm_iv -> vol_5d from ~16 windows to ~30,
    and every one of the new windows would have been NaN. The run would
    have completed, reported no improvement, and the conclusion would
    have been about a missing column rather than about the market.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .nse_prices import fetch_bhav_day

    syms = [str(s).upper().replace(".NS", "") for s in symbols]
    dates = pd.bdate_range(start, end or pd.Timestamp.today().normalize())

    def one(d):
        raw, reason = fetch_fo_raw(d, use_cache=use_cache)
        if raw is None:
            return d, None, None, reason
        eq, _ = fetch_bhav_day(d, syms, use_cache=use_cache)
        spot = (eq.set_index("TckrSymb")["ClsPric"].to_dict()
                if eq is not None else None)
        return d, parse_fo(raw), spot, reason

    rows: Dict[pd.Timestamp, pd.DataFrame] = {}
    stats = {"ok": 0, "cached": 0, "absent": 0, "throttled": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for d, df, spot, reason in ex.map(one, dates):
            stats[reason] = stats.get(reason, 0) + 1
            if df is not None:
                rows[d] = day_features(df, syms, spot=spot)

    if not rows:
        logger.warning("[derivatives] no days fetched: %s", stats)
        return {f: pd.DataFrame() for f in FEATURE_NAMES}

    logger.info("[derivatives] %d/%d sessions (%s)", len(rows), len(dates), stats)

    # Per-channel coverage, logged because the failure above is invisible
    # otherwise: a channel that is empty for a whole epoch has the right
    # shape, the right name and no values.
    for f in ("atm_iv", "basis_annualised"):
        have = sum(1 for d in rows if rows[d][f].notna().any())
        if have < 0.5 * len(rows):
            logger.warning("[derivatives] %s present on only %d/%d sessions "
                           "- check that spot is reaching day_features",
                           f, have, len(rows))
    idx = sorted(rows)
    return {
        f: pd.DataFrame({d: rows[d][f] for d in idx}).T.reindex(columns=syms)
        for f in FEATURE_NAMES
    }
