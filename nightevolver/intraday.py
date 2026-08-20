"""
Intraday structure as daily-frequency features.

THE DESIGN CHOICE THAT MATTERS
------------------------------
Minute bars could be used to build a minute-frequency strategy. They are
not used that way here. Instead each trading day's minute bars are
collapsed into a handful of numbers describing the SHAPE of that day -
where the volume sat, how the day trended against its own VWAP, how much
of the range came in the first thirty minutes - and those numbers become
daily features.

Two reasons, and the second is the important one:

1. They drop straight into the existing audit, targets and FDR
   correction. No parallel stack to keep in sync, and the result is
   directly comparable with the daily-indicator numbers already
   measured.

2. **It is the cheap decisive test.** The open question is whether
   intraday data carries information that daily OHLCV does not. That
   question is answered by 26 daily features vs these, on the same
   targets, with the same correction - not by building a minute-level
   trading system first and discovering the answer six weeks later.
   If day-shape carries nothing about tomorrow, a finer-grained strategy
   over the same information will not save it.

If something here does survive correction, THEN a minute-frequency
strategy is worth building, and this module has told you which channels
to build it on.

CAUSALITY. Feature at day t is computed from day t's minute bars only -
all of them, including the close - and is scored against the return from
t to t+1. That is the same convention the daily indicators already use
(`data_loader` computes row t from bars <= t). A feature that used any
part of day t+1 would be look-ahead, and there is a test for it.

ORDER FLOW is separate and comes from `depth_recorder` output, not from
candles. Those features are marked in ORDERFLOW_FEATURE_NAMES and will
be unavailable until the recorder has been running for a while - which
is the whole point of starting it early.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("nightevolver.intraday")

IST_OPEN_MINUTES = 9 * 60 + 15
IST_CLOSE_MINUTES = 15 * 60 + 30

INTRADAY_FEATURE_NAMES: Tuple[str, ...] = (
    "overnight_gap",
    "open_range_pct",
    "first_hour_ret",
    "last_hour_ret",
    "intraday_rvol",
    "close_vs_vwap",
    "vwap_dispersion",
    "volume_concentration",
    "close_volume_share",
    "intraday_ret_skew",
    "signed_volume_frac",
    "path_efficiency",
)
N_INTRADAY_FEATURES = len(INTRADAY_FEATURE_NAMES)

ORDERFLOW_FEATURE_NAMES: Tuple[str, ...] = (
    "depth_imbalance_mean",
    "depth_imbalance_close",
    "spread_mean_bps",
    "book_pressure_slope",
)
N_ORDERFLOW_FEATURES = len(ORDERFLOW_FEATURE_NAMES)


def _minutes_since_midnight(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.hour.to_numpy() * 60 + idx.minute.to_numpy()


def day_shape_features(bars: pd.DataFrame,
                       prev_close: Optional[float] = None) -> Dict[str, float]:
    """Collapse one day's minute bars into the day-shape features.

    `bars` must be that single day's minute candles, indexed by
    timestamp, with columns open/high/low/close/volume.
    """
    if bars.empty:
        return {name: np.nan for name in INTRADAY_FEATURE_NAMES}

    bars = bars.sort_index()
    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    v = bars["volume"].to_numpy(dtype=float)
    mins = _minutes_since_midnight(bars.index)

    day_open, day_close = o[0], c[-1]
    out: Dict[str, float] = {}

    out["overnight_gap"] = (day_open / prev_close - 1.0) if prev_close else np.nan

    first30 = mins < IST_OPEN_MINUTES + 30
    if first30.any() and day_close > 0:
        out["open_range_pct"] = (h[first30].max() - lo[first30].min()) / day_close
    else:
        out["open_range_pct"] = np.nan

    first60 = mins < IST_OPEN_MINUTES + 60
    out["first_hour_ret"] = (c[first60][-1] / day_open - 1.0) \
        if first60.any() and day_open > 0 else np.nan

    last60 = mins >= IST_CLOSE_MINUTES - 60
    out["last_hour_ret"] = (day_close / o[last60][0] - 1.0) \
        if last60.any() and o[last60][0] > 0 else np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.diff(c) / c[:-1]
    rets = rets[np.isfinite(rets)]
    out["intraday_rvol"] = float(rets.std(ddof=1)) if rets.size > 2 else np.nan

    typical = (h + lo + c) / 3.0
    vsum = v.sum()
    vwap = float((typical * v).sum() / vsum) if vsum > 0 else np.nan
    out["close_vs_vwap"] = (day_close / vwap - 1.0) if vwap and vwap > 0 else np.nan
    out["vwap_dispersion"] = float(np.std(typical / vwap - 1.0)) \
        if vwap and vwap > 0 else np.nan

    # Herfindahl of per-minute volume: 1/n_bars means perfectly even
    # trading, higher means the day's volume arrived in a few bursts.
    if vsum > 0:
        share = v / vsum
        out["volume_concentration"] = float((share ** 2).sum())
        close15 = mins >= IST_CLOSE_MINUTES - 15
        out["close_volume_share"] = float(v[close15].sum() / vsum) \
            if close15.any() else np.nan
    else:
        out["volume_concentration"] = np.nan
        out["close_volume_share"] = np.nan

    if rets.size > 3 and rets.std(ddof=1) > 1e-12:
        z = (rets - rets.mean()) / rets.std(ddof=1)
        out["intraday_ret_skew"] = float((z ** 3).mean())
    else:
        out["intraday_ret_skew"] = np.nan

    # Tick-rule signed volume from minute bars. A genuine proxy, not the
    # real thing: true signed volume needs trade-level data against the
    # quote. It is a stand-in until the depth recorder has history.
    if v.size > 1 and vsum > 0:
        sign = np.sign(np.diff(c))
        out["signed_volume_frac"] = float((sign * v[1:]).sum() / vsum)
    else:
        out["signed_volume_frac"] = np.nan

    # Path efficiency: net move divided by total distance travelled.
    # Near 1 = a clean trend, near 0 = a day that went nowhere loudly.
    travelled = float(np.abs(np.diff(c)).sum())
    out["path_efficiency"] = (abs(day_close - day_open) / travelled) \
        if travelled > 1e-12 else np.nan

    return out


def minutes_to_daily_features(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """All minute bars for ONE symbol -> daily feature frame.

    Index is the trading date; columns are INTRADAY_FEATURE_NAMES.
    """
    if minute_bars.empty:
        return pd.DataFrame(columns=list(INTRADAY_FEATURE_NAMES))

    bars = minute_bars.sort_index()
    dates = pd.DatetimeIndex(bars.index).normalize()
    rows: List[Dict[str, float]] = []
    index: List[pd.Timestamp] = []
    prev_close: Optional[float] = None

    for day, chunk in bars.groupby(dates):
        feats = day_shape_features(chunk, prev_close=prev_close)
        rows.append(feats)
        index.append(pd.Timestamp(day))
        prev_close = float(chunk["close"].iloc[-1])

    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))[
        list(INTRADAY_FEATURE_NAMES)]


def depth_file_to_daily_features(records: Iterable[Dict],
                                 token_to_symbol: Dict[int, str],
                                 ) -> Dict[str, Dict[str, float]]:
    """One recorded session -> per-symbol order-flow features.

    Consumes `depth_recorder.read_depth_file` output. Returns
    {symbol: {feature: value}} for the session the records came from.
    """
    acc: Dict[int, Dict[str, List[float]]] = {}
    for r in records:
        tok = r.get("tk")
        sym = token_to_symbol.get(tok)
        if sym is None:
            continue
        bids, asks = r.get("b") or [], r.get("a") or []
        if not bids or not asks:
            continue
        bq = float(sum(e[0] for e in bids))
        aq = float(sum(e[0] for e in asks))
        tot = bq + aq
        if tot <= 0:
            continue
        best_bid, best_ask = float(bids[0][1]), float(asks[0][1])
        mid = 0.5 * (best_bid + best_ask)
        slot = acc.setdefault(tok, {"imb": [], "spread": [], "slope": []})
        slot["imb"].append((bq - aq) / tot)
        if mid > 0:
            slot["spread"].append(1e4 * (best_ask - best_bid) / mid)
        # Book pressure slope: near-touch imbalance minus far imbalance.
        # Positive means the pressure sits close to the touch, which is
        # the part that actually moves price.
        near = float(bids[0][0]) - float(asks[0][0])
        far = (float(sum(e[0] for e in bids[1:])) -
               float(sum(e[0] for e in asks[1:])))
        denom = abs(near) + abs(far)
        if denom > 0:
            slot["slope"].append((near - far) / denom)

    out: Dict[str, Dict[str, float]] = {}
    for tok, slot in acc.items():
        sym = token_to_symbol[tok]
        imb = slot["imb"]
        out[sym] = {
            "depth_imbalance_mean": float(np.mean(imb)) if imb else np.nan,
            # Last 2% of the session's snapshots, i.e. near the close.
            "depth_imbalance_close": float(np.mean(imb[-max(1, len(imb) // 50):]))
            if imb else np.nan,
            "spread_mean_bps": float(np.mean(slot["spread"])) if slot["spread"] else np.nan,
            "book_pressure_slope": float(np.mean(slot["slope"])) if slot["slope"] else np.nan,
        }
    return out


def _causal_zscore(s: pd.Series, min_periods: int = 60) -> pd.Series:
    mean = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std().replace(0.0, np.nan)
    return (s - mean) / std


def normalise_features(per_symbol: Dict[str, pd.DataFrame],
                       symbols: Sequence[str],
                       dates: pd.DatetimeIndex,
                       feature_names: Sequence[str]) -> np.ndarray:
    """{symbol: daily feature frame} -> [T, A, F], causally normalised.

    Same treatment as the technical indicators: expanding-window z-score
    per (symbol, channel) then tanh-squash to ~[-1, 1], so every channel
    sits on the scale the genome's thresholds are defined against and
    the audit compares like with like.
    """
    T, A, F = len(dates), len(symbols), len(feature_names)
    out = np.zeros((T, A, F))
    for a, sym in enumerate(symbols):
        df = per_symbol.get(sym)
        if df is None or df.empty:
            continue
        df = df.reindex(dates)
        for f, name in enumerate(feature_names):
            if name not in df.columns:
                continue
            z = _causal_zscore(df[name].astype(float))
            out[:, a, f] = np.tanh(z.to_numpy(dtype=np.float64) / 2.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def load_orderflow_features(depth_dir: Path, token_to_symbol: Dict[int, str],
                            symbols: Sequence[str],
                            ) -> Dict[str, pd.DataFrame]:
    """Read every recorded session in `depth_dir` into per-symbol frames.

    Returns {} when nothing has been recorded yet, which is the expected
    state until the recorder has been running for a while. The caller
    should say so plainly rather than reporting a null result on
    order flow as if it were a measurement.
    """
    from nightevolver.depth_recorder import read_depth_file

    depth_dir = Path(depth_dir)
    files = sorted(depth_dir.glob("depth_*.jsonl.gz"))
    if not files:
        return {}

    rows: Dict[str, Dict[pd.Timestamp, Dict[str, float]]] = {s: {} for s in symbols}
    for path in files:
        try:
            day = pd.Timestamp(path.stem.split("_")[1][:8])
        except (IndexError, ValueError):
            logger.warning("[intraday] cannot parse a date from %s - skipping", path)
            continue
        per_sym = depth_file_to_daily_features(read_depth_file(path), token_to_symbol)
        for sym, feats in per_sym.items():
            if sym in rows:
                rows[sym][day] = feats

    out: Dict[str, pd.DataFrame] = {}
    for sym, by_day in rows.items():
        if by_day:
            out[sym] = pd.DataFrame.from_dict(by_day, orient="index")[
                list(ORDERFLOW_FEATURE_NAMES)].sort_index()
    logger.info("[intraday] order-flow features from %d session files, "
                "%d symbols with data", len(files), len(out))
    return out
