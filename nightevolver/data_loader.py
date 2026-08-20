"""
NSE data + the 20 classical indicators the genome votes on.

Indicators are implemented directly in numpy/pandas rather than via
TA-Lib. TA-Lib is not installed, needs a C library, and every indicator
here is a few lines of arithmetic - taking a system dependency that can
fail on an unattended Hetzner box, to avoid writing an EMA, is a bad
trade.

TWO NON-NEGOTIABLES, both learned the hard way in this repo:

1. EVERY indicator is causal. Row t is computed from bars <= t and is
   used to predict bar t+1. pandas' `.rolling()` includes the current
   row, which is correct here ONLY because the signal at t is applied to
   the return from t to t+1. The harness enforces that offset, and a
   test mutates future bars to prove no earlier row moves.

2. NORMALISATION IS PER-ASSET AND CAUSAL. Indicators are z-scored
   against their own EXPANDING history, not the full-sample mean. A
   full-sample z-score leaks the future into every row - the single most
   common silent look-ahead in technical-strategy backtests, and it
   inflates results enough to manufacture an edge from nothing.

Output is squashed to roughly [-1, 1] by tanh(z/2) so the genome's
entry/exit thresholds live on a fixed, comparable scale across every
indicator and every asset. (Scale mismatch between a threshold and the
quantity it gates has now caused two separate bugs in this project.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nightevolver.genome import N_FLOW, N_TECHNICAL

logger = logging.getLogger("nightevolver.data")

# Bars discarded at the start so every indicator has a full lookback.
WARMUP_BARS = 60


@dataclass(frozen=True)
class MarketData:
    """Aligned prices, forward returns and normalised indicators.

    indicators: [T, n_assets, N_INDICATORS] causal, in ~[-1, 1]
    forward_returns: [T, n_assets] the return from bar t to t+1 - the
        thing a signal at t is scored against. Last row is NaN-free by
        construction (truncated).
    """

    dates: pd.DatetimeIndex
    tickers: Tuple[str, ...]
    close: np.ndarray                # [T, n_assets]
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray
    indicators: np.ndarray           # [T, n_assets, N_INDICATORS]
    forward_returns: np.ndarray      # [T, n_assets]

    @property
    def n_bars(self) -> int:
        return self.close.shape[0]

    @property
    def n_assets(self) -> int:
        return self.close.shape[1]

    def slice(self, start: int, end: int) -> "MarketData":
        return MarketData(
            dates=self.dates[start:end], tickers=self.tickers,
            close=self.close[start:end], high=self.high[start:end],
            low=self.low[start:end], volume=self.volume[start:end],
            indicators=self.indicators[start:end],
            forward_returns=self.forward_returns[start:end],
        )


# -- individual indicators (all causal) -------------------------------------

def _ema(x: pd.DataFrame, span: int) -> pd.DataFrame:
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.DataFrame, period: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing (alpha = 1/period), the standard RSI definition.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
         period: int = 14) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).stack(),
        (high - prev_close).abs().stack(),
        (low - prev_close).abs().stack(),
    ], axis=1).max(axis=1).unstack()
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _adx_di(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
            period: int = 14) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (adx, di_spread). Wilder's directional movement system."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    atr = _atr(high, low, close, period).replace(0.0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False,
                                min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False,
                                  min_periods=period).mean() / atr

    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di - minus_di


def _causal_zscore(x: pd.DataFrame, min_periods: int = 20) -> pd.DataFrame:
    """Expanding-window z-score - uses only bars up to and including t.

    Expanding rather than rolling so the statistic stabilises over a long
    backtest, and CAUSAL so it cannot leak. A full-sample z-score here
    would be look-ahead, and would silently inflate every result.
    """
    mean = x.expanding(min_periods=min_periods).mean()
    std = x.expanding(min_periods=min_periods).std()
    return (x - mean) / std.replace(0.0, np.nan)


def compute_indicators(close: pd.DataFrame, high: pd.DataFrame,
                       low: pd.DataFrame, volume: pd.DataFrame) -> np.ndarray:
    """-> [T, n_assets, N_INDICATORS], causal, squashed to ~[-1, 1].

    Channel order MUST match genome.INDICATOR_NAMES exactly; a test
    asserts the count and the assembly order is written to mirror it
    line for line.
    """
    raw: List[pd.DataFrame] = []

    # --- oscillators -------------------------------------------------
    raw.append(_rsi(close, 14) - 50.0)                      # rsi_14
    raw.append(_rsi(close, 28) - 50.0)                      # rsi_28

    ema12, ema26 = _ema(close, 12), _ema(close, 26)
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    raw.append((macd - macd_signal) / close)                # macd_hist
    raw.append(np.sign(macd - macd_signal))                 # macd_signal_cross

    # --- bands / trend ----------------------------------------------
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    raw.append((close - sma20) / (2 * std20).replace(0.0, np.nan))   # bb_position
    raw.append((4 * std20) / sma20.replace(0.0, np.nan))             # bb_width

    ema9, ema21, ema50 = _ema(close, 9), _ema(close, 21), _ema(close, 50)
    raw.append((ema9 - ema21) / close)                      # ema_9_21_cross
    raw.append((ema21 - ema50) / close)                     # ema_21_50_cross
    raw.append((close - ema50) / close)                     # price_vs_ema50
    raw.append(sma20.pct_change(5))                         # sma_20_slope

    # --- directional strength ---------------------------------------
    adx, di_spread = _adx_di(high, low, close, 14)
    raw.append(adx - 25.0)                                  # adx_14 (25 = trend/no-trend line)
    raw.append(di_spread)                                   # di_spread

    # --- stochastic --------------------------------------------------
    low14 = low.rolling(14, min_periods=14).min()
    high14 = high.rolling(14, min_periods=14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14).replace(0.0, np.nan)
    stoch_d = stoch_k.rolling(3, min_periods=3).mean()
    raw.append(stoch_k - 50.0)                              # stoch_k
    raw.append(stoch_k - stoch_d)                           # stoch_d_cross

    # --- volatility / volume ----------------------------------------
    raw.append(_atr(high, low, close, 14) / close)          # atr_pct

    # Rolling VWAP. NOTE: this repo has a MEASURED result that a raw
    # volume channel made walk-forward Sharpe worse (-1.51 vs -0.43).
    # VWAP is a different construction (price weighted by volume, not
    # volume as a level), but it is the one volume-derived channel here
    # and is worth ablating if results look volume-driven.
    typical = (high + low + close) / 3.0
    vol_safe = volume.replace(0.0, np.nan)
    vwap = ((typical * vol_safe).rolling(20, min_periods=20).sum()
            / vol_safe.rolling(20, min_periods=20).sum())
    raw.append((close - vwap) / close)                      # vwap_gap

    # --- momentum / dispersion --------------------------------------
    raw.append(close.pct_change(5))                         # mom_5
    raw.append(close.pct_change(21))                        # mom_21
    raw.append(vol_safe / vol_safe.rolling(20, min_periods=20).mean())   # vol_ratio
    rets = close.pct_change()
    raw.append(rets / rets.rolling(20, min_periods=20).std().replace(0.0, np.nan))  # ret_zscore

    if len(raw) != N_TECHNICAL:
        raise RuntimeError(
            f"assembled {len(raw)} indicators but genome expects {N_TECHNICAL} "
            "technical channels - TECHNICAL_INDICATOR_NAMES and "
            "compute_indicators() have drifted apart"
        )

    # Causal z-score then tanh-squash, per asset. Puts every channel on
    # the same [-1, 1] scale the genome's thresholds are defined against.
    channels = []
    for df in raw:
        z = _causal_zscore(df.astype(float))
        channels.append(np.tanh(z.to_numpy(dtype=np.float64) / 2.0))

    out = np.stack(channels, axis=2)                        # [T, A, N]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_market_data(close: pd.DataFrame, high: Optional[pd.DataFrame] = None,
                      low: Optional[pd.DataFrame] = None,
                      volume: Optional[pd.DataFrame] = None,
                      flows: Optional[np.ndarray] = None) -> MarketData:
    """Assemble MarketData from OHLCV frames (high/low/volume optional).

    Missing high/low/volume are synthesised from close so the system can
    still run on close-only history (which is what kronos/backtest.py's
    loader provides). Synthesised inputs are clearly degraded - ATR and
    stochastic become near-degenerate - so this is a fallback, not a
    mode to rely on.
    """
    # PRICE VALIDATION, before anything derives from these numbers.
    #
    # Found by audit. A single Inf or 0.0 in the close panel produced a
    # forward return of 1.8e308 - float max - because the guard at the
    # bottom of this function is `np.nan_to_num(fwd, nan=0.0)`, and
    # nan_to_num maps +inf to FLOAT MAX, not to 0. The indicators looked
    # clean (they are tanh-squashed), so nothing downstream showed a
    # symptom; only the TARGET was poisoned, and one 1.8e308 return
    # dominates any mean, Sharpe or fitness it touches.
    #
    # Negative prices were also accepted silently.
    bad = ~np.isfinite(close.to_numpy(dtype=np.float64)) | \
        (close.to_numpy(dtype=np.float64) <= 0.0)
    if bad.any():
        n_bad = int(bad.sum())
        close = close.mask(pd.DataFrame(bad, index=close.index,
                                        columns=close.columns))
        logger.warning("[data] %d non-finite or non-positive close price(s) "
                       "masked to NaN before any derived quantity is computed",
                       n_bad)

    close = close.dropna(how="any")
    if close.empty or close.shape[1] == 0:
        raise ValueError(
            "no usable price rows after dropping NaNs. An all-NaN column "
            "silently emptied the whole panel here before this check "
            "existed, returning a MarketData with 0 bars rather than "
            "failing - so downstream code computed statistics on empty "
            "arrays. Check per-symbol coverage before calling this.")
    idx = close.index
    if high is None:
        high = close
    if low is None:
        low = close
    if volume is None:
        volume = pd.DataFrame(1.0, index=idx, columns=close.columns)
    high = high.reindex(idx).ffill().fillna(close)
    low = low.reindex(idx).ffill().fillna(close)
    volume = volume.reindex(idx).ffill().fillna(1.0)

    indicators = compute_indicators(close, high, low, volume)

    # Append the market-wide flow channels. When flows are unavailable
    # these are zeros, which cast a zero vote and are therefore inert -
    # the genome layout stays fixed either way, so a genome never
    # decodes against a different channel ordering than it evolved on.
    if flows is None:
        flow_block = np.zeros((len(idx), len(close.columns), N_FLOW))
    else:
        flows = np.asarray(flows, dtype=np.float64)
        if flows.shape != (len(idx), N_FLOW):
            raise ValueError(
                f"flows must be [{len(idx)}, {N_FLOW}] (one row per bar, "
                f"market-wide); got {flows.shape}")
        flow_block = np.repeat(flows[:, None, :], len(close.columns), axis=1)
    indicators = np.concatenate([indicators, flow_block], axis=2)

    c = close.to_numpy(dtype=np.float64)
    # forward_returns[t] = return from t to t+1. A signal computed at t
    # is scored against this, which is the only alignment that is not
    # look-ahead. The final bar has no forward return, so drop it.
    fwd = np.full_like(c, np.nan)
    fwd[:-1] = c[1:] / c[:-1] - 1.0

    keep = slice(WARMUP_BARS, len(idx) - 1)
    return MarketData(
        dates=idx[keep], tickers=tuple(close.columns),
        close=c[keep], high=high.to_numpy(dtype=np.float64)[keep],
        low=low.to_numpy(dtype=np.float64)[keep],
        volume=volume.to_numpy(dtype=np.float64)[keep],
        indicators=indicators[keep],
        # posinf/neginf are set explicitly. The default nan_to_num maps
        # inf to FLOAT MAX (1.8e308), which is not a sane return and
        # would swamp every downstream statistic - see the validation
        # note at the top of this function.
        forward_returns=np.nan_to_num(fwd[keep], nan=0.0,
                                      posinf=0.0, neginf=0.0),
    )


def fetch_nse_data(tickers: Sequence[str], start: str,
                   end: Optional[str] = None) -> MarketData:
    """Fetch NSE OHLCV via yfinance and build MarketData."""
    import yfinance as yf

    logger.info("[nightevolver] fetching %d tickers from %s", len(tickers), start)
    data = yf.download(list(tickers), start=start, end=end, interval="1d",
                       progress=False, auto_adjust=True, group_by="column")
    if data is None or data.empty:
        raise RuntimeError("yfinance returned no data")

    def field(name: str) -> Optional[pd.DataFrame]:
        if name not in data:
            return None
        f = data[name]
        return f.to_frame(tickers[0]) if isinstance(f, pd.Series) else f

    close = field("Close")
    if close is None:
        raise RuntimeError("no Close column in yfinance response")
    return build_market_data(close.dropna(how="all"), field("High"),
                             field("Low"), field("Volume"))
