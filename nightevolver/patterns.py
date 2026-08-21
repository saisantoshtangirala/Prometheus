"""
Classical TA and microstructure estimators, as testable channels.

WHAT THIS IS FOR. The existing 26 channels are momentum, mean-reversion
and trend transforms plus six flow series. Absent entirely: candlestick
pattern recognition, Ichimoku/Keltner/Donchian channel systems, and the
OHLC-based volatility and liquidity estimators from the microstructure
literature. Those are the standard content of every technical-analysis
and quant text, and they had never been implemented here, so "we tried
technical analysis and found nothing" was not quite true - a large and
well-known part of it had not been tried.

This module makes that claim testable rather than assumed.

WHAT TO EXPECT, stated before the measurement rather than after.

Groups 1 and 2 - candlesticks and channel systems - are DETERMINISTIC
FUNCTIONS OF DAILY OHLCV, the same input already measured across 45
channels, 16,800 GA trials and a 30-draw null cloud with no directional
edge found. A new function of an input that carries no information about
tomorrow's direction cannot manufacture information; it can only add
surface for a search to overfit. The GA already demonstrated that
appetite by naming 15 different "best" indicators across 16 windows. So
the prior on these is low, and they are here to be REFUTED cheaply
rather than because they are expected to work.

Group 3 is different and is the reason this module is worth writing.

  VOLATILITY ESTIMATORS. Garman-Klass, Parkinson and Rogers-Satchell use
  the high and low, not just the close. A close-to-close estimate throws
  away the entire intraday range, and these are 5-8x more EFFICIENT -
  the same accuracy from a fraction of the observations. That matters
  here specifically: the one near-miss in this project is atm_iv against
  vol_5d at p=0.065, where the binding constraint is statistical power.
  A better-measured target is a direct attack on that constraint, and
  unlike everything else in this file it has a mechanism for helping
  that is not "maybe this pattern works".

  LIQUIDITY ESTIMATORS. Amihud illiquidity, Roll's effective spread and
  a Kyle's-lambda proxy estimate PRICE IMPACT - how far the price moves
  per unit of volume. That is a property of the order book inferred from
  daily bars, which is closer to genuinely new information than another
  moving-average crossover, though far weaker than the real tape.

Everything is CAUSAL: every value at bar t uses only bars <= t. The
pattern functions read the current and immediately preceding bars, which
are both known at t's close.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("nightevolver.patterns")

CANDLESTICK_NAMES = (
    "doji", "hammer", "shooting_star", "marubozu", "engulfing", "harami",
    "morning_star", "evening_star", "piercing", "dark_cloud",
    "three_soldiers", "three_crows",
)

CHANNEL_NAMES = (
    "ichimoku_tk_cross", "ichimoku_cloud_pos", "keltner_pos",
    "donchian_pos", "pivot_dist",
)

MICROSTRUCTURE_NAMES = (
    "gk_vol", "parkinson_vol", "rogers_satchell", "vol_of_vol",
    "amihud_illiq", "roll_spread", "kyle_lambda",
    "close_loc_value", "gap_open", "range_expansion",
)

FEATURE_NAMES = CANDLESTICK_NAMES + CHANNEL_NAMES + MICROSTRUCTURE_NAMES

_EPS = 1e-12


def _safe(x: np.ndarray) -> np.ndarray:
    """Finite, or NaN. Never inf.

    np.nan_to_num maps +inf to 1.8e308 rather than to 0, and this
    codebase has already been bitten by that once: a single zero price
    produced a forward return of FLOAT MAX which the tanh-squashed
    indicators hid completely, so only the TARGET was poisoned.
    """
    return np.where(np.isfinite(x), x, np.nan)


# ---------------------------------------------------------------------
# 1. Candlestick patterns
# ---------------------------------------------------------------------

def candlestick_features(open_: np.ndarray, high: np.ndarray,
                         low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
    """Twelve classical formations, as CONTINUOUS strengths in [-1, 1].

    Not booleans. A pattern library that emits 0/1 throws away how
    decisively the shape was met, and a rank statistic on a mostly-zero
    binary column has almost no resolution - which would make a real
    effect undetectable for a reason that has nothing to do with the
    market. Sign carries direction where the pattern has one: positive
    is bullish.
    """
    o, h, l, c = (np.asarray(x, dtype=float) for x in (open_, high, low, close))
    rng = np.maximum(h - l, _EPS)
    body = c - o
    abs_body = np.abs(body)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    prev = lambda a: np.vstack([np.full((1,) + a.shape[1:], np.nan), a[:-1]])
    po, pc, pbody = prev(o), prev(c), prev(body)
    pabs = np.abs(pbody)
    p2c, p2o = prev(pc), prev(po)

    out: Dict[str, np.ndarray] = {}

    # Indecision: body tiny relative to the day's range.
    out["doji"] = _safe(1.0 - np.minimum(abs_body / rng, 1.0))

    # HAMMER: a long LOWER shadow with the body near the top - the low
    # was rejected. SHOOTING STAR: a long UPPER shadow with the body near
    # the bottom - the high was rejected.
    #
    # These must be defined from their OWN shadow, not as mirrors of each
    # other. The first version used clip((lower-upper)/rng) and
    # -clip((upper-lower)/rng); since clip is symmetric, -clip(-x) ==
    # clip(x) and the two channels were ALGEBRAICALLY IDENTICAL -
    # measured correlation +1.000000, np.allclose True. Two identical
    # columns add a duplicate for the search to pick between while
    # carrying one channel's worth of information.
    small_body = 1.0 - np.minimum(abs_body / rng, 1.0)
    out["hammer"] = _safe(np.clip(lower / rng, 0.0, 1.0)
                          * np.clip(1.0 - 2.0 * upper / rng, 0.0, 1.0)
                          * small_body)
    out["shooting_star"] = _safe(-np.clip(upper / rng, 0.0, 1.0)
                                 * np.clip(1.0 - 2.0 * lower / rng, 0.0, 1.0)
                                 * small_body)

    # Body fills the range: conviction, signed by direction.
    out["marubozu"] = _safe(np.sign(body) * np.minimum(abs_body / rng, 1.0))

    # Today's body swallows yesterday's, opposite colour.
    engulf = (abs_body / np.maximum(pabs, _EPS)) - 1.0
    opposite = np.sign(body) != np.sign(pbody)
    out["engulfing"] = _safe(np.where(opposite,
                                      np.sign(body) * np.clip(engulf, 0.0, 1.0), 0.0))

    # Today's body inside yesterday's: contraction after a move.
    out["harami"] = _safe(np.where(opposite,
                                   -np.sign(pbody) * np.clip(1.0 - abs_body /
                                                             np.maximum(pabs, _EPS),
                                                             0.0, 1.0), 0.0))

    # Three-bar reversals: down, indecision, up (and the mirror).
    small_mid = (pabs / np.maximum(prev(rng), _EPS)) < 0.3
    d2 = np.sign(prev(pbody))
    out["morning_star"] = _safe(np.where(small_mid & (d2 < 0) & (body > 0),
                                         np.minimum(abs_body / rng, 1.0), 0.0))
    out["evening_star"] = _safe(np.where(small_mid & (d2 > 0) & (body < 0),
                                         -np.minimum(abs_body / rng, 1.0), 0.0))

    # Close back through the midpoint of the prior opposite body.
    pmid = (po + pc) / 2.0
    out["piercing"] = _safe(np.where((pbody < 0) & (body > 0) & (c > pmid),
                                     np.clip((c - pmid) / np.maximum(pabs, _EPS),
                                             0.0, 1.0), 0.0))
    out["dark_cloud"] = _safe(np.where((pbody > 0) & (body < 0) & (c < pmid),
                                       -np.clip((pmid - c) / np.maximum(pabs, _EPS),
                                                0.0, 1.0), 0.0))

    # Three consecutive same-direction closes, strength = mean body.
    up3 = (c > o) & (pc > po) & (p2c > p2o)
    dn3 = (c < o) & (pc < po) & (p2c < p2o)
    mean_body = (abs_body + pabs + np.abs(p2c - p2o)) / (3.0 * rng)
    out["three_soldiers"] = _safe(np.where(up3, np.minimum(mean_body, 1.0), 0.0))
    out["three_crows"] = _safe(np.where(dn3, -np.minimum(mean_body, 1.0), 0.0))
    return out


# ---------------------------------------------------------------------
# 2. Channel systems
# ---------------------------------------------------------------------

def _roll(a: np.ndarray, n: int, fn) -> np.ndarray:
    df = pd.DataFrame(a)
    return getattr(df.rolling(n, min_periods=max(2, n // 2)), fn)().to_numpy()


def channel_features(high: np.ndarray, low: np.ndarray,
                     close: np.ndarray) -> Dict[str, np.ndarray]:
    """Ichimoku, Keltner, Donchian and pivot distance.

    Ichimoku's forward-shifted spans (senkou A/B, normally plotted 26
    bars AHEAD) are used UNSHIFTED here. Plotting them forward is a
    charting convention; shifting them into the feature matrix would
    place a value derived from bar t at bar t+26, which reads as a
    prediction and is in fact a look-ahead in the other direction. The
    cloud position below compares today's close to spans computed from
    data up to today.
    """
    h, l, c = (np.asarray(x, dtype=float) for x in (high, low, close))
    out: Dict[str, np.ndarray] = {}

    def hh(n): return _roll(h, n, "max")
    def ll(n): return _roll(l, n, "min")

    tenkan = (hh(9) + ll(9)) / 2.0
    kijun = (hh(26) + ll(26)) / 2.0
    senkou_a = (tenkan + kijun) / 2.0
    senkou_b = (hh(52) + ll(52)) / 2.0

    out["ichimoku_tk_cross"] = _safe(np.tanh((tenkan - kijun) /
                                             np.maximum(np.abs(kijun), _EPS) * 50.0))
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bot = np.minimum(senkou_a, senkou_b)
    width = np.maximum(cloud_top - cloud_bot, _EPS)
    out["ichimoku_cloud_pos"] = _safe(np.clip((c - cloud_bot) / width, -2.0, 3.0))

    ema20 = pd.DataFrame(c).ewm(span=20, adjust=False).mean().to_numpy()
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.vstack([np.full((1,) + c.shape[1:], np.nan), c[:-1]])),
        np.abs(l - np.vstack([np.full((1,) + c.shape[1:], np.nan), c[:-1]]))))
    atr = pd.DataFrame(tr).ewm(span=20, adjust=False).mean().to_numpy()
    out["keltner_pos"] = _safe(np.clip((c - ema20) / np.maximum(2.0 * atr, _EPS),
                                       -3.0, 3.0))

    dc_hi, dc_lo = hh(20), ll(20)
    out["donchian_pos"] = _safe(np.clip((c - dc_lo) /
                                        np.maximum(dc_hi - dc_lo, _EPS), 0.0, 1.0))

    pivot = (h + l + c) / 3.0
    out["pivot_dist"] = _safe(np.tanh((c - pivot) /
                                      np.maximum(np.abs(pivot), _EPS) * 100.0))
    return out


# ---------------------------------------------------------------------
# 3. Microstructure and efficient volatility estimators
# ---------------------------------------------------------------------

def microstructure_features(open_: np.ndarray, high: np.ndarray,
                            low: np.ndarray, close: np.ndarray,
                            volume: Optional[np.ndarray] = None
                            ) -> Dict[str, np.ndarray]:
    """OHLC volatility estimators and daily-bar liquidity proxies.

    THE VOLATILITY ESTIMATORS ARE THE POINT. A close-to-close estimate
    discards the entire intraday range; Garman-Klass is roughly 7x more
    efficient, Parkinson ~5x. "Efficient" here means the same precision
    from a fraction of the observations, which is a direct attack on the
    binding constraint of the one near-miss this project has (atm_iv ->
    vol_5d at p=0.065, limited by power, not by effect size).

    Rogers-Satchell is included because, unlike the other two, it is
    unbiased in the presence of DRIFT - and a trending name is exactly
    where the others overstate volatility.
    """
    o, h, l, c = (np.asarray(x, dtype=float) for x in (open_, high, low, close))
    out: Dict[str, np.ndarray] = {}

    log_hl = np.log(np.maximum(h, _EPS) / np.maximum(l, _EPS))
    log_co = np.log(np.maximum(c, _EPS) / np.maximum(o, _EPS))

    gk = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    out["gk_vol"] = _safe(np.sqrt(np.maximum(gk, 0.0) * 252.0))
    out["parkinson_vol"] = _safe(np.sqrt(
        np.maximum(log_hl ** 2 / (4.0 * np.log(2.0)), 0.0) * 252.0))

    # RS = log(H/C)*log(H/O) + log(L/C)*log(L/O). The first version wrote
    # log(H/O)*log(C/H) + log(L/O)*log(C/L), which is the exact NEGATIVE
    # of this - and since RS is non-negative by construction, the
    # subsequent max(rs, 0) clamped every value to zero. The channel was
    # identically 0.000 with sd 0.000 across the whole panel: present,
    # named, and carrying nothing.
    log_hc = np.log(np.maximum(h, _EPS) / np.maximum(c, _EPS))
    log_ho = np.log(np.maximum(h, _EPS) / np.maximum(o, _EPS))
    log_lc = np.log(np.maximum(l, _EPS) / np.maximum(c, _EPS))
    log_lo = np.log(np.maximum(l, _EPS) / np.maximum(o, _EPS))
    rs = log_hc * log_ho + log_lc * log_lo
    out["rogers_satchell"] = _safe(np.sqrt(np.maximum(rs, 0.0) * 252.0))

    # Volatility of volatility - regime instability, not level.
    out["vol_of_vol"] = _safe(_roll(out["gk_vol"], 20, "std"))

    prev_c = np.vstack([np.full((1,) + c.shape[1:], np.nan), c[:-1]])
    ret = c / np.maximum(prev_c, _EPS) - 1.0

    if volume is not None:
        v = np.asarray(volume, dtype=float)
        turnover = np.maximum(v * c, _EPS)
        # Amihud: |return| per unit of turnover - price impact per rupee.
        out["amihud_illiq"] = _safe(np.log1p(
            np.abs(ret) / turnover * 1e9))
        # Kyle's lambda proxy: regression slope of |return| on volume,
        # approximated per bar and smoothed. A real lambda needs signed
        # order flow, which daily bars do not carry - this is the
        # available shadow of it, and is labelled as a proxy for that
        # reason.
        # Log-scaled: the raw ratio is ~1e-5 with a standard deviation
        # that rounds to zero at display precision, which is not wrong
        # but leaves a rank statistic almost no resolution to work with.
        out["kyle_lambda"] = _safe(np.log1p(_roll(
            np.abs(ret) / np.maximum(np.sqrt(np.maximum(v, _EPS)), _EPS),
            20, "mean") * 1e6))
    else:
        nanl = np.full_like(c, np.nan)
        out["amihud_illiq"], out["kyle_lambda"] = nanl, nanl.copy()

    # Roll's effective spread: 2*sqrt(-cov(r_t, r_{t-1})) when the serial
    # covariance is negative, which is the bid-ask bounce. A POSITIVE
    # covariance means the estimator does not apply - that is trending,
    # not spread - and yields NaN rather than a fabricated zero.
    prev_ret = np.vstack([np.full((1,) + c.shape[1:], np.nan), ret[:-1]])
    cov = _roll(ret * prev_ret, 20, "mean")
    out["roll_spread"] = _safe(np.where(cov < 0, 2.0 * np.sqrt(np.maximum(-cov, 0.0)),
                                        np.nan))

    # Where in the day's range did it close - the "close location value".
    out["close_loc_value"] = _safe(np.clip(
        ((c - l) - (h - c)) / np.maximum(h - l, _EPS), -1.0, 1.0))

    out["gap_open"] = _safe(np.tanh(
        (o - prev_c) / np.maximum(np.abs(prev_c), _EPS) * 50.0))

    rng = h - l
    out["range_expansion"] = _safe(np.log1p(
        np.maximum(rng, _EPS) / np.maximum(_roll(rng, 20, "mean"), _EPS)))
    return out


def build_pattern_features(close: np.ndarray, high: np.ndarray,
                           low: np.ndarray,
                           volume: Optional[np.ndarray] = None,
                           open_: Optional[np.ndarray] = None
                           ) -> Dict[str, np.ndarray]:
    """All three groups as {name: [T, A]}.

    `open_` is synthesised from the previous close when absent. That is a
    real degradation and it is why it is stated here: every candlestick
    body becomes the close-to-close move, so doji and marubozu collapse
    toward measuring the same thing, and gap_open becomes identically
    zero. The bhavcopy carries a true open, so pass it - the fallback
    exists so the module runs on close-only history, not because the
    result is equivalent.
    """
    c = np.asarray(close, dtype=float)
    if open_ is None:
        open_ = np.vstack([c[:1], c[:-1]])
        logger.warning("[patterns] no OPEN supplied - synthesising from the "
                       "previous close. Candlestick bodies degenerate to "
                       "close-to-close moves and gap_open becomes zero.")
    feats: Dict[str, np.ndarray] = {}
    feats.update(candlestick_features(open_, high, low, c))
    feats.update(channel_features(high, low, c))
    feats.update(microstructure_features(open_, high, low, c, volume))
    return feats
