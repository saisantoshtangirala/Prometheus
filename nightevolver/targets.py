"""
Alternative prediction targets.

The premise being tested here is that direction is the hardest thing on
the menu and we have been insisting on it. Volatility clusters;
cross-sectional dispersion is more stable than the level; regimes
persist. So this module builds four targets from the same price history
and lets the information audit measure which, if any, is predictable
from features we can actually compute.

A WARNING THAT BELONGS AT THE TOP, because the audit will almost
certainly report a large number for volatility and it must not be
misread:

    PREDICTING VOLATILITY IS NOT THE SAME AS MAKING MONEY.

Realised volatility is strongly autocorrelated - tomorrow's vol is close
to today's vol - so *any* vol-ish feature will score well against a vol
target. That is a real statistical property (it is why GARCH works), but
it is nearly free information: a constant "tomorrow looks like today"
rule captures most of it. To convert a vol forecast into P&L you need
either an instrument whose price is a function of vol (options, which
this system does not trade) or a sizing rule whose edge comes from
somewhere else.

So `vol_5d` scoring highly is the EXPECTED result and is not evidence of
an edge. The audit therefore reports every target against a persistence
baseline (see information_audit.PERSISTENCE_BASELINE) - the question is
never "is it predictable" but "is it predictable beyond trivially
extrapolating its own past".

Directional and cross-sectional targets do not have this escape hatch:
they are close to unautocorrelated, so a high score there would be a
genuine finding.

CAUSALITY. Targets are forward-looking by definition - that is what makes
them targets. The contract is:

    target[t] is a function of bars STRICTLY AFTER t.
    features[t] is a function of bars <= t.

`valid[t]` is False wherever there are not enough future bars, and the
audit drops those rows rather than filling them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger("nightevolver.targets")

TARGET_NAMES: Tuple[str, ...] = (
    "direction_1d",
    "rel_strength_1d",
    "vol_5d",
    "regime_shift_5d",
)


@dataclass(frozen=True)
class Target:
    """A forward-looking quantity to be predicted.

    values: [T, A] the target
    valid:  [T, A] bool - False where the target is undefined (not
            enough future bars). Never fill these; drop them.
    kind:   "continuous" or "signed" - signed targets are ones where
            getting the sign right is the economically meaningful part.
    autocorr_baseline: whether this target is largely predictable from
            its own recent history. True means a high score is cheap.
    """

    name: str
    values: np.ndarray
    valid: np.ndarray
    kind: str
    autocorr_baseline: bool

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())


def _forward_returns_matrix(close: np.ndarray) -> np.ndarray:
    """[T, A] simple return from t to t+1; last row undefined (NaN)."""
    fwd = np.full_like(close, np.nan, dtype=np.float64)
    fwd[:-1] = close[1:] / close[:-1] - 1.0
    return fwd


def direction_1d(close: np.ndarray) -> Target:
    """Next-bar return. The target this project has always used, kept
    here as the control that every other target is compared against."""
    fwd = _forward_returns_matrix(close)
    return Target("direction_1d", np.nan_to_num(fwd), np.isfinite(fwd),
                  kind="signed", autocorr_baseline=False)


def rel_strength_1d(close: np.ndarray) -> Target:
    """Next-bar return MINUS the cross-sectional mean next-bar return.

    This is the 'which asset beats the basket' target. It is worth
    testing separately from direction because it strips out the market
    factor, which is the dominant and least predictable component of a
    single stock's daily move. A strategy that cannot time the market
    may still rank names within it - and rank information is tradeable
    long/short, or as an overweight within a long-only book.

    Requires >= 2 assets to mean anything; with one asset the
    cross-sectional mean IS the asset and the target is identically 0.
    """
    fwd = _forward_returns_matrix(close)
    valid = np.isfinite(fwd)
    if close.shape[1] < 2:
        logger.warning("[targets] rel_strength_1d needs >=2 assets; got %d - "
                       "target is identically zero and carries no information",
                       close.shape[1])
        return Target("rel_strength_1d", np.zeros_like(fwd), valid,
                      kind="signed", autocorr_baseline=False)
    # Rows with no valid asset (the final bar) would make nanmean warn
    # and return NaN; compute the mean only where something is valid.
    masked = np.where(valid, fwd, np.nan)
    counts = valid.sum(axis=1, keepdims=True)
    sums = np.nansum(masked, axis=1, keepdims=True)
    xs_mean = np.divide(sums, counts, out=np.zeros_like(sums),
                        where=counts > 0)
    rel = fwd - xs_mean
    return Target("rel_strength_1d", np.nan_to_num(rel), valid,
                  kind="signed", autocorr_baseline=False)


def vol_5d(close: np.ndarray, horizon: int = 5) -> Target:
    """Realised volatility over the NEXT `horizon` bars.

    std of daily returns from t+1 to t+horizon. Undefined for the last
    `horizon` rows.

    Expect this to be highly predictable and read the module docstring
    before celebrating: vol is autocorrelated, so this is the cheap one.

    VALIDITY IS PER ASSET, NOT PER DATE. `np.isfinite(window).all()`
    tests the whole [horizon, A] slice, so on a ragged panel ONE name
    with a gap invalidates that date for every other name. Measured
    after delisted names stopped being forward-filled: vol_5d went to
    0.0% valid across a 1,764 x 100 panel - the target vanished
    entirely, and a walk-forward on it simply reported no windows, which
    reads as "no signal" rather than "no data".
    """
    rets = np.full_like(close, np.nan, dtype=np.float64)
    rets[1:] = close[1:] / close[:-1] - 1.0

    T, A = close.shape
    out = np.full((T, A), np.nan)
    for t in range(T - horizon):
        window = rets[t + 1:t + 1 + horizon]
        ok = np.isfinite(window).all(axis=0)          # per asset
        if not ok.any():
            continue
        vals = window.std(axis=0, ddof=1) if horizon > 1 else np.zeros(A)
        out[t] = np.where(ok, vals, np.nan)
    valid = np.isfinite(out)
    return Target("vol_5d", np.nan_to_num(out), valid,
                  kind="continuous", autocorr_baseline=True)


def regime_shift_5d(close: np.ndarray, horizon: int = 5) -> Target:
    """Log-ratio of next-`horizon` realised vol to trailing-`horizon`
    realised vol.

    This is the target that asks 'is the market about to change
    character', as distinct from 'how volatile will it be'. Taking the
    ratio against trailing vol removes the level - and with it most of
    the autocorrelation that makes raw vol easy - so this is a
    substantially harder and more interesting target than vol_5d.

    A positive value means volatility expanded relative to the recent
    past; negative means it contracted.
    """
    rets = np.full_like(close, np.nan, dtype=np.float64)
    rets[1:] = close[1:] / close[:-1] - 1.0

    T, A = close.shape
    out = np.full((T, A), np.nan)
    eps = 1e-8
    for t in range(T - horizon):
        fut = rets[t + 1:t + 1 + horizon]
        past = rets[max(0, t + 1 - horizon):t + 1]
        if len(past) < horizon:
            continue
        ok = np.isfinite(fut).all(axis=0) & np.isfinite(past).all(axis=0)
        if not ok.any():
            continue
        fv = fut.std(axis=0, ddof=1)
        pv = past.std(axis=0, ddof=1)
        out[t] = np.where(ok, np.log((fv + eps) / (pv + eps)), np.nan)
    valid = np.isfinite(out)
    return Target("regime_shift_5d", np.nan_to_num(out), valid,
                  kind="continuous", autocorr_baseline=False)


def build_targets(close: np.ndarray, horizon: int = 5) -> Dict[str, Target]:
    """All four targets from a [T, A] close matrix."""
    close = np.asarray(close, dtype=np.float64)
    if close.ndim != 2:
        raise ValueError(f"close must be [T, A], got shape {close.shape}")
    ts = [direction_1d(close), rel_strength_1d(close),
          vol_5d(close, horizon), regime_shift_5d(close, horizon)]
    out = {t.name: t for t in ts}
    missing = set(TARGET_NAMES) - set(out)
    if missing:
        raise RuntimeError(f"build_targets did not produce {missing}")
    return out


def persistence_baseline(target: Target, close: np.ndarray,
                         horizon: int = 5) -> np.ndarray:
    """The 'tomorrow looks like today' predictor for a target: the same
    quantity measured over the TRAILING window instead of the forward
    one, using only bars <= t.

    This is what any real forecast has to beat. For vol_5d it is
    trailing 5-day realised vol - which is a strong predictor and free.

    Returns [T, A]; rows where it is undefined are zero and the caller
    should mask them with the target's own `valid`.
    """
    close = np.asarray(close, dtype=np.float64)
    rets = np.full_like(close, np.nan)
    rets[1:] = close[1:] / close[:-1] - 1.0
    T, A = close.shape
    out = np.full((T, A), np.nan)

    if target.name == "vol_5d":
        for t in range(T):
            past = rets[max(0, t + 1 - horizon):t + 1]
            if len(past) != horizon:
                continue
            ok = np.isfinite(past).all(axis=0)        # per asset
            out[t] = np.where(ok, past.std(axis=0, ddof=1), np.nan)
    elif target.name == "regime_shift_5d":
        # log(TRAILING VOL), not a ratio of past vols.
        #
        # This is a correction, and the run that forced it is worth
        # recording. regime_shift_5d[t] = log(fwd_vol[t]) -
        # log(trail_vol[t]). Any feature that proxies trailing
        # volatility - atr_pct is one, almost by definition - therefore
        # correlates with the target THROUGH ITS OWN DENOMINATOR,
        # whether or not it forecasts anything.
        #
        # The first version of this baseline used log(vol[t-h..t] /
        # vol[t-2h..t-h]), a ratio of two past windows. Partialling that
        # out does not remove the denominator term, so the artefact
        # survived: on a PURE RANDOM WALK the calibration run reported
        # atr_pct -> regime_shift_5d at incremental rho = -0.3864,
        # q = 0.04. A significant result on data with no structure.
        #
        # Controlling for log(trail_vol) itself removes exactly the
        # mechanical component and leaves the part of forward vol that
        # trailing vol does not already explain - which is the thing
        # actually worth asking about.
        for t in range(T):
            a = rets[max(0, t + 1 - horizon):t + 1]
            if len(a) != horizon:
                continue
            ok = np.isfinite(a).all(axis=0)           # per asset
            out[t] = np.where(ok, np.log(a.std(axis=0, ddof=1) + 1e-8), np.nan)
    else:
        # Directional targets: yesterday's return as today's forecast.
        out[1:] = rets[1:]

    return np.nan_to_num(out)
