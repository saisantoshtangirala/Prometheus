"""
Feature bank: assembles the candidate features the QUBO selector chooses
from.

Two design rules, both learned from this project's own history:

1. EVERY feature is causal by construction. Feature row t uses bars
   strictly before t. There is no `.shift(-1)`, no centred rolling
   window, no `.rolling(...).mean()` that peeks. This is asserted by
   tests, not trusted.

2. The bank is DELIBERATELY WIDE and unfiltered. Component 2 (QUBO
   selection) exists to pick a subset; pre-filtering here on a hunch
   would just move the selection decision somewhere less principled and
   less measurable. Redundant features are fine - the redundancy term in
   the QUBO is designed to handle exactly that.

Per-asset feature groups:
  momentum        returns over several horizons
  volatility      realised vol over several windows
  mean-reversion  z-score of price vs its own moving average
  volume-free     (deliberately no volume: this repo has a MEASURED
                  result that adding a volume channel made walk-forward
                  Sharpe worse, -1.51 vs -0.43. Not re-litigating that
                  without a reason.)
  connectome      the four per-node network features from component 1

Market-wide features (broadcast to every asset):
  the five global connectome statistics
  cross-sectional dispersion of returns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from chimera.connectome import (
    GLOBAL_FEATURE_NAMES, NODE_FEATURE_NAMES, FinancialConnectome,
)

MOMENTUM_HORIZONS = (1, 2, 5, 10, 21)
VOL_WINDOWS = (5, 10, 21)
ZSCORE_WINDOWS = (10, 21)


def _causal_rolling(x: np.ndarray, window: int, fn) -> np.ndarray:
    """Apply `fn` over a strictly-backward window. [T, N] -> [T, N].

    Row t uses x[t-window:t] - EXCLUDING t. Rows before `window` are 0.
    Written explicitly rather than via pandas.rolling because
    pandas' default window INCLUDES the current row, which is the single
    most common way look-ahead enters a feature pipeline.
    """
    T, N = x.shape
    out = np.zeros((T, N), dtype=np.float64)
    for t in range(window, T):
        out[t] = fn(x[t - window : t])
    return out


def feature_names(n_assets: int) -> List[str]:
    names = [f"mom_{h}" for h in MOMENTUM_HORIZONS]
    names += [f"vol_{w}" for w in VOL_WINDOWS]
    names += [f"zscore_{w}" for w in ZSCORE_WINDOWS]
    names += [f"conn_{n}" for n in NODE_FEATURE_NAMES]
    names += [f"mkt_{n}" for n in GLOBAL_FEATURE_NAMES]
    names += ["mkt_dispersion"]
    return names


@dataclass(frozen=True)
class FeatureBank:
    """values: [T, n_assets, n_features]; names: len n_features."""

    values: np.ndarray
    names: Tuple[str, ...]
    warmup: int

    @property
    def n_features(self) -> int:
        return self.values.shape[2]

    def flat(self) -> np.ndarray:
        """[T * n_assets, n_features] - the QUBO selector's input shape.

        Pooling assets into one design matrix is intentional: the model
        is cross-sectional (one shared function applied per asset), so
        feature relevance should be judged pooled, not per-name.
        """
        T, N, F = self.values.shape
        return self.values.reshape(T * N, F)


def build_features(
    returns: np.ndarray,
    prices: Optional[np.ndarray] = None,
    connectome_window: int = 60,
) -> FeatureBank:
    """Build the causal feature bank from a return series.

    returns: [T, n_assets] simple returns
    prices:  [T, n_assets] optional; reconstructed from returns if absent
    """
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError(f"returns must be [T, n_assets], got {R.shape}")
    T, N = R.shape
    P = np.cumprod(1.0 + np.nan_to_num(R), axis=0) if prices is None \
        else np.asarray(prices, dtype=np.float64)

    layers: List[np.ndarray] = []

    # Momentum: cumulative return over the h bars ENDING at t-1.
    for h in MOMENTUM_HORIZONS:
        m = np.zeros((T, N))
        for t in range(h, T):
            m[t] = np.prod(1.0 + R[t - h : t], axis=0) - 1.0
        layers.append(m)

    for w in VOL_WINDOWS:
        layers.append(_causal_rolling(R, w, lambda a: a.std(axis=0)))

    # Mean-reversion z-score of price vs its own trailing mean.
    for w in ZSCORE_WINDOWS:
        z = np.zeros((T, N))
        for t in range(w, T):
            win = P[t - w : t]
            mu, sd = win.mean(axis=0), win.std(axis=0)
            z[t] = np.where(sd > 1e-12, (P[t - 1] - mu) / np.where(sd > 1e-12, sd, 1.0), 0.0)
        layers.append(z)

    per_asset = np.stack(layers, axis=2)                       # [T, N, k]

    fc = FinancialConnectome(window=connectome_window)
    node_feats, glob_feats = fc.rolling_features(R)            # [T,N,4], [T,5]

    glob_b = np.repeat(glob_feats[:, None, :], N, axis=1)      # [T, N, 5]

    dispersion = np.zeros((T, 1))
    for t in range(1, T):
        dispersion[t, 0] = float(np.std(R[t - 1]))
    disp_b = np.repeat(dispersion[:, None, :], N, axis=1)      # [T, N, 1]

    values = np.concatenate([per_asset, node_feats, glob_b, disp_b], axis=2)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    warmup = max(max(MOMENTUM_HORIZONS), max(VOL_WINDOWS),
                 max(ZSCORE_WINDOWS), fc.warmup)
    return FeatureBank(values=values, names=tuple(feature_names(N)), warmup=warmup)


def standardise(train: np.ndarray, apply_to: np.ndarray) -> np.ndarray:
    """Z-score `apply_to` using statistics from `train` ONLY.

    The signature enforces the discipline: you cannot standardise a test
    window without explicitly handing over a train window to fit on.
    Fitting the scaler on the full series is a classic, quiet look-ahead
    leak and this makes it awkward to do by accident.
    """
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (apply_to - mu) / sd
