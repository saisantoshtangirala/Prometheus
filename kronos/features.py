"""
Shared feature construction: returns + volume, concatenated into one
[T, 2*n_assets] input for the SNN/causal_transformer/LTC. One place both
training (scripts/train.py) and inference (kronos/reflex.py,
kronos/bias_estimator.py, kronos/backtest.py) build this from, so the
input contract can't silently drift out of sync between callers the way
the SNN's output shape once did (see PrometheusEngine.snn_output_size's
docstring for that history).

Feature layout per timestep: [return_1..return_A, vol_z_1..vol_z_A].

Why volume at all: every model in this system was, until now, trained and
run on nothing but each asset's own past returns over a several-week
window - predicting tomorrow's return from nothing but the last month of
the same handful of stocks' own returns. That is close to the hardest
version of this problem there is. Volume is cheap, already-fetched real
information (every yfinance pull already includes it) that price-only
models never saw. Sentiment/macro are NOT added here even though
kronos/data_pipeline.py's DailyMemory already carries them - there is no
historical archive to honestly backtest either against, and wiring
unverifiable inputs into a live trading model would repeat the exact
mistake this project spent this session correcting (declaring victory
without held-out proof).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

VOLUME_ZSCORE_CLIP = 5.0   # guard against one extreme print dominating the feature


def n_input_features(n_assets: int) -> int:
    """Width of build_features()'s output for n_assets assets."""
    return n_assets * 2


def build_features(
    returns: np.ndarray,
    volumes: Optional[np.ndarray],
) -> np.ndarray:
    """
    returns: [T, A] per-bar returns.
    volumes: [T, A] per-bar volumes, or None/wrong-shape (falls back to
      zeros - keeps the output shape contract stable even when volume
      isn't available, e.g. an offline CSV backtest with no volume
      column, rather than raising or silently returning returns-only).

    Returns [T, 2*A] float32: returns concatenated with each asset's own
    WITHIN-WINDOW volume z-score. Normalizing against the window's own
    mean/std (rather than needing extra history beyond what's already
    passed in) never looks outside [0, T) - safe for both the strict
    walk-forward backtest (which already bounds every window to "data up
    to and including t-1") and live per-tick inference (whose window is
    already "recent past up to now").
    """
    returns = np.asarray(returns, dtype=np.float32)
    T, A = returns.shape
    if volumes is None or volumes.shape != returns.shape:
        vol_z = np.zeros_like(returns)
    else:
        volumes = np.asarray(volumes, dtype=np.float32)
        mean = volumes.mean(axis=0, keepdims=True)
        std = volumes.std(axis=0, keepdims=True)
        vol_z = (volumes - mean) / (std + 1e-8)
        vol_z = np.clip(vol_z, -VOLUME_ZSCORE_CLIP, VOLUME_ZSCORE_CLIP)
    return np.concatenate([returns, vol_z], axis=1).astype(np.float32)
