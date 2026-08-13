"""
Data Validator – pre-processing guards for the Prometheus pipeline.

Provides:
  - KalmanFilter1D: missing data imputation
  - floor_nanoseconds_to_microseconds: timestamp normalization
  - detect_illiquid_periods: volume-based liquidity flag
  - detect_flash_crash: single-bar extreme move detection
  - chronological_split: leakage-free train/test split
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class KalmanFilter1D:
    """
    Scalar Kalman filter for 1-D time-series imputation.

    Uses a random-walk process model: x_t = x_{t-1} + process_noise.
    Observations: z_t = x_t + obs_noise.
    """

    def __init__(self, process_noise: float = 1e-3, obs_noise: float = 1e-1):
        self.Q = process_noise
        self.R = obs_noise

    def fill(self, series: pd.Series) -> pd.Series:
        """Fill NaN values using the Kalman smoother. Non-NaN values are kept as-is."""
        result = series.copy().astype(float)
        valid = series.dropna()
        x = float(valid.iloc[0]) if not valid.empty else 0.0
        P = 1.0

        for i in range(len(series)):
            # Predict
            x_pred = x
            P_pred = P + self.Q

            if pd.isna(series.iloc[i]):
                # No observation → use prediction
                result.iloc[i] = x_pred
                x, P = x_pred, P_pred
            else:
                # Kalman update
                K = P_pred / (P_pred + self.R)
                x = x_pred + K * (float(series.iloc[i]) - x_pred)
                P = (1.0 - K) * P_pred
                result.iloc[i] = x

        return result

    def fill_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self.fill, axis=0)


def floor_nanoseconds_to_microseconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Floor a DatetimeIndex from nanosecond to microsecond precision.
    Raises ValueError if index is not a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    df = df.copy()
    df.index = df.index.floor("us")
    return df


def standardize_timezone(df: pd.DataFrame, target_tz: str = "UTC") -> pd.DataFrame:
    """Convert any timezone (or naive UTC) to target_tz."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(target_tz)
    df.index = df.index.floor("us")
    return df


def detect_illiquid_periods(
    volume: pd.Series,
    consecutive_zeros: int = 3,
) -> Tuple[pd.Series, bool]:
    """
    Detect illiquid periods (≥ N consecutive zero-volume bars).

    Returns (filled_volume, is_illiquid).
    Fills zero-volume periods with the median of non-zero bars.
    """
    med = float(volume[volume > 0].median()) if (volume > 0).any() else 1.0
    zero_run = (volume == 0).astype(int).rolling(consecutive_zeros).sum()
    is_illiquid = bool((zero_run >= consecutive_zeros).any())
    filled = volume.where(volume > 0, med)
    return filled, is_illiquid


def detect_flash_crash(
    returns: np.ndarray,
    threshold: float = -0.20,
) -> bool:
    """
    Return True if any single-bar return is below threshold (e.g., -20%).
    Used to trigger CortisolSystem.trigger_flash_crash_lockout().
    """
    return bool(np.any(np.asarray(returns) < threshold))


def chronological_split(
    df: pd.DataFrame,
    train_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time-series strictly by date to prevent look-ahead leakage.

    Args:
        df: DataFrame with DatetimeIndex.
        train_end: ISO date string (exclusive end of training set).

    Returns:
        (train_df, test_df) — no overlap.
    """
    cutoff = pd.Timestamp(train_end)
    train = df[df.index < cutoff]
    test = df[df.index >= cutoff]
    return train, test


def validate_no_future_edges(
    train_corr: np.ndarray,
    test_corr: np.ndarray,
    threshold: float = 0.3,
    leakage_tol: float = 0.1,
) -> bool:
    """
    Check that edges present in the training correlation graph are not
    suspiciously over-fitted to the test period.

    A leakage flag is raised if:
      edges_in_test_but_not_train / total_test_edges > leakage_tol

    Returns True if clean (no leakage), False if suspicious.
    """
    train_edges = np.abs(train_corr) > threshold
    test_edges = np.abs(test_corr) > threshold
    np.fill_diagonal(train_edges, False)
    np.fill_diagonal(test_edges, False)

    novel_in_test = test_edges & ~train_edges
    total_test = test_edges.sum()
    if total_test == 0:
        return True
    leakage_ratio = novel_in_test.sum() / total_test
    return bool(leakage_ratio <= leakage_tol)
