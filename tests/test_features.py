"""Tests for kronos/features.py - the shared returns+volume feature builder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone

from kronos.data_pipeline import DailyMemory
from kronos.features import build_features, n_input_features


def test_n_input_features_is_double_n_assets():
    assert n_input_features(10) == 20
    assert n_input_features(1) == 2


def test_output_shape_is_2a_wide():
    returns = np.random.randn(30, 10).astype(np.float32) * 0.01
    volumes = np.random.randint(1_000_000, 50_000_000, (30, 10)).astype(np.float32)
    feats = build_features(returns, volumes)
    assert feats.shape == (30, 20)


def test_first_half_is_returns_unchanged():
    returns = np.random.randn(30, 10).astype(np.float32) * 0.01
    volumes = np.random.randint(1_000_000, 50_000_000, (30, 10)).astype(np.float32)
    feats = build_features(returns, volumes)
    np.testing.assert_array_equal(feats[:, :10], returns)


def test_second_half_is_zscored_volume():
    returns = np.zeros((30, 10), dtype=np.float32)
    volumes = np.random.randint(1_000_000, 50_000_000, (30, 10)).astype(np.float32)
    feats = build_features(returns, volumes)
    vol_z = feats[:, 10:]
    # a proper z-score has ~zero mean and ~unit std per column
    assert np.allclose(vol_z.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(vol_z.std(axis=0), 1.0, atol=1e-3)


def test_none_volumes_falls_back_to_zeros_not_raise():
    returns = np.random.randn(30, 10).astype(np.float32) * 0.01
    feats = build_features(returns, None)
    assert feats.shape == (30, 20)
    np.testing.assert_array_equal(feats[:, 10:], np.zeros((30, 10), dtype=np.float32))


def test_mismatched_volume_shape_falls_back_to_zeros_not_raise():
    returns = np.random.randn(30, 10).astype(np.float32) * 0.01
    wrong_shape_volumes = np.random.randn(30, 5).astype(np.float32)
    feats = build_features(returns, wrong_shape_volumes)
    assert feats.shape == (30, 20)
    np.testing.assert_array_equal(feats[:, 10:], np.zeros((30, 10), dtype=np.float32))


def test_extreme_volume_spike_is_clipped():
    returns = np.zeros((30, 3), dtype=np.float32)
    volumes = np.ones((30, 3), dtype=np.float32) * 1_000_000
    volumes[0, 0] = 1e12   # one absurd print
    feats = build_features(returns, volumes)
    assert np.all(np.abs(feats[:, 3:]) <= 5.0 + 1e-6)


def test_never_leaks_outside_window_deterministic_given_same_window():
    """Same fixed window in -> same features out (no external state)."""
    rng = np.random.default_rng(3)
    returns = rng.standard_normal((30, 10)).astype(np.float32) * 0.01
    volumes = rng.integers(1_000_000, 50_000_000, (30, 10)).astype(np.float32)
    a = build_features(returns, volumes)
    b = build_features(returns, volumes)
    np.testing.assert_array_equal(a, b)


class TestDailyMemoryVolumesWindow:
    def _memory(self, vol_column_order):
        rng = np.random.default_rng(9)
        tickers = ["AAA", "BBB", "CCC"]
        dates = pd.bdate_range("2026-01-01", periods=20)
        returns = pd.DataFrame(
            rng.standard_normal((20, 3)).astype(np.float32) * 0.01,
            index=dates, columns=tickers,
        )
        volumes = pd.DataFrame(
            rng.integers(1_000_000, 50_000_000, (20, 3)).astype(np.float32),
            index=dates, columns=vol_column_order,
        )
        prices = (1 + returns).cumprod() * 100
        vix = pd.Series(rng.uniform(15, 25, 20), index=dates)
        return DailyMemory(
            as_of=datetime.now(timezone.utc),
            prices=prices, volumes=volumes, returns=returns, vix=vix,
            sentiment={t: 0.0 for t in tickers}, macro={},
            source_used="synthetic",
        )

    def test_shape_matches_returns_window(self):
        mem = self._memory(vol_column_order=["AAA", "BBB", "CCC"])
        assert mem.volumes_window(5).shape == mem.returns_window(5).shape

    def test_aligns_to_returns_column_order_even_if_volumes_differ(self):
        """volumes DataFrame has a DIFFERENT ticker column order than
        returns - volumes_window() must still align to returns' order,
        not silently hand back mismatched columns that would let
        build_features() concatenate the wrong asset's volume onto the
        wrong asset's return."""
        mem = self._memory(vol_column_order=["CCC", "AAA", "BBB"])
        vw = mem.volumes_window(20)
        # Reconstruct what AAA's volume should be (from the raw DataFrame,
        # not through the accessor) and confirm it lands in AAA's column
        # position (index 0, matching returns.columns == ["AAA","BBB","CCC"]).
        expected_aaa = mem.volumes["AAA"].values.astype(np.float32)
        np.testing.assert_array_equal(vw[:, 0], expected_aaa)

    def test_combines_cleanly_with_build_features(self):
        mem = self._memory(vol_column_order=["AAA", "BBB", "CCC"])
        feats = build_features(mem.returns_window(10), mem.volumes_window(10))
        assert feats.shape == (10, n_input_features(3))
