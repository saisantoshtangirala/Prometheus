"""
Tests for kronos/bias_estimator.py - the once-per-adoption
causal_transformer "second opinion" ReflexArc checks its own SNN
signal against, and for ReflexArc actually blending it in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos import load_config
from kronos.bias_estimator import compute_daily_bias
from kronos.reflex import ReflexArc
from prometheus.engine import PrometheusEngine, PrometheusConfig


@pytest.fixture
def saved_engine_dir(tmp_path):
    """A real PrometheusEngine.save() output - arch.json + weight files,
    the actual contract compute_daily_bias reads."""
    cfg = PrometheusConfig(
        n_assets=10, seq_len=64, horizon=5, d_model=32, n_heads=2, n_layers=2,
        device="cpu", output_dir=str(tmp_path / "out"),
        snn_layer_sizes=[32, 16], snn_output_size=10,
    )
    engine = PrometheusEngine(cfg)
    meta_dir = tmp_path / "meta"
    engine.save(str(meta_dir))
    return tmp_path


class TestComputeDailyBias:
    def test_happy_path_returns_n_assets_vector(self, saved_engine_dir):
        recent_returns = np.random.randn(80, 10).astype(np.float32) * 0.01
        bias = compute_daily_bias(recent_returns, checkpoint_dir=saved_engine_dir)
        assert bias is not None
        assert bias.shape == (10,)

    def test_insufficient_bars_returns_none(self, saved_engine_dir):
        recent_returns = np.random.randn(10, 10).astype(np.float32) * 0.01
        assert compute_daily_bias(recent_returns, checkpoint_dir=saved_engine_dir) is None

    def test_missing_arch_json_returns_none(self, tmp_path):
        (tmp_path / "meta").mkdir()
        recent_returns = np.random.randn(80, 10).astype(np.float32) * 0.01
        assert compute_daily_bias(recent_returns, checkpoint_dir=tmp_path) is None

    def test_wrong_n_assets_returns_none(self, saved_engine_dir):
        recent_returns = np.random.randn(80, 5).astype(np.float32) * 0.01
        assert compute_daily_bias(recent_returns, checkpoint_dir=saved_engine_dir) is None

    def test_corrupt_weight_file_returns_none_not_raise(self, saved_engine_dir):
        (saved_engine_dir / "meta" / "ltc.pt").write_bytes(b"not a torch checkpoint")
        recent_returns = np.random.randn(80, 10).astype(np.float32) * 0.01
        assert compute_daily_bias(recent_returns, checkpoint_dir=saved_engine_dir) is None


class TestReflexArcConfidenceBlend:
    @pytest.fixture
    def reflex(self, tmp_path):
        cfg = load_config()
        cfg.override("trading.db_path", str(tmp_path / "trades.db"))
        return ReflexArc(cfg)

    def test_no_bias_set_leaves_signals_unchanged(self, reflex):
        n_assets = len(reflex.cfg.data.tickers)
        recent_returns = np.random.randn(30, n_assets).astype(np.float32) * 0.01
        decision = reflex.infer(recent_returns, vix_value=15.0)
        assert decision.confidence_blended is False

    def test_agreeing_bias_boosts_magnitude_same_sign(self, reflex):
        n_assets = len(reflex.cfg.data.tickers)
        recent_returns = np.random.randn(30, n_assets).astype(np.float32) * 0.01
        before = reflex.infer(recent_returns, vix_value=15.0)

        reflex.set_daily_bias(np.sign(before.signals) * 0.5)
        after = reflex.infer(recent_returns, vix_value=15.0)

        assert after.confidence_blended is True
        for b, a in zip(before.signals, after.signals):
            if abs(b) > 1e-6:
                assert np.sign(a) == np.sign(b)
                assert abs(a) >= abs(b) - 1e-6

    def test_disagreeing_bias_dampens_but_does_not_flip_sign(self, reflex):
        n_assets = len(reflex.cfg.data.tickers)
        recent_returns = np.random.randn(30, n_assets).astype(np.float32) * 0.01
        before = reflex.infer(recent_returns, vix_value=15.0)

        reflex.set_daily_bias(-np.sign(before.signals) * 0.5)
        after = reflex.infer(recent_returns, vix_value=15.0)

        assert after.confidence_blended is True
        for b, a in zip(before.signals, after.signals):
            if abs(b) > 1e-6:
                assert np.sign(a) == np.sign(b)   # dampened, never vetoed
                assert abs(a) <= abs(b) + 1e-6

    def test_mismatched_length_bias_skipped_gracefully(self, reflex):
        n_assets = len(reflex.cfg.data.tickers)
        recent_returns = np.random.randn(30, n_assets).astype(np.float32) * 0.01
        reflex.set_daily_bias(np.array([0.1, 0.2]))   # wrong length
        decision = reflex.infer(recent_returns, vix_value=15.0)
        assert decision.confidence_blended is False

    def test_clearing_bias_reverts_to_unblended(self, reflex):
        n_assets = len(reflex.cfg.data.tickers)
        recent_returns = np.random.randn(30, n_assets).astype(np.float32) * 0.01
        reflex.set_daily_bias(np.ones(n_assets))
        reflex.infer(recent_returns, vix_value=15.0)
        reflex.set_daily_bias(None)
        decision = reflex.infer(recent_returns, vix_value=15.0)
        assert decision.confidence_blended is False
