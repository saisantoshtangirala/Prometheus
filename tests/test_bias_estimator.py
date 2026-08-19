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
    the actual contract compute_daily_bias reads. Returns-only
    (n_input_features defaults to n_assets) - the historical contract,
    still what most checkpoints look like."""
    cfg = PrometheusConfig(
        n_assets=10, seq_len=64, horizon=5, d_model=32, n_heads=2, n_layers=2,
        device="cpu", output_dir=str(tmp_path / "out"),
        snn_layer_sizes=[32, 16], snn_output_size=10,
    )
    engine = PrometheusEngine(cfg)
    meta_dir = tmp_path / "meta"
    engine.save(str(meta_dir))
    return tmp_path


@pytest.fixture
def saved_engine_dir_with_volume_features(tmp_path):
    """Same as saved_engine_dir but built with n_input_features=n_assets*2
    (kronos/features.py's returns+volume contract) - what a checkpoint
    trained after the feature-richness change looks like."""
    from kronos.features import n_input_features as _n_feat
    cfg = PrometheusConfig(
        n_assets=10, seq_len=64, horizon=5, d_model=32, n_heads=2, n_layers=2,
        device="cpu", output_dir=str(tmp_path / "out"),
        snn_layer_sizes=[32, 16], snn_output_size=10,
        n_input_features=_n_feat(10),
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

    def test_volume_feature_checkpoint_with_volumes_provided(self, saved_engine_dir_with_volume_features):
        recent_returns = np.random.randn(80, 10).astype(np.float32) * 0.01
        recent_volumes = np.random.randint(1_000_000, 50_000_000, (80, 10)).astype(np.float32)
        bias = compute_daily_bias(
            recent_returns, checkpoint_dir=saved_engine_dir_with_volume_features,
            recent_volumes=recent_volumes,
        )
        assert bias is not None
        assert bias.shape == (10,)

    def test_volume_feature_checkpoint_without_volumes_still_works(self, saved_engine_dir_with_volume_features):
        """Omitting recent_volumes must degrade gracefully (zeros channel
        via build_features), not raise or silently mismatch shapes."""
        recent_returns = np.random.randn(80, 10).astype(np.float32) * 0.01
        bias = compute_daily_bias(
            recent_returns, checkpoint_dir=saved_engine_dir_with_volume_features,
        )
        assert bias is not None
        assert bias.shape == (10,)


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


class TestReflexArcSizeCalibration:
    """calibrate_size_scale() - fits raw-pred -> position-size scaling
    against the SNN's own realized track record. Replaces an earlier
    normalize-by-its-own-volatility approach that was built and rejected:
    it was mathematically scale-invariant (400 ticks of pure white noise
    still produced 57% of cap average conviction, identical whether the
    noise was scaled x1 or x10) because it only ever compared a signal to
    itself. This version is anchored to actual realized outcomes instead."""

    @pytest.fixture
    def reflex(self, tmp_path):
        cfg = load_config()
        cfg.override("trading.db_path", str(tmp_path / "trades.db"))
        return ReflexArc(cfg)

    def test_default_scale_before_any_calibration(self, reflex):
        from kronos.reflex import DEFAULT_SIZE_SCALE
        assert reflex._size_scale == DEFAULT_SIZE_SCALE

    def test_insufficient_samples_falls_back_to_default(self, reflex):
        from kronos.reflex import DEFAULT_SIZE_SCALE
        n_assets = len(reflex.cfg.data.tickers)
        reflex._size_scale = 5.0   # simulate a prior real calibration
        tiny_returns = np.random.randn(3, n_assets).astype(np.float32) * 0.01
        reflex.calibrate_size_scale(tiny_returns)
        assert reflex._size_scale == DEFAULT_SIZE_SCALE

    def test_pure_noise_calibration_stays_safe_not_saturated(self, reflex):
        """The critical regression test: pure noise in, and the RESULTING
        inference signals must stay well short of saturation - not
        reproduce the rejected approach's 57%-of-cap-on-noise behavior.
        torch's global RNG is seeded so the SNN's untrained init weights
        (unrelated to the noise_returns seeding below) don't make this
        flaky run to run."""
        import torch
        torch.manual_seed(0)
        n_assets = len(reflex.cfg.data.tickers)
        rng = np.random.default_rng(3)
        noise_returns = rng.standard_normal(
            (reflex.cfg.data.lookback_days, n_assets)
        ).astype(np.float32) * 0.01
        reflex.calibrate_size_scale(noise_returns)

        mags = []
        for _ in range(50):
            recent = rng.standard_normal((30, n_assets)).astype(np.float32) * 0.01
            decision = reflex.infer(recent, vix_value=15.0)
            mags.append(np.abs(decision.signals))
        mags = np.array(mags)
        # 0.3 still leaves a wide margin under the rejected approach's 0.57
        # mean-of-cap on the same kind of pure-noise input.
        assert mags.mean() < 0.3
        assert (mags > 0.9).mean() < 0.05  # no near-saturation on pure noise

    def test_calibration_never_raises_on_malformed_input(self, reflex):
        from kronos.reflex import DEFAULT_SIZE_SCALE
        reflex.calibrate_size_scale(np.zeros((0, 0)))
        assert reflex._size_scale == DEFAULT_SIZE_SCALE

    def test_infer_applies_size_scale_before_tanh(self, reflex, monkeypatch):
        """Directly verifies the wiring in infer(): with size_scale set,
        signals must equal tanh(raw_pred * size_scale), not plain
        tanh(raw_pred)."""
        import torch
        n_assets = len(reflex.cfg.data.tickers)
        raw = torch.full((1, n_assets), 0.01)

        def fake_forward(x):
            return raw

        monkeypatch.setattr(reflex.snn, "forward", fake_forward)
        reflex._size_scale = 3.0
        recent_returns = np.random.randn(30, n_assets).astype(np.float32) * 0.01
        decision = reflex.infer(recent_returns, vix_value=15.0)
        expected = np.tanh(0.01 * 3.0)
        assert np.allclose(decision.signals, expected, atol=1e-5)
