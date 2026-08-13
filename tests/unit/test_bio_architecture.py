"""
Phase 3: Biological Architecture — LTC & Spiking Neural Networks
Tests: LTC-01, LTC-02, SNN-01, SNN-02
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prometheus.neuro.ltc_network import LiquidTimeConstantNetwork
from prometheus.neuro.neuromodulation import CortisolSystem, DopamineSystem, NeuromodulationSystem
from prometheus.neuro.spiking_network import SpikingMarketEncoder


# ---------------------------------------------------------------------------
# LTC-01: Gradient Flow — no NaN / Inf after 100-step forward pass
# ---------------------------------------------------------------------------

class TestLTC01GradientFlow:
    @staticmethod
    def _get_pred(out):
        """Extract prediction tensor from LTC output (tensor, list, dict) or dict."""
        if isinstance(out, tuple):
            return out[0]
        if isinstance(out, dict):
            return out["predictions"]
        return out

    def test_no_nan_in_gradients(self, ltc_net):
        """100-step forward pass must not produce NaN/Inf gradients."""
        seq_len = 100
        n_assets = 5
        x = torch.randn(1, seq_len, n_assets, requires_grad=False)

        # Forward
        out = ltc_net(x)
        pred = self._get_pred(out)

        # Scalar loss
        loss = pred.mean()
        loss.backward()

        for name, param in ltc_net.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_gradient_clipping_at_1(self, ltc_net):
        """After gradient clipping at 1.0 the L2 norm must be ≤ 1.0."""
        x = torch.randn(1, 20, 5)
        out = ltc_net(x)
        pred = self._get_pred(out)
        pred.sum().backward()

        torch.nn.utils.clip_grad_norm_(ltc_net.parameters(), max_norm=1.0)

        total_norm = 0.0
        for p in ltc_net.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        assert total_norm <= 1.01, f"Gradient norm {total_norm:.4f} exceeds clip threshold"

    def test_output_shape(self, ltc_net):
        x = torch.randn(2, 30, 5)
        out = ltc_net(x)
        pred = self._get_pred(out)
        assert pred.shape[0] == 2, "Batch dimension must be preserved"

    def test_no_nan_in_output(self, ltc_net):
        x = torch.randn(1, 50, 5)
        out = ltc_net(x)
        pred = self._get_pred(out)
        assert not torch.isnan(pred).any()


# ---------------------------------------------------------------------------
# LTC-02: Dopamine Bounds — RPE clamped to [-5, +5]
# ---------------------------------------------------------------------------

class TestLTC02DopamineBounds:
    def test_raw_rpe_clamped_lower(self, dopamine):
        """Extreme negative return error must not exceed -5."""
        dopamine.update(predicted_return=0.5, actual_return=-10.0)
        assert dopamine._raw_rpe >= -DopamineSystem.DA_RPE_CLAMP, (
            f"RPE {dopamine._raw_rpe} < -5.0 — dopamine not clamped"
        )

    def test_raw_rpe_clamped_upper(self, dopamine):
        """Extreme positive return error must not exceed +5."""
        dopamine.update(predicted_return=-0.5, actual_return=10.0)
        assert dopamine._raw_rpe <= DopamineSystem.DA_RPE_CLAMP, (
            f"RPE {dopamine._raw_rpe} > +5.0 — dopamine not clamped"
        )

    def test_da_level_stays_in_range(self, dopamine):
        """DA level (0..1) must remain in bounds after many updates."""
        rng = np.random.default_rng(7)
        for _ in range(200):
            p = float(rng.uniform(-2, 2))
            a = float(rng.uniform(-2, 2))
            dopamine.update(p, a)
        lvl = dopamine.get_level()
        assert 0.0 <= lvl <= 1.0, f"DA level {lvl} out of [0, 1]"

    def test_loss_does_not_explode_at_da_limit(self, dopamine):
        """Position multiplier must stay finite even at DA extremes."""
        # Drive DA to maximum
        for _ in range(50):
            dopamine.update(predicted_return=-5.0, actual_return=5.0)
        mult = dopamine.get_position_multiplier()
        assert np.isfinite(mult), "Position multiplier must be finite"
        assert 0.0 <= mult <= 5.0, f"Position multiplier {mult} out of range"


# ---------------------------------------------------------------------------
# SNN-01: Spike Decay — spikes fade to zero after tau=20 steps
# ---------------------------------------------------------------------------

class TestSNN01SpikeDecay:
    def test_spikes_decay_with_zero_input(self, snn_encoder):
        """
        Feed a non-zero input to build up membrane potential,
        then switch to zero input — spikes must fade within tau=20 steps.
        """
        torch.manual_seed(0)
        n_assets, seq_len = 5, 16
        tau = 20

        # First prime the network with some signal
        priming = torch.randn(1, seq_len, n_assets)
        with torch.no_grad():
            _ = snn_encoder(priming)

        # Now send zero input for tau steps
        zeros = torch.zeros(1, tau, n_assets)
        with torch.no_grad():
            out = snn_encoder(zeros)

        # SNN returns (tensor, meta_dict)
        meta = out[1] if isinstance(out, tuple) else (out.get("meta", {}) if isinstance(out, dict) else {})
        firing_rate = meta.get("firing_rate", 0.0)
        # After tau steps of silence, firing rate must be near zero
        assert firing_rate <= 0.15, (
            f"Firing rate {firing_rate:.4f} still high after {tau} zero-input steps. "
            "SNN spike decay (refractory period) not working."
        )

    def test_no_spikes_on_constant_zero_input(self, snn_encoder):
        """Pure zero input from cold start must produce no spikes."""
        x = torch.zeros(1, 30, 5)
        with torch.no_grad():
            out = snn_encoder(x)
        meta = out[1] if isinstance(out, tuple) else {}
        firing_rate = meta.get("firing_rate", 0.0)
        assert firing_rate <= 0.05, f"Zero input produced non-zero firing rate: {firing_rate}"


# ---------------------------------------------------------------------------
# SNN-02: Cortisol Stress — exactly 70% position reduction in fear mode
# ---------------------------------------------------------------------------

class TestSNN02CortisolStress:
    def test_fear_mode_caps_position_at_30_pct(self, cortisol):
        """
        When cortisol is driven above fear_threshold,
        get_position_cap() must return exactly FEAR_POSITION_CAP (0.30).
        """
        # Force cortisol into fear mode by setting internal state directly
        cortisol._cort_level = 0.9
        cortisol._in_fear_mode = True

        cap = cortisol.get_position_cap()
        assert cap == CortisolSystem.FEAR_POSITION_CAP, (
            f"Position cap in fear mode = {cap:.3f}, expected {CortisolSystem.FEAR_POSITION_CAP}. "
            "Biological 70% reduction constraint not met."
        )

    def test_extreme_volatility_triggers_fear(self, cortisol):
        """
        Extreme market entropy (>20% daily move mapped to entropy≈0.95)
        must drive cortisol above fear_threshold within a few updates.
        """
        # Simulate 5 extreme volatility bars
        for _ in range(5):
            cortisol.update(
                market_entropy=0.97,
                drawdown=0.85,
                corr_breakdown=0.90,
                causal_confidence=0.05,
            )
        assert cortisol._cort_level >= cortisol.fear_threshold or cortisol._in_fear_mode, (
            "Extreme inputs should push cortisol into fear mode"
        )

    def test_position_reduction_is_70_pct(self):
        """End-to-end: position_cap in fear mode = 0.30 → 70% reduction from 1.0."""
        cort = CortisolSystem(fear_threshold=0.7)
        cort._cort_level = 0.95
        cort._in_fear_mode = True

        baseline = 1.0  # fully invested
        cap = cort.get_position_cap()
        reduction_pct = (baseline - cap) / baseline
        assert abs(reduction_pct - 0.70) < 1e-9, (
            f"Reduction = {reduction_pct:.2%}, expected exactly 70%"
        )

    def test_lockout_prevents_buy_signals(self, cortisol):
        """After flash crash lockout, fear mode persists for lockout_duration steps."""
        cortisol.trigger_flash_crash_lockout()

        fear_count = 0
        for _ in range(cortisol.lockout_duration + 1):
            _, in_fear = cortisol.update(
                market_entropy=0.1,  # now calm
                drawdown=0.0,
                corr_breakdown=0.0,
                causal_confidence=0.9,
            )
            if in_fear:
                fear_count += 1

        assert fear_count >= cortisol.lockout_duration - 1, (
            f"Lockout lasted {fear_count} steps, expected ≥ {cortisol.lockout_duration - 1}"
        )
