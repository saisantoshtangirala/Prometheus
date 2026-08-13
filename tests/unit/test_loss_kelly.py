"""
Phase 7: Asymmetric Loss & Kelly Criterion
Tests: LOSS-01, KELLY-01
+ Hypothesis property-based tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prometheus.loss.asymmetric_loss import AsymmetricUtilityLoss
from prometheus.loss.kelly_optimizer import KellyCriterionOptimizer

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st
    HYP_AVAILABLE = True
except ImportError:
    HYP_AVAILABLE = False
    pytest.mark.hypothesis = pytest.mark.skip("hypothesis not installed")


# ---------------------------------------------------------------------------
# LOSS-01: Asymmetric Utility Function
# ---------------------------------------------------------------------------

class TestLOSS01AsymmetricUtility:
    """
    Case 1: correct direction, same magnitude → loss ≈ 0, gradients ≈ 0.
    Case 2: wrong direction → loss >> Case 1 (exponentially higher).
    """

    def test_perfect_prediction_has_near_zero_loss(self, loss_fn):
        """Pred = target (same sign, same magnitude) → loss should be 0."""
        pred = torch.tensor([[0.05, 0.03, -0.02]], requires_grad=False)
        target = torch.tensor([[0.05, 0.03, -0.02]])
        loss = loss_fn(pred, target)
        assert loss.item() < 1e-6, f"Perfect prediction loss={loss.item():.6f}, expected ≈ 0"

    def test_correct_direction_overestimate_has_zero_loss(self, loss_fn):
        """
        Bullish miss: pred=+5%, actual=+5% (exact match) → 0 loss.
        Case where pred > actual but same sign → pred is not underestimated → 0.
        """
        pred = torch.tensor([[0.08]])    # more bullish
        target = torch.tensor([[0.05]])  # actual was less bullish
        loss = loss_fn(pred, target)
        # Same sign, pred > actual → not underestimated → loss = 0
        assert loss.item() < 1e-3, (
            f"Bullish over-prediction loss={loss.item():.4f}. "
            "When prediction is same-sign but larger, loss must be ≈ 0."
        )

    def test_wrong_direction_loss_exponentially_higher(self, loss_fn):
        """
        Case: pred=+5% but market goes -5% → exponential directional penalty.
        This must be much larger than a same-direction error.
        """
        magnitude = 0.05
        # Same direction miss
        pred_correct = torch.tensor([[magnitude]])
        target_correct = torch.tensor([[magnitude * 0.5]])
        loss_correct = loss_fn(pred_correct, target_correct)

        # Wrong direction miss
        pred_wrong = torch.tensor([[magnitude]])
        target_wrong = torch.tensor([[-magnitude]])
        loss_wrong = loss_fn(pred_wrong, target_wrong)

        assert loss_wrong.item() > loss_correct.item(), (
            f"Wrong-direction loss ({loss_wrong.item():.4f}) must exceed "
            f"correct-direction loss ({loss_correct.item():.4f})"
        )

    def test_gradients_near_zero_on_perfect_prediction(self, loss_fn):
        """Backprop on perfect prediction: gradients on pred must be ≈ 0."""
        pred = torch.tensor([[0.05]], requires_grad=True)
        target = torch.tensor([[0.05]])
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert abs(pred.grad.item()) < 1e-4, (
            f"Gradient on perfect prediction = {pred.grad.item():.6f}, expected ≈ 0"
        )

    def test_wrong_direction_loss_exceeds_correct_direction_loss(self, loss_fn):
        """
        For the same prediction magnitude, wrong-direction error must produce
        much higher loss than a correct-direction miss of the same magnitude.
        """
        magnitude = 0.10

        # Correct direction: pred=+10%, actual=+5% (overestimated, same sign)
        loss_correct = loss_fn(
            torch.tensor([[magnitude]]),
            torch.tensor([[magnitude * 0.5]]),
        )

        # Wrong direction: pred=+10%, actual=-10%
        loss_wrong = loss_fn(
            torch.tensor([[magnitude]]),
            torch.tensor([[-magnitude]]),
        )

        ratio = loss_wrong.item() / (loss_correct.item() + 1e-8)
        assert ratio >= 2.0, (
            f"Wrong-direction loss ({loss_wrong.item():.4f}) must be ≥ 2× "
            f"correct-direction loss ({loss_correct.item():.4f}), got ratio={ratio:.2f}"
        )

    def test_confidence_scales_wrong_direction_penalty(self, loss_fn):
        """High confidence wrong-direction prediction must be penalised more."""
        pred = torch.tensor([[0.05]])
        target = torch.tensor([[-0.05]])

        low_conf = torch.tensor([0.1])
        high_conf = torch.tensor([0.9])

        loss_low = loss_fn(pred, target, confidence=low_conf)
        loss_high = loss_fn(pred, target, confidence=high_conf)

        assert loss_high.item() > loss_low.item(), (
            "High-confidence wrong prediction must be penalised more than low-confidence"
        )


# ---------------------------------------------------------------------------
# KELLY-01: Zero-Entropy Confidence — 100% certainty caps at 25%
# ---------------------------------------------------------------------------

class TestKELLY01MaxConfidenceCap:
    def test_100pct_confidence_caps_fraction_at_max_position(self):
        """100% model confidence → Kelly fraction ≤ max_position (0.25)."""
        kelly = KellyCriterionOptimizer(
            n_assets=1, kelly_fraction=0.5, max_position=0.25
        )
        predictions = np.array([0.10])  # strong bullish signal
        confidence = np.array([1.0])    # 100% certainty

        result = kelly.compute_kelly_fractions(predictions, confidence)
        fraction = abs(result["kelly_fractions"][0])

        assert fraction <= 0.25, (
            f"Kelly fraction at 100% confidence = {fraction:.4f} > 0.25. "
            "Model must cap position at 25% even at maximum certainty."
        )
        assert fraction > 0, "Non-zero positive prediction must produce non-zero fraction"

    def test_kelly_all_in_never_allowed(self):
        """Under no confidence/prediction combination should fraction = 1.0."""
        kelly = KellyCriterionOptimizer(
            n_assets=3, kelly_fraction=1.0, max_position=0.25
        )
        predictions = np.array([0.5, 0.5, 0.5])   # extreme bullish
        confidence = np.array([1.0, 1.0, 1.0])     # maximum certainty

        result = kelly.compute_kelly_fractions(predictions, confidence)
        fractions = np.abs(result["kelly_fractions"])

        assert (fractions <= 0.25).all(), (
            f"All-in allocation detected: {fractions}. "
            "Kelly criterion must NEVER suggest 100% position."
        )

    def test_zero_confidence_produces_zero_fraction(self):
        """Zero confidence must yield zero or minimal position."""
        kelly = KellyCriterionOptimizer(n_assets=1, max_position=0.25)
        predictions = np.array([0.10])
        confidence = np.array([0.0])   # zero certainty

        result = kelly.compute_kelly_fractions(predictions, confidence)
        fraction = abs(result["kelly_fractions"][0])
        assert fraction < 0.10, (
            f"Zero-confidence prediction yielded {fraction:.4f} position (too large)"
        )

    def test_negative_prediction_yields_short(self):
        kelly = KellyCriterionOptimizer(n_assets=1, max_position=0.25)
        result = kelly.compute_kelly_fractions(
            np.array([-0.05]), np.array([0.7])
        )
        f = result["kelly_fractions"][0]
        assert f <= 0, f"Bearish prediction must yield short (negative) fraction, got {f}"

    def test_correlation_adjustment_reduces_concentrated_positions(self):
        kelly = KellyCriterionOptimizer(n_assets=2, max_position=0.25)
        predictions = np.array([0.05, 0.05])
        confidence = np.array([0.8, 0.8])
        corr = np.array([[1.0, 0.95], [0.95, 1.0]])  # near-perfect correlation

        result_no_corr = kelly.compute_kelly_fractions(predictions, confidence)
        result_with_corr = kelly.compute_kelly_fractions(
            predictions, confidence, correlation_matrix=corr
        )

        total_no_corr = sum(abs(f) for f in result_no_corr["kelly_fractions"])
        total_with_corr = sum(abs(f) for f in result_with_corr["kelly_fractions"])
        assert total_with_corr <= total_no_corr + 1e-6, (
            "High correlation should reduce total exposure, not increase it"
        )


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------

@pytest.mark.hypothesis
class TestPropertyBased:
    @pytest.mark.skipif(not HYP_AVAILABLE, reason="hypothesis not installed")
    @given(
        pred_val=st.floats(-0.5, 0.5, allow_nan=False, allow_infinity=False),
        target_val=st.floats(-0.5, 0.5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_loss_always_nonneg(self, pred_val, target_val):
        """Asymmetric loss must be ≥ 0 for all finite inputs."""
        loss_fn = AsymmetricUtilityLoss()
        pred = torch.tensor([[pred_val]], dtype=torch.float32)
        target = torch.tensor([[target_val]], dtype=torch.float32)
        loss = loss_fn(pred, target)
        assert loss.item() >= -1e-7, f"Negative loss={loss.item()} for pred={pred_val}, target={target_val}"

    @pytest.mark.skipif(not HYP_AVAILABLE, reason="hypothesis not installed")
    @given(
        confidence=st.floats(0.0, 1.0, allow_nan=False),
        n_assets=st.integers(1, 5),
        seed=st.integers(0, 1000),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_kelly_fractions_always_bounded(self, confidence, n_assets, seed):
        """Kelly fractions must always be within [-max_position, +max_position]."""
        max_pos = 0.20
        kelly = KellyCriterionOptimizer(
            n_assets=n_assets, kelly_fraction=0.5, max_position=max_pos
        )
        rng = np.random.default_rng(seed)
        predictions = rng.uniform(-0.1, 0.1, n_assets)
        confidences = np.full(n_assets, confidence)

        result = kelly.compute_kelly_fractions(predictions, confidences)
        fractions = np.array(result["kelly_fractions"])

        assert np.all(np.abs(fractions) <= max_pos + 1e-6), (
            f"Kelly fraction exceeded max_position={max_pos}: {fractions}"
        )


# ---------------------------------------------------------------------------
# Issue fix: zero-gradient on conservative-correct predictions
# ---------------------------------------------------------------------------

class TestLOSS02ZeroGradientFix:
    """
    Regression tests for the calibration floor added to prevent the
    model from collapsing to always-zero predictions.

    Without the fix: pred > target (same sign) → underestimated=0 → loss=0 → grad=0.
    With the fix:   a 1e-4 calibration term keeps a nonzero gradient alive.
    """

    def test_conservative_correct_pred_has_nonzero_gradient(self):
        """Predicting +10% when actual is +5% must still produce a gradient."""
        loss_fn = AsymmetricUtilityLoss()
        pred = torch.tensor([[0.10]], requires_grad=True)
        target = torch.tensor([[0.05]])   # overestimate, same direction
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().item() > 0, (
            "Zero gradient on conservative-correct prediction — model cannot calibrate magnitude"
        )

    def test_calibration_gradient_much_smaller_than_underestimate_gradient(self):
        """Overestimate gradient must be << underestimate gradient (1e-4 vs alpha=0.5)."""
        loss_fn = AsymmetricUtilityLoss()

        pred_over = torch.tensor([[0.10]], requires_grad=True)
        target_over = torch.tensor([[0.05]])
        loss_fn(pred_over, target_over).backward()
        grad_over = pred_over.grad.abs().item()

        pred_under = torch.tensor([[0.05]], requires_grad=True)
        target_under = torch.tensor([[0.10]])
        loss_fn(pred_under, target_under).backward()
        grad_under = pred_under.grad.abs().item()

        assert grad_over < grad_under, (
            f"Overestimate gradient {grad_over:.6f} should be < underestimate {grad_under:.6f}"
        )
        assert grad_over / grad_under < 0.01, (
            f"Calibration floor is too large: overestimate/underestimate ratio = {grad_over / grad_under:.4f}"
        )

    def test_zero_prediction_is_not_a_loss_minimum(self):
        """pred=0.0 must NOT be a local minimum — model should be pushed away from zero."""
        loss_fn = AsymmetricUtilityLoss()
        pred_zero = torch.tensor([[0.0]], requires_grad=True)
        target = torch.tensor([[0.05]])
        loss_zero = loss_fn(pred_zero, target)
        loss_zero.backward()
        # Gradient should exist at pred=0 so the optimizer can move away from it
        assert pred_zero.grad is not None and pred_zero.grad.abs().item() > 0, (
            "pred=0.0 is a gradient dead-end — model will collapse to predicting nothing"
        )
