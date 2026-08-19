"""
Tests for kronos/backtest.py's signal-direction diagnostic
(WalkForwardBacktester.diagnose_signal_direction, SignalDiagnostic,
render_signal_diagnostic_report).

This diagnostic exists to answer one specific question raised by the real
walk-forward backtest results: kronos's Hit Rate (fraction of trading days
with positive NET portfolio PnL) was ~47-48%, below a coin flip. That
metric conflates position sizing, transaction costs, and cross-asset
netting - it doesn't by itself say whether the model's raw predicted
DIRECTION is wrong. This diagnostic strips all of that away.

Before trusting it on the real, ambiguous question, these tests validate
the diagnostic ON KNOWN GROUND TRUTH: synthetic strategies with a
deliberately-constructed correct, inverted, or absent signal, checked
against synthetic data with a known momentum structure. If the tool can't
correctly identify a strategy that's right by construction, wrong by
construction, or a coin flip by construction, it can't be trusted on the
real ambiguous case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos.backtest import (
    KronosStrategy,
    Strategy,
    WalkForwardBacktester,
    WalkForwardConfig,
    _compute_signal_diagnostic,
    _hit_rate,
    render_signal_diagnostic_report,
)


def momentum_history(n_days: int = 500, n_assets: int = 2, rho: float = 0.6,
                     seed: int = 3):
    """Synthetic returns with KNOWN positive autocorrelation:
    rets[t] = rho * rets[t-1] + noise. A strategy that predicts
    sign(rets[t-1]) is correct by construction more often than not, at a
    rate that grows with rho - this is ground truth, not an assumption."""
    rng = np.random.default_rng(seed)
    rets = np.zeros((n_days, n_assets))
    rets[0] = rng.normal(0, 0.01, n_assets)
    for t in range(1, n_days):
        rets[t] = rho * rets[t - 1] + rng.normal(0, 0.01, n_assets)
    import pandas as pd
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"T{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=cols)


class FollowMomentumStrategy(Strategy):
    """Predicts sign(yesterday's return) - correct by construction on
    momentum_history()'s positively-autocorrelated synthetic data."""
    name = "follow_momentum"

    def weights_for(self, recent_returns: np.ndarray) -> np.ndarray:
        return np.sign(recent_returns[-1]) * 0.1


class AntiMomentumStrategy(Strategy):
    """The exact opposite call every time - wrong by construction on the
    same data."""
    name = "anti_momentum"

    def weights_for(self, recent_returns: np.ndarray) -> np.ndarray:
        return -np.sign(recent_returns[-1]) * 0.1


class CoinFlipStrategy(Strategy):
    """A fixed-seed random sign every call - no relationship to the data
    at all, by construction."""
    name = "coin_flip"

    def __init__(self, seed=99):
        self.rng = np.random.default_rng(seed)

    def weights_for(self, recent_returns: np.ndarray) -> np.ndarray:
        n = recent_returns.shape[1]
        return self.rng.choice([-1.0, 1.0], size=n) * 0.1


class AbstainStrategy(Strategy):
    """Never makes a directional call at all."""
    name = "abstain"

    def weights_for(self, recent_returns: np.ndarray) -> np.ndarray:
        return np.zeros(recent_returns.shape[1])


@pytest.fixture
def momentum_closes():
    return momentum_history(n_days=500, n_assets=2, rho=0.6)


@pytest.fixture
def bt(momentum_closes):
    return WalkForwardBacktester(
        momentum_closes, WalkForwardConfig(train_window=100, test_window=20)
    )


# ---------------------------------------------------------------------------
# Ground-truth validation: does the diagnostic detect what's true by
# construction?
# ---------------------------------------------------------------------------

class TestGroundTruthDetection:
    def test_correct_signal_shows_significant_positive_hit_rate(self, bt):
        diag = bt.diagnose_signal_direction(FollowMomentumStrategy())
        assert diag.hit_rate > 0.5
        assert diag.hit_rate_p_value < 0.05, (
            "a signal that's correct by construction on strongly "
            "autocorrelated data must be statistically significant"
        )
        assert diag.pearson_r > 0
        assert diag.spearman_r > 0

    def test_inverted_signal_shows_significant_negative_hit_rate(self, bt):
        diag = bt.diagnose_signal_direction(AntiMomentumStrategy())
        assert diag.hit_rate < 0.5
        assert diag.hit_rate_p_value < 0.05
        assert diag.pearson_r < 0
        assert diag.spearman_r < 0

    def test_inverting_the_wrong_signal_recovers_the_right_one(self, bt):
        """The core actionable claim the diagnostic makes: if hit rate is
        significantly BELOW 50%, hit_rate_if_sign_inverted should recover
        (approximately) what the correct-by-construction strategy gets."""
        anti = bt.diagnose_signal_direction(AntiMomentumStrategy())
        correct = bt.diagnose_signal_direction(FollowMomentumStrategy())
        assert anti.hit_rate_if_sign_inverted == pytest.approx(
            correct.hit_rate, abs=1e-9
        )

    def test_coin_flip_signal_is_not_statistically_significant(self, bt):
        diag = bt.diagnose_signal_direction(CoinFlipStrategy())
        assert diag.hit_rate_p_value >= 0.05, (
            "a signal with no relationship to the data must NOT be "
            "reported as significant - a real no-edge result looks like "
            "this, not like a below-50% hit rate"
        )

    def test_hit_rate_near_fifty_for_coin_flip(self, bt):
        diag = bt.diagnose_signal_direction(CoinFlipStrategy())
        assert 0.35 < diag.hit_rate < 0.65   # loose band, still a sanity bound

    def test_abstaining_strategy_has_zero_calls(self, bt):
        diag = bt.diagnose_signal_direction(AbstainStrategy())
        assert diag.n_calls == 0
        assert diag.hit_rate == 0.0

    def test_per_ticker_breakdown_present_for_every_ticker(self, bt):
        diag = bt.diagnose_signal_direction(FollowMomentumStrategy())
        assert set(diag.per_ticker.keys()) == {"T0", "T1"}
        for stats in diag.per_ticker.values():
            assert stats.n_obs > 0


class TestHitRateHelper:
    def test_excludes_zero_predictions_as_ties(self):
        rate, n_calls, n_hits = _hit_rate([1.0, 0.0, -1.0], [1.0, 5.0, 1.0])
        assert n_calls == 2   # the 0.0 prediction is excluded
        assert n_hits == 1    # +1 vs +1 hits, -1 vs +1 misses
        assert rate == pytest.approx(0.5)

    def test_all_hits(self):
        rate, n_calls, n_hits = _hit_rate([1.0, -1.0, 1.0], [2.0, -3.0, 0.5])
        assert rate == pytest.approx(1.0)

    def test_no_observations_returns_zero_not_raise(self):
        rate, n_calls, n_hits = _hit_rate([], [])
        assert rate == 0.0
        assert n_calls == 0


class TestComputeSignalDiagnosticDirectly:
    def test_perfectly_correlated_data(self):
        preds = [1.0, 2.0, 3.0, -1.0, -2.0]
        actuals = [1.0, 2.0, 3.0, -1.0, -2.0]
        diag = _compute_signal_diagnostic(
            "x", preds, actuals, {"T": preds}, {"T": actuals},
        )
        assert diag.hit_rate == pytest.approx(1.0)
        assert diag.pearson_r == pytest.approx(1.0, abs=1e-6)
        assert diag.spearman_r == pytest.approx(1.0, abs=1e-6)

    def test_perfectly_anti_correlated_data(self):
        preds = [1.0, 2.0, 3.0, -1.0, -2.0]
        actuals = [-1.0, -2.0, -3.0, 1.0, 2.0]
        diag = _compute_signal_diagnostic(
            "x", preds, actuals, {"T": preds}, {"T": actuals},
        )
        assert diag.hit_rate == pytest.approx(0.0)
        assert diag.hit_rate_if_sign_inverted == pytest.approx(1.0)
        assert diag.pearson_r == pytest.approx(-1.0, abs=1e-6)


class TestReportRendering:
    def test_significant_negative_verdict_mentions_sign_bug(self, bt):
        diag = bt.diagnose_signal_direction(AntiMomentumStrategy())
        report = render_signal_diagnostic_report(diag)
        assert "sign" in report.lower()
        assert f"{diag.hit_rate_if_sign_inverted:.1%}" in report

    def test_insignificant_verdict_says_no_detectable_signal(self, bt):
        diag = bt.diagnose_signal_direction(CoinFlipStrategy())
        report = render_signal_diagnostic_report(diag)
        assert "NOT statistically distinguishable" in report

    def test_zero_calls_handled_without_raising(self, bt):
        diag = bt.diagnose_signal_direction(AbstainStrategy())
        report = render_signal_diagnostic_report(diag)   # must not raise
        assert "never made a directional call" in report


# ---------------------------------------------------------------------------
# Real (small) KronosStrategy wiring - structural only, no edge assumed
# ---------------------------------------------------------------------------

class TestKronosStrategyWiring:
    def test_runs_end_to_end_without_raising(self, bt):
        strat = KronosStrategy(population=4, generations=1, top_k=2, n_futures=16)
        diag = bt.diagnose_signal_direction(strat)
        assert diag.n_obs > 0
        assert 0.0 <= diag.hit_rate <= 1.0
        assert np.isfinite(diag.pearson_r)
        assert np.isfinite(diag.spearman_r)
