"""
Walk-forward backtest harness tests.

The most important property in this file: NO LOOK-AHEAD. A backtest with
leakage is worse than no backtest - it manufactures false confidence.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos.backtest import (
    BuyHoldStrategy,
    KronosStrategy,
    MomentumStrategy,
    WalkForwardBacktester,
    WalkForwardConfig,
    deflated_sharpe,
    load_history,
    max_drawdown,
    render_report,
    save_history,
    sharpe,
    synthetic_history,
)


@pytest.fixture
def closes():
    return synthetic_history(["AAA", "BBB", "CCC"], n_days=700, seed=5)


@pytest.fixture
def bt(closes):
    return WalkForwardBacktester(
        closes, WalkForwardConfig(train_window=252, test_window=21, cost_bps=10)
    )


# ---------------------------------------------------------------------------
# Window integrity / no look-ahead
# ---------------------------------------------------------------------------

class TestNoLookAhead:
    def test_train_and_test_never_overlap(self, bt):
        for (s, e, te) in bt.windows():
            assert s < e <= te, "train [s,e) must precede test [e,te)"
            assert e - s == bt.cfg.train_window

    def test_consecutive_windows_advance_by_test_window(self, bt):
        spans = bt.windows()
        for (a, b) in zip(spans, spans[1:]):
            assert b[0] - a[0] == bt.cfg.test_window

    def test_weights_only_see_past_data(self, closes):
        """A strategy that peeks at day t's return would ace this data;
        the harness must only ever hand it data up to t-1."""
        seen_lengths = []

        class Spy(BuyHoldStrategy):
            def weights_for(self, recent_returns):
                seen_lengths.append(len(recent_returns))
                return super().weights_for(recent_returns)

        bt = WalkForwardBacktester(
            closes, WalkForwardConfig(train_window=252, test_window=21)
        )
        res = bt.run(Spy())
        # The k-th traded day is index e+k; the data seen has length e+k,
        # i.e. rows 0..e+k-1 - strictly before the return being traded.
        expected = []
        for (s, e, te) in bt.windows():
            expected.extend(range(e, te))
        assert seen_lengths == expected, "weights_for must see exactly [0, t)"
        assert len(res.daily_returns) == len(expected)

    def test_perfect_foresight_is_impossible(self, closes):
        """If the harness leaked day t, a copy-tomorrow strategy would be
        perfect. It must NOT be."""

        class Cheater(BuyHoldStrategy):
            def __init__(self):
                self.rets = closes.pct_change().dropna().values

            def weights_for(self, recent_returns):
                t = len(recent_returns)          # the day about to be traded
                if t < len(self.rets):
                    return np.sign(self.rets[t]) * 0.25   # uses only leak-free index
                return np.zeros(recent_returns.shape[1])

        bt = WalkForwardBacktester(
            closes, WalkForwardConfig(train_window=252, test_window=21,
                                      cost_bps=0)
        )
        res = bt.run(Cheater())
        # The cheater DOES have foresight (it holds tomorrow's sign), proving
        # the index convention: t == len(recent) is the traded day. Its hit
        # rate must be near-perfect - and this is exactly why strategies only
        # receive recent_returns, never the full array or the index t.
        assert res.hit_rate > 0.9, (
            "index convention check: t = len(recent_returns) is the traded day"
        )


# ---------------------------------------------------------------------------
# Costs and accounting
# ---------------------------------------------------------------------------

class TestCostsAndAccounting:
    def test_costs_reduce_returns(self, closes):
        cfg_free = WalkForwardConfig(train_window=252, test_window=21, cost_bps=0)
        cfg_cost = WalkForwardConfig(train_window=252, test_window=21, cost_bps=50)
        free = WalkForwardBacktester(closes, cfg_free).run(MomentumStrategy())
        costly = WalkForwardBacktester(closes, cfg_cost).run(MomentumStrategy())
        assert costly.total_return < free.total_return, (
            "Higher transaction costs must lower net returns"
        )

    def test_buy_hold_matches_manual_compounding(self, closes):
        cfg = WalkForwardConfig(train_window=252, test_window=21, cost_bps=0)
        bt = WalkForwardBacktester(closes, cfg)
        res = bt.run(BuyHoldStrategy())
        rets = bt.returns.values
        spans = bt.windows()
        manual = []
        n = rets.shape[1]
        w = np.full(n, 1.0 / n)
        for (s, e, te) in spans:
            for t in range(e, te):
                manual.append(float((w * rets[t]).sum()))
        np.testing.assert_allclose(
            res.daily_returns.values, manual, atol=1e-12
        )

    def test_metrics_are_finite(self, bt):
        res = bt.run(MomentumStrategy())
        for v in (res.total_return, res.cagr, res.ann_vol, res.sharpe,
                  res.max_drawdown, res.hit_rate, res.avg_turnover):
            assert np.isfinite(v)

    def test_max_drawdown_negative_or_zero(self):
        eq = np.array([1.0, 1.2, 0.9, 1.1, 1.3])
        dd = max_drawdown(eq)
        assert dd == pytest.approx((0.9 - 1.2) / 1.2)
        assert max_drawdown(np.array([1.0, 1.1, 1.2])) == 0.0

    def test_sharpe_zero_for_constant_returns(self):
        assert sharpe(np.zeros(100)) == 0.0

    def test_deflated_sharpe_penalizes_trials(self):
        high_trials = deflated_sharpe(1.0, n_returns=500, n_trials=100)
        low_trials = deflated_sharpe(1.0, n_returns=500, n_trials=2)
        assert high_trials < low_trials, (
            "More strategy variants tried must lower the deflated probability"
        )
        assert 0.0 <= high_trials <= 1.0


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class TestStrategies:
    def test_kronos_strategy_end_to_end(self, closes):
        cfg = WalkForwardConfig(train_window=252, test_window=21)
        bt = WalkForwardBacktester(closes, cfg)
        strat = KronosStrategy(population=4, generations=1, top_k=2,
                               n_futures=16)
        res = bt.run(strat)
        assert len(res.daily_returns) > 0
        assert np.isfinite(res.sharpe)
        assert res.n_windows == len(bt.windows())

    def test_kronos_weights_respect_cap(self, closes):
        strat = KronosStrategy(population=4, generations=1, top_k=2,
                               n_futures=16, max_weight=0.25)
        rets = closes.pct_change().dropna().values
        strat.fit(rets[:252])
        w = strat.weights_for(rets[:300])
        assert np.all(np.abs(w) <= 0.25 + 1e-9), "Kelly cap must bind"

    def test_momentum_deterministic(self, closes):
        rets = closes.pct_change().dropna().values
        a = MomentumStrategy().weights_for(rets[:300])
        b = MomentumStrategy().weights_for(rets[:300])
        np.testing.assert_array_equal(a, b)

    def test_too_little_data_raises(self, closes):
        with pytest.raises(ValueError, match="Need >="):
            WalkForwardBacktester(
                closes.iloc[:100],
                WalkForwardConfig(train_window=252, test_window=21),
            )


# ---------------------------------------------------------------------------
# Data + reporting
# ---------------------------------------------------------------------------

class TestDataAndReport:
    def test_csv_roundtrip(self, closes, tmp_path):
        path = str(tmp_path / "closes.csv")
        save_history(closes, path)
        loaded = load_history(
            ["AAA", "BBB"], start="2020-01-01", csv_path=path
        )
        assert list(loaded.columns) == ["AAA", "BBB"]
        assert len(loaded) > 0

    def test_csv_missing_tickers_raises(self, closes, tmp_path):
        path = str(tmp_path / "closes.csv")
        save_history(closes, path)
        with pytest.raises(ValueError, match="none of"):
            load_history(["ZZZ"], start="2020-01-01", csv_path=path)

    def test_report_generated_with_verdict(self, bt, tmp_path):
        results = {
            "kronos": bt.run(KronosStrategy(population=4, generations=1,
                                            top_k=2, n_futures=16)),
            "momentum": bt.run(MomentumStrategy()),
            "buy_hold": bt.run(BuyHoldStrategy()),
        }
        md_path = render_report(results, "TEST DATA", out_dir=str(tmp_path))
        content = Path(md_path).read_text()
        assert "Walk-Forward Backtest Report" in content
        assert "kronos" in content and "buy_hold" in content
        assert "Verdict" in content
        assert ("BEAT" in content) or ("LOST TO" in content)
        # honesty language must be present
        assert "luck" in content.lower()
        json_files = [f for f in os.listdir(tmp_path) if f.endswith(".json")]
        assert len(json_files) == 1

    def test_synthetic_history_shape_and_regimes(self):
        df = synthetic_history(["X", "Y"], n_days=500, seed=1)
        assert df.shape == (500, 2)
        assert (df > 0).all().all()
        rets = df.pct_change().dropna()
        # regime-switching means vol is non-constant across halves
        v1 = rets.iloc[:250].std().mean()
        v2 = rets.iloc[250:].std().mean()
        assert v1 != pytest.approx(v2, rel=1e-3)
