"""
Tests for kronos/risk_guard.py - the independent risk layer that must keep
working even when the ML pipeline (signals, evolution, checkpoints) is
producing garbage. Two things are tested at two levels:

1. RiskGuard in isolation: the pure daily-loss/drawdown/sanity-check math,
   and that halt state is a file (survives a fresh instance = a restart).
2. Wired into KronosOrchestrator: a real breach during run_reflex_tick
   actually flattens positions and blocks further trading, and a manual
   kill-switch file has the same immediate effect - no 24h veto delay.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos.config import load_config
from kronos.data_pipeline import DailyMemory
from kronos.orchestrator import KronosOrchestrator
from kronos.paper_trader import PaperTrader
from kronos.risk_guard import RiskGuard


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg.override("data.tickers", ["AAA", "BBB"])
    cfg.override("trading.db_path", str(tmp_path / "trades.db"))
    cfg.override("orchestrator.checkpoint_dir", str(tmp_path / "models"))
    cfg.override("orchestrator.report_dir", str(tmp_path / "reports"))
    cfg.override("orchestrator.veto_file", str(tmp_path / "veto.txt"))
    cfg.override("risk.halt_file", str(tmp_path / "risk_halt.flag"))
    cfg.override("risk.kill_switch_file", str(tmp_path / "KILL_SWITCH"))
    cfg.override("risk.max_daily_loss_pct", 0.05)
    cfg.override("risk.max_drawdown_pct", 0.20)
    cfg.override("risk.max_single_order_pct", 0.30)
    cfg.override("risk.max_price_deviation_pct", 0.20)
    return cfg


def make_memory(config, n_days: int = 10) -> DailyMemory:
    rng = np.random.default_rng(3)
    tickers = list(config.data.tickers)
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0, 0.001, (n_days, len(tickers))), axis=0)),
        index=dates, columns=tickers,
    )
    volumes = pd.DataFrame(
        rng.integers(10_000_000, 50_000_000, (n_days, len(tickers))).astype(float),
        index=dates, columns=tickers,
    )
    returns = prices.pct_change().fillna(0.0)
    vix = pd.Series(rng.uniform(15, 20, n_days), index=dates, name="VIX")
    return DailyMemory(
        as_of=datetime.now(timezone.utc),
        prices=prices, volumes=volumes, returns=returns, vix=vix,
        sentiment={t: 0.0 for t in tickers},
        macro={"vix_last": float(vix.iloc[-1]), "vix_mean_20d": float(vix.mean()),
               "market_return_1d": 0.0, "market_vol_20d": 0.01},
        source_used="synthetic",
    )


# ---------------------------------------------------------------------------
# RiskGuard in isolation
# ---------------------------------------------------------------------------

class TestRiskGuardChecks:
    def test_no_breach_when_flat(self, config):
        guard = RiskGuard(config)
        trader = PaperTrader(config)
        assert guard.check(trader) is None
        trader.close()

    def test_daily_loss_breach_detected(self, config):
        guard = RiskGuard(config)
        trader = PaperTrader(config)
        trader.execute(day=1, ticker="AAA", target_weight=0.5,
                       price=100.0, bar_volume=50_000_000)
        # mark the position down > 5% of starting equity
        trader.last_prices["AAA"] = 80.0
        reason = guard.check(trader)
        assert reason is not None and "daily loss" in reason
        trader.close()

    def test_drawdown_breach_detected_against_historical_peak(self, config):
        """A cumulative decline from the campaign's all-time peak past the
        20% limit must be caught even when the MOST RECENT day's own loss
        (4.76%, from $105k to $100k) is comfortably under the 5%
        daily-loss limit - drawdown is a distinct check from daily loss,
        not a special case of it. Manipulates trader state directly
        (rather than via execute(), which clips any single position to
        the 25% Kelly cap) to isolate check_drawdown's own math."""
        guard = RiskGuard(config)
        trader = PaperTrader(config)
        trader._equity_history = [100_000.0, 130_000.0, 125_000.0, 120_000.0,
                                   115_000.0, 110_000.0, 105_000.0, 100_000.0]
        trader.cash = 100_000.0
        trader.positions = {}
        reason = guard.check(trader)
        assert reason is not None and "drawdown" in reason
        trader.close()

    def test_kill_switch_file_reports_breach(self, config, tmp_path):
        guard = RiskGuard(config)
        trader = PaperTrader(config)
        assert guard.check(trader) is None
        Path(config.risk.kill_switch_file).write_text("stop")
        reason = guard.check(trader)
        assert reason is not None and "KILL_SWITCH" in reason
        trader.close()

    def test_disabled_guard_never_breaches(self, config):
        cfg = config
        cfg.override("risk.enabled", False)
        guard = RiskGuard(cfg)
        trader = PaperTrader(cfg)
        trader.execute(day=1, ticker="AAA", target_weight=0.5,
                       price=100.0, bar_volume=50_000_000)
        trader.last_prices["AAA"] = 1.0   # catastrophic - would breach if enabled
        assert guard.check(trader) is None
        trader.close()


class TestRiskGuardSanityCheck:
    def test_price_deviation_rejected(self, config):
        guard = RiskGuard(config)
        reason = guard.sanity_check_order("AAA", price=150.0, last_price=100.0,
                                          target_weight=0.1)
        assert reason is not None and "deviates" in reason

    def test_price_within_tolerance_accepted(self, config):
        guard = RiskGuard(config)
        reason = guard.sanity_check_order("AAA", price=105.0, last_price=100.0,
                                          target_weight=0.1)
        assert reason is None

    def test_oversized_weight_rejected(self, config):
        guard = RiskGuard(config)
        reason = guard.sanity_check_order("AAA", price=100.0, last_price=100.0,
                                          target_weight=0.9)
        assert reason is not None and "max_single_order_pct" in reason

    def test_no_last_price_skips_deviation_check(self, config):
        """First-ever quote for a ticker has nothing to compare against -
        must not be treated as an infinite deviation."""
        guard = RiskGuard(config)
        reason = guard.sanity_check_order("AAA", price=100.0, last_price=None,
                                          target_weight=0.1)
        assert reason is None


class TestRiskGuardHaltPersistence:
    def test_trip_creates_halt_file_with_reason(self, config):
        guard = RiskGuard(config)
        assert not guard.halted
        guard.trip("test breach")
        assert guard.halted
        assert "test breach" in guard.halt_reason

    def test_halt_survives_a_fresh_instance(self, config):
        """A fresh RiskGuard pointed at the same halt_file (simulating a
        service restart) must still see the halt - state lives in the
        file, not memory, precisely so a restart can't silently clear a
        real risk trip."""
        first = RiskGuard(config)
        first.trip("breach before restart")
        second = RiskGuard(config)
        assert second.halted
        assert "breach before restart" in second.halt_reason

    def test_second_trip_does_not_overwrite_original_reason(self, config):
        guard = RiskGuard(config)
        guard.trip("first reason")
        guard.trip("second reason")
        assert "first reason" in guard.halt_reason
        assert "second reason" not in guard.halt_reason

    def test_manual_reset_by_deleting_halt_file(self, config):
        guard = RiskGuard(config)
        guard.trip("breach")
        assert guard.halted
        os.remove(guard.halt_file)
        assert not guard.halted


# ---------------------------------------------------------------------------
# Wired into KronosOrchestrator
# ---------------------------------------------------------------------------

class _LongDecision:
    """A deterministic go-long-both-tickers decision, so the test controls
    position direction instead of depending on an untrained model's
    effectively-random signal sign."""
    signals = np.array([1.0, 1.0])
    position_cap = 1.0
    asset_caps = {}


class TestOrchestratorRiskEnforcement:
    def test_reflex_tick_auto_flattens_and_halts_on_daily_loss_breach(self, config, monkeypatch):
        orch = KronosOrchestrator(config)
        orch.state.day = 1
        orch.state.memory = make_memory(config)
        monkeypatch.setattr(orch.reflex, "infer", lambda *a, **kw: _LongDecision())

        # establish a long position via a normal-looking tick
        orch.run_reflex_tick(18.0, {"AAA": 100.0, "BBB": 50.0},
                             {"AAA": 20_000_000, "BBB": 20_000_000})
        assert orch.trader.positions.get("AAA", 0.0) > 0.0 \
            or orch.trader.positions.get("BBB", 0.0) > 0.0

        # crash the marks far enough to blow the daily-loss limit
        crashed = {"AAA": 40.0, "BBB": 20.0}
        orch.run_reflex_tick(18.0, crashed,
                             {"AAA": 20_000_000, "BBB": 20_000_000})

        assert orch.risk_guard.halted
        assert orch.trader.positions.get("AAA", 0.0) == 0.0
        assert orch.trader.positions.get("BBB", 0.0) == 0.0

        # a further tick must not open any new position while halted
        orch.run_reflex_tick(18.0, crashed,
                             {"AAA": 20_000_000, "BBB": 20_000_000})
        assert orch.trader.positions.get("AAA", 0.0) == 0.0
        assert orch.trader.positions.get("BBB", 0.0) == 0.0

    def test_halt_survives_a_simulated_restart(self, config, monkeypatch):
        orch = KronosOrchestrator(config)
        orch.state.day = 1
        orch.state.memory = make_memory(config)
        monkeypatch.setattr(orch.reflex, "infer", lambda *a, **kw: _LongDecision())
        orch.run_reflex_tick(18.0, {"AAA": 100.0, "BBB": 50.0},
                             {"AAA": 20_000_000, "BBB": 20_000_000})
        orch.run_reflex_tick(18.0, {"AAA": 40.0, "BBB": 20.0},
                             {"AAA": 20_000_000, "BBB": 20_000_000})
        assert orch.risk_guard.halted

        second = KronosOrchestrator(config)   # simulates a process restart
        assert second.risk_guard.halted

    def test_kill_switch_file_halts_immediately_with_no_prior_breach(self, config):
        orch = KronosOrchestrator(config)
        orch.state.day = 1
        orch.state.memory = make_memory(config)
        Path(config.risk.kill_switch_file).write_text("operator stop")

        orch.run_reflex_tick(18.0, {"AAA": 100.0, "BBB": 50.0},
                             {"AAA": 20_000_000, "BBB": 20_000_000})

        assert orch.risk_guard.halted
        assert "KILL_SWITCH" in orch.risk_guard.halt_reason
        assert orch.trader.positions.get("AAA", 0.0) == 0.0
        assert orch.trader.positions.get("BBB", 0.0) == 0.0

    def test_oversized_target_weight_order_rejected_not_executed(self, config, monkeypatch):
        """A signal-computation bug that produces a too-large target
        weight must be rejected by the sanity check before it ever
        reaches PaperTrader.execute() - independent of execute()'s own
        Kelly-cap clip."""
        orch = KronosOrchestrator(config)
        orch.state.day = 1
        orch.state.memory = make_memory(config)

        class FakeDecision:
            signals = np.array([100.0, 0.0])  # absurd - would clip to 0.9 weight
            position_cap = 1.0
            asset_caps = {}

        monkeypatch.setattr(orch.reflex, "infer", lambda *a, **kw: FakeDecision())
        orch.run_reflex_tick(18.0, {"AAA": 100.0, "BBB": 50.0},
                             {"AAA": 20_000_000, "BBB": 20_000_000})
        assert orch.trader.positions.get("AAA", 0.0) == 0.0
