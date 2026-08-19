"""
Project Kronos test suite.

Covers the 7 required cases:
  1. test_orchestrator_state_transition
  2. test_data_pipeline_fallback
  3. test_nightmare_variance
  4. test_evolver_population
  5. test_reflex_lockout
  6. test_paper_trader_slippage
  7. test_end_to_end_24h_simulation
plus supporting edge cases (veto delay, Kelly cap, retry-once, ensemble).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos.config import load_config
from kronos.data_pipeline import (
    DailyMemory,
    DataPipeline,
    SourceError,
    YFinanceSource,
)
from kronos.evolver import KronosEvolver, WeightedEnsemble
from kronos.nightmare_generator import NightmareGenerator
from kronos.orchestrator import KronosOrchestrator, Phase
from kronos.paper_trader import PaperTrader
from kronos.reflex import RegimeSwitchGate, ReflexArc
from kronos.warmer import KronosWarmer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    # Shrink everything for test speed; redirect all writes into tmp
    cfg.override("data.tickers", ["AAA", "BBB", "CCC"])
    cfg.override("nightmare.n_futures", 64)
    cfg.override("nightmare.batch_size", 32)
    cfg.override("nightmare.diffusion_steps", 4)
    cfg.override("nightmare.horizon_days", 5)
    cfg.override("evolution.population_size", 20)
    cfg.override("evolution.n_generations", 1)
    cfg.override("evolution.top_k", 5)
    cfg.override("trading.db_path", str(tmp_path / "trades.db"))
    cfg.override("orchestrator.checkpoint_dir", str(tmp_path / "models"))
    cfg.override("orchestrator.report_dir", str(tmp_path / "reports"))
    cfg.override("orchestrator.veto_file", str(tmp_path / "veto.txt"))
    cfg.override("risk.halt_file", str(tmp_path / "risk_halt.flag"))
    cfg.override("risk.kill_switch_file", str(tmp_path / "KILL_SWITCH"))
    return cfg


def make_memory(config, n_days: int = 30, seed: int = 7) -> DailyMemory:
    """Synthetic DailyMemory - no network required."""
    rng = np.random.default_rng(seed)
    tickers = list(config.data.tickers)
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (n_days, len(tickers))), axis=0)),
        index=dates, columns=tickers,
    )
    volumes = pd.DataFrame(
        rng.integers(1_000_000, 50_000_000, (n_days, len(tickers))).astype(float),
        index=dates, columns=tickers,
    )
    returns = prices.pct_change().fillna(0.0)
    vix = pd.Series(rng.uniform(15, 25, n_days), index=dates, name="VIX")
    return DailyMemory(
        as_of=datetime.now(timezone.utc),
        prices=prices, volumes=volumes, returns=returns, vix=vix,
        sentiment={t: 0.0 for t in tickers},
        macro={"vix_last": float(vix.iloc[-1]),
               "vix_mean_20d": float(vix.tail(20).mean()),
               "market_return_1d": float(returns.iloc[-1].mean()),
               "market_vol_20d": float(returns.tail(20).std().mean())},
        source_used="synthetic",
    )


@pytest.fixture
def memory(config):
    return make_memory(config)


# ---------------------------------------------------------------------------
# 1. Orchestrator state transitions
# ---------------------------------------------------------------------------

class TestOrchestratorStateTransition:
    def test_orchestrator_state_transition(self, config):
        """Phases must switch at the configured simulated times."""
        orch = KronosOrchestrator(config)
        base = datetime(2026, 3, 2)  # a Monday

        expectations = [
            (base.replace(hour=0, minute=30), Phase.DIGESTION),
            (base.replace(hour=2, minute=30), Phase.NIGHTMARE),
            (base.replace(hour=4, minute=15), Phase.EVOLUTION),
            (base.replace(hour=5, minute=30), Phase.ADAPTATION),
            (base.replace(hour=6, minute=5), Phase.REPORT),
            (base.replace(hour=9, minute=45), Phase.REFLEX),
            (base.replace(hour=12, minute=0), Phase.REFLEX),
            (base.replace(hour=16, minute=1), Phase.LOGGING),
            (base.replace(hour=23, minute=59), Phase.LOGGING),
        ]
        for when, expected in expectations:
            got = orch.phase_for(when)
            assert got == expected, (
                f"At {when.time()} expected {expected.value}, got {got.value}"
            )

    def test_nightmare_to_neat_to_maml_ordering(self, config):
        """Sequence integrity: NIGHTMARE < EVOLUTION < ADAPTATION in the day."""
        orch = KronosOrchestrator(config)
        base = datetime(2026, 3, 2)
        t_nightmare = orch.phase_for(base.replace(hour=3))
        t_neat = orch.phase_for(base.replace(hour=4, minute=30))
        t_maml = orch.phase_for(base.replace(hour=5, minute=30))
        assert (t_nightmare, t_neat, t_maml) == (
            Phase.NIGHTMARE, Phase.EVOLUTION, Phase.ADAPTATION
        )


# ---------------------------------------------------------------------------
# 2. Data pipeline fallback
# ---------------------------------------------------------------------------

class TestDataPipelineFallback:
    def test_data_pipeline_fallback(self, config, memory):
        """If the primary source fails, the next source wins - no crash."""
        config.override("data.sources", ["yfinance", "polygon", "alphavantage"])
        pipeline = DataPipeline(config)

        # Simulate: yfinance fails, polygon succeeds
        good_frame = pd.concat(
            {"Close": memory.prices, "Volume": memory.volumes}, axis=1
        )
        frames = {"polygon": good_frame}   # yfinance absent = it failed
        name, frame, flags = pipeline.cross_validate(frames)
        assert name == "polygon", "Fallback source must be selected"

    def test_all_sources_failed_raises(self, config):
        pipeline = DataPipeline(config)
        with pytest.raises(SourceError, match="ALL data sources failed"):
            pipeline.cross_validate({})

    def test_yfinance_source_error_on_missing_lib(self, config):
        src = YFinanceSource()
        with patch.dict(sys.modules, {"yfinance": None}):
            with pytest.raises((SourceError, ImportError)):
                src.fetch(["SPY"], 5)

    def test_kalman_repairs_missing_values(self, config, memory):
        pipeline = DataPipeline(config)
        frame = pd.concat(
            {"Close": memory.prices.copy(), "Volume": memory.volumes}, axis=1
        )
        # poke holes in one ticker
        frame.loc[frame.index[5:8], ("Close", "AAA")] = np.nan
        closes, volumes, flags = pipeline._clean(frame)
        assert not closes["AAA"].isna().any(), "Kalman must repair NaNs"
        assert any("kalman_repaired:AAA" in f for f in flags)


# ---------------------------------------------------------------------------
# 3. Nightmare variance
# ---------------------------------------------------------------------------

class TestNightmareVariance:
    def test_nightmare_variance(self, config, memory):
        """Generated futures must not be identical copies (variance > 0)."""
        gen = NightmareGenerator(config)
        buffer = gen.generate(memory, n_futures=32)
        assert buffer.variance > 0.0, "Nightmare futures collapsed to identical paths"
        assert buffer.n_futures == 32

    def test_worst_slice_is_worst(self, config, memory):
        gen = NightmareGenerator(config)
        buffer = gen.generate(memory, n_futures=32)
        worst = buffer.worst(5)
        assert worst.shape[0] == 5
        # The mean P&L of the worst-5 must be <= overall mean
        weights = torch.full((buffer.futures.shape[-1],), 1.0 / buffer.futures.shape[-1])
        worst_pnl = (worst * weights).sum(dim=(1, 2)).mean()
        all_pnl = buffer.portfolio_pnl.mean()
        assert worst_pnl <= all_pnl + 1e-6


# ---------------------------------------------------------------------------
# 4. Evolver population
# ---------------------------------------------------------------------------

class TestEvolverPopulation:
    def test_evolver_population(self, config, memory):
        """Exactly 20 variants spawned; exactly 5 combined into the master."""
        evolver = KronosEvolver(config)
        variants = evolver.spawn_variants()
        assert len(variants) == 20, f"Expected 20 variants, got {len(variants)}"

        gen = NightmareGenerator(config)
        buffer = gen.generate(memory, n_futures=32)
        result = evolver.evolve(buffer)
        assert len(result.top_genomes) == 5, (
            f"Expected top-5 selection, got {len(result.top_genomes)}"
        )
        assert isinstance(result.master_model, WeightedEnsemble)
        assert len(result.master_model.members) == 5

    def test_ensemble_weights_sum_to_one(self, config, memory):
        evolver = KronosEvolver(config)
        gen = NightmareGenerator(config)
        buffer = gen.generate(memory, n_futures=32)
        result = evolver.evolve(buffer)
        total = float(result.master_model.weights.sum())
        assert abs(total - 1.0) < 1e-5

    def test_degraded_mode_shrinks_population(self, config, memory):
        evolver = KronosEvolver(config)
        variants = evolver.spawn_variants(degraded=True)
        assert len(variants) == config.evolution.fallback_population_size

    def test_master_model_forward_pass(self, config, memory):
        evolver = KronosEvolver(config)
        gen = NightmareGenerator(config)
        buffer = gen.generate(memory, n_futures=32)
        result = evolver.evolve(buffer)
        x = torch.randn(4, evolver.input_dim)
        out = result.master_model(x)
        assert out.shape == (4, evolver.output_dim)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# 5. Reflex lockout
# ---------------------------------------------------------------------------

class TestReflexLockout:
    def test_reflex_lockout(self, config):
        """VIX spike > 2 std must set position_cap = 0.0 for 30 minutes."""
        gate = RegimeSwitchGate(config)
        now = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)

        # Build a calm baseline
        for i in range(15):
            gate.update(20.0 + 0.1 * (i % 3), now=now + timedelta(seconds=i))

        # Spike far beyond 2 sigma
        state = gate.update(45.0, now=now + timedelta(minutes=1))
        assert state.position_cap == 0.0, "Spike must zero the position cap"
        assert state.regime == "panic"

        # 29 minutes later: still locked
        state = gate.update(20.0, now=now + timedelta(minutes=30))
        assert not gate.allows_new_longs(now=now + timedelta(minutes=30))

        # 31+ minutes after the spike: lockout expired
        state = gate.update(20.0, now=now + timedelta(minutes=32))
        assert state.position_cap == 1.0, "Lockout must expire after 30 minutes"
        assert gate.allows_new_longs(now=now + timedelta(minutes=32))

    def test_reflex_gate_kills_long_signals(self, config, memory):
        arc = ReflexArc(config)
        now = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)
        recent = memory.returns_window(5)
        for i in range(15):
            arc.gate.update(20.0, now=now + timedelta(seconds=i))
        decision = arc.infer(recent, vix_value=45.0, now=now + timedelta(minutes=1))
        assert decision.position_cap == 0.0
        assert (decision.signals <= 0.0).all(), (
            "In panic, all long (positive) signals must be suppressed"
        )

    def test_calm_market_no_lockout(self, config):
        gate = RegimeSwitchGate(config)
        now = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)
        for i in range(30):
            state = gate.update(20.0 + 0.2 * np.sin(i), now=now + timedelta(seconds=i))
        assert state.position_cap == 1.0
        assert state.regime == "calm"


# ---------------------------------------------------------------------------
# 6. Paper trader slippage
# ---------------------------------------------------------------------------

class TestPaperTraderSlippage:
    def test_paper_trader_slippage(self, config):
        """A $100k trade on a low-liquidity stock incurs 0.5% slippage."""
        trader = PaperTrader(config)
        low_liquidity_volume = 50_000   # below mid threshold of 1M

        fill = trader.execute(
            day=1, ticker="THIN", target_weight=0.25,
            price=100.0, bar_volume=low_liquidity_volume,
        )
        assert fill is not None
        assert abs(fill.slippage_pct - 0.5) < 1e-9, (
            f"Low-liquidity slippage must be 0.5%, got {fill.slippage_pct}%"
        )
        assert fill.fill_price == pytest.approx(100.0 * 1.005), (
            "Buy fill must be signal price + 0.5%"
        )
        trader.close()

    def test_high_liquidity_slippage_smaller(self, config):
        trader = PaperTrader(config)
        fill = trader.execute(
            day=1, ticker="SPY", target_weight=0.20,
            price=500.0, bar_volume=50_000_000,
        )
        assert fill is not None
        assert fill.slippage_pct == pytest.approx(0.05)
        trader.close()

    def test_kelly_cap_enforced(self, config):
        """No trade may exceed max_position_pct of equity."""
        trader = PaperTrader(config)
        trader.execute(
            day=1, ticker="AAA", target_weight=0.90,   # tries to blow the cap
            price=100.0, bar_volume=50_000_000,
        )
        assert trader.position_pct("AAA") <= config.trading.max_position_pct + 0.01
        trader.close()

    def test_trades_persisted_to_sqlite(self, config):
        trader = PaperTrader(config)
        trader.execute(day=1, ticker="AAA", target_weight=0.10,
                       price=50.0, bar_volume=5_000_000)
        rows = trader._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        assert rows == 1
        stats = trader.close_day(1, {"AAA": 51.0})
        assert stats["n_trades"] == 1
        assert stats["equity"] > 0
        trader.close()

    def test_sell_side_slippage_negative(self, config):
        trader = PaperTrader(config)
        trader.execute(day=1, ticker="AAA", target_weight=0.20,
                       price=100.0, bar_volume=50_000_000)
        fill = trader.execute(day=1, ticker="AAA", target_weight=0.0,
                              price=100.0, bar_volume=50_000_000)
        assert fill is not None and fill.side == "sell"
        assert fill.fill_price < 100.0, "Sell fills must be below signal price"
        trader.close()


class TestPaperTraderResume:
    """A fresh PaperTrader(config) pointed at the same db_path as a prior
    one must pick up where that prior one left off - not silently start a
    new initial_capital account. This is what makes a multi-month paper
    campaign survive a service restart (every deploy restarts the
    process) instead of resetting to day 1 every time."""

    def test_fresh_db_starts_clean(self, config):
        """No prior history - a fresh account is the CORRECT behavior,
        not a bug. Confirms resume_from_db() doesn't invent state."""
        trader = PaperTrader(config)
        assert trader.cash == config.trading.initial_capital
        assert trader.positions == {}
        assert trader._equity_history == [config.trading.initial_capital]
        trader.close()

    def test_cash_and_positions_survive_a_restart(self, config):
        first = PaperTrader(config)
        first.execute(day=1, ticker="AAA", target_weight=0.20,
                     price=100.0, bar_volume=50_000_000)
        first.execute(day=1, ticker="BBB", target_weight=0.10,
                     price=50.0, bar_volume=50_000_000)
        cash_before = first.cash
        positions_before = dict(first.positions)
        first.close()

        second = PaperTrader(config)   # simulates a process restart
        assert second.cash == pytest.approx(cash_before)
        assert second.positions == pytest.approx(positions_before)
        second.close()

    def test_equity_history_survives_a_restart(self, config):
        first = PaperTrader(config)
        first.execute(day=1, ticker="AAA", target_weight=0.10,
                     price=100.0, bar_volume=50_000_000)
        first.close_day(1, {"AAA": 105.0})
        first.close_day(2, {"AAA": 103.0})
        first.close()

        second = PaperTrader(config)
        # one entry per closed day persisted to daily_performance - no
        # synthetic leading initial_capital marker, since the persisted
        # rows already fully capture the real equity progression.
        assert len(second._equity_history) == 2
        second.close()

    def test_written_off_position_not_revived_on_resume(self, config):
        first = PaperTrader(config)
        first.execute(day=1, ticker="AAA", target_weight=0.20,
                     price=100.0, bar_volume=50_000_000)
        first.write_off(day=1, ticker="AAA", price=0.0005)
        first.close()

        second = PaperTrader(config)
        assert "AAA" not in second.positions
        second.close()

    def test_second_position_untouched_by_first_ticker_trading_after(self, config):
        """positions must be reconstructed PER TICKER from that ticker's
        own most recent trade, not just the single globally-latest trade
        row - otherwise a ticker traded earlier (but not most recently)
        would incorrectly disappear on resume."""
        first = PaperTrader(config)
        first.execute(day=1, ticker="AAA", target_weight=0.20,
                     price=100.0, bar_volume=50_000_000)
        first.execute(day=1, ticker="BBB", target_weight=0.10,
                     price=50.0, bar_volume=50_000_000)
        # AAA trades again - the most recent row overall is now AAA's
        first.execute(day=1, ticker="AAA", target_weight=0.15,
                     price=101.0, bar_volume=50_000_000)
        aaa_before = first.positions["AAA"]
        bbb_before = first.positions["BBB"]
        first.close()

        second = PaperTrader(config)
        assert second.positions["AAA"] == pytest.approx(aaa_before)
        assert second.positions["BBB"] == pytest.approx(bbb_before)
        second.close()

    def test_old_schema_db_without_cash_after_falls_back_to_fresh(self, config):
        """A trades.db that predates this fix has trades but no cash_after
        values (NULL after migration) - must fall back to a fresh account
        rather than crash or guess a wrong cash figure."""
        first = PaperTrader(config)
        first.execute(day=1, ticker="AAA", target_weight=0.20,
                     price=100.0, bar_volume=50_000_000)
        with first._db_lock:
            first._conn.execute("UPDATE trades SET cash_after = NULL")
            first._conn.commit()
        first.close()

        second = PaperTrader(config)
        assert second.cash == config.trading.initial_capital
        second.close()


# ---------------------------------------------------------------------------
# 7. End-to-end 24h simulation
# ---------------------------------------------------------------------------

class TestEndToEnd24hSimulation:
    def test_end_to_end_24h_simulation(self, config, memory, tmp_path):
        """A full compressed day must produce a GodsEye report without errors."""
        orch = KronosOrchestrator(config)

        # Inject the synthetic memory instead of hitting the network
        with patch.object(orch.pipeline, "run_sync", return_value=memory):
            rng = np.random.default_rng(3)
            tickers = memory.tickers
            base = {t: float(memory.prices[t].iloc[-1]) for t in tickers}
            ticks = []
            for _ in range(5):
                base = {t: p * (1 + rng.normal(0, 0.002)) for t, p in base.items()}
                volumes = {t: float(rng.integers(1_000_000, 50_000_000))
                           for t in tickers}
                ticks.append((float(rng.uniform(18, 22)), dict(base), volumes))

            state = orch.run_full_day(day=1, market_ticks=ticks)

        assert state.report_path is not None, "God's Eye report must be generated"
        assert os.path.exists(state.report_path)
        content = Path(state.report_path).read_text()
        assert "God's Eye Report" in content
        assert "Human Summary" in content
        assert state.phase_failures == {}, (
            f"No phase may fail in the happy path: {state.phase_failures}"
        )
        # Audit trail must have entries for the day
        rows = orch.trader._conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE day = 1"
        ).fetchone()[0]
        assert rows > 0, "Audit log must capture the day"
        orch.trader.close()

    def test_phase_failure_does_not_kill_the_day(self, config, memory):
        """Retry-once then continue: a broken nightmare phase must not stop the day."""
        orch = KronosOrchestrator(config)
        with patch.object(orch.pipeline, "run_sync", return_value=memory), \
             patch.object(orch.nightmare_gen, "generate",
                          side_effect=RuntimeError("diffusion exploded")):
            state = orch.run_full_day(day=2)
        assert "nightmare" in state.phase_failures
        # Day still completed and logged
        rows = orch.trader._conn.execute(
            "SELECT COUNT(*) FROM daily_performance WHERE day = 2"
        ).fetchone()[0]
        assert rows == 1
        orch.trader.close()

    def test_veto_takes_24_hours(self, config, memory, tmp_path):
        """A veto.txt must NOT apply immediately."""
        orch = KronosOrchestrator(config)
        veto_path = config.orchestrator.veto_file
        Path(veto_path).write_text("FLATTEN")

        with patch.object(orch.pipeline, "run_sync", return_value=memory):
            orch.run_full_day(day=1)

        # Directive is scheduled, not applied: file still present, pending set
        assert orch._pending_veto is not None
        assert "FLATTEN" in orch._pending_veto["directive"]
        assert os.path.exists(veto_path), (
            "Veto file must survive until the 24h delay elapses"
        )
        orch.trader.close()


# ---------------------------------------------------------------------------
# MAML warm-up contract
# ---------------------------------------------------------------------------

class TestWarmerContract:
    def test_exactly_three_gradient_steps(self, config, memory):
        evolver = KronosEvolver(config)
        gen = NightmareGenerator(config)
        buffer = gen.generate(memory, n_futures=32)
        master = evolver.evolve(buffer).master_model

        warmer = KronosWarmer(config)
        result = warmer.warm(master, memory)
        assert result.n_steps == 3
        assert len(result.inner_losses) == 3, (
            "MAML warm-up must run exactly 3 inner steps"
        )
        assert result.regime_estimate in ("bull", "bear", "sideways")

    def test_warmup_does_not_mutate_master(self, config, memory):
        evolver = KronosEvolver(config)
        gen = NightmareGenerator(config)
        buffer = gen.generate(memory, n_futures=32)
        master = evolver.evolve(buffer).master_model
        before = {k: v.clone() for k, v in master.named_parameters()}

        warmer = KronosWarmer(config)
        warmer.warm(master, memory)

        for k, v in master.named_parameters():
            assert torch.allclose(v, before[k]), (
                f"warm() must not mutate the master model in place ({k})"
            )
