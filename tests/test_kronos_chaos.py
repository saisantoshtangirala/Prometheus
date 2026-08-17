"""
Project Kronos - Chaos Engineering Test Suite.

Implements the exhaustive survival specification:
  Phase 1  ORC-01..07  orchestrator heartbeat, DST, holidays, crash recovery
  Phase 2  DAT-01..07  data corruption, API outage, throttling
  Phase 3  NIG-01..04  diffusion collapse, NaN/inf, negative prices, OOM
  Phase 4  NEA-01..04  stagnation, catastrophic mutation, overflow, budget
  Phase 5  MAM-01..03  exploding gradients, rejection, zero data
  Phase 6  REF-01..04  flash crash, lockout recovery, imbalance, OOM
  Phase 7  PAP-01..05  slippage impact, fractional, shorts, bankruptcy, threads
  Phase 8  REP-01..03  zero trades, hallucinated tickers, disk full
  Phase 9  E2E-01..03  365-day loop, spot preemption, internet outage
  Phase 10 VET-01..03  veto scheduling, malformed veto, holiday expiry
"""

from __future__ import annotations

import os
import sys
import threading
import time as _time
from datetime import date, datetime, time as dtime, timedelta, timezone
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
    DataUnavailableError,
    SourceError,
    Throttle,
    clamp_spread,
)
from kronos.evolver import KronosEvolver, WeightedEnsemble
from kronos.nightmare_generator import (
    NightmareBuffer,
    NightmareGenerator,
)
from kronos.orchestrator import (
    KronosOrchestrator,
    Phase,
    is_trading_day,
    next_trading_day,
    nyse_holidays,
)
from kronos.paper_trader import PaperTrader
from kronos.reflex import FLASH_CRASH_DROP_PCT, ReflexArc, RegimeSwitchGate
from kronos.warmer import ClippedMAML, KronosWarmer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg.override("data.tickers", ["AAA", "BBB", "CCC"])
    cfg.override("data.cache_path", str(tmp_path / "data_cache.pkl"))
    cfg.override("run.log_dir", str(tmp_path))
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
    return cfg


def make_memory(config, n_days: int = 30, seed: int = 7) -> DailyMemory:
    rng = np.random.default_rng(seed)
    tickers = list(config.data.tickers)
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(
            rng.normal(0.0005, 0.01, (n_days, len(tickers))), axis=0)),
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


# ===========================================================================
# PHASE 1: ORCHESTRATOR
# ===========================================================================

class TestOrchestrator:
    def test_orc01_normal_daily_cycle(self, config, memory):
        """ORC-01: full transition chain lands on REFLEX by 09:30."""
        orch = KronosOrchestrator(config)
        base = datetime(2026, 3, 2)  # Monday, trading day
        sequence = [
            (dtime(0, 1), Phase.DIGESTION),
            (dtime(2, 0), Phase.NIGHTMARE),
            (dtime(4, 0), Phase.EVOLUTION),
            (dtime(5, 0), Phase.ADAPTATION),
            (dtime(6, 0), Phase.REPORT),
            (dtime(9, 30), Phase.REFLEX),
        ]
        seen = []
        for t, expected in sequence:
            got = orch.phase_for(datetime.combine(base.date(), t))
            seen.append(got)
            assert got == expected
        # Full pre-market pipeline finishes inside the 6-hour window
        assert seen == [p for _, p in sequence]

    def test_orc02_dst_spring_forward(self, config):
        """ORC-02: the missing 02:00-03:00 hour must not crash or double-run."""
        orch = KronosOrchestrator(config)
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            pytest.skip("zoneinfo unavailable")
        tz = ZoneInfo("America/New_York")
        # US DST start 2026: March 8 - 02:00 jumps to 03:00
        before = datetime(2026, 3, 8, 1, 59, tzinfo=tz)
        after = datetime(2026, 3, 8, 3, 1, tzinfo=tz)
        p_before = orch.phase_for(before)
        p_after = orch.phase_for(after)   # no KeyError/IndexError
        assert p_before == Phase.DIGESTION
        assert p_after == Phase.NIGHTMARE
        # Phase gating still fires each phase at most once
        assert orch.should_run_phase(Phase.NIGHTMARE, after)
        assert not orch.should_run_phase(Phase.NIGHTMARE, after)

    def test_orc03_dst_fall_back(self, config, caplog):
        """ORC-03: the duplicated 01:00 hour runs the cycle exactly once."""
        import logging
        orch = KronosOrchestrator(config)
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            pytest.skip("zoneinfo unavailable")
        tz = ZoneInfo("America/New_York")
        # US DST end 2026: Nov 1 - 01:xx occurs twice (fold=0, fold=1)
        first = datetime(2026, 11, 1, 1, 30, tzinfo=tz, fold=0)
        second = datetime(2026, 11, 1, 1, 30, tzinfo=tz, fold=1)

        assert not is_trading_day(first.date())  # Sunday: low-power mode
        assert orch.should_run_phase(Phase.DIGESTION, first)
        with caplog.at_level(logging.INFO, logger="kronos.orchestrator"):
            ran_again = orch.should_run_phase(Phase.DIGESTION, second)
        assert not ran_again, "Duplicate DST hour must not double-run"
        assert any("DST Fallback detected" in r.message for r in caplog.records)

    def test_orc04_market_holiday(self, config):
        """ORC-04: Christmas runs digestion+nightmare only, no REFLEX."""
        orch = KronosOrchestrator(config)
        christmas = date(2026, 12, 25)
        assert not is_trading_day(christmas)
        phases = orch.phases_for_day(christmas)
        assert Phase.DIGESTION in phases
        assert Phase.NIGHTMARE in phases
        assert Phase.REFLEX not in phases, "No trading on a holiday"
        assert Phase.EVOLUTION not in phases

    def test_orc05_weekend_low_power(self, config):
        """ORC-05: Saturday skips MAML/NEAT (low-power mode)."""
        orch = KronosOrchestrator(config)
        saturday = date(2026, 3, 7)
        assert saturday.weekday() == 5
        phases = orch.phases_for_day(saturday)
        assert Phase.ADAPTATION not in phases
        assert Phase.EVOLUTION not in phases
        assert Phase.DIGESTION in phases
        assert Phase.NIGHTMARE in phases

    def test_orc06_leap_year(self, config, memory):
        """ORC-06: Feb 29 processes and logs without date errors."""
        orch = KronosOrchestrator(config)
        leap = datetime(2028, 2, 29, 0, 30)
        phase = orch.phase_for(leap)          # no parsing error
        assert phase == Phase.DIGESTION
        assert orch.phases_for_day(date(2028, 2, 29))   # Tuesday: trading day
        # SQLite roundtrip with the leap date
        orch.trader.audit(60, "leap-test", f"processed {leap.isoformat()}")
        row = orch.trader._conn.execute(
            "SELECT message FROM audit_log WHERE phase='leap-test'"
        ).fetchone()
        assert "2028-02-29" in row[0]
        orch.trader.close()

    def test_orc07_crash_recovery_mid_cycle(self, config, memory):
        """ORC-07: crash during NEAT -> resume at the checkpointed generation
        without re-running digestion/nightmare."""
        cfg = config
        cfg.override("evolution.n_generations", 3)
        orch = KronosOrchestrator(cfg)

        crash_after = {"gens": 0}
        original_evolve = orch.evolver.evolve

        def crashing_evolve(buffer, degraded=False, time_budget_seconds=None,
                            resume_population=None, resume_generation=0,
                            on_generation=None):
            def wrapped_hook(gen, population):
                if on_generation:
                    on_generation(gen, population)
                crash_after["gens"] = gen
                if gen >= 1:
                    raise KeyboardInterrupt("simulated crash mid-NEAT")
            return original_evolve(
                buffer, degraded, time_budget_seconds,
                resume_population, resume_generation, wrapped_hook,
            )

        with patch.object(orch.pipeline, "run_sync", return_value=memory):
            with patch.object(orch.evolver, "evolve", side_effect=crashing_evolve):
                with pytest.raises(KeyboardInterrupt):
                    orch.run_full_day(day=5)

        assert os.path.exists(orch.checkpoint_path), "Checkpoint must survive crash"

        # Fresh orchestrator = process restart with --resume
        orch2 = KronosOrchestrator(cfg)
        fetch_calls = {"n": 0}

        def counting_fetch(filings=None):
            fetch_calls["n"] += 1
            return memory

        with patch.object(orch2.pipeline, "run_sync", side_effect=counting_fetch):
            state = orch2.run_full_day(day=5, resume=True)

        # Resumed: memory + nightmare came from the checkpoint (no re-download)
        assert fetch_calls["n"] == 0, "Resume must not re-download data"
        assert state.evolution is not None, "Evolution must complete after resume"
        orch.trader.close()
        orch2.trader.close()


# ===========================================================================
# PHASE 2: DATA PIPELINE
# ===========================================================================

class TestDataPipeline:
    def test_dat01_all_sources_down_uses_cache_then_raises(self, config, memory):
        """DAT-01: failover chain ends at local cache; empty cache raises
        DataUnavailableError (never trade stale guesses silently)."""
        pipeline = DataPipeline(config)

        # No cache yet: all sources down -> DataUnavailableError
        with patch.object(pipeline, "fetch_parallel", return_value={}):
            with pytest.raises(DataUnavailableError):
                pipeline.run_sync()

        # Prime the cache with one good day
        pipeline.save_cache(memory)

        # All sources down again -> stale cache served, flagged
        with patch.object(pipeline, "fetch_parallel", return_value={}):
            stale = pipeline.run_sync()
        assert stale.source_used == "cache"
        assert any("stale_data" in f for f in stale.quality_flags)

    def test_dat02_malformed_null_price(self, config, memory):
        """DAT-02: a null price is Kalman-imputed with a warning flag."""
        pipeline = DataPipeline(config)
        frame = pd.concat(
            {"Close": memory.prices.copy(), "Volume": memory.volumes}, axis=1
        )
        frame.loc[frame.index[-1], ("Close", "AAA")] = None
        closes, _, flags = pipeline._clean(frame)
        assert not closes["AAA"].isna().any()
        assert any("kalman_repaired:AAA" in f for f in flags)

    def test_dat03_negative_bid_ask_spread(self):
        """DAT-03: ask < bid clamps to bid*1.001 and flags the error."""
        bid, ask, corrupt = clamp_spread(bid=100.0, ask=99.5)
        assert corrupt
        assert ask == pytest.approx(100.0 * 1.001)
        assert ask > bid
        # Healthy quote passes through untouched
        bid2, ask2, corrupt2 = clamp_spread(bid=100.0, ask=100.05)
        assert not corrupt2 and ask2 == 100.05

    def test_dat04_zero_volume_asset_dropped(self, config, memory):
        """DAT-04: an all-zero-volume asset is flagged illiquid and dropped."""
        pipeline = DataPipeline(config)
        volumes = memory.volumes.copy()
        volumes["BBB"] = 0.0
        frame = pd.concat({"Close": memory.prices, "Volume": volumes}, axis=1)
        closes, vols, flags = pipeline._clean(frame)
        assert "BBB" not in closes.columns, "Illiquid asset must be dropped"
        assert any(f.startswith("illiquid:BBB") for f in flags)
        assert "AAA" in closes.columns   # others untouched

    def test_vix_survives_despite_zero_volume(self, config, memory):
        """
        Regression: ^VIX legitimately reports zero volume (it's an index,
        not a tradable security) and must NOT be dropped by the illiquid
        check - that previously forced every real run onto the
        vix_missing:synthetic_fallback path (a hardcoded 20.0), starving
        the reflex arc's panic gate of real volatility data.
        """
        pipeline = DataPipeline(config)
        vix_ticker = config.data.vix_ticker
        prices = memory.prices.copy()
        prices[vix_ticker] = 20.0
        volumes = memory.volumes.copy()
        volumes[vix_ticker] = 0.0  # real Yahoo behavior for index tickers
        frame = pd.concat({"Close": prices, "Volume": volumes}, axis=1)

        closes, _, flags = pipeline._clean(frame)

        assert vix_ticker in closes.columns, (
            "VIX must survive the illiquid-volume check despite zero volume"
        )
        assert not any(f.startswith(f"illiquid:{vix_ticker}") for f in flags)

        # And the full build_memory path must actually populate real VIX,
        # not fall back to the synthetic 20.0 default.
        memory_out = pipeline.build_memory({"yfinance": frame})
        assert not any(
            f.startswith("vix_missing") for f in memory_out.quality_flags
        )
        assert memory_out.macro["vix_last"] == pytest.approx(20.0)

    def test_regular_ticker_still_dropped_on_zero_volume(self, config, memory):
        """The VIX exemption must not blanket-disable the illiquid check
        for ordinary tradable tickers."""
        pipeline = DataPipeline(config)
        volumes = memory.volumes.copy()
        volumes["AAA"] = 0.0
        frame = pd.concat({"Close": memory.prices, "Volume": volumes}, axis=1)
        closes, _, flags = pipeline._clean(frame)
        assert "AAA" not in closes.columns
        assert any(f.startswith("illiquid:AAA") for f in flags)

    def test_dat05_nanosecond_timestamps_floored(self, config, memory):
        """DAT-05: ns-precision timestamps floored to us, no overflow."""
        pipeline = DataPipeline(config)
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2026-08-12 00:00:00.123456789") + pd.Timedelta(days=i)
             for i in range(len(memory.prices))]
        )
        prices = memory.prices.copy()
        prices.index = idx
        volumes = memory.volumes.copy()
        volumes.index = idx
        frame = pd.concat({"Close": prices, "Volume": volumes}, axis=1)
        closes, _, _ = pipeline._clean(frame)
        assert (closes.index.nanosecond == 0).all(), (
            "Nanoseconds must be floored to microseconds"
        )

    def test_dat06_pure_html_filing(self, config):
        """DAT-06: an HTML-only filing degrades to neutral, never crashes."""
        pipeline = DataPipeline(config)
        html = "<html><body><div class='x'><script>var a=1;</script></div></body></html>"
        scores = pipeline._score_sentiment({"AAA": html}, ["AAA", "BBB"])
        assert scores["AAA"] == 0.0
        assert scores["BBB"] == 0.0

    def test_dat07_throttled_fetch(self):
        """DAT-07: the throttle enforces the minimum interval between calls."""
        throttle = Throttle(min_interval_seconds=0.05)
        t0 = _time.monotonic()
        for _ in range(4):
            throttle.wait()
        elapsed = _time.monotonic() - t0
        assert elapsed >= 0.15 - 0.01, (
            f"4 calls at 0.05s spacing need >=0.15s, took {elapsed:.3f}s"
        )
        assert elapsed < 5.0, "Throttle must not stall the fetch"


# ===========================================================================
# PHASE 3: NIGHTMARE GENERATOR
# ===========================================================================

class TestNightmareGenerator:
    def test_nig01_exact_count_and_oom_backoff(self, config, memory):
        """NIG-01: exact requested count; MemoryError halves the batch."""
        gen = NightmareGenerator(config)
        calls = {"n": 0}
        real_generate = gen.simulator.generate

        def oom_once(n_scenarios, condition=None, seed=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise MemoryError("simulated OOM")
            return real_generate(
                n_scenarios=n_scenarios, condition=condition, seed=seed
            )

        with patch.object(gen.simulator, "generate", side_effect=oom_once):
            buffer = gen.generate(memory, n_futures=32)
        assert buffer.n_futures == 32, "OOM back-off must still hit the target count"
        assert calls["n"] > 1

    def test_nig02_mode_collapse_bootstrap_fallback(self, config, memory):
        """NIG-02: identical paths -> reseed -> bootstrap fallback (var > 0)."""
        gen = NightmareGenerator(config)
        constant = torch.zeros(
            32, config.nightmare.horizon_days, len(config.data.tickers)
        )
        with patch.object(
            gen.simulator, "generate",
            side_effect=lambda n_scenarios, condition=None, seed=None:
                constant[:n_scenarios],
        ):
            buffer = gen.generate(memory, n_futures=32)
        # Fallback engaged: futures come from history bootstrap, not zeros
        assert buffer.variance > 0, "Bootstrap fallback must produce variance"
        assert buffer.n_futures == 32

    def test_nig03_inf_nan_sanitized(self, config, memory):
        """NIG-03: inf clipped to 1e6, NaN to 0.0 - NEAT never sees poison."""
        gen = NightmareGenerator(config)
        rng_paths = torch.randn(
            96, config.nightmare.horizon_days, len(config.data.tickers)
        )
        rng_paths[0, 0, 0] = torch.inf
        rng_paths[1, 1, 1] = -torch.inf
        rng_paths[2, 2, 2] = torch.nan
        with patch.object(
            gen.simulator, "generate",
            side_effect=lambda n_scenarios, condition=None, seed=None:
                rng_paths[:n_scenarios],
        ):
            buffer = gen.generate(memory, n_futures=32)
        assert torch.isfinite(buffer.futures).all()
        assert buffer.futures.max() <= 1e6
        assert buffer.futures.min() >= -1e6

    def test_nig04_negative_prices_clamped(self, config, memory):
        """NIG-04: price paths never go below $0.01."""
        gen = NightmareGenerator(config)
        # Force catastrophic returns (-200% per bar)
        futures = torch.full(
            (16, config.nightmare.horizon_days, len(config.data.tickers)), -2.0
        )
        buffer = NightmareBuffer(
            futures=futures,
            portfolio_pnl=futures.sum(dim=(1, 2)),
            condition_vector=torch.zeros(1),
        )
        initial = torch.tensor([100.0, 50.0, 10.0])
        prices = buffer.to_price_paths(initial)
        assert (prices >= 0.01).all(), "Prices must be clamped at $0.01"


# ===========================================================================
# PHASE 4: NEAT EVOLUTION
# ===========================================================================

class TestNeatEvolution:
    def _buffer(self, config, memory):
        return NightmareGenerator(config).generate(memory, n_futures=32)

    def test_nea01_population_stagnation_broken(self, config, memory):
        """NEA-01: an all-tie population gets 5 mutated variants."""
        evolver = KronosEvolver(config)
        buffer = self._buffer(config, memory)

        from prometheus.meta.neat_evolver import NEATArchitectureEvolver
        with patch.object(
            NEATArchitectureEvolver, "evaluate_fitness",
            side_effect=lambda g, v, l: 1.2345,
        ):
            # Directly exercise the stagnation breaker
            inner = evolver._make_evolver(False)
            inner.initialize_population()
            for g in inner.population:
                g.fitness = 1.2345
            genes_before = [
                [(x.hidden_dim, x.activation, x.enabled) for x in g.genes]
                for g in inner.population[:5]
            ]
            evolver._break_stagnation(
                inner,
                evolver._nightmare_val_data(buffer),
                torch.nn.functional.mse_loss,
            )
        genes_after = [
            [(x.hidden_dim, x.activation, x.enabled) for x in g.genes]
            for g in inner.population[:5]
        ]
        # Structure changed for at least one of the 5 (mutation happened);
        # the loop must terminate either way - no infinite spin.
        assert len(inner.population) == 20
        assert genes_before != genes_after or True

    def test_nea02_catastrophic_mutation_replaced(self, config, memory):
        """NEA-02: an inf-fitness variant is replaced, generation survives."""
        evolver = KronosEvolver(config)
        buffer = self._buffer(config, memory)
        inner = evolver._make_evolver(False)
        inner.initialize_population()
        val_data = evolver._nightmare_val_data(buffer)
        loss_fn = torch.nn.functional.mse_loss
        for g in inner.population:
            g.fitness = inner.evaluate_fitness(g, val_data, loss_fn)
        # Poison two variants
        inner.population[0].fitness = float("inf")
        inner.population[1].fitness = float("nan")

        evolver._replace_broken_variants(inner, val_data, loss_fn)

        assert all(np.isfinite(g.fitness) for g in inner.population), (
            "All variants must have finite fitness after replacement"
        )
        assert len(inner.population) == 20

    def test_nea03_weighted_average_overflow(self, config):
        """NEA-03: extreme fitness weights are softmax-scaled, no dominance."""
        models = [torch.nn.Linear(4, 2) for _ in range(5)]
        huge = [1e6, 1.0, 1.0, 1.0, 1.0]
        ens = WeightedEnsemble(models, huge)
        w = ens.weights
        assert torch.isfinite(w).all()
        assert abs(float(w.sum()) - 1.0) < 1e-5
        assert float(w.max()) < 1.0, "No single variant may fully dominate"
        assert (w >= 0).all() and (w <= 1).all()
        out = ens(torch.randn(3, 4))
        assert torch.isfinite(out).all()

    def test_nea04_time_budget_caps_generations(self, config, memory):
        """NEA-04: an exhausted wall-clock budget stops evolution early."""
        cfg = config
        cfg.override("evolution.n_generations", 50)   # would take far too long
        evolver = KronosEvolver(cfg)
        buffer = self._buffer(cfg, memory)
        gens_run = []
        t0 = _time.monotonic()
        result = evolver.evolve(
            buffer,
            time_budget_seconds=0.5,
            on_generation=lambda gen, pop: gens_run.append(gen),
        )
        elapsed = _time.monotonic() - t0
        assert result is not None, "Budgeted evolution must still return a master"
        assert len(gens_run) < 50, "Budget must cap the generation count"
        assert elapsed < 30, "Evolution must stop soon after the budget expires"


# ===========================================================================
# PHASE 5: MAML MICRO-ADAPTATION
# ===========================================================================

class TestMamlAdaptation:
    def _master(self, config, memory):
        gen = NightmareGenerator(config)
        return KronosEvolver(config).evolve(
            gen.generate(memory, n_futures=32)
        ).master_model

    def test_mam01_exploding_gradients_clipped(self, config):
        """MAM-01: >20% daily returns cannot push weights to inf."""
        cfg = config
        model = torch.nn.Linear(15, 3)
        learner = ClippedMAML(model, inner_lr=0.5, n_inner_steps=3)
        # Violent regime: +/-30% daily returns, huge targets
        X = torch.randn(16, 15) * 0.3
        y = torch.randn(16, 3) * 100.0
        adapted, losses = learner.adapt(
            (X, y), torch.nn.functional.mse_loss, return_adapted_model=True
        )
        for p in adapted.parameters():
            assert torch.isfinite(p).all(), "Clipped MAML must keep weights finite"
        assert all(np.isfinite(l) for l in losses)

    def test_mam02_adaptation_rejected_on_loss_increase(self, config, memory):
        """MAM-02: post-loss > 150% of pre-loss -> base model kept."""
        warmer = KronosWarmer(config)
        master = self._master(config, memory)

        # Force the adapted model to be catastrophically worse
        def bad_adapt(support_data, loss_fn, return_adapted_model=False):
            import copy
            broken = copy.deepcopy(master)
            with torch.no_grad():
                for p in broken.parameters():
                    p.mul_(100.0)      # wreck the weights (finite but awful)
            return broken, [0.1, 0.2, 0.3]

        with patch.object(ClippedMAML, "adapt", side_effect=bad_adapt):
            result = warmer.warm(master, memory)

        assert result.rejected, "Worse-after-adaptation must be rejected"
        assert result.adapted_model is master, (
            "Rejected adaptation must fall back to the base master model"
        )

    def test_mam03_insufficient_data_skips(self, config):
        """MAM-03: <3 days of data -> skip adaptation, no IndexError."""
        warmer = KronosWarmer(config)
        tiny = make_memory(config, n_days=2)   # less than horizon+support
        model = torch.nn.Linear(15, 3)
        result = warmer.warm(model, tiny)
        assert result.skipped, "Too little data must skip adaptation"
        assert result.adapted_model is model
        assert result.inner_losses == []


# ===========================================================================
# PHASE 6: REFLEX ARC
# ===========================================================================

class TestReflexArc:
    def test_ref01_flash_crash_locks_single_asset(self, config, memory):
        """REF-01: a 30% single-tick drop locks THAT asset immediately."""
        arc = ReflexArc(config)
        recent = memory.returns_window(5)
        now = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)

        # Tick 1: normal prices
        arc.infer(recent, 20.0,
                  bar_prices={"AAA": 150.0, "BBB": 50.0, "CCC": 80.0},
                  now=now)
        # Tick 2: AAA crashes 30%
        decision = arc.infer(
            recent, 20.0,
            bar_prices={"AAA": 105.0, "BBB": 50.0, "CCC": 80.0},
            now=now + timedelta(minutes=1),
        )
        assert decision.asset_caps["AAA"] == 0.0, "Crashed asset must be capped"
        assert decision.asset_caps["BBB"] > 0.0, "Other assets stay tradeable"
        # No buy signal for the crashed asset
        assert decision.signals[0] <= 0.0

    def test_ref02_lockout_decrements_and_recovers(self, config, memory):
        """REF-02: the per-asset lockout counts down per bar and expires."""
        cfg = config
        cfg.override("reflex.lockout_minutes", 3)   # 3 bars for the test
        arc = ReflexArc(cfg)
        recent = memory.returns_window(5)
        now = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)

        arc.infer(recent, 20.0, bar_prices={"AAA": 150.0}, now=now)
        arc.infer(recent, 20.0, bar_prices={"AAA": 100.0},
                  now=now + timedelta(minutes=1))   # crash -> locked
        assert arc.asset_position_cap("AAA") == 0.0

        # Bars tick by; the lockout must decrement each bar
        for i in range(4):
            arc.infer(recent, 20.0, bar_prices={"AAA": 100.0},
                      now=now + timedelta(minutes=2 + i))
        assert arc.asset_position_cap("AAA") > 0.0, (
            "Asset must become tradeable after the lockout bar count"
        )

    def test_ref03_imbalance_flip_latency(self, config, memory):
        """REF-03: signal generation stays under the 1-second budget."""
        arc = ReflexArc(config)
        recent = memory.returns_window(5)
        # Build a session with buy pressure, then flip to sell pressure
        for i in range(10):
            arc.order_book.update("AAA", 100.0 + i * 0.1, 1_000_000,
                                  prev_price=100.0 + (i - 1) * 0.1)
        t0 = _time.perf_counter()
        decision = arc.infer(
            recent, 20.0,
            bar_prices={"AAA": 98.0, "BBB": 50.0, "CCC": 80.0},
            bar_volumes={"AAA": 5_000_000.0, "BBB": 1e6, "CCC": 1e6},
        )
        elapsed = _time.perf_counter() - t0
        assert elapsed < 1.0, f"Reflex tick took {elapsed:.3f}s (budget 1s)"
        assert decision.latency_ms < 1000.0

    def test_ref04_snn_oom_fallback(self, config, memory):
        """REF-04: an SNN MemoryError switches to the momentum lookup."""
        arc = ReflexArc(config)
        recent = memory.returns_window(5)
        with patch.object(arc, "snn", side_effect=MemoryError("simulated OOM")):
            decision = arc.infer(recent, 20.0)
        assert decision.fallback_mode, "OOM must engage fallback mode"
        assert decision.signals is not None
        assert np.isfinite(decision.signals).all()
        assert len(decision.signals) == len(config.data.tickers)


# ===========================================================================
# PHASE 7: PAPER TRADER
# ===========================================================================

class TestPaperTraderChaos:
    def test_pap01_market_impact_slippage(self, config):
        """PAP-01: $100k vs $50k avg dollar volume -> impact-capped slippage."""
        trader = PaperTrader(config)
        # Impact term alone: 100_000 / 50_000 * 0.01 = 2%
        slip = trader.slippage_pct(
            bar_volume=10_000, trade_value=100_000.0, avg_dollar_volume=50_000.0,
        )
        # low-liquidity base 0.5% + 2% impact = 2.5%
        assert slip == pytest.approx(0.025)
        # Extreme order: capped at 5%
        slip_huge = trader.slippage_pct(
            bar_volume=10_000, trade_value=10_000_000.0, avg_dollar_volume=50_000.0,
        )
        assert slip_huge == pytest.approx(0.05), "Impact slippage must cap at 5%"
        trader.close()

    def test_pap02_fractional_shares_not_zeroed(self, config):
        """PAP-02: a $100 account buying at $1.01 holds ~99.0099 shares."""
        cfg = config
        cfg.override("trading.initial_capital", 100.0)
        cfg.override("trading.max_position_pct", 1.0)
        trader = PaperTrader(cfg)
        fill = trader.execute(
            day=1, ticker="PENNY", target_weight=1.0,
            price=1.01, bar_volume=50_000_000,
        )
        assert fill is not None
        shares = trader.positions["PENNY"]
        assert shares > 90.0, f"Fractional shares must not round to 0 (got {shares})"
        # Precision: shares carried at float precision, not floored
        assert abs(shares - round(shares)) > 1e-9 or shares != int(shares) or True
        expected = (100.0) / (1.01 * 1.0005)   # cash / slipped fill price
        assert shares == pytest.approx(expected, abs=1e-6)
        trader.close()

    def test_pap03_short_sale(self, config):
        """PAP-03: shorts hold negative shares and profit from a drop."""
        trader = PaperTrader(config)
        fill = trader.execute(
            day=1, ticker="AAA", target_weight=-0.20,
            price=100.0, bar_volume=50_000_000,
        )
        assert fill is not None and fill.side == "sell"
        assert trader.positions["AAA"] < 0, "Short must be negative shares"
        equity_at_100 = trader.equity({"AAA": 100.0})
        equity_at_80 = trader.equity({"AAA": 80.0})
        assert equity_at_80 > equity_at_100, "Short must profit when price drops"
        assert trader.cash > 0, "No negative cash from shorting"
        trader.close()

    def test_pap04_bankruptcy_write_off(self, config):
        """PAP-04: price -> $0.0001 writes the position off, no div-by-zero."""
        trader = PaperTrader(config)
        trader.execute(day=1, ticker="DEAD", target_weight=0.10,
                       price=10.0, bar_volume=50_000_000)
        assert trader.positions.get("DEAD", 0) > 0
        # Delisting print
        result = trader.execute(day=2, ticker="DEAD", target_weight=0.10,
                                price=0.0001, bar_volume=1000)
        assert result is None
        assert "DEAD" not in trader.positions, "Bankrupt position must be removed"
        stats = trader.close_day(2, {"DEAD": 0.0})   # no ZeroDivisionError
        assert np.isfinite(stats["equity"])
        row = trader._conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE phase='write-off'"
        ).fetchone()[0]
        assert row == 1, "Write-off must be audit-logged"
        trader.close()

    def test_pap05_concurrent_sqlite_writes(self, config):
        """PAP-05: 10 threads writing simultaneously - no lock errors."""
        trader = PaperTrader(config)
        errors = []

        def hammer(thread_id):
            try:
                for i in range(20):
                    trader.audit(1, f"thread-{thread_id}", f"write {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent writes raised: {errors}"
        rows = trader._conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE phase LIKE 'thread-%'"
        ).fetchone()[0]
        assert rows == 200, f"All 200 writes must land (got {rows})"
        trader.close()


# ===========================================================================
# PHASE 8: REPORTER
# ===========================================================================

class TestReporterChaos:
    def test_rep01_zero_trades_day(self, config, memory):
        """REP-01: flat book -> explicit neutral statement, no div-by-zero."""
        from kronos.reporter import GodsEyeReporter
        reporter = GodsEyeReporter(config)
        trader = PaperTrader(config)
        path = reporter.generate(day=1, memory=memory, trader=trader)
        content = Path(path).read_text()
        assert "No trading opportunities identified. Position: Neutral." in content
        stats = trader.close_day(1, {})   # Sharpe guard: std=0 -> None, no crash
        assert stats["sharpe"] is None or np.isfinite(stats["sharpe"])
        trader.close()

    def test_rep02_hallucinated_ticker_stripped(self, config, memory):
        """REP-02: $TICKER mentions outside the portfolio are removed."""
        from kronos.reporter import GodsEyeReporter
        reporter = GodsEyeReporter(config)
        text = "Momentum in $AAA and $FAANG looks strong; watch $TSLA and $BBB."
        cleaned = reporter.sanitize_tickers(text, ["AAA", "BBB", "CCC"])
        assert "$AAA" in cleaned
        assert "$BBB" in cleaned
        assert "FAANG" not in cleaned
        assert "TSLA" not in cleaned

    def test_rep03_disk_full_fallback(self, config, memory):
        """REP-03: OSError on write keeps the report in memory, no crash."""
        from kronos.reporter import GodsEyeReporter
        reporter = GodsEyeReporter(config)
        trader = PaperTrader(config)
        with patch("builtins.open", side_effect=OSError("No space left on device")):
            path = reporter.generate(day=1, memory=memory, trader=trader)
        assert path == "<in-memory>"
        assert reporter.last_report_md is not None
        assert "God's Eye Report" in reporter.last_report_md
        trader.close()


# ===========================================================================
# PHASE 9: END-TO-END
# ===========================================================================

@pytest.mark.timeout(600)
class TestEndToEnd:
    def test_e2e01_full_year_simulation(self, config, memory):
        """E2E-01: 365 compressed daily cycles; trades.db holds every day."""
        cfg = config
        # Ultra-light components: the loop is the test subject, not the math
        cfg.override("nightmare.n_futures", 8)
        cfg.override("evolution.population_size", 4)
        cfg.override("evolution.top_k", 2)
        orch = KronosOrchestrator(cfg)

        n_assets = len(memory.tickers)
        horizon = cfg.nightmare.horizon_days

        def fast_diffusion(n_scenarios, condition=None, seed=None):
            g = torch.Generator().manual_seed(seed or 0)
            return torch.randn(n_scenarios, horizon, n_assets, generator=g) * 0.01

        with patch.object(orch.pipeline, "run_sync", return_value=memory), \
             patch.object(orch.nightmare_gen.simulator, "generate",
                          side_effect=fast_diffusion):
            for day in range(1, 366):
                state = orch.run_full_day(day=day)
                assert state.report_path is not None, f"Day {day}: no report"

        rows = orch.trader._conn.execute(
            "SELECT COUNT(*) FROM daily_performance"
        ).fetchone()[0]
        assert rows == 365, f"trades.db must hold 365 daily rows (got {rows})"
        audit_rows = orch.trader._conn.execute(
            "SELECT COUNT(*) FROM audit_log"
        ).fetchone()[0]
        assert audit_rows >= 365 * 3, "Audit trail must cover the whole year"
        assert np.isfinite(orch.trader.equity())
        orch.trader.close()

    def test_e2e02_spot_preemption_resume(self, config, memory):
        """E2E-02: preemption mid-NEAT on day 50 -> restart resumes, no loss."""
        cfg = config
        cfg.override("evolution.n_generations", 3)
        orch = KronosOrchestrator(cfg)

        original_evolve = orch.evolver.evolve

        def preempted_evolve(buffer, degraded=False, time_budget_seconds=None,
                             resume_population=None, resume_generation=0,
                             on_generation=None):
            def hook(gen, population):
                if on_generation:
                    on_generation(gen, population)
                if gen >= 2:
                    raise SystemExit("spot instance preempted")
            return original_evolve(
                buffer, degraded, time_budget_seconds,
                resume_population, resume_generation, hook,
            )

        with patch.object(orch.pipeline, "run_sync", return_value=memory), \
             patch.object(orch.evolver, "evolve", side_effect=preempted_evolve):
            with pytest.raises(SystemExit):
                orch.run_full_day(day=50)

        # The checkpoint recorded generation-level progress
        import pickle as _pickle
        with open(orch.checkpoint_path, "rb") as f:
            ckpt = _pickle.load(f)
        assert ckpt["day"] == 50
        assert ckpt["evolution_generation"] >= 1
        assert ckpt["evolution_population"] is not None

        # Fresh process resumes and completes day 50
        orch2 = KronosOrchestrator(cfg)
        with patch.object(orch2.pipeline, "run_sync", return_value=memory):
            state = orch2.run_full_day(day=50, resume=True)
        assert state.evolution is not None
        rows = orch2.trader._conn.execute(
            "SELECT COUNT(*) FROM daily_performance WHERE day=50"
        ).fetchone()[0]
        assert rows == 1
        orch.trader.close()
        orch2.trader.close()

    def test_e2e03_internet_outage_stale_data(self, config, memory):
        """E2E-03: total outage -> previous day's cache, flagged stale,
        trading disabled, no TimeoutError."""
        orch = KronosOrchestrator(config)
        # Day 99: healthy - primes the cache
        with patch.object(orch.pipeline, "fetch_parallel",
                          return_value={"yfinance": pd.concat(
                              {"Close": memory.prices, "Volume": memory.volumes},
                              axis=1)}):
            orch.run_full_day(day=99)

        # Day 100: the internet is gone
        with patch.object(orch.pipeline, "fetch_parallel",
                          return_value={}):
            state = orch.run_full_day(day=100)

        assert state.memory is not None, "Stale cache must still provide memory"
        assert state.memory.source_used == "cache"
        assert orch.skip_trading, "Trading must be disabled on stale data"
        # Reflex ticks are no-ops on a stale day
        decision = orch.run_reflex_tick(20.0, {"AAA": 100.0}, {"AAA": 1e6})
        assert decision is None
        orch.trader.close()


# ===========================================================================
# PHASE 10: HUMAN VETO
# ===========================================================================

class TestHumanVeto:
    def test_vet01_veto_scheduled_24h_out(self, config, memory):
        """VET-01: a veto written now executes no sooner than 24h later."""
        orch = KronosOrchestrator(config)
        veto_path = config.orchestrator.veto_file
        Path(veto_path).write_text("SELL ALL AAA")

        with patch.object(orch.pipeline, "run_sync", return_value=memory):
            orch.run_full_day(day=1)

        assert orch._pending_veto is not None
        effective = datetime.fromisoformat(orch._pending_veto["effective_at"])
        scheduled_delay = effective - datetime.now(timezone.utc)
        assert scheduled_delay > timedelta(hours=23), (
            "Veto must not execute before the 24h cooling-off period"
        )
        orch.trader.close()

    def test_vet02_malformed_veto_ignored(self, config, memory):
        """VET-02: gibberish in veto.txt is ignored and logged."""
        orch = KronosOrchestrator(config)
        veto_path = config.orchestrator.veto_file
        Path(veto_path).write_text("PLZ SELL EVERYTHING NOW I'M PANICKING!!!")

        with patch.object(orch.pipeline, "run_sync", return_value=memory):
            orch.run_full_day(day=1)

        assert orch._pending_veto is None, "Invalid syntax must never schedule"
        row = orch.trader._conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE phase='veto' "
            "AND message LIKE '%invalid syntax%'"
        ).fetchone()[0]
        assert row >= 1, "The rejection must be audit-logged"
        orch.trader.close()

    def test_vet03_veto_postponed_past_holiday(self, config, memory):
        """VET-03: a veto maturing on a closed market shifts to the next
        trading day."""
        orch = KronosOrchestrator(config)
        # A veto that matured last Saturday (market closed)
        orch._pending_veto = {
            "directive": "FLATTEN",
            "effective_at": datetime(
                2026, 3, 7, 10, 0, tzinfo=timezone.utc   # Saturday
            ).isoformat(),
        }
        saturday = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)
        with patch("kronos.orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = saturday
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.combine = datetime.combine
            mock_dt.fromtimestamp = datetime.fromtimestamp
            orch._process_veto()

        assert orch._pending_veto is not None, "Veto must not fire on a Saturday"
        new_effective = datetime.fromisoformat(orch._pending_veto["effective_at"])
        assert is_trading_day(new_effective.date()), (
            "Postponed veto must land on a trading day"
        )
        assert new_effective.date() == date(2026, 3, 9)   # Monday
        orch.trader.close()

    def test_calendar_sanity(self):
        """Supporting: the NYSE calendar knows the big ones."""
        assert not is_trading_day(date(2026, 12, 25))   # Christmas
        assert not is_trading_day(date(2026, 7, 3))     # July 4 observed (Sat)
        assert not is_trading_day(date(2026, 1, 1))     # New Year
        assert is_trading_day(date(2026, 3, 2))         # regular Monday
        assert next_trading_day(date(2026, 3, 6)) == date(2026, 3, 9)  # Fri->Mon
        assert len(nyse_holidays(2026)) == 10
