"""
Tests for scripts/run_kronos.py - the actual CLI entry point, not just the
KronosOrchestrator class underneath it.

This file exists because of a real bug: the FileHandler crash and the
realtime-loop mid-day-restart gap were BOTH invisible to every other test
in this repo, since every other test exercises KronosOrchestrator directly
and never actually runs this script. Testing the entry point separately is
the fix for that blind spot, not just a fix for the bugs it found.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from kronos import KronosOrchestrator, Phase, load_config
from kronos.data_pipeline import DailyMemory
import run_kronos


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg.override("data.tickers", ["AAA", "BBB", "CCC"])
    cfg.override("nightmare.n_futures", 32)
    cfg.override("nightmare.batch_size", 16)
    cfg.override("evolution.population_size", 6)
    cfg.override("evolution.n_generations", 1)
    cfg.override("evolution.top_k", 3)
    cfg.override("trading.db_path", str(tmp_path / "trades.db"))
    cfg.override("orchestrator.checkpoint_dir", str(tmp_path / "models"))
    cfg.override("orchestrator.report_dir", str(tmp_path / "reports"))
    cfg.override("orchestrator.veto_file", str(tmp_path / "veto.txt"))
    cfg.override("run.log_dir", str(tmp_path))
    return cfg


def make_memory(config) -> DailyMemory:
    rng = np.random.default_rng(4)
    tickers = list(config.data.tickers)
    dates = pd.bdate_range("2026-01-01", periods=30)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (30, len(tickers))), axis=0)),
        index=dates, columns=tickers,
    )
    volumes = pd.DataFrame(
        rng.integers(1_000_000, 50_000_000, (30, len(tickers))).astype(float),
        index=dates, columns=tickers,
    )
    returns = prices.pct_change().fillna(0.0)
    vix = pd.Series(20.0, index=dates, name="VIX")
    return DailyMemory(
        as_of=datetime.now(timezone.utc),
        prices=prices, volumes=volumes, returns=returns, vix=vix,
        sentiment={t: 0.0 for t in tickers},
        macro={"vix_last": 20.0, "vix_mean_20d": 20.0,
               "market_return_1d": 0.0, "market_vol_20d": 0.01},
        source_used="synthetic",
    )


class TestCatchUp:
    def test_starting_mid_reflex_window_catches_up_all_pre_market_phases(self, config):
        """The exact bug: booting at 11:49 UTC (inside REFLEX) must not
        skip today's digestion/nightmare/evolution/adaptation/report."""
        orch = KronosOrchestrator(config)
        memory = make_memory(config)
        mid_day = datetime(2026, 3, 2, 11, 49, tzinfo=timezone.utc)

        with patch.object(orch.pipeline, "run_sync", return_value=memory):
            executed = set()
            run_kronos.catch_up(orch, executed, day=1, now=mid_day)

        assert executed == {
            "digestion", "nightmare", "evolution", "adaptation", "report",
        }
        assert orch.state.memory is not None, (
            "Digestion must have actually run - this is what REFLEX ticks "
            "depend on for the rest of the day"
        )
        assert orch.master_model is not None

    def test_starting_early_only_catches_up_phases_already_open(self, config):
        """Booting at 03:00 UTC (inside NIGHTMARE, before EVOLUTION opens
        at 04:00) must run digestion+nightmare only - not jump ahead."""
        orch = KronosOrchestrator(config)
        memory = make_memory(config)
        early = datetime(2026, 3, 2, 3, 0, tzinfo=timezone.utc)

        with patch.object(orch.pipeline, "run_sync", return_value=memory):
            executed = set()
            run_kronos.catch_up(orch, executed, day=1, now=early)

        assert executed == {"digestion", "nightmare"}
        assert "evolution" not in executed
        assert orch.state.evolution is None, (
            "Evolution's window hasn't opened yet - catch-up must not run it early"
        )

    def test_starting_at_midnight_catches_up_nothing(self, config):
        """Booting exactly at 00:00 (DIGESTION just opened) needs no
        catch-up beyond what the main loop will do on its own."""
        orch = KronosOrchestrator(config)
        memory = make_memory(config)
        midnight = datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)

        with patch.object(orch.pipeline, "run_sync", return_value=memory):
            executed = set()
            run_kronos.catch_up(orch, executed, day=1, now=midnight)

        assert executed == {"digestion"}

    def test_already_executed_phases_are_not_rerun(self, config):
        """If digestion already ran (e.g. a prior catch-up), a second call
        must not repeat it."""
        orch = KronosOrchestrator(config)
        memory = make_memory(config)
        mid_day = datetime(2026, 3, 2, 11, 0, tzinfo=timezone.utc)

        call_count = {"n": 0}

        def counting_run_sync(filings=None):
            call_count["n"] += 1
            return memory

        with patch.object(orch.pipeline, "run_sync", side_effect=counting_run_sync):
            executed = {"digestion"}
            run_kronos.catch_up(orch, executed, day=1, now=mid_day)

        assert call_count["n"] == 0, "Already-executed digestion must not re-run"
        assert executed == {
            "digestion", "nightmare", "evolution", "adaptation", "report",
        }

    def test_starting_in_logging_window_still_catches_up_everything(self, config):
        """Booting at 23:00 UTC (LOGGING window) must still catch up the
        full pre-market sequence, not skip it entirely."""
        orch = KronosOrchestrator(config)
        memory = make_memory(config)
        late = datetime(2026, 3, 2, 23, 0, tzinfo=timezone.utc)

        with patch.object(orch.pipeline, "run_sync", return_value=memory):
            executed = set()
            run_kronos.catch_up(orch, executed, day=1, now=late)

        assert executed == {
            "digestion", "nightmare", "evolution", "adaptation", "report",
        }


class TestRunpodWiring:
    """RunPod pod orchestration now runs entirely in GitHub Actions
    (.github/workflows/train-runpod.yml), which delivers a checkpoint
    onto this box directly - scripts/run_kronos.py has nothing to kick
    off, only maybe_adopt_runpod_checkpoint() to poll every main-loop
    iteration, cheaply, never blocking."""

    def test_run_realtime_polls_adoption_every_iteration_without_blocking(self, config):
        from unittest.mock import patch as _patch
        from zoneinfo import ZoneInfo

        orch = KronosOrchestrator(config)
        et_tz = ZoneInfo("America/New_York")
        during_market_hours = datetime(2026, 3, 2, 10, 0, tzinfo=et_tz)

        call_count = {"n": 0}

        def fake_sleep(seconds):
            call_count["n"] += 1
            if call_count["n"] >= 3:
                run_kronos._shutdown = True

        with _patch("run_kronos._exchange_now", return_value=during_market_hours), \
             _patch.object(orch, "phase_for", return_value=Phase.REFLEX), \
             _patch.object(orch.pipeline, "run_sync", return_value=make_memory(config)), \
             _patch("run_kronos.fetch_live_bar", return_value=({}, {})), \
             _patch.object(orch, "maybe_adopt_runpod_checkpoint") as mock_adopt, \
             _patch("run_kronos.time.sleep", side_effect=fake_sleep):
            try:
                run_kronos.run_realtime(orch, n_days=1)
            finally:
                run_kronos._shutdown = False

        assert mock_adopt.call_count == 3, (
            "must be polled once per loop iteration, cheaply - not skipped, "
            "not blocked on"
        )


class TestExchangeTimezone:
    """Regression: run_realtime()/catch_up() used to feed raw UTC wall-clock
    time into orchestrator.phase_for(), which reads only the wall-clock
    digits off whatever datetime it's given and compares them against
    schedule boundaries that are exchange-local (America/New_York) by
    design (config.yaml: "24h clock, exchange timezone"; see also
    TestOrchestrator.test_orc01_normal_daily_cycle in test_kronos_chaos.py,
    which feeds phase_for() naive ET wall-clock times directly). A server
    always runs in UTC, so this silently ran the REFLEX trading window
    hours off from real NYSE hours every single day."""

    def test_exchange_now_returns_et_digits_not_utc_digits(self, config):
        """At a known instant, _exchange_now() must return the configured
        exchange's wall clock, not the server's UTC wall clock. Uses ET
        explicitly here (independent of Kronos's actual deployed
        exchange) since this is testing the generic UTC-to-exchange-time
        conversion mechanism, not any particular timezone."""
        from datetime import datetime, timezone
        from unittest.mock import patch as _patch
        from zoneinfo import ZoneInfo

        config.override("run.timezone", "America/New_York")
        orch = KronosOrchestrator(config)
        # 2026-08-17 13:25 UTC = 09:25 ET (EDT, UTC-4 in August)
        fixed_utc = datetime(2026, 8, 17, 13, 25, tzinfo=timezone.utc)

        class _FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_utc.astimezone(tz) if tz else fixed_utc

        with _patch("run_kronos.datetime", _FixedDatetime):
            et_now = run_kronos._exchange_now(orch)

        assert et_now.tzinfo is not None
        assert et_now.astimezone(ZoneInfo("America/New_York")).hour == 9
        assert et_now.hour == 9, (
            "_exchange_now() must return ET wall-clock digits (09:xx), "
            "not the UTC wall-clock digits (13:xx) - phase_for() only "
            "ever looks at .time(), it does not convert timezones itself"
        )

    def test_utc_digits_fed_directly_would_wrongly_resolve_to_reflex(self, config):
        """Demonstrates the exact bug this fixes: feeding phase_for() the
        raw UTC datetime at real ET market-open-minus-5-minutes wrongly
        reports REFLEX (already trading) instead of REPORT (pre-market),
        because 13:25 (UTC digits) >= the 09:30 REFLEX boundary even
        though the real ET time is only 09:25 - market hasn't opened."""
        from datetime import datetime, timezone

        orch = KronosOrchestrator(config)
        real_et_time_is_9_25_am = datetime(2026, 8, 17, 13, 25, tzinfo=timezone.utc)

        buggy_phase = orch.phase_for(real_et_time_is_9_25_am)
        assert buggy_phase == Phase.REFLEX, (
            "sanity check: confirms phase_for() ignores tzinfo and reads "
            "raw wall-clock digits, which is exactly why the caller must "
            "convert to exchange time first"
        )

    def test_catch_up_default_now_resolves_via_exchange_time(self, config):
        """catch_up() with no explicit now= must ask phase_for() using
        exchange-local time (via _exchange_now), not datetime.now(utc)."""
        from datetime import datetime, timezone
        from unittest.mock import patch as _patch
        from zoneinfo import ZoneInfo

        orch = KronosOrchestrator(config)
        memory = make_memory(config)
        et_tz = ZoneInfo("America/New_York")
        # Real ET time: 09:25 - five minutes before market open, still
        # inside the REPORT window (opens 06:00, REFLEX opens 09:30).
        fixed_et = datetime(2026, 3, 2, 9, 25, tzinfo=et_tz)

        seen_now = []
        real_phase_for = orch.phase_for

        def spying_phase_for(now):
            seen_now.append(now)
            return real_phase_for(now)

        with _patch.object(orch, "phase_for", side_effect=spying_phase_for), \
             _patch("run_kronos._exchange_now", return_value=fixed_et), \
             _patch.object(orch.pipeline, "run_sync", return_value=memory):
            executed = set()
            run_kronos.catch_up(orch, executed, day=1)

        assert seen_now == [fixed_et], (
            "catch_up() must resolve its own now via _exchange_now(), not "
            "compute a fresh datetime.now(timezone.utc) internally"
        )
        # 09:25 ET is before REFLEX (09:30) - REPORT is the last window
        # open, so catch-up must run all five pre-market phases, but must
        # NOT have wrongly concluded the market was already open.
        assert executed == {
            "digestion", "nightmare", "evolution", "adaptation", "report",
        }


class TestFetchLiveBar:
    """Regression: run_reflex_tick() only executes a trade when bar_prices
    is non-empty (kronos/orchestrator.py's `if bar_prices:` gate). The
    real-time loop used to call run_reflex_tick(vix) with no price data
    at all, so it computed a signal every minute and never acted on it -
    equity stayed frozen at the starting balance for the entire 365-day
    run. fetch_live_bar() is what closes that gap."""

    def test_single_ticker_flat_columns(self):
        from unittest.mock import MagicMock, patch as _patch

        idx = pd.date_range("2026-01-01 09:30", periods=2, freq="1min")
        fake_data = pd.DataFrame(
            {"Close": [101.0, 101.5], "Volume": [1000.0, 1200.0]}, index=idx,
        )
        with _patch("yfinance.download", return_value=fake_data):
            prices, volumes = run_kronos.fetch_live_bar(["AAA"])
        assert prices == {"AAA": 101.5}
        assert volumes == {"AAA": 1200.0}

    def test_multi_ticker_multiindex_columns(self):
        from unittest.mock import patch as _patch

        idx = pd.date_range("2026-01-01 09:30", periods=2, freq="1min")
        cols = pd.MultiIndex.from_product(
            [["Close", "Volume"], ["AAA", "BBB"]]
        )
        fake_data = pd.DataFrame(
            [[100.0, 200.0, 1000.0, 2000.0], [100.5, 199.5, 1100.0, 2100.0]],
            index=idx, columns=cols,
        )
        with _patch("yfinance.download", return_value=fake_data):
            prices, volumes = run_kronos.fetch_live_bar(["AAA", "BBB"])
        assert prices == {"AAA": 100.5, "BBB": 199.5}
        assert volumes == {"AAA": 1100.0, "BBB": 2100.0}

    def test_missing_import_returns_empty_not_raise(self):
        from unittest.mock import patch as _patch

        with _patch.dict("sys.modules", {"yfinance": None}):
            prices, volumes = run_kronos.fetch_live_bar(["AAA"])
        assert prices == {}
        assert volumes == {}

    def test_network_exception_returns_empty_not_raise(self):
        from unittest.mock import patch as _patch

        with _patch("yfinance.download", side_effect=ConnectionError("down")):
            prices, volumes = run_kronos.fetch_live_bar(["AAA"])
        assert prices == {}
        assert volumes == {}

    def test_empty_frame_returns_empty(self):
        from unittest.mock import patch as _patch

        with _patch("yfinance.download", return_value=pd.DataFrame()):
            prices, volumes = run_kronos.fetch_live_bar(["AAA"])
        assert prices == {}
        assert volumes == {}

    def test_download_called_with_threads_disabled(self):
        """Regression: yf.download()'s default threaded mode leaks one
        never-closed sqlite connection (yfinance's own tz/cookie cache)
        per worker thread per call. Called every REFLEX minute, that
        accumulated open fds until the process could no longer open its
        own trades.db (~61 minutes after each restart, confirmed live on
        the Hetzner box). threads=False must always be passed."""
        from unittest.mock import MagicMock, patch as _patch

        idx = pd.date_range("2026-01-01 09:30", periods=1, freq="1min")
        fake_data = pd.DataFrame({"Close": [101.0], "Volume": [1000.0]}, index=idx)
        mock_download = MagicMock(return_value=fake_data)
        with _patch("yfinance.download", mock_download):
            run_kronos.fetch_live_bar(["AAA"])
        assert mock_download.call_args.kwargs.get("threads") is False


class TestRealtimeExecutesTrades:
    def test_reflex_tick_receives_live_prices_and_can_trade(self, config):
        """End-to-end: run_realtime()'s REFLEX branch must fetch a live
        bar and pass it into run_reflex_tick, so trader.execute() is
        reachable - not called with vix only, which silently disabled
        all trading in real-time mode."""
        from datetime import datetime
        from unittest.mock import patch as _patch
        from zoneinfo import ZoneInfo

        orch = KronosOrchestrator(config)
        memory = make_memory(config)
        et_tz = ZoneInfo("America/New_York")
        during_market_hours = datetime(2026, 3, 2, 10, 0, tzinfo=et_tz)
        live_prices = {t: 100.0 for t in memory.tickers}
        live_volumes = {t: 5_000_000.0 for t in memory.tickers}

        call_count = {"n": 0}

        def fake_sleep(seconds):
            call_count["n"] += 1
            if call_count["n"] >= 1:
                run_kronos._shutdown = True

        captured = {}
        real_run_reflex_tick = orch.run_reflex_tick

        def spying_run_reflex_tick(vix_value, bar_prices=None, bar_volumes=None, now=None):
            captured["bar_prices"] = bar_prices
            captured["bar_volumes"] = bar_volumes
            return real_run_reflex_tick(vix_value, bar_prices, bar_volumes, now=now)

        with _patch("run_kronos._exchange_now", return_value=during_market_hours), \
             _patch.object(orch, "phase_for", return_value=Phase.REFLEX), \
             _patch.object(orch.pipeline, "run_sync", return_value=memory), \
             _patch("run_kronos.fetch_live_bar", return_value=(live_prices, live_volumes)), \
             _patch.object(orch, "run_reflex_tick", side_effect=spying_run_reflex_tick), \
             _patch("run_kronos.time.sleep", side_effect=fake_sleep):
            try:
                run_kronos.run_realtime(orch, n_days=1)
            finally:
                run_kronos._shutdown = False

        assert captured["bar_prices"] == live_prices, (
            "run_reflex_tick must receive the fetched live prices, not "
            "be called with vix alone"
        )
        assert captured["bar_volumes"] == live_volumes


class TestReflexTickLogging:
    def test_every_reflex_tick_logs_a_progress_line(self, config, caplog):
        """Regression: the heartbeat log used to fire only every 15th tick,
        leaving up to a 15-minute silent gap after every restart (exactly
        what looked like a hang after redeploying the exchange-timezone
        fix). Every REFLEX iteration must log now, so a fresh boot shows
        progress within ~60s instead of a long unexplained pause."""
        import logging
        from datetime import datetime
        from unittest.mock import patch as _patch
        from zoneinfo import ZoneInfo

        orch = KronosOrchestrator(config)
        memory = make_memory(config)
        et_tz = ZoneInfo("America/New_York")
        during_market_hours = datetime(2026, 3, 2, 10, 0, tzinfo=et_tz)

        call_count = {"n": 0}

        def fake_sleep(seconds):
            call_count["n"] += 1
            if call_count["n"] >= 3:
                run_kronos._shutdown = True

        with _patch("run_kronos._exchange_now", return_value=during_market_hours), \
             _patch.object(orch, "phase_for", return_value=Phase.REFLEX), \
             _patch.object(orch.pipeline, "run_sync", return_value=memory), \
             _patch("run_kronos.fetch_live_bar", return_value=({}, {})), \
             _patch("run_kronos.time.sleep", side_effect=fake_sleep), \
             caplog.at_level(logging.INFO, logger="kronos.main"):
            try:
                run_kronos.run_realtime(orch, n_days=1)
            finally:
                run_kronos._shutdown = False  # don't leak into other tests

        tick_lines = [r for r in caplog.records if "reflex tick" in r.message]
        assert len(tick_lines) == 3, (
            "every REFLEX iteration must produce a log line, not just "
            "every 15th"
        )


class TestResumeDayCount:
    """Regression: run_realtime() used to hardcode day = 1 on every
    process start, with no attempt to recover how far a prior run had
    gotten. Since deploy-hetzner.yml restarts kronos.service on every
    push, this silently reset the entire paper-trading campaign back to
    day 1 with a fresh account on every single deploy - the campaign
    could never accumulate more than the time between two deploys.
    trades.db itself survives restarts on disk (only git-tracked files
    are touched by `git reset --hard`); this is about actually reading
    it back at startup instead of ignoring it."""

    def test_resume_day_starts_at_1_with_no_prior_history(self, config):
        orch = KronosOrchestrator(config)
        assert run_kronos._resume_day(orch) == 1

    def test_resume_day_continues_after_a_closed_day(self, config):
        orch = KronosOrchestrator(config)
        orch.trader.close_day(1, {t: 100.0 for t in config.data.tickers})
        assert run_kronos._resume_day(orch) == 2

    def test_resume_day_continues_after_several_closed_days(self, config):
        orch = KronosOrchestrator(config)
        for day in (1, 2, 3):
            orch.trader.close_day(day, {t: 100.0 for t in config.data.tickers})
        assert run_kronos._resume_day(orch) == 4

    def test_resume_day_survives_a_simulated_process_restart(self, config):
        """The actual regression scenario: one orchestrator/trader closes
        a day and shuts down (simulating a deploy killing the process);
        a SECOND, freshly-constructed orchestrator pointed at the same
        db_path (simulating the next process start) must pick up where
        it left off, not reset to day 1."""
        first = KronosOrchestrator(config)
        first.trader.close_day(1, {t: 100.0 for t in config.data.tickers})
        first.trader.close()

        second = KronosOrchestrator(config)
        assert run_kronos._resume_day(second) == 2

    def test_run_realtime_starts_at_the_resumed_day_not_hardcoded_1(self, config):
        """End-to-end: run_realtime() itself (not just _resume_day() in
        isolation) must actually use the resumed day count."""
        from datetime import datetime
        from unittest.mock import patch as _patch
        from zoneinfo import ZoneInfo

        orch = KronosOrchestrator(config)
        orch.trader.close_day(1, {t: 100.0 for t in config.data.tickers})
        memory = make_memory(config)
        et_tz = ZoneInfo("America/New_York")
        during_market_hours = datetime(2026, 3, 2, 10, 0, tzinfo=et_tz)

        def fake_sleep(seconds):
            run_kronos._shutdown = True

        with _patch("run_kronos._exchange_now", return_value=during_market_hours), \
             _patch.object(orch, "phase_for", return_value=Phase.REFLEX), \
             _patch.object(orch.pipeline, "run_sync", return_value=memory), \
             _patch("run_kronos.fetch_live_bar", return_value=({}, {})), \
             _patch("run_kronos.time.sleep", side_effect=fake_sleep):
            try:
                run_kronos.run_realtime(orch, n_days=365)
            finally:
                run_kronos._shutdown = False

        assert orch.state.day == 2, (
            "a restart after day 1 already closed must resume at day 2, "
            "not silently restart the whole campaign at day 1"
        )


class TestLoggingSetup:
    def test_log_directory_created_before_file_handler(self, tmp_path, monkeypatch):
        """Regression: logging.FileHandler("logs/kronos.log") used to run
        at import time, before the logs/ directory existed, crashing
        every single start under systemd (WorkingDirectory=/opt/prometheus,
        fresh checkout, no logs/ dir yet). Simulates that exact scenario:
        a cwd with no logs/ directory."""
        monkeypatch.chdir(tmp_path)
        assert not (tmp_path / "logs").exists()

        import importlib
        import run_kronos as rk
        importlib.reload(rk)

        assert (tmp_path / "logs").is_dir(), (
            "logs/ must exist by the time the module-level FileHandler runs"
        )
        assert (tmp_path / "logs" / "kronos.log").exists()
