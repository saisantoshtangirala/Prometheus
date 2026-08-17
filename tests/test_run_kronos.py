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
        """At a known instant, _exchange_now() must return the ET wall
        clock, not the server's UTC wall clock."""
        from datetime import datetime, timezone
        from unittest.mock import patch as _patch
        from zoneinfo import ZoneInfo

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
