"""
Tests for the NSE (India) trading calendar added in kronos/calendar_utils.py,
and for KronosOrchestrator actually using it (not the NYSE one) to gate
live trading, now that this deployment trades NSE.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos import (
    KronosOrchestrator,
    Phase,
    is_nse_trading_day,
    is_trading_day,
    load_config,
    next_nse_trading_day,
    nse_holidays,
)
from kronos.calendar_utils import NSE_HOLIDAYS_BY_YEAR


class TestNseCalendarSanity:
    def test_known_2026_holidays(self):
        assert not is_nse_trading_day(date(2026, 1, 26))    # Republic Day
        assert not is_nse_trading_day(date(2026, 3, 3))     # Holi
        assert not is_nse_trading_day(date(2026, 10, 2))    # Gandhi Jayanti
        assert not is_nse_trading_day(date(2026, 12, 25))   # Christmas

    def test_regular_weekday_is_a_trading_day(self):
        monday = date(2026, 3, 2)
        assert monday.weekday() == 0
        assert is_nse_trading_day(monday)

    def test_weekend_is_not_a_trading_day(self):
        saturday = date(2026, 3, 7)
        assert saturday.weekday() == 5
        assert not is_nse_trading_day(saturday)

    def test_next_trading_day_skips_weekend_and_holiday(self):
        # Republic Day 2026 (Mon Jan 26) preceded by a Sunday
        assert next_nse_trading_day(date(2026, 1, 24)) == date(2026, 1, 27)

    def test_nse_and_nyse_calendars_are_genuinely_independent(self):
        """Holi is an NSE holiday but an ordinary NYSE trading day - and
        vice versa for, say, Thanksgiving - proving the swap in
        orchestrator.py actually changed which calendar governs, not
        just coincidentally landing on shared dates like Christmas."""
        holi = date(2026, 3, 3)
        assert not is_nse_trading_day(holi)
        assert is_trading_day(holi)   # NYSE: unaffected, ordinary Tuesday

        thanksgiving = date(2026, 11, 26)
        assert not is_trading_day(thanksgiving)     # NYSE holiday
        assert is_nse_trading_day(thanksgiving)      # NSE: ordinary Thursday

    def test_unknown_year_falls_back_to_weekday_only_with_one_warning(self, caplog):
        assert 2099 not in NSE_HOLIDAYS_BY_YEAR
        with caplog.at_level(logging.WARNING, logger="kronos.calendar_utils"):
            assert nse_holidays(2099) == set()
            assert nse_holidays(2099) == set()   # second call must not warn again
        warnings = [r for r in caplog.records if "no NSE holiday list" in r.message]
        assert len(warnings) == 1


class TestOrchestratorUsesNseCalendar:
    @pytest.fixture
    def config(self, tmp_path):
        cfg = load_config()
        cfg.override("trading.db_path", str(tmp_path / "trades.db"))
        cfg.override("orchestrator.checkpoint_dir", str(tmp_path / "models"))
        return cfg

    def test_holi_is_low_power_no_reflex(self, config):
        """Holi is an NSE holiday but not a weekend and not a NYSE
        holiday - if orchestrator.py still used the old NYSE calendar,
        this date would wrongly report a full trading day."""
        orch = KronosOrchestrator(config)
        phases = orch.phases_for_day(date(2026, 3, 3))
        assert Phase.REFLEX not in phases
        assert Phase.EVOLUTION not in phases
        assert Phase.DIGESTION in phases
        orch.trader.close()

    def test_thanksgiving_is_a_full_nse_trading_day(self, config):
        """The mirror case: a NYSE holiday that is an ordinary NSE
        trading day must run the full cycle, not low-power mode."""
        orch = KronosOrchestrator(config)
        phases = orch.phases_for_day(date(2026, 11, 26))
        assert phases == set(Phase)
        orch.trader.close()
