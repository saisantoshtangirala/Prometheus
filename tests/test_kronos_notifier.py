"""
Tests for kronos/notifier.py - the daily WhatsApp progress digest.

Every test mocks the actual HTTP call; nothing here should ever hit a
real network endpoint. The core invariant under test: a notification
problem (missing config, network failure, malformed response) must NEVER
propagate out and disrupt the trading loop - send() always returns a
bool, never raises, and the orchestrator's integration point treats
failures as fully silent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos.config import load_config
from kronos.notifier import MAX_MESSAGE_CHARS, WhatsAppNotifier


@pytest.fixture
def config():
    cfg = load_config()
    cfg.override("notifications.enabled", True)
    cfg.override("run.total_days", 365)
    return cfg


class TestEnabledGating:
    def test_disabled_without_env_vars(self, config, monkeypatch):
        monkeypatch.delenv("KRONOS_WHATSAPP_PHONE", raising=False)
        monkeypatch.delenv("KRONOS_WHATSAPP_APIKEY", raising=False)
        notifier = WhatsAppNotifier(config)
        assert not notifier.enabled

    def test_disabled_with_only_phone_set(self, config, monkeypatch):
        monkeypatch.setenv("KRONOS_WHATSAPP_PHONE", "15551234567")
        monkeypatch.delenv("KRONOS_WHATSAPP_APIKEY", raising=False)
        notifier = WhatsAppNotifier(config)
        assert not notifier.enabled

    def test_disabled_when_config_flag_off(self, monkeypatch):
        cfg = load_config()
        cfg.override("notifications.enabled", False)
        monkeypatch.setenv("KRONOS_WHATSAPP_PHONE", "15551234567")
        monkeypatch.setenv("KRONOS_WHATSAPP_APIKEY", "999999")
        notifier = WhatsAppNotifier(cfg)
        assert not notifier.enabled

    def test_enabled_with_everything_set(self, config, monkeypatch):
        monkeypatch.setenv("KRONOS_WHATSAPP_PHONE", "15551234567")
        monkeypatch.setenv("KRONOS_WHATSAPP_APIKEY", "999999")
        notifier = WhatsAppNotifier(config)
        assert notifier.enabled

    def test_missing_notifications_section_disabled_by_default(self, monkeypatch):
        """A config built without any notifications key at all must not crash."""
        cfg = load_config()
        cfg._data.pop("notifications", None)
        monkeypatch.setenv("KRONOS_WHATSAPP_PHONE", "15551234567")
        monkeypatch.setenv("KRONOS_WHATSAPP_APIKEY", "999999")
        notifier = WhatsAppNotifier(cfg)
        assert not notifier.enabled


class TestSend:
    def test_send_noop_when_disabled(self, config, monkeypatch):
        monkeypatch.delenv("KRONOS_WHATSAPP_PHONE", raising=False)
        notifier = WhatsAppNotifier(config)
        with patch("requests.get") as mock_get:
            result = notifier.send("test")
        assert result is False
        mock_get.assert_not_called()

    def test_send_success(self, config, monkeypatch):
        monkeypatch.setenv("KRONOS_WHATSAPP_PHONE", "15551234567")
        monkeypatch.setenv("KRONOS_WHATSAPP_APIKEY", "999999")
        notifier = WhatsAppNotifier(config)
        mock_resp = MagicMock(status_code=200, text="Message queued")
        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = notifier.send("hello")
        assert result is True
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["phone"] == "15551234567"
        assert kwargs["params"]["apikey"] == "999999"
        assert kwargs["params"]["text"] == "hello"

    def test_send_failure_status_code_returns_false_not_raise(self, config, monkeypatch):
        monkeypatch.setenv("KRONOS_WHATSAPP_PHONE", "15551234567")
        monkeypatch.setenv("KRONOS_WHATSAPP_APIKEY", "bad")
        notifier = WhatsAppNotifier(config)
        mock_resp = MagicMock(status_code=401, text="Invalid apikey")
        with patch("requests.get", return_value=mock_resp):
            result = notifier.send("hello")
        assert result is False

    def test_send_network_exception_returns_false_not_raise(self, config, monkeypatch):
        monkeypatch.setenv("KRONOS_WHATSAPP_PHONE", "15551234567")
        monkeypatch.setenv("KRONOS_WHATSAPP_APIKEY", "999999")
        notifier = WhatsAppNotifier(config)
        with patch("requests.get", side_effect=ConnectionError("network unreachable")):
            result = notifier.send("hello")
        assert result is False, "A network failure must never raise out of send()"

    def test_message_truncated_to_max_length(self, config, monkeypatch):
        monkeypatch.setenv("KRONOS_WHATSAPP_PHONE", "15551234567")
        monkeypatch.setenv("KRONOS_WHATSAPP_APIKEY", "999999")
        notifier = WhatsAppNotifier(config)
        huge_text = "x" * (MAX_MESSAGE_CHARS + 500)
        mock_resp = MagicMock(status_code=200, text="ok")
        with patch("requests.get", return_value=mock_resp) as mock_get:
            notifier.send(huge_text)
        sent_text = mock_get.call_args.kwargs["params"]["text"]
        assert len(sent_text) <= MAX_MESSAGE_CHARS


class TestReportFormatting:
    def test_percentage_calculation(self, config):
        notifier = WhatsAppNotifier(config)
        text = notifier.build_daily_report(
            day=73, stats={"equity": 100000.0, "pnl": 0.0, "n_trades": 0},
            total_days=365,
        )
        assert "Day 73/365" in text
        assert "20.0%" in text   # 73/365 = 20.0%

    def test_core_fields_present(self, config):
        notifier = WhatsAppNotifier(config)
        text = notifier.build_daily_report(
            day=1,
            stats={"equity": 101234.56, "pnl": 456.78, "sharpe": 1.42,
                   "n_trades": 3, "directional_accuracy": 0.55,
                   "max_position_pct": 0.18},
            total_days=365,
        )
        assert "$101,234.56" in text
        assert "+456.78" in text
        assert "1.42" in text
        assert "Trades today: 3" in text
        assert "55%" in text

    def test_optional_fields_omitted_when_not_given(self, config):
        notifier = WhatsAppNotifier(config)
        text = notifier.build_daily_report(
            day=1, stats={"equity": 100000.0, "pnl": 0.0, "n_trades": 0},
            total_days=365,
        )
        assert "Sharpe" not in text
        assert "regime" not in text.lower() or "Market regime" not in text
        assert "NEAT" not in text

    def test_phase_failures_surfaced(self, config):
        notifier = WhatsAppNotifier(config)
        text = notifier.build_daily_report(
            day=5, stats={"equity": 99000.0, "pnl": -50.0, "n_trades": 0},
            total_days=365,
            phase_failures={"nightmare": "diffusion collapse"},
        )
        assert "WARNINGS" in text
        assert "nightmare" in text

    def test_kalman_repair_flags_filtered_out_as_noise(self, config):
        """Routine data repair flags shouldn't clutter a personal digest -
        only genuinely notable flags (dropped tickers, stale data) should."""
        notifier = WhatsAppNotifier(config)
        text = notifier.build_daily_report(
            day=1, stats={"equity": 100000.0, "pnl": 0.0, "n_trades": 0},
            total_days=365,
            quality_flags=["kalman_repaired:AAPL", "kalman_repaired:SPY"],
        )
        assert "kalman_repaired" not in text

    def test_notable_flags_still_shown(self, config):
        notifier = WhatsAppNotifier(config)
        text = notifier.build_daily_report(
            day=1, stats={"equity": 100000.0, "pnl": 0.0, "n_trades": 0},
            total_days=365,
            quality_flags=["kalman_repaired:AAPL", "illiquid:XYZ:dropped"],
        )
        assert "illiquid:XYZ:dropped" in text
        assert "kalman_repaired:AAPL" not in text

    def test_reflex_regime_panic_surfaced_but_calm_hidden(self, config):
        notifier = WhatsAppNotifier(config)
        calm_text = notifier.build_daily_report(
            day=1, stats={"equity": 100000.0, "pnl": 0.0, "n_trades": 0},
            total_days=365, reflex_regime="calm",
        )
        panic_text = notifier.build_daily_report(
            day=1, stats={"equity": 100000.0, "pnl": 0.0, "n_trades": 0},
            total_days=365, reflex_regime="panic",
        )
        assert "calm" not in calm_text.lower() or "Reflex gate" not in calm_text
        assert "PANIC" in panic_text


class TestOrchestratorIntegration:
    def test_orchestrator_has_notifier(self):
        from kronos import KronosOrchestrator
        cfg = load_config()
        cfg.override("trading.db_path", ":memory:")
        orch = KronosOrchestrator(cfg)
        assert hasattr(orch, "notifier")
        orch.trader.close()

    def test_run_logging_sends_notification_when_enabled(self, tmp_path, monkeypatch):
        from kronos import KronosOrchestrator
        monkeypatch.setenv("KRONOS_WHATSAPP_PHONE", "15551234567")
        monkeypatch.setenv("KRONOS_WHATSAPP_APIKEY", "999999")

        cfg = load_config()
        cfg.override("notifications.enabled", True)
        cfg.override("trading.db_path", str(tmp_path / "trades.db"))

        orch = KronosOrchestrator(cfg)
        orch.state.day = 1
        mock_resp = MagicMock(status_code=200, text="ok")
        with patch("requests.get", return_value=mock_resp) as mock_get:
            orch.run_logging()
        mock_get.assert_called_once()
        orch.trader.close()

    def test_run_logging_never_raises_when_notifier_broken(self, tmp_path, monkeypatch):
        """A notifier exception must never take down day-close bookkeeping."""
        from kronos import KronosOrchestrator
        monkeypatch.setenv("KRONOS_WHATSAPP_PHONE", "15551234567")
        monkeypatch.setenv("KRONOS_WHATSAPP_APIKEY", "999999")

        cfg = load_config()
        cfg.override("notifications.enabled", True)
        cfg.override("trading.db_path", str(tmp_path / "trades.db"))

        orch = KronosOrchestrator(cfg)
        orch.state.day = 1
        with patch.object(orch.notifier, "send", side_effect=RuntimeError("boom")):
            stats = orch.run_logging()  # must not raise
        assert stats["equity"] > 0
        orch.trader.close()

    def test_run_logging_skips_notification_when_disabled(self, tmp_path):
        from kronos import KronosOrchestrator
        cfg = load_config()
        cfg.override("notifications.enabled", False)
        cfg.override("trading.db_path", str(tmp_path / "trades.db"))

        orch = KronosOrchestrator(cfg)
        orch.state.day = 1
        with patch("requests.get") as mock_get:
            orch.run_logging()
        mock_get.assert_not_called()
        orch.trader.close()
