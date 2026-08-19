"""
Tests for Phase 3 of the algo-trading-ready plan: reliability/observability
additions on top of the existing kronos/notifier.py daily digest.

  - a large single-day PnL swing gets an immediate out-of-band alert
  - trades.db is backed up once per day, with old backups pruned
  - a RunPod checkpoint that fails to adopt sends an alert (not just a log line)
  - run_kronos.py's main() sends a crash alert before re-raising an
    exception that's about to take the process down
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from kronos import KronosOrchestrator, load_config


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg.override("data.tickers", ["AAA", "BBB"])
    cfg.override("trading.db_path", str(tmp_path / "trades.db"))
    cfg.override("orchestrator.checkpoint_dir", str(tmp_path / "models"))
    cfg.override("orchestrator.report_dir", str(tmp_path / "reports"))
    cfg.override("backup.dir", str(tmp_path / "backups"))
    cfg.override("backup.max_backups", 3)
    return cfg


def _enable_notifications(cfg, monkeypatch):
    monkeypatch.setenv("KRONOS_TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("KRONOS_TELEGRAM_CHAT_ID", "999999")
    cfg.override("notifications.enabled", True)


class TestLargePnlAlert:
    def test_large_loss_triggers_extra_alert(self, config, monkeypatch):
        _enable_notifications(config, monkeypatch)
        config.override("notifications.large_pnl_alert_pct", 0.05)
        orch = KronosOrchestrator(config)
        orch.state.day = 1
        orch.trader.execute(day=1, ticker="AAA", target_weight=0.25,
                            price=100.0, bar_volume=50_000_000)
        mock_resp = MagicMock(status_code=200, text="ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            orch.run_logging({"AAA": 20.0})   # an 80% crash on the position
        # one call for the daily digest, one for the large-pnl alert
        assert mock_post.call_count == 2
        bodies = [c.kwargs["json"]["text"] for c in mock_post.call_args_list]
        assert any("large PnL move" in b for b in bodies)
        orch.trader.close()

    def test_small_move_does_not_trigger_extra_alert(self, config, monkeypatch):
        _enable_notifications(config, monkeypatch)
        orch = KronosOrchestrator(config)
        orch.state.day = 1
        mock_resp = MagicMock(status_code=200, text="ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            orch.run_logging()   # flat account, day 1, zero pnl
        assert mock_post.call_count == 1   # just the daily digest
        orch.trader.close()

    def test_never_raises_on_zero_day_start_equity(self, config, monkeypatch):
        """An edge case (day-start equity of zero) must be handled, not
        divide-by-zero crash day-close bookkeeping."""
        _enable_notifications(config, monkeypatch)
        orch = KronosOrchestrator(config)
        orch.state.day = 1
        orch.trader.cash = 0.0
        orch.trader._equity_history = [0.0]
        with patch("requests.post", return_value=MagicMock(status_code=200, text="ok")):
            stats = orch.run_logging()   # must not raise
        assert stats is not None
        orch.trader.close()


class TestTradeDbBackup:
    def test_backup_created_after_run_logging(self, config):
        orch = KronosOrchestrator(config)
        orch.state.day = 1
        orch.run_logging()
        backups = list(Path(config.backup.dir).glob("trades_day*.db"))
        assert len(backups) == 1
        orch.trader.close()

    def test_old_backups_pruned_beyond_max(self, config):
        orch = KronosOrchestrator(config)
        for day in range(1, 6):   # 5 days, max_backups=3
            orch.state.day = day
            orch.run_logging()
        backups = sorted(Path(config.backup.dir).glob("trades_day*.db"))
        assert len(backups) == 3
        assert backups[-1].name == "trades_day0005.db"
        orch.trader.close()

    def test_backup_failure_never_raises(self, config, monkeypatch):
        """A backup problem (disk full, permissions) must never take down
        day-close bookkeeping - the same discipline as the notifier."""
        orch = KronosOrchestrator(config)
        orch.state.day = 1
        with patch("shutil.copyfile", side_effect=OSError("disk full")):
            stats = orch.run_logging()   # must not raise
        assert stats is not None
        orch.trader.close()


class TestCheckpointAdoptionFailureAlert:
    def test_shape_mismatch_sends_alert(self, config, monkeypatch, tmp_path):
        _enable_notifications(config, monkeypatch)
        runpod_dir = tmp_path / "runpod"
        monkeypatch.setattr("kronos.orchestrator.RUNPOD_CHECKPOINT_DIR", runpod_dir)
        orch = KronosOrchestrator(config)

        mismatched = nn.Linear(2, 2)
        meta_dir = runpod_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        torch.save(mismatched.state_dict(), meta_dir / "snn.pt")

        with patch.object(orch.notifier, "send") as mock_send:
            orch.maybe_adopt_runpod_checkpoint()

        mock_send.assert_called_once()
        assert "FAILED" in mock_send.call_args.args[0]
        orch.trader.close()

    def test_successful_adoption_does_not_send_failure_alert(self, config, monkeypatch, tmp_path):
        _enable_notifications(config, monkeypatch)
        runpod_dir = tmp_path / "runpod"
        monkeypatch.setattr("kronos.orchestrator.RUNPOD_CHECKPOINT_DIR", runpod_dir)
        orch = KronosOrchestrator(config)

        source = type(orch.reflex.snn)(
            input_size=len(config.data.tickers), layer_sizes=[32, 16],
            output_size=len(config.data.tickers),
        )
        meta_dir = runpod_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        torch.save(source.state_dict(), meta_dir / "snn.pt")

        with patch.object(orch.notifier, "send") as mock_send:
            orch.maybe_adopt_runpod_checkpoint()

        mock_send.assert_not_called()
        orch.trader.close()


class TestCrashAlert:
    def test_uncaught_exception_sends_alert_then_reraises(self, config, monkeypatch):
        _enable_notifications(config, monkeypatch)
        import run_kronos

        orch = KronosOrchestrator(config)
        orch.state.day = 3

        monkeypatch.setattr(run_kronos, "load_config", lambda *a, **kw: config)
        monkeypatch.setattr(run_kronos, "KronosOrchestrator", lambda *a, **kw: orch)
        monkeypatch.setattr(
            run_kronos, "run_realtime",
            MagicMock(side_effect=RuntimeError("simulated fd exhaustion")),
        )
        monkeypatch.setattr(sys, "argv", ["run_kronos.py", "--mode", "paper"])

        with patch.object(orch.notifier, "send") as mock_send:
            with pytest.raises(RuntimeError, match="simulated fd exhaustion"):
                run_kronos.main()

        mock_send.assert_called_once()
        assert "CRASHED" in mock_send.call_args.args[0]
        assert "day 3" in mock_send.call_args.args[0].lower() \
            or "Day 3" in mock_send.call_args.args[0]
        orch.trader.close()

    def test_notifier_disabled_still_reraises_cleanly(self, config, monkeypatch):
        """No Telegram configured must not change the crash behavior -
        the real exception still propagates so systemd still restarts it."""
        import run_kronos

        orch = KronosOrchestrator(config)
        monkeypatch.setattr(run_kronos, "load_config", lambda *a, **kw: config)
        monkeypatch.setattr(run_kronos, "KronosOrchestrator", lambda *a, **kw: orch)
        monkeypatch.setattr(
            run_kronos, "run_realtime",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        monkeypatch.setattr(sys, "argv", ["run_kronos.py", "--mode", "paper"])

        with pytest.raises(RuntimeError, match="boom"):
            run_kronos.main()
        orch.trader.close()
