"""
Tests for KronosOrchestrator's RunPod checkpoint adoption
(kronos/orchestrator.py: maybe_adopt_runpod_checkpoint and friends).

Pod orchestration itself now runs entirely in GitHub Actions
(.github/workflows/train-runpod.yml), which scp's a checkpoint straight
onto the box this process runs on. From KronosOrchestrator's point of
view that's indistinguishable from "a file showed up on disk" - so these
tests just create/update files under a fake CHECKPOINT_DIR and check
that maybe_adopt_runpod_checkpoint() notices and reacts correctly. No
network, no subprocess, no threads.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos import KronosOrchestrator, load_config
from kronos.features import n_input_features


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg.override("data.tickers", ["AAA", "BBB", "CCC"])
    cfg.override("trading.db_path", str(tmp_path / "trades.db"))
    cfg.override("orchestrator.checkpoint_dir", str(tmp_path / "models"))
    return cfg


def _write_snn_checkpoint(checkpoint_dir: Path, snn: torch.nn.Module) -> None:
    meta_dir = checkpoint_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    torch.save(snn.state_dict(), meta_dir / "snn.pt")


class TestNoLongerKicksOffTraining:
    def test_orchestrator_has_no_kick_off_method(self, config):
        """RunPod pod orchestration moved to GitHub Actions entirely -
        Kronos should have nothing left that starts a training run."""
        orch = KronosOrchestrator(config)
        assert not hasattr(orch, "kick_off_runpod_training")
        orch.trader.close()


class TestMaybeAdoptRunpodCheckpoint:
    def test_noop_when_no_checkpoint_file_exists(self, config, monkeypatch, tmp_path):
        monkeypatch.setattr("kronos.orchestrator.RUNPOD_CHECKPOINT_DIR", tmp_path / "runpod")
        orch = KronosOrchestrator(config)
        before = {k: v.clone() for k, v in orch.reflex.snn.state_dict().items()}

        orch.maybe_adopt_runpod_checkpoint()   # must not raise

        after = orch.reflex.snn.state_dict()
        assert all(torch.equal(after[k], v) for k, v in before.items())
        orch.trader.close()

    def test_adopts_a_fresh_checkpoint(self, config, monkeypatch, tmp_path):
        runpod_dir = tmp_path / "runpod"
        monkeypatch.setattr("kronos.orchestrator.RUNPOD_CHECKPOINT_DIR", runpod_dir)
        orch = KronosOrchestrator(config)

        source = type(orch.reflex.snn)(
            input_size=n_input_features(len(config.data.tickers)), layer_sizes=[32, 16],
            output_size=len(config.data.tickers),
        )
        _write_snn_checkpoint(runpod_dir, source)

        orch.maybe_adopt_runpod_checkpoint()

        after = orch.reflex.snn.state_dict()
        expected = source.state_dict()
        assert all(torch.equal(after[k], v) for k, v in expected.items())
        orch.trader.close()

    def test_persists_active_snn_after_adopting(self, config, monkeypatch, tmp_path):
        runpod_dir = tmp_path / "runpod"
        monkeypatch.setattr("kronos.orchestrator.RUNPOD_CHECKPOINT_DIR", runpod_dir)
        orch = KronosOrchestrator(config)

        source = type(orch.reflex.snn)(
            input_size=n_input_features(len(config.data.tickers)), layer_sizes=[32, 16],
            output_size=len(config.data.tickers),
        )
        _write_snn_checkpoint(runpod_dir, source)
        orch.maybe_adopt_runpod_checkpoint()

        assert Path(orch._active_snn_path()).exists()
        orch.trader.close()

    def test_does_not_reload_the_same_file_twice(self, config, monkeypatch, tmp_path):
        """GitHub Actions overwrites the checkpoint in place each night -
        if the mtime hasn't changed since we last adopted it, don't
        reload (and don't log a fresh 'adopted' line) every 30s poll."""
        runpod_dir = tmp_path / "runpod"
        monkeypatch.setattr("kronos.orchestrator.RUNPOD_CHECKPOINT_DIR", runpod_dir)
        orch = KronosOrchestrator(config)

        source = type(orch.reflex.snn)(
            input_size=n_input_features(len(config.data.tickers)), layer_sizes=[32, 16],
            output_size=len(config.data.tickers),
        )
        _write_snn_checkpoint(runpod_dir, source)
        orch.maybe_adopt_runpod_checkpoint()

        call_count = {"n": 0}
        real_load = torch.load

        def counting_load(*args, **kwargs):
            call_count["n"] += 1
            return real_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", counting_load)
        for _ in range(3):
            orch.maybe_adopt_runpod_checkpoint()

        assert call_count["n"] == 0, "must not re-load an unchanged checkpoint file"
        orch.trader.close()

    def test_adopts_again_once_the_file_is_updated(self, config, monkeypatch, tmp_path):
        runpod_dir = tmp_path / "runpod"
        monkeypatch.setattr("kronos.orchestrator.RUNPOD_CHECKPOINT_DIR", runpod_dir)
        orch = KronosOrchestrator(config)

        first = type(orch.reflex.snn)(
            input_size=n_input_features(len(config.data.tickers)), layer_sizes=[32, 16],
            output_size=len(config.data.tickers),
        )
        _write_snn_checkpoint(runpod_dir, first)
        orch.maybe_adopt_runpod_checkpoint()

        second = type(orch.reflex.snn)(
            input_size=n_input_features(len(config.data.tickers)), layer_sizes=[32, 16],
            output_size=len(config.data.tickers),
        )
        time.sleep(0.01)   # ensure a distinct mtime
        _write_snn_checkpoint(runpod_dir, second)
        orch.maybe_adopt_runpod_checkpoint()

        after = orch.reflex.snn.state_dict()
        expected = second.state_dict()
        assert all(torch.equal(after[k], v) for k, v in expected.items())
        orch.trader.close()

    def test_shape_mismatch_does_not_crash_and_is_not_retried_forever(self, config, monkeypatch, tmp_path):
        import torch.nn as nn
        runpod_dir = tmp_path / "runpod"
        monkeypatch.setattr("kronos.orchestrator.RUNPOD_CHECKPOINT_DIR", runpod_dir)
        orch = KronosOrchestrator(config)
        before = {k: v.clone() for k, v in orch.reflex.snn.state_dict().items()}

        mismatched = nn.Linear(2, 2)
        _write_snn_checkpoint(runpod_dir, mismatched)

        call_count = {"n": 0}
        real_load = torch.load

        def counting_load(*args, **kwargs):
            call_count["n"] += 1
            return real_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", counting_load)
        for _ in range(3):
            orch.maybe_adopt_runpod_checkpoint()   # must not raise

        after = orch.reflex.snn.state_dict()
        assert all(torch.equal(after[k], v) for k, v in before.items())
        assert call_count["n"] == 1, "a known-bad file must be tried once, then remembered, not retried every poll"
        orch.trader.close()


class TestRestartPersistence:
    def test_restart_restores_last_adopted_snn(self, config):
        """A persisted reflex_snn_active.pt from a prior process must be
        picked back up on the next KronosOrchestrator() construction -
        surviving a systemd restart."""
        orch1 = KronosOrchestrator(config)
        with torch.no_grad():
            for p in orch1.reflex.snn.parameters():
                p.add_(1.0)
        expected = {k: v.clone() for k, v in orch1.reflex.snn.state_dict().items()}
        orch1._persist_active_snn()
        orch1.trader.close()

        orch2 = KronosOrchestrator(config)
        for k, v in orch2.reflex.snn.state_dict().items():
            assert torch.equal(v, expected[k])
        orch2.trader.close()


class TestDailyDigestRunpodStatus:
    """The daily Telegram digest should say whether a fresh RunPod
    checkpoint was adopted that day, so the user learns the pipeline
    worked from the one message they already read - not by checking
    GitHub Actions separately."""

    def test_reports_adopted_when_checkpoint_landed_today(self, config, monkeypatch, tmp_path):
        from unittest.mock import MagicMock, patch

        runpod_dir = tmp_path / "runpod"
        monkeypatch.setattr("kronos.orchestrator.RUNPOD_CHECKPOINT_DIR", runpod_dir)
        monkeypatch.setenv("KRONOS_TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("KRONOS_TELEGRAM_CHAT_ID", "999999")
        config.override("notifications.enabled", True)

        orch = KronosOrchestrator(config)
        source = type(orch.reflex.snn)(
            input_size=n_input_features(len(config.data.tickers)), layer_sizes=[32, 16],
            output_size=len(config.data.tickers),
        )
        _write_snn_checkpoint(runpod_dir, source)
        orch.maybe_adopt_runpod_checkpoint()
        assert orch._runpod_adopted_today is True

        mock_resp = MagicMock(status_code=200, text="ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            orch.run_logging()

        sent_text = mock_post.call_args.kwargs["json"]["text"]
        assert "RunPod: adopted fresh checkpoint" in sent_text
        orch.trader.close()

    def test_flag_resets_after_run_logging(self, config, monkeypatch, tmp_path):
        runpod_dir = tmp_path / "runpod"
        monkeypatch.setattr("kronos.orchestrator.RUNPOD_CHECKPOINT_DIR", runpod_dir)
        orch = KronosOrchestrator(config)

        source = type(orch.reflex.snn)(
            input_size=n_input_features(len(config.data.tickers)), layer_sizes=[32, 16],
            output_size=len(config.data.tickers),
        )
        _write_snn_checkpoint(runpod_dir, source)
        orch.maybe_adopt_runpod_checkpoint()
        assert orch._runpod_adopted_today is True

        orch.run_logging()
        assert orch._runpod_adopted_today is False, (
            "must reset once the day's digest has gone out, so day 2 "
            "doesn't wrongly claim day 1's adoption"
        )
        orch.trader.close()

    def test_reports_unchanged_when_no_new_checkpoint_but_one_exists(self, config, monkeypatch, tmp_path):
        from unittest.mock import MagicMock, patch

        runpod_dir = tmp_path / "runpod"
        monkeypatch.setattr("kronos.orchestrator.RUNPOD_CHECKPOINT_DIR", runpod_dir)
        monkeypatch.setenv("KRONOS_TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("KRONOS_TELEGRAM_CHAT_ID", "999999")
        config.override("notifications.enabled", True)

        orch = KronosOrchestrator(config)
        source = type(orch.reflex.snn)(
            input_size=n_input_features(len(config.data.tickers)), layer_sizes=[32, 16],
            output_size=len(config.data.tickers),
        )
        _write_snn_checkpoint(runpod_dir, source)
        orch.maybe_adopt_runpod_checkpoint()
        orch.run_logging()   # day 1: adopted, then flag resets

        mock_resp = MagicMock(status_code=200, text="ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            orch.run_logging()   # day 2: nothing new landed

        sent_text = mock_post.call_args.kwargs["json"]["text"]
        assert "RunPod: none today (kept yesterday's)" in sent_text
        orch.trader.close()

    def test_no_runpod_line_when_never_adopted_anything(self, config, monkeypatch, tmp_path):
        from unittest.mock import MagicMock, patch

        monkeypatch.setattr("kronos.orchestrator.RUNPOD_CHECKPOINT_DIR", tmp_path / "runpod")
        monkeypatch.setenv("KRONOS_TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("KRONOS_TELEGRAM_CHAT_ID", "999999")
        config.override("notifications.enabled", True)

        orch = KronosOrchestrator(config)   # no checkpoint ever appears
        mock_resp = MagicMock(status_code=200, text="ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            orch.run_logging()

        sent_text = mock_post.call_args.kwargs["json"]["text"]
        assert "RunPod" not in sent_text
        orch.trader.close()
