"""
Tests for kronos/runpod_trigger.py and its wiring into KronosOrchestrator.

Every network/subprocess boundary (requests, ssh, scp, rsync) is mocked -
nothing here ever touches a real RunPod pod. The core invariant under
test, mirroring every other module built this session: a RunPod problem
(API error, unreachable pod, SSH failure, timeout, an outright exception)
must NEVER crash the daily cycle or leave a pod running - it always
returns a TrainingResult(success=False, ...) and always deletes any pod
that was created, via try/finally.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date as ddate, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos import runpod_trigger
from kronos.config import load_config
from kronos.orchestrator import RUNPOD_EVOLUTION_BUDGET_SECONDS, KronosOrchestrator
from kronos.runpod_trigger import TrainingResult


@pytest.fixture(autouse=True)
def _isolate_module_paths(tmp_path, monkeypatch):
    """Every test gets its own lock file / checkpoint dir - never the
    real repo-relative paths this module defaults to. Also shrinks the
    reachability/SSH timeouts: those loops key off time.monotonic(), not
    time.sleep(), so mocking only time.sleep still leaves a real
    multi-minute busy-loop unless these are cut down too."""
    monkeypatch.setattr(runpod_trigger, "LOCK_FILE", tmp_path / "runpod.lock")
    monkeypatch.setattr(runpod_trigger, "CHECKPOINT_DIR", tmp_path / "checkpoints" / "runpod")
    monkeypatch.setattr(runpod_trigger, "POD_REACHABLE_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(runpod_trigger, "SSH_READY_TIMEOUT_SECONDS", 1)
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    yield


def _pod_response(status="RUNNING", ip="1.2.3.4", port=22):
    return {"id": "pod123", "desiredStatus": status, "publicIp": ip,
            "portMappings": {"22": port}}


class TestLock:
    def test_already_running_skips_without_any_api_call(self, tmp_path):
        runpod_trigger.LOCK_FILE.write_text("existing,fresh")
        with patch("requests.post") as mock_post:
            result = runpod_trigger.trigger_training_and_wait()
        assert result.success is False
        assert result.reason == "already_running"
        mock_post.assert_not_called()

    def test_stale_lock_is_cleared_and_proceeds(self, tmp_path):
        runpod_trigger.LOCK_FILE.write_text("stale")
        stale_time = time.time() - runpod_trigger.STALE_LOCK_SECONDS - 60
        os.utime(runpod_trigger.LOCK_FILE, (stale_time, stale_time))

        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200, json=lambda: {"id": "podX"}, raise_for_status=lambda: None,
            )
            with patch("requests.get") as mock_get:
                # never becomes reachable - just proves we got past the lock
                mock_get.return_value = MagicMock(
                    status_code=200, json=lambda: {"desiredStatus": "PROVISIONING"},
                    raise_for_status=lambda: None,
                )
                with patch("time.sleep"):
                    result = runpod_trigger.trigger_training_and_wait()

        assert result.reason == "pod_unreachable"
        mock_post.assert_called_once()

    def test_lock_always_released_even_on_exception(self, tmp_path):
        with patch("requests.post", side_effect=ConnectionError("network down")):
            result = runpod_trigger.trigger_training_and_wait()
        assert result.success is False
        assert not runpod_trigger.LOCK_FILE.exists()

    def test_missing_api_key_returns_cleanly(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        with patch("requests.post") as mock_post:
            result = runpod_trigger.trigger_training_and_wait()
        assert result.success is False
        assert result.reason == "no_api_key"
        mock_post.assert_not_called()
        assert not runpod_trigger.LOCK_FILE.exists()


class TestPodLifecycle:
    def _mock_create(self):
        return MagicMock(status_code=200, json=lambda: {"id": "podX"}, raise_for_status=lambda: None)

    def test_pod_deleted_when_never_reachable(self, tmp_path):
        with patch("requests.post", return_value=self._mock_create()) as mock_post, \
             patch("requests.get", return_value=MagicMock(
                 status_code=200, json=lambda: {"desiredStatus": "PROVISIONING"},
                 raise_for_status=lambda: None)), \
             patch("requests.delete") as mock_delete, \
             patch("time.sleep"):
            result = runpod_trigger.trigger_training_and_wait()

        assert result.reason == "pod_unreachable"
        mock_delete.assert_called_once()
        assert "podX" in mock_delete.call_args[0][0]

    def test_pod_deleted_when_ssh_never_comes_up(self, tmp_path):
        with patch("requests.post", return_value=self._mock_create()), \
             patch("requests.get", return_value=MagicMock(
                 status_code=200, json=lambda: _pod_response(), raise_for_status=lambda: None)), \
             patch("requests.delete") as mock_delete, \
             patch("subprocess.run", return_value=MagicMock(returncode=1)), \
             patch("time.sleep"):
            result = runpod_trigger.trigger_training_and_wait()

        assert result.reason == "ssh_unreachable"
        mock_delete.assert_called_once()

    def test_pod_deleted_even_on_unexpected_exception(self, tmp_path):
        with patch("requests.post", return_value=self._mock_create()), \
             patch("requests.get", side_effect=RuntimeError("boom mid-poll")), \
             patch("requests.delete") as mock_delete:
            result = runpod_trigger.trigger_training_and_wait()

        assert result.success is False
        assert "exception" in result.reason
        mock_delete.assert_called_once()

    def test_delete_failure_does_not_raise(self, tmp_path):
        """Even if terminating the pod itself fails, trigger_training_and_
        wait() must still return cleanly and release the lock - a failed
        delete is a critical-log emergency, not a crash."""
        with patch("requests.post", return_value=self._mock_create()), \
             patch("requests.get", side_effect=RuntimeError("boom")), \
             patch("requests.delete", side_effect=ConnectionError("also down")):
            result = runpod_trigger.trigger_training_and_wait()

        assert result.success is False
        assert not runpod_trigger.LOCK_FILE.exists()


class TestTrainingSucceeds:
    def test_full_success_path_extracts_checkpoint_and_deletes_pod(self, tmp_path):
        def fake_ssh_run(args, **kwargs):
            # _check_remote_marker parses stdout; everything else just needs returncode 0
            if "echo done" in " ".join(args) or "kronos_train_done" in " ".join(args):
                return MagicMock(returncode=0, stdout="done", stderr="")
            return MagicMock(returncode=0, stdout="ready", stderr="")

        checkpoint_tar = tmp_path / "fake.tar.gz"
        import tarfile
        src_dir = tmp_path / "src_checkpoints"
        (src_dir / "meta").mkdir(parents=True)
        (src_dir / "meta" / "snn.pt").write_bytes(b"not a real tensor, just bytes")
        with tarfile.open(checkpoint_tar, "w:gz") as tar:
            tar.add(src_dir, arcname="checkpoints")

        def fake_scp(args, **kwargs):
            # simulate scp landing the tarball at the requested dest path
            dest = Path(args[-1])
            dest.write_bytes(checkpoint_tar.read_bytes())
            return MagicMock(returncode=0)

        def fake_subprocess(args, **kwargs):
            if args[0] == "ssh":
                return fake_ssh_run(args, **kwargs)
            if args[0] == "scp":
                return fake_scp(args, **kwargs)
            if args[0] == "rsync":
                return MagicMock(returncode=0)
            raise AssertionError(f"unexpected subprocess call: {args}")

        with patch("requests.post", return_value=MagicMock(
                status_code=200, json=lambda: {"id": "podX"}, raise_for_status=lambda: None)), \
             patch("requests.get", return_value=MagicMock(
                 status_code=200, json=lambda: _pod_response(), raise_for_status=lambda: None)), \
             patch("requests.delete") as mock_delete, \
             patch("subprocess.run", side_effect=fake_subprocess), \
             patch("time.sleep"):
            result = runpod_trigger.trigger_training_and_wait()

        assert result.success is True
        assert result.tarball_path.exists()
        assert (runpod_trigger.CHECKPOINT_DIR / "meta" / "snn.pt").exists()
        mock_delete.assert_called_once()
        assert not runpod_trigger.LOCK_FILE.exists()

    def test_training_failure_marker_returns_false_and_deletes_pod(self, tmp_path):
        def fake_ssh_run(args, **kwargs):
            if "kronos_train_done" in " ".join(args):
                return MagicMock(returncode=0, stdout="failed", stderr="")
            return MagicMock(returncode=0, stdout="ready", stderr="")

        with patch("requests.post", return_value=MagicMock(
                status_code=200, json=lambda: {"id": "podX"}, raise_for_status=lambda: None)), \
             patch("requests.get", return_value=MagicMock(
                 status_code=200, json=lambda: _pod_response(), raise_for_status=lambda: None)), \
             patch("requests.delete") as mock_delete, \
             patch("subprocess.run", side_effect=lambda args, **kw: (
                 MagicMock(returncode=0) if args[0] in ("rsync",) else fake_ssh_run(args, **kw)
             )), \
             patch("time.sleep"):
            result = runpod_trigger.trigger_training_and_wait()

        assert result.success is False
        assert result.reason == "training_failed"
        mock_delete.assert_called_once()

    def test_timeout_returns_false_and_deletes_pod(self, tmp_path, monkeypatch):
        """Simulates a job that never produces a done/failed marker within
        budget - must give up, not hang forever. Uses real (tiny) sleeps
        rather than mocking the clock, since POD_REACHABLE/SSH_READY
        succeed on their first check either way - only POLL_INTERVAL_
        SECONDS needs shrinking to keep this fast."""
        monkeypatch.setattr(runpod_trigger, "POLL_INTERVAL_SECONDS", 0.02)

        def fake_ssh_run(args, **kwargs):
            return MagicMock(returncode=0, stdout="running", stderr="")

        with patch("requests.post", return_value=MagicMock(
                status_code=200, json=lambda: {"id": "podX"}, raise_for_status=lambda: None)), \
             patch("requests.get", return_value=MagicMock(
                 status_code=200, json=lambda: _pod_response(), raise_for_status=lambda: None)), \
             patch("requests.delete") as mock_delete, \
             patch("subprocess.run", side_effect=lambda args, **kw: (
                 MagicMock(returncode=0) if args[0] in ("rsync",) else fake_ssh_run(args, **kw)
             )):
            result = runpod_trigger.trigger_training_and_wait(timeout_seconds=0.1)

        assert result.success is False
        assert result.reason == "timeout"
        mock_delete.assert_called_once()


class TestLoadRunpodCheckpoint:
    def test_missing_file_returns_false_and_leaves_model_unchanged(self, tmp_path):
        model = nn.Linear(4, 4)
        before = model.weight.clone()
        loaded = runpod_trigger.load_runpod_checkpoint(model, checkpoint_dir=tmp_path)
        assert loaded is False
        assert torch.equal(model.weight, before)

    def test_valid_checkpoint_loads_in_place(self, tmp_path):
        source = nn.Linear(4, 4)
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        torch.save(source.state_dict(), meta_dir / "snn.pt")

        target = nn.Linear(4, 4)
        loaded = runpod_trigger.load_runpod_checkpoint(target, checkpoint_dir=tmp_path)
        assert loaded is True
        assert torch.equal(target.weight, source.weight)

    def test_shape_mismatch_returns_false_not_raise(self, tmp_path):
        """Regression for the real PrometheusEngine-vs-ReflexArc SNN
        output_size mismatch (n_assets//2 vs n_assets) - must degrade to
        'no checkpoint', never crash the caller."""
        source = nn.Linear(4, 4)
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        torch.save(source.state_dict(), meta_dir / "snn.pt")

        target = nn.Linear(4, 2)   # incompatible shape
        before = target.weight.clone()
        loaded = runpod_trigger.load_runpod_checkpoint(target, checkpoint_dir=tmp_path)
        assert loaded is False
        assert torch.equal(target.weight, before)


class TestOrchestratorWiring:
    @pytest.fixture
    def config(self, tmp_path):
        cfg = load_config()
        cfg.override("data.tickers", ["AAA", "BBB", "CCC"])
        cfg.override("trading.db_path", str(tmp_path / "trades.db"))
        cfg.override("orchestrator.checkpoint_dir", str(tmp_path / "models"))
        return cfg

    def test_disabled_without_api_key(self, config, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        orch = KronosOrchestrator(config)
        assert orch._runpod_enabled is False
        orch.kick_off_runpod_training(ddate(2026, 3, 2))  # a Monday
        assert orch._runpod_thread is None
        orch.trader.close()

    def test_skips_weekend(self, config, monkeypatch):
        monkeypatch.setenv("RUNPOD_API_KEY", "key")
        orch = KronosOrchestrator(config)
        orch.kick_off_runpod_training(ddate(2026, 3, 7))  # a Saturday
        assert orch._runpod_thread is None
        orch.trader.close()

    def test_kicks_off_background_thread_on_trading_day(self, config, monkeypatch):
        monkeypatch.setenv("RUNPOD_API_KEY", "key")
        orch = KronosOrchestrator(config)
        with patch("kronos.orchestrator.trigger_training_and_wait",
                    return_value=TrainingResult(False, reason="test_stub")) as mock_trigger:
            orch.kick_off_runpod_training(ddate(2026, 3, 2))
            assert orch._runpod_thread is not None
            orch._runpod_thread.join(timeout=5)
        mock_trigger.assert_called_once()
        orch.trader.close()

    def test_maybe_adopt_is_noop_before_any_kickoff(self, config):
        orch = KronosOrchestrator(config)
        orch.maybe_adopt_runpod_checkpoint()   # must not raise
        orch.trader.close()

    def test_maybe_adopt_loads_successful_checkpoint_into_reflex_snn(self, config, monkeypatch):
        monkeypatch.setenv("RUNPOD_API_KEY", "key")
        orch = KronosOrchestrator(config)

        with patch("kronos.orchestrator.load_runpod_checkpoint", return_value=True) as mock_load:
            orch._runpod_thread = MagicMock(is_alive=lambda: False)
            orch._runpod_result = TrainingResult(True, reason="ok")
            orch._runpod_adopted_today = False
            orch.maybe_adopt_runpod_checkpoint()

        mock_load.assert_called_once_with(orch.reflex.snn)
        assert orch._runpod_adopted_today is True
        assert (Path(config.orchestrator.checkpoint_dir) / "reflex_snn_active.pt").exists()
        orch.trader.close()

    def test_maybe_adopt_keeps_existing_weights_on_failure(self, config, monkeypatch, caplog):
        import logging
        monkeypatch.setenv("RUNPOD_API_KEY", "key")
        orch = KronosOrchestrator(config)
        before = {k: v.clone() for k, v in orch.reflex.snn.state_dict().items()}

        orch._runpod_thread = MagicMock(is_alive=lambda: False)
        orch._runpod_result = TrainingResult(False, reason="training_failed")
        orch._runpod_adopted_today = False
        with caplog.at_level(logging.CRITICAL, logger="kronos.orchestrator"):
            orch.maybe_adopt_runpod_checkpoint()

        assert any("keeping today's existing SNN weights" in r.message for r in caplog.records)
        assert orch._runpod_adopted_today is True
        after = orch.reflex.snn.state_dict()
        assert all(torch.equal(after[k], v) for k, v in before.items())
        orch.trader.close()

    def test_maybe_adopt_gives_up_after_budget_expires_without_blocking(self, config, monkeypatch):
        """Must never actually sleep/block for RUNPOD_EVOLUTION_BUDGET_
        SECONDS - it's a monotonic-clock deadline check, evaluated once,
        non-blocking."""
        monkeypatch.setenv("RUNPOD_API_KEY", "key")
        orch = KronosOrchestrator(config)
        orch._runpod_thread = MagicMock(is_alive=lambda: True)
        orch._runpod_kickoff_monotonic = time.monotonic() - RUNPOD_EVOLUTION_BUDGET_SECONDS - 5
        orch._runpod_adopted_today = False

        start = time.monotonic()
        orch.maybe_adopt_runpod_checkpoint()
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, "must be a non-blocking check, not a real wait"
        assert orch._runpod_adopted_today is True
        orch.trader.close()

    def test_maybe_adopt_keeps_waiting_within_budget(self, config, monkeypatch):
        monkeypatch.setenv("RUNPOD_API_KEY", "key")
        orch = KronosOrchestrator(config)
        orch._runpod_thread = MagicMock(is_alive=lambda: True)
        orch._runpod_kickoff_monotonic = time.monotonic()   # just started
        orch._runpod_adopted_today = False

        orch.maybe_adopt_runpod_checkpoint()

        assert orch._runpod_adopted_today is False   # still waiting, not given up
        orch.trader.close()

    def test_restart_restores_last_adopted_snn(self, config, monkeypatch):
        """A persisted reflex_snn_active.pt from a prior process must be
        picked back up on the next KronosOrchestrator() construction -
        surviving a systemd restart, not just an in-process day change."""
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
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
