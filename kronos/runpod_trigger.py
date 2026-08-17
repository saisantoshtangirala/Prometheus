"""
Nightly RunPod GPU-training orchestrator for Kronos.

Flow: KronosOrchestrator.kick_off_runpod_training() starts
trigger_training_and_wait() on a background thread when DIGESTION opens
on a trading day. That function is a full, blocking, synchronous nightly
job: create a pod, push this checkout onto it, run scripts/train.py,
pull the checkpoint back, always delete the pod - success, failure, or
exception. It can legitimately run for hours; that is why it is started
off the main thread rather than called directly from a phase method.

KronosOrchestrator.maybe_adopt_runpod_checkpoint() is polled from the
main realtime loop (scripts/run_kronos.py) once per iteration - a cheap,
non-blocking check, never a long join. The moment the background thread
finishes (or a fixed budget expires), it calls load_runpod_checkpoint()
here to pull the trained weights into the live trading path.

Two deliberate deviations from a literal "block Evolution on RunPod"
design, both explained in kronos/orchestrator.py where they're wired in:

1. RunPod's Pods REST API (the same one .github/workflows/train-runpod.yml
   already uses) has no "job status" concept - only pod infrastructure
   status (RUNNING/EXITED/...). There is also no output-URL endpoint for
   a custom training script. So "poll for job status" and "download the
   result" both work the way the existing GitHub Actions workflow already
   does it: SSH in, run training in the background inside the pod, poll a
   sentinel file over SSH, then scp the tarball out when it appears.

2. This module pushes the current checkout to the pod via rsync over the
   same SSH connection, instead of having the pod `git clone` with a
   token. Hetzner already has a full checkout of the target branch sitting
   right here - reusing it avoids inventing a new long-lived GitHub
   credential just for this, and avoids a rate-limited clone from a
   throwaway pod.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
import torch

logger = logging.getLogger("kronos.runpod_trigger")

RUNPOD_API = "https://rest.runpod.io/v1"
POD_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-devel-ubuntu22.04"

LOCK_FILE = Path("runpod.lock")
STALE_LOCK_SECONDS = 4 * 60 * 60          # older than this = presumed crashed, not "still running"
CHECKPOINT_DIR = Path("checkpoints/runpod")

POD_REACHABLE_TIMEOUT_SECONDS = 10 * 60   # pod must get a public IP+port within this
SSH_READY_TIMEOUT_SECONDS = 5 * 60        # sshd must come up inside the container within this
POLL_INTERVAL_SECONDS = 60
JOB_TIMEOUT_SECONDS = 3 * 60 * 60         # scripts/train.py itself must finish within this

REMOTE_DIR = "/opt/prometheus"
REMOTE_MARKER_DONE = "/tmp/kronos_train_done"
REMOTE_MARKER_FAILED = "/tmp/kronos_train_failed"
REMOTE_TARBALL = "/tmp/checkpoints_job.tar.gz"

LOCAL_REPO_DIR = Path(__file__).resolve().parent.parent


class TrainingResult:
    """Outcome of one trigger_training_and_wait() run. Never an exception -
    every failure mode is represented here so callers can fall back
    without a try/except of their own."""

    def __init__(self, success: bool, tarball_path: Optional[Path] = None, reason: str = ""):
        self.success = success
        self.tarball_path = tarball_path
        self.reason = reason

    def __repr__(self) -> str:
        return f"TrainingResult(success={self.success}, reason={self.reason!r})"


# ---------------------------------------------------------------------------
# RunPod REST API
# ---------------------------------------------------------------------------

def _api_headers() -> dict:
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY not set")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _create_pod(gpu_type: str, cloud_type: str, run_id: str) -> str:
    resp = requests.post(
        f"{RUNPOD_API}/pods",
        headers=_api_headers(),
        json={
            "name": f"kronos-nightly-{run_id}",
            "imageName": POD_IMAGE,
            "gpuTypeIds": [gpu_type],
            "gpuCount": 1,
            "cloudType": cloud_type,
            "containerDiskInGb": 20,
            "ports": ["22/tcp"],
        },
        timeout=30,
    )
    resp.raise_for_status()
    pod_id = resp.json().get("id")
    if not pod_id:
        raise RuntimeError(f"pod creation returned no id: {resp.text}")
    return pod_id


def _get_pod(pod_id: str) -> dict:
    resp = requests.get(f"{RUNPOD_API}/pods/{pod_id}", headers=_api_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _delete_pod(pod_id: str) -> None:
    """Best-effort pod deletion. Never raises - the caller (always a
    finally: block) must not itself fail to finish cleaning up the lock
    file just because this also failed. A failed delete is a cost-control
    emergency, not a normal warning - logged at CRITICAL on purpose."""
    try:
        requests.delete(f"{RUNPOD_API}/pods/{pod_id}", headers=_api_headers(), timeout=30)
        logger.info("[runpod] pod %s deleted", pod_id)
    except Exception as e:
        logger.critical(
            "[runpod] FAILED TO DELETE POD %s - check the RunPod console "
            "NOW to avoid idle billing: %s", pod_id, e,
        )


# ---------------------------------------------------------------------------
# SSH / rsync helpers - all best-effort, all raise on failure (caught by
# the caller's single try/except in trigger_training_and_wait)
# ---------------------------------------------------------------------------

def _ssh_key_path() -> str:
    return os.environ.get("RUNPOD_SSH_KEY_PATH", "/root/.ssh/runpod_key")


def _ssh_cmd_args(host: str, port: int, remote_command: str) -> list:
    return ["ssh", "-i", _ssh_key_path(), "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10", "-p", str(port), f"root@{host}", remote_command]


def _ssh_cmd(host: str, port: int, remote_command: str, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        _ssh_cmd_args(host, port, remote_command),
        capture_output=True, text=True, timeout=timeout,
    )


def _wait_for_ssh(host: str, port: int, timeout_s: Optional[int] = None) -> bool:
    deadline = time.monotonic() + (timeout_s if timeout_s is not None else SSH_READY_TIMEOUT_SECONDS)
    while time.monotonic() < deadline:
        result = _ssh_cmd(host, port, "echo ready", timeout=15)
        if result.returncode == 0:
            return True
        time.sleep(5)
    return False


def _push_repo_to_pod(host: str, port: int) -> None:
    """Rsync this checkout (minus .git/logs/venv) onto the fresh pod - see
    module docstring for why this replaces a git-clone-with-token."""
    subprocess.run(_ssh_cmd_args(host, port, f"mkdir -p {REMOTE_DIR}"), check=True, timeout=30)
    subprocess.run(
        ["rsync", "-az", "--delete",
         "-e", f"ssh -i {_ssh_key_path()} -p {port} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10",
         "--exclude", ".git", "--exclude", "logs", "--exclude", ".venv",
         "--exclude", "__pycache__", "--exclude", "checkpoints",
         f"{LOCAL_REPO_DIR}/", f"root@{host}:{REMOTE_DIR}/"],
        check=True, timeout=300,
    )


def _launch_remote_training(host: str, port: int, mode: str, n_assets: int) -> None:
    remote_cmd = (
        f"cd {REMOTE_DIR} && "
        f"(pip install -q -e . || pip install -q torch numpy scipy scikit-learn networkx pandas) && "
        f"pip install -q pyyaml yfinance && "
        f"nohup bash -c '"
        f"python scripts/train.py -m {mode} -d cuda --n-assets {n_assets} "
        f"&& tar czf {REMOTE_TARBALL} checkpoints "
        f"&& touch {REMOTE_MARKER_DONE} "
        f"|| touch {REMOTE_MARKER_FAILED}"
        f"' > /tmp/train.log 2>&1 & disown"
    )
    result = subprocess.run(_ssh_cmd_args(host, port, remote_cmd), capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"failed to launch remote training: {result.stderr}")


def _check_remote_marker(host: str, port: int) -> Optional[str]:
    """Returns 'done', 'failed', or None (still running / unreachable)."""
    result = _ssh_cmd(
        host, port,
        f"[ -f {REMOTE_MARKER_DONE} ] && echo done || "
        f"([ -f {REMOTE_MARKER_FAILED} ] && echo failed || echo running)",
        timeout=20,
    )
    status = result.stdout.strip()
    return status if status in ("done", "failed") else None


def _pull_tarball(host: str, port: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["scp", "-i", _ssh_key_path(), "-o", "StrictHostKeyChecking=accept-new",
         "-P", str(port), f"root@{host}:{REMOTE_TARBALL}", str(dest)],
        check=True, timeout=300,
    )


def _extract_checkpoint(tarball_path: Path) -> None:
    """Extracts the tarball into CHECKPOINT_DIR, stripping its own
    top-level "checkpoints/" wrapper - the remote side builds it with
    `tar czf ... checkpoints` from REMOTE_DIR, so every member starts
    with that prefix. Without stripping it, contents would land one
    level too deep (CHECKPOINT_DIR/checkpoints/meta/snn.pt instead of
    CHECKPOINT_DIR/meta/snn.pt, where load_runpod_checkpoint looks).
    Also rejects any member whose stripped path would escape
    CHECKPOINT_DIR (path traversal / "tar slip")."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "r:gz") as tar:
        safe_members = []
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if parts and parts[0] == "checkpoints":
                parts = parts[1:]
            if not parts:
                continue
            dest = (CHECKPOINT_DIR / Path(*parts)).resolve()
            if not str(dest).startswith(str(CHECKPOINT_DIR.resolve())):
                logger.warning("[runpod] skipping tar member outside checkpoint dir: %s", member.name)
                continue
            member.name = str(Path(*parts))
            safe_members.append(member)
        tar.extractall(CHECKPOINT_DIR, members=safe_members)
    logger.info("[runpod] extracted %s into %s", tarball_path, CHECKPOINT_DIR)


# ---------------------------------------------------------------------------
# Lock file - atomic acquire, self-healing on a crash that skipped cleanup
# ---------------------------------------------------------------------------

def _lock_is_stale() -> bool:
    try:
        mtime = LOCK_FILE.stat().st_mtime
    except OSError:
        return True
    return (time.time() - mtime) > STALE_LOCK_SECONDS


def _acquire_lock() -> bool:
    if LOCK_FILE.exists():
        if not _lock_is_stale():
            return False
        logger.warning(
            "[runpod] lock file older than %dh - assuming a crashed prior "
            "run (not a live one) and clearing it", STALE_LOCK_SECONDS // 3600,
        )
        try:
            LOCK_FILE.unlink()
        except OSError:
            pass
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()},{datetime.now(timezone.utc).isoformat()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _release_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def trigger_training_and_wait(
    mode: str = "full",
    gpu_type: str = "NVIDIA RTX A5000",
    cloud_type: str = "COMMUNITY",
    n_assets: int = 10,
    timeout_seconds: int = JOB_TIMEOUT_SECONDS,
) -> TrainingResult:
    """
    Blocking, synchronous, end-to-end nightly GPU training run. Meant to
    be called off the main thread (see
    KronosOrchestrator.kick_off_runpod_training) - it can legitimately
    take up to `timeout_seconds` (default 3h) to return.

    Never raises. Every failure mode - already running, no API key, API
    error, unreachable pod, SSH failure, training failure, timeout, or
    any unexpected exception - returns TrainingResult(success=False, ...)
    so the caller can fall back to yesterday's weights without the daily
    cycle ever crashing. The pod is ALWAYS deleted (try/finally) whenever
    one was actually created, regardless of how this function exits.
    """
    if not _acquire_lock():
        logger.warning("[runpod] training already running (%s exists) - skipping", LOCK_FILE)
        return TrainingResult(False, reason="already_running")

    pod_id: Optional[str] = None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    start = time.monotonic()

    try:
        try:
            _api_headers()
        except RuntimeError as e:
            logger.critical("[runpod] %s", e)
            return TrainingResult(False, reason="no_api_key")

        logger.info("[runpod] creating pod (gpu=%s cloud=%s mode=%s n_assets=%d)",
                    gpu_type, cloud_type, mode, n_assets)
        pod_id = _create_pod(gpu_type, cloud_type, run_id)

        host = port = None
        deadline = start + POD_REACHABLE_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            info = _get_pod(pod_id)
            if info.get("desiredStatus") == "RUNNING" and info.get("publicIp"):
                port = (info.get("portMappings") or {}).get("22")
                if port:
                    host = info["publicIp"]
                    break
            time.sleep(10)

        if not host or not port:
            logger.critical("[runpod] pod %s never became reachable within %d min",
                             pod_id, POD_REACHABLE_TIMEOUT_SECONDS // 60)
            return TrainingResult(False, reason="pod_unreachable")

        if not _wait_for_ssh(host, port):
            logger.critical("[runpod] sshd never came up inside pod %s", pod_id)
            return TrainingResult(False, reason="ssh_unreachable")

        logger.info("[runpod] pod %s reachable at %s:%s - pushing checkout", pod_id, host, port)
        _push_repo_to_pod(host, port)

        logger.info("[runpod] launching training on pod %s", pod_id)
        _launch_remote_training(host, port, mode, n_assets)

        while time.monotonic() - start < timeout_seconds:
            status = _check_remote_marker(host, port)
            elapsed_min = (time.monotonic() - start) / 60
            if status == "done":
                logger.info("[runpod] training complete after %.1f min - pulling checkpoint", elapsed_min)
                dest = CHECKPOINT_DIR / f"checkpoints_job-{run_id}.tar.gz"
                _pull_tarball(host, port, dest)
                _extract_checkpoint(dest)
                return TrainingResult(True, tarball_path=dest, reason="ok")
            if status == "failed":
                logger.critical("[runpod] training script failed inside pod %s after %.1f min",
                                 pod_id, elapsed_min)
                return TrainingResult(False, reason="training_failed")
            logger.info("[runpod] still running (%.1f min elapsed)", elapsed_min)
            time.sleep(POLL_INTERVAL_SECONDS)

        logger.critical("[runpod] training timed out after %d min - falling back to yesterday's checkpoint",
                         timeout_seconds // 60)
        return TrainingResult(False, reason="timeout")

    except Exception as e:
        logger.critical("[runpod] unexpected failure, falling back to yesterday's checkpoint: %s", e)
        return TrainingResult(False, reason=f"exception:{e}")

    finally:
        if pod_id is not None:
            _delete_pod(pod_id)
        _release_lock()


def load_runpod_checkpoint(reflex_snn: torch.nn.Module, checkpoint_dir: Optional[Path] = None) -> bool:
    """
    Loads tonight's RunPod-trained SpikingMarketEncoder weights straight
    into reflex_snn (KronosOrchestrator.reflex.snn - the model that
    actually decides trades) in place. Returns True if loaded, False
    otherwise; reflex_snn is left exactly as it was passed in on any
    failure, so the caller keeps trading on whatever it already had.

    Looks for <checkpoint_dir>/meta/snn.pt - PrometheusEngine.save()'s
    "full" pipeline output after its meta (MAML) phase, which is exactly
    what "python scripts/train.py -m full" leaves in checkpoints/meta/.

    NOTE - a known, pre-existing shape risk this function does not try to
    paper over: PrometheusEngine builds its own SNN with
    output_size=n_assets // 2 and a configurable layer_sizes
    (prometheus/engine.py), while ReflexArc builds its SNN with
    output_size=n_assets and a hardcoded [32, 16] (kronos/reflex.py).
    Training with a matching --n-assets (this module always passes one)
    removes one source of mismatch but not that one - if the shapes still
    don't line up, load_state_dict raises here, and this function treats
    that exactly like "no checkpoint found": log a warning, return False,
    change nothing. It does not silently truncate or reshape a tensor to
    make a mismatched checkpoint fit.
    """
    path = (checkpoint_dir or CHECKPOINT_DIR) / "meta" / "snn.pt"
    if not path.exists():
        logger.warning("WARNING: No RunPod checkpoint found at %s. Starting from scratch.", path)
        return False
    try:
        state_dict = torch.load(path, map_location="cpu")
        reflex_snn.load_state_dict(state_dict)
        logger.info("[runpod] loaded RunPod-trained SNN weights from %s", path)
        return True
    except Exception as e:
        logger.warning(
            "WARNING: RunPod checkpoint at %s could not be loaded (%s) - "
            "likely a shape mismatch between PrometheusEngine's SNN and "
            "ReflexArc's. Starting from scratch.", path, e,
        )
        return False
