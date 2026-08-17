"""
Loads nightly RunPod-trained checkpoints into Kronos's live trading path.

Pod orchestration (create/train/pull/delete) runs entirely in GitHub
Actions - .github/workflows/train-runpod.yml is scheduled nightly, skips
weekends/holidays via kronos/calendar_utils.py, and scp's the resulting
checkpoint directory straight onto the Hetzner box at CHECKPOINT_DIR.
Kronos itself never talks to the RunPod API and holds no RunPod
credentials - KronosOrchestrator.maybe_adopt_runpod_checkpoint() just
watches CHECKPOINT_DIR for a newer file and calls load_runpod_checkpoint()
below when one appears.

(An earlier version of this module ran the pod lifecycle directly from
Hetzner - a background thread doing REST API calls, SSH, rsync, a lock
file. That's gone: scheduling and infra orchestration belong in a CI/CD
system, not embedded in the process that's also responsible for trading
every day. GitHub Actions already had the SSH secrets to reach Hetzner
via deploy-hetzner.yml, so extending that trust relationship to also
deliver a checkpoint was the smaller, safer change.)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("kronos.runpod_trigger")

CHECKPOINT_DIR = Path("checkpoints/runpod")


def load_runpod_checkpoint(reflex_snn: torch.nn.Module, checkpoint_dir: Optional[Path] = None) -> bool:
    """
    Loads tonight's RunPod-trained SpikingMarketEncoder weights straight
    into reflex_snn (KronosOrchestrator.reflex.snn - the model that
    actually decides trades) in place. Returns True if loaded, False
    otherwise; reflex_snn is left exactly as it was passed in on any
    failure, so the caller keeps trading on whatever it already had.

    Looks for <checkpoint_dir>/meta/snn.pt - PrometheusEngine.save()'s
    "full" pipeline output after its meta (MAML) phase, which is exactly
    what "python scripts/train.py -m full" leaves in checkpoints/meta/,
    and what train-runpod.yml's push-to-Hetzner step lands at
    checkpoints/runpod/meta/snn.pt on this box.

    NOTE - a known, pre-existing shape risk this function does not try to
    paper over: PrometheusEngine builds its own SNN with
    output_size=n_assets // 2 and a configurable layer_sizes
    (prometheus/engine.py), while ReflexArc builds its SNN with
    output_size=n_assets and a hardcoded [32, 16] (kronos/reflex.py).
    Training with a matching --n-assets (train-runpod.yml always passes
    one) removes one source of mismatch but not that one - if the shapes
    still don't line up, load_state_dict raises here, and this function
    treats that exactly like "no checkpoint found": log a warning, return
    False, change nothing. It does not silently truncate or reshape a
    tensor to make a mismatched checkpoint fit.
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
