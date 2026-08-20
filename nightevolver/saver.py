"""
Checkpoint contract between RunPod (training) and Hetzner (execution).

This is the seam where the previous project failed hardest: the model
that was validated was not the model that traded. So the checkpoint
carries not just the genome but everything needed to verify that the
thing being loaded is the thing that was tested:

  genome_version   - refuses to load a genome encoded under a different
                     gene layout. Gene 7 must mean the same thing on
                     both machines or the strategy is silently scrambled.
  indicator_names  - the exact channel ordering used at training time.
  metrics          - in-sample AND out-of-sample stats, the search
                     budget, the noise benchmark and the deflated
                     Sharpe, so Hetzner can REFUSE a checkpoint that did
                     not clear the statistical gate.

`load_checkpoint` fails closed on every mismatch: log, raise, change
nothing. Adopting a checkpoint you cannot verify is worse than trading
yesterday's.

Format is JSON, not pickle. A pickle checkpoint is arbitrary code
execution on load, and this file is fetched over the network from an
ephemeral pod onto the box that holds the trading account.
"""

from __future__ import annotations

import json
import logging
import os
import tarfile
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from nightevolver.genome import GENOME_LENGTH, GENOME_VERSION, INDICATOR_NAMES, decode

logger = logging.getLogger("nightevolver.saver")

CHECKPOINT_DIR = Path("checkpoints/nightevolver")
CHECKPOINT_NAME = "nightevolver_best.json"

# Hetzner refuses to trade a checkpoint below this deflated-Sharpe
# probability. 0.95 is this project's own pre-existing gate, applied
# here to the GA's real search budget rather than a token n_trials.
MIN_DEFLATED_SHARPE_PROB = 0.95


def _stats_dict(stats) -> Optional[Dict]:
    if stats is None:
        return None
    return {
        "sharpe": float(stats.sharpe),
        "total_return": float(stats.total_return),
        "max_drawdown": float(stats.max_drawdown),
        "win_rate": float(stats.win_rate),
        "n_trades": int(stats.n_trades),
        "avg_turnover": float(stats.avg_turnover),
        "n_obs": int(len(stats.daily_returns)),
    }


def save_checkpoint(result, tickers, checkpoint_dir: Optional[Path] = None,
                    mode: str = "ga", extra: Optional[Dict] = None) -> Path:
    """Write an EvolutionResult to a verifiable JSON checkpoint."""
    d = Path(checkpoint_dir or CHECKPOINT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    path = d / CHECKPOINT_NAME

    payload = {
        "genome_version": GENOME_VERSION,
        "genome_length": GENOME_LENGTH,
        "indicator_names": list(INDICATOR_NAMES),
        "mode": mode,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "tickers": list(tickers),
        "genome": np.asarray(result.best_genome, dtype=float).tolist(),
        "strategy": result.best_strategy.to_dict(),
        "metrics": {
            "in_sample": _stats_dict(result.in_sample),
            "out_of_sample": _stats_dict(result.out_of_sample),
            "search_budget": int(result.search_budget),
            "noise_benchmark_sharpe": float(result.noise_benchmark_sharpe),
            "deflated_sharpe_prob": float(result.deflated_sharpe_prob),
            "overfitting_gap": float(result.overfitting_gap)
            if result.out_of_sample is not None else None,
            "beats_noise": bool(result.beats_noise),
        },
        "history": result.history,
    }
    if extra:
        payload["extra"] = extra

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("[nightevolver] checkpoint written to %s", path)
    return path


def load_checkpoint(path: Optional[Path] = None,
                    require_gate: bool = True) -> Dict:
    """Load and VERIFY a checkpoint. Raises on any mismatch.

    require_gate: if True (the default for live execution), refuse a
    checkpoint whose deflated Sharpe probability is below the gate. This
    is the mechanism that stops an overfit nightly run from reaching the
    trading account just because it happened to be the newest file.
    """
    p = Path(path or (CHECKPOINT_DIR / CHECKPOINT_NAME))
    if not p.exists():
        raise FileNotFoundError(f"no nightevolver checkpoint at {p}")

    with open(p) as f:
        ck = json.load(f)

    ver = ck.get("genome_version")
    if ver != GENOME_VERSION:
        raise ValueError(
            f"checkpoint genome_version {ver} != running code's {GENOME_VERSION} - "
            "the gene layout changed, so this genome would be mis-decoded. Retrain."
        )
    names = tuple(ck.get("indicator_names", ()))
    if names != INDICATOR_NAMES:
        raise ValueError(
            "checkpoint indicator ordering differs from the running code - "
            "gene weights would attach to the wrong indicators. Retrain."
        )
    genome = np.asarray(ck["genome"], dtype=float)
    if genome.size != GENOME_LENGTH:
        raise ValueError(f"genome length {genome.size} != {GENOME_LENGTH}")

    if require_gate:
        dsr = float(ck.get("metrics", {}).get("deflated_sharpe_prob", 0.0))
        if dsr < MIN_DEFLATED_SHARPE_PROB:
            raise ValueError(
                f"checkpoint deflated Sharpe probability {dsr:.3f} is below the "
                f"{MIN_DEFLATED_SHARPE_PROB} gate - this strategy is not "
                f"statistically distinguishable from the best of "
                f"{ck.get('metrics', {}).get('search_budget', '?')} random searches. "
                "Refusing to trade it."
            )

    ck["genome"] = genome
    ck["decoded"] = decode(genome)
    return ck


def package_checkpoint(checkpoint_dir: Optional[Path] = None,
                       out_path: Optional[Path] = None) -> Path:
    """Tar+gzip the checkpoint dir for artifact upload / scp pickup."""
    d = Path(checkpoint_dir or CHECKPOINT_DIR)
    out = Path(out_path or f"checkpoints_nightevolver_{int(datetime.now().timestamp())}.tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        tar.add(d, arcname=d.name)
    logger.info("[nightevolver] packaged %s -> %s", d, out)
    return out
