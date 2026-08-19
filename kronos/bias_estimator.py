"""
Computes a once-per-adoption directional bias from the nightly
RunPod checkpoint's causal_transformer + LTC pair (PrometheusEngine's
own multi-step-ahead forecaster), for ReflexArc to blend into its
per-tick SNN signal as an agreement-based confidence modifier - the
"second opinion" capability that was missing entirely before this:
ReflexArc.infer() previously used only self.snn's output, with no
way to check it against anything.

Deliberately NOT run per reflex tick - the causal_transformer is a
real multi-layer attention model (d_model~128, several layers),
too heavy for reflex.inference_budget_ms's low-latency budget (the
SNN it's checked against is a ~50-neuron network by comparison).
Computed once whenever a fresh checkpoint is adopted, cached on the
orchestrator, and reused for every reflex tick until the next
adoption.

Reconstructs LTC + CausalTransformer directly (not the full
PrometheusEngine - that would also pull in the DAG/diffusion/
hive-mind/sentiment subsystems Kronos's live path has no use for)
from <checkpoint_dir>/meta/arch.json, the architecture sidecar
PrometheusEngine.save() writes alongside the weight files. Missing
sidecar, missing weight files, or a shape mismatch all fail closed:
log a warning, return None - never raises, never guesses a shape.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from kronos.features import build_features
from prometheus.causal.causal_transformer import CausalTransformer
from prometheus.neuro.ltc_network import LiquidTimeConstantNetwork

logger = logging.getLogger("kronos.bias_estimator")


def compute_daily_bias(
    recent_returns: np.ndarray,   # [T, n_assets], T >= arch's seq_len
    checkpoint_dir: Path,
    recent_volumes: Optional[np.ndarray] = None,   # [T, n_assets], see kronos/features.py
) -> Optional[np.ndarray]:
    """
    Returns a [n_assets] array - the checkpoint's causal_transformer's
    nearest-horizon-step predicted return per asset - or None if the
    sidecar/weights are missing, mismatched, or anything else about
    this doesn't line up. Never raises.
    """
    meta_dir = checkpoint_dir / "meta"
    arch_path = meta_dir / "arch.json"
    if not arch_path.exists():
        logger.warning(
            "No arch.json at %s - skipping daily bias (no confidence "
            "blending until the next checkpoint that has one).", arch_path,
        )
        return None

    try:
        with open(arch_path) as f:
            arch = json.load(f)

        n_assets = arch["n_assets"]
        seq_len = arch["seq_len"]
        # Older checkpoints saved before kronos/features.py existed have no
        # n_input_features key - their only historical behavior was
        # returns-only (n_input_features == n_assets), so that's the
        # correct fallback, not a hard failure.
        n_feat = arch.get("n_input_features", n_assets)
        if recent_returns.ndim != 2 or recent_returns.shape[-1] != n_assets:
            logger.warning(
                "Daily bias skipped: recent_returns shape %s doesn't match "
                "checkpoint's %d assets.", recent_returns.shape, n_assets,
            )
            return None
        if recent_returns.shape[0] < seq_len:
            logger.info(
                "Daily bias skipped: only %d bars available, checkpoint "
                "needs %d.", recent_returns.shape[0], seq_len,
            )
            return None

        ltc = LiquidTimeConstantNetwork(
            input_size=n_feat,
            hidden_sizes=arch["ltc_hidden"],
            output_size=n_assets,
        )
        ltc.load_state_dict(torch.load(meta_dir / "ltc.pt", map_location="cpu"))
        ltc.eval()

        causal_transformer = CausalTransformer(
            # NOT n_feat: causal_transformer's actual input is always
            # ltc's OUTPUT (ltc_out, below), which is n_assets-wide
            # regardless of how wide ltc's own input is. Only ltc itself
            # (which consumes raw x) uses n_feat. Mirrors
            # PrometheusEngine.__init__'s construction exactly - see its
            # comment for why.
            n_features=n_assets,
            n_targets=n_assets,
            horizon=arch["horizon"],
            d_model=arch["d_model"],
            n_heads=arch["n_heads"],
            n_layers=arch["n_layers"],
            d_ff=arch["d_ff"],
        )
        causal_transformer.load_state_dict(
            torch.load(meta_dir / "causal_transformer.pt", map_location="cpu")
        )
        causal_transformer.eval()

        window = recent_returns[-seq_len:]
        vol_window = recent_volumes[-seq_len:] if recent_volumes is not None else None
        feats = build_features(window, vol_window) if n_feat != n_assets else window
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            ltc_out, _, _ = ltc(x)
            causal_out = causal_transformer(ltc_out, return_attributions=False)
        predictions = causal_out["predictions"]   # [1, horizon, n_assets]
        return predictions[0, 0, :].numpy()
    except Exception as e:
        logger.warning(
            "Daily bias computation failed (%s) - skipping confidence "
            "blending until the next successful attempt.", e,
        )
        return None
