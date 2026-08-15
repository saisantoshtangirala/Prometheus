"""
Kronos MAML Micro-Adaptation - phase 4 of the daily cycle (05:00 - 06:00).

Takes the master model produced by evolution and runs EXACTLY
config.adaptation.n_inner_steps (3) gradient steps on the real returns of
the last few trading days. This snaps the model onto the current regime
(bull / bear / sideways) without giving it enough rope to overfit noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

from prometheus.meta.maml_engine import MAMLMetaLearner

logger = logging.getLogger(__name__)


@dataclass
class WarmupResult:
    adapted_model: nn.Module
    inner_losses: List[float]
    regime_estimate: str          # "bull" | "bear" | "sideways"
    n_steps: int


class KronosWarmer:
    """3-step MAML warm-up of the daily master model on recent real data."""

    def __init__(self, config):
        self.cfg = config

    # -- data prep ----------------------------------------------------------

    def support_set(self, memory) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the support set from the last `support_days` trading days.

        The master model maps flattened nightmare-shaped windows
        [horizon * n_assets] -> next-bar returns [n_assets], so the support
        set uses the same shape from REAL data: sliding windows of length
        `horizon` predicting the following bar.
        """
        horizon = self.cfg.nightmare.horizon_days
        support_days = self.cfg.adaptation.support_days
        window = memory.returns_window(horizon + support_days)

        xs, ys = [], []
        for start in range(len(window) - horizon):
            xs.append(window[start:start + horizon].reshape(-1))
            ys.append(window[start + horizon])
        X = torch.tensor(xs, dtype=torch.float32)
        y = torch.tensor(ys, dtype=torch.float32)
        return X, y

    def estimate_regime(self, memory) -> str:
        """Cheap regime read from the mean drift of the support window."""
        support_days = self.cfg.adaptation.support_days
        recent = memory.returns_window(support_days)
        drift = float(recent.mean())
        vol = float(recent.std())
        if drift > 0.5 * vol / max(len(recent), 1) ** 0.5 and drift > 0:
            return "bull"
        if drift < -0.5 * vol / max(len(recent), 1) ** 0.5 and drift < 0:
            return "bear"
        return "sideways"

    # -- warm-up ------------------------------------------------------------

    def warm(self, master_model: nn.Module, memory) -> WarmupResult:
        """Run exactly n_inner_steps MAML adaptation steps. Non-destructive."""
        n_steps = int(self.cfg.adaptation.n_inner_steps)
        learner = MAMLMetaLearner(
            model=master_model,
            inner_lr=self.cfg.adaptation.inner_lr,
            n_inner_steps=n_steps,
        )
        X, y = self.support_set(memory)

        adapted, inner_losses = learner.adapt(
            support_data=(X, y),
            loss_fn=nn.functional.mse_loss,
            return_adapted_model=True,
        )

        regime = self.estimate_regime(memory)
        logger.info(
            "[warmup] %d steps on %d support samples, regime=%s, losses=%s",
            n_steps, X.shape[0], regime,
            [f"{l:.5f}" for l in inner_losses],
        )
        return WarmupResult(
            adapted_model=adapted,
            inner_losses=inner_losses,
            regime_estimate=regime,
            n_steps=n_steps,
        )
