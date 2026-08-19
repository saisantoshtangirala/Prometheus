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

MAX_GRAD_NORM = 1.0
REJECTION_THRESHOLD = 1.5    # reject if post-adaptation loss > 150% of pre


class ClippedMAML(MAMLMetaLearner):
    """
    MAML with inner-loop gradient clipping (MAM-01).

    Extends (does not modify) the Prometheus MAMLMetaLearner: each inner
    gradient is rescaled to a global norm of MAX_GRAD_NORM before the
    parameter update, so a 20%-daily-return regime shift cannot blow the
    weights out to inf.
    """

    def adapt(self, support_data, loss_fn, return_adapted_model=False):
        import copy as _copy
        X, y = support_data
        adapted_params = {
            name: p.clone() for name, p in self.model.named_parameters()
        }
        inner_losses = []
        for _ in range(self.n_inner_steps):
            loss = self._forward_with_params(X, y, adapted_params, loss_fn)
            inner_losses.append(float(loss.item()))
            grads = torch.autograd.grad(
                loss, adapted_params.values(),
                create_graph=not self.first_order, allow_unused=True,
            )
            grads = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, adapted_params.values())
            ]
            # Global-norm clip at MAX_GRAD_NORM
            total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
            clip_coef = MAX_GRAD_NORM / (total_norm + 1e-8)
            if clip_coef < 1.0:
                grads = [g * clip_coef for g in grads]
            adapted_params = {
                name: p - self.inner_lr * g
                for (name, p), g in zip(adapted_params.items(), grads)
            }

        if return_adapted_model:
            adapted = _copy.deepcopy(self.model)
            with torch.no_grad():
                for name, p in adapted.named_parameters():
                    if name in adapted_params:
                        p.copy_(adapted_params[name])
            return adapted, inner_losses
        return self.model, inner_losses


@dataclass
class WarmupResult:
    adapted_model: nn.Module
    inner_losses: List[float]
    regime_estimate: str          # "bull" | "bear" | "sideways"
    n_steps: int
    rejected: bool = False        # MAM-02: adaptation made things worse
    skipped: bool = False         # MAM-03: not enough data to adapt


class KronosWarmer:
    """3-step MAML warm-up of the daily master model on recent real data."""

    def __init__(self, config):
        self.cfg = config

    # -- data prep ----------------------------------------------------------

    def support_set(self, memory) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the support set from the last `support_days` trading days.

        The master model maps flattened nightmare-shaped windows
        [(horizon - 1) * n_assets] -> next-bar returns [n_assets], so the
        support set uses the same shape from REAL data: sliding windows of
        length `horizon - 1` predicting the following bar. (AUDIT-1A: the
        master model's input width is horizon-1, not horizon - see
        kronos/evolver.py's KronosEvolver.__init__ for why - so this must
        match or every nightly warm-up would fail with a shape mismatch.)
        """
        horizon = self.cfg.nightmare.horizon_days - 1
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
        """
        Run exactly n_inner_steps MAML adaptation steps. Non-destructive.

        Safety rails:
          MAM-01: inner gradients clipped to global norm 1.0 (ClippedMAML)
          MAM-02: if the adapted loss exceeds REJECTION_THRESHOLD x the
                  pre-adaptation loss, the adaptation is rejected and the
                  base master model is used for the day
          MAM-03: with too little data to build a support set, adaptation
                  is skipped entirely (no IndexError, base model used)
        """
        n_steps = int(self.cfg.adaptation.n_inner_steps)
        regime = self.estimate_regime(memory)

        try:
            X, y = self.support_set(memory)
        except (ValueError, IndexError):
            X = torch.zeros(0)
            y = torch.zeros(0)
        if X.numel() == 0 or X.shape[0] < 1:
            logger.warning(
                "[warmup] Insufficient data for MAML. Skipping."
            )
            return WarmupResult(
                adapted_model=master_model, inner_losses=[],
                regime_estimate=regime, n_steps=0, skipped=True,
            )

        learner = ClippedMAML(
            model=master_model,
            inner_lr=self.cfg.adaptation.inner_lr,
            n_inner_steps=n_steps,
        )
        loss_fn = nn.functional.mse_loss
        with torch.no_grad():
            pre_loss = float(loss_fn(self._predict(master_model, X), y).item())

        adapted, inner_losses = learner.adapt(
            support_data=(X, y), loss_fn=loss_fn, return_adapted_model=True,
        )

        with torch.no_grad():
            post_loss = float(loss_fn(self._predict(adapted, X), y).item())

        # MAM-01 hard check: adapted weights must be finite
        weights_finite = all(
            torch.isfinite(p).all() for p in adapted.parameters()
        )
        if not weights_finite or (
            pre_loss > 1e-12 and post_loss > pre_loss * REJECTION_THRESHOLD
        ):
            logger.warning(
                "[warmup] MAML adaptation rejected. Using base model. "
                "(pre=%.6f post=%.6f finite=%s)",
                pre_loss, post_loss, weights_finite,
            )
            return WarmupResult(
                adapted_model=master_model, inner_losses=inner_losses,
                regime_estimate=regime, n_steps=n_steps, rejected=True,
            )

        logger.info(
            "[warmup] %d steps on %d support samples, regime=%s, "
            "loss %.6f -> %.6f",
            n_steps, X.shape[0], regime, pre_loss, post_loss,
        )
        return WarmupResult(
            adapted_model=adapted,
            inner_losses=inner_losses,
            regime_estimate=regime,
            n_steps=n_steps,
        )

    @staticmethod
    def _predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        out = model(X)
        if isinstance(out, dict):
            out = out.get("predictions", list(out.values())[0])
        return out
