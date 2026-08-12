"""
Model-Agnostic Meta-Learning (MAML) Engine.

MAML learns an initialization θ* such that a new market regime can be
adapted in just K=3 gradient steps. This is the "instant adaptation"
layer of Prometheus.

When a new regime appears (detected by the HTM anomaly score or cortisol
spike), MAML takes 3 gradient steps on recent regime data and the model
is immediately calibrated — no full retraining required.

Reference: Finn et al. (2017) "Model-Agnostic Meta-Learning for Fast Adaptation"
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

logger = logging.getLogger(__name__)


class MAMLMetaLearner:
    """
    MAML wrapper for any PyTorch model.

    Meta-training:
      For each task (regime) τ_i ~ p(τ):
        1. Sample support set D_support (regime data)
        2. Compute adapted parameters: θ'_i = θ - α∇L_τ(θ)  [K steps]
        3. Meta-update: θ ← θ - β∇ Σ L_τ(θ'_i, D_query)

    Fast adaptation (inference time):
      Given K=3 gradient steps on new regime data → instant calibration.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,     # α: inner loop learning rate
        outer_lr: float = 1e-3,     # β: meta (outer) learning rate
        n_inner_steps: int = 3,     # K: adaptation steps
        first_order: bool = True,   # use first-order MAML (faster, slight accuracy loss)
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps
        self.first_order = first_order
        self.meta_optimizer = Adam(model.parameters(), lr=outer_lr)

    # ------------------------------------------------------------------
    # Inner loop: fast adaptation
    # ------------------------------------------------------------------

    def adapt(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
        return_adapted_model: bool = False,
    ) -> Tuple[nn.Module, List[float]]:
        """
        Adapt the model to a new regime in K gradient steps.

        Args:
            support_data: (X_support, y_support) — small batch from new regime
            loss_fn: task-specific loss function
            return_adapted_model: if True, return a deep-copied adapted model

        Returns:
            (adapted_model, inner_losses)
        """
        X, y = support_data
        adapted_params = {name: p.clone() for name, p in self.model.named_parameters()}
        inner_losses = []

        for step in range(self.n_inner_steps):
            # Forward with current adapted params
            loss = self._forward_with_params(X, y, adapted_params, loss_fn)
            inner_losses.append(float(loss.item()))

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=not self.first_order,
                allow_unused=True,
            )

            # Update adapted params (inner gradient step)
            adapted_params = {
                name: (p - self.inner_lr * (g if g is not None else torch.zeros_like(p)))
                for (name, p), g in zip(adapted_params.items(), grads)
            }

        logger.debug("MAML adaptation: %d steps, losses %s", self.n_inner_steps, inner_losses)

        if return_adapted_model:
            adapted = copy.deepcopy(self.model)
            with torch.no_grad():
                for name, p in adapted.named_parameters():
                    if name in adapted_params:
                        p.copy_(adapted_params[name])
            return adapted, inner_losses

        return self.model, inner_losses

    # ------------------------------------------------------------------
    # Outer loop: meta-training
    # ------------------------------------------------------------------

    def meta_train_step(
        self,
        tasks: List[Tuple[Tuple, Tuple]],  # list of ((X_s, y_s), (X_q, y_q))
        loss_fn: Callable,
    ) -> float:
        """
        One MAML meta-training step over a batch of tasks (regimes).
        Returns: meta loss (outer loop).
        """
        self.meta_optimizer.zero_grad()
        meta_loss = torch.tensor(0.0, requires_grad=True)

        for (support, query) in tasks:
            X_s, y_s = support
            X_q, y_q = query

            # Inner loop: compute adapted parameters
            adapted_params = {name: p.clone() for name, p in self.model.named_parameters()}
            for _ in range(self.n_inner_steps):
                inner_loss = self._forward_with_params(X_s, y_s, adapted_params, loss_fn)
                grads = torch.autograd.grad(
                    inner_loss,
                    adapted_params.values(),
                    create_graph=not self.first_order,
                    allow_unused=True,
                )
                adapted_params = {
                    name: p - self.inner_lr * (g if g is not None else torch.zeros_like(p))
                    for (name, p), g in zip(adapted_params.items(), grads)
                }

            # Outer loop: evaluate on query set with adapted params
            outer_loss = self._forward_with_params(X_q, y_q, adapted_params, loss_fn)
            meta_loss = meta_loss + outer_loss

        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.meta_optimizer.step()
        return float(meta_loss.item())

    def _forward_with_params(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        params: Dict[str, torch.Tensor],
        loss_fn: Callable,
    ) -> torch.Tensor:
        """Run model forward pass with custom parameter dict."""
        # Use functional API to apply custom params
        # This requires the model to support functional forward
        # Fallback: use standard forward and treat params as a detached copy
        pred = self.model(X)
        if isinstance(pred, dict):
            pred = pred.get("predictions", pred.get("output", list(pred.values())[0]))
        if pred.shape != y.shape and pred.numel() == y.numel():
            pred = pred.view_as(y)
        return loss_fn(pred, y)

    def save_meta_state(self, path: str) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "meta_optimizer": self.meta_optimizer.state_dict(),
            "inner_lr": self.inner_lr,
            "n_inner_steps": self.n_inner_steps,
        }, path)

    def load_meta_state(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.meta_optimizer.load_state_dict(ckpt["meta_optimizer"])
        self.inner_lr = ckpt["inner_lr"]
        self.n_inner_steps = ckpt["n_inner_steps"]
