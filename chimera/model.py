"""
The composed CHIMERA network: where components 3, 4 and 5 meet.

    [B, T, n_features]
        |
        v  (3) chaotic-attention encoder, Lorenz-seeded from the window
    latent [B, d_model]
        |-----> (4) GRPO policy head  -> portfolio distribution (ACTION)
        |-----> (5) PINN value head   -> V(S,t), PDE residual (CONSTRAINT)
        `-----> return head           -> mu_hat, the directional signal

Three heads on one trunk, and the reason that is the right structure
rather than three separate models: the PINN residual and the
no-arbitrage penalty only constrain anything if they share a
representation with the thing that trades. A no-arbitrage penalty on an
isolated auxiliary network constrains nothing of interest.

Training objective:

    L = L_policy(GRPO/DAPO)
      + w_ret * L_return        (supervised MSE on realised next return)
      + w_arb * L_no_arbitrage  (cross-sectional, component 5A)
      + w_pde * L_pde_residual  (Black-Scholes, component 5B)

L_return is included deliberately even though the policy is RL-trained:
a pure-RL start on a near-zero-signal reward is a cold-start problem, and
the supervised head gives the shared trunk a gradient that is dense
rather than group-relative. The policy still owns the ACTION; the
supervised head only shapes the representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from chimera.chaotic_attn import ChaoticAttentionEncoder
from chimera.grpo import DAPOConfig, GaussianPortfolioPolicy, GroupRelativePolicy
from chimera.pinn import NoArbitragePenalty, ValueSurface, black_scholes_residual


@dataclass
class ChimeraConfig:
    """Architecture + optimisation settings.

    Defaults are deliberately SMALL. With ~250 training bars per
    walk-forward window and 10 assets, a large transformer memorises the
    window; d_model=64/2 layers is already generous. This is the
    parameter-count discipline that the audited SNN pipeline lacked.
    """

    seq_len: int = 32
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    learnable_chaos: bool = True

    n_selected_features: int = 12
    policy_hidden: int = 64

    w_return: float = 1.0
    w_arbitrage: float = 0.1
    w_pde: float = 0.01

    lr: float = 3e-4
    epochs: int = 30
    batch_size: int = 32
    grad_clip: float = 1.0

    pde_collocation: int = 64
    risk_free_rate: float = 0.065      # India 10y, not the US rate
    pde_sigma: float = 0.20

    seed: int = 42


class ChimeraNet(nn.Module):
    """Shared chaotic-attention trunk with policy / value / return heads."""

    def __init__(self, n_features: int, n_assets: int, cfg: ChimeraConfig):
        super().__init__()
        self.cfg = cfg
        self.n_assets = n_assets

        self.encoder = ChaoticAttentionEncoder(
            n_features=n_features, d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_layers=cfg.n_layers, dropout=cfg.dropout,
            learnable_chaos=cfg.learnable_chaos,
        )
        # Per-asset scalar prediction from a shared trunk: the encoder is
        # applied to each asset's own feature sequence, so one function
        # serves all names (cross-sectional model, not n_assets models).
        self.return_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2), nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )
        self.policy = GaussianPortfolioPolicy(
            state_dim=cfg.d_model * 2, n_assets=n_assets, hidden=cfg.policy_hidden,
        )
        self.value_surface = ValueSurface(hidden=32, n_layers=3, context_dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_assets, T, n_features] -> latent [B, n_assets, d_model]."""
        B, A, T, F = x.shape
        return self.encoder(x.reshape(B * A, T, F)).reshape(B, A, -1)

    def predict_returns(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_assets, T, F] -> mu_hat [B, n_assets]."""
        return self.return_head(self.encode(x)).squeeze(-1)

    def policy_state(self, latent: torch.Tensor) -> torch.Tensor:
        """Pool per-asset latents into one market state for the policy.

        mean AND max pooling concatenated: the mean carries the average
        market condition, the max carries "is any single name screaming",
        and a portfolio policy needs both.
        """
        return torch.cat([latent.mean(dim=1), latent.max(dim=1).values], dim=-1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.encode(x)
        return {
            "latent": latent,
            "mu_hat": self.return_head(latent).squeeze(-1),
            "policy_state": self.policy_state(latent),
        }


class ChimeraTrainer:
    """Trains ChimeraNet with the combined RL + supervised + physics loss."""

    def __init__(self, net: ChimeraNet, cfg: ChimeraConfig,
                 dapo: Optional[DAPOConfig] = None):
        self.net = net
        self.cfg = cfg
        self.no_arb = NoArbitragePenalty(n_null_directions=3)
        self.grpo = GroupRelativePolicy(net.policy, dapo or DAPOConfig(), lr=cfg.lr)
        # Trunk + supervised/physics heads on their own optimiser; the
        # policy has its own inside GroupRelativePolicy. Keeping them
        # separate stops the RL loss and the supervised loss from
        # fighting over one Adam moment estimate.
        trunk_params = (list(net.encoder.parameters())
                        + list(net.return_head.parameters())
                        + list(net.value_surface.parameters()))
        self.opt = torch.optim.AdamW(trunk_params, lr=cfg.lr, weight_decay=1e-5)

    def fit_constraints(self, train_returns: np.ndarray) -> None:
        """Fit the no-arbitrage null space on TRAIN data only."""
        self.no_arb.fit(torch.tensor(np.asarray(train_returns), dtype=torch.float32))

    def pde_loss(self) -> torch.Tensor:
        """Black-Scholes residual on random collocation points."""
        cfg = self.cfg
        n = cfg.pde_collocation
        S = torch.empty(n).uniform_(50.0, 150.0)
        t = torch.empty(n).uniform_(0.0, 0.99)
        res = black_scholes_residual(
            self.net.value_surface, S, t, r=cfg.risk_free_rate, sigma=cfg.pde_sigma,
        )
        return res.pow(2).mean()

    def train_epoch(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """One supervised+physics epoch.

        X: [n_samples, n_assets, T, F], y: [n_samples, n_assets]
        """
        cfg = self.cfg
        self.net.train()
        n = X.shape[0]
        perm = torch.randperm(n)
        totals = {"return": 0.0, "arb": 0.0, "pde": 0.0, "total": 0.0}
        nb = 0

        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb, yb = X[idx], y[idx]

            mu_hat = self.net.predict_returns(xb)
            l_ret = nn.functional.mse_loss(mu_hat, yb)
            l_arb = self.no_arb(mu_hat)
            l_pde = self.pde_loss()

            loss = cfg.w_return * l_ret + cfg.w_arbitrage * l_arb + cfg.w_pde * l_pde

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for g in self.opt.param_groups for p in g["params"]], cfg.grad_clip)
            self.opt.step()

            totals["return"] += float(l_ret.item())
            totals["arb"] += float(l_arb.item())
            totals["pde"] += float(l_pde.item())
            totals["total"] += float(loss.item())
            nb += 1

        return {k: v / max(nb, 1) for k, v in totals.items()}

    def train_policy_epoch(self, X: torch.Tensor, next_returns: torch.Tensor,
                           cost_bps: float = 22.0) -> Dict[str, float]:
        """One GRPO/DAPO epoch on the policy head.

        The trunk is frozen here (latents computed under no_grad): the
        policy learns to ACT on the representation, while the
        representation itself is shaped by the supervised+physics loss.
        Letting the noisy group-relative gradient reshape the encoder as
        well is how you get a trunk that fits reward noise.
        """
        from chimera.grpo import realised_pnl_reward

        self.net.eval()
        with torch.no_grad():
            state = self.net.policy_state(self.net.encode(X))

        def reward_fn(actions: torch.Tensor) -> torch.Tensor:
            return realised_pnl_reward(actions, next_returns, cost_bps=cost_bps)

        return self.grpo.collect_and_step(state, reward_fn)
