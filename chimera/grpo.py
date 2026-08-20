"""
Component 4/6 - Critic-free policy optimisation: GRPO + DAPO.

GRPO (Group Relative Policy Optimization, DeepSeekMath) removes the value
network. Instead of learning a critic to estimate a baseline, it samples
a GROUP of G actions from the same state and uses the group's own reward
statistics as the baseline:

    A_g = (r_g - mean(r)) / (std(r) + eps)

Why that is the right call specifically for trading, rather than just
cheaper: a value critic has to learn E[return | market state]. That
quantity is close to zero and swamped by noise - it is, more or less,
the very thing this project has spent the session failing to predict.
Asking a critic to learn it in order to reduce variance is circular. The
group-relative baseline sidesteps it entirely: it only ever asks "which
of these G portfolios did better THIS bar", a comparison that is
well-posed even when the absolute expected return is unlearnable.

DAPO (2025) adds four fixes to GRPO, three of which are implemented here
(the fourth, overlong-reward shaping, is about token budgets in text
generation and has no trading analogue):

  1. CLIP-HIGHER - decouple the clip range: eps_low != eps_high. With a
     symmetric clip, low-probability actions can never gain enough ratio
     to be reinforced, and the policy collapses to low entropy. Allowing
     a larger upward clip preserves exploration. Directly relevant here:
     entropy collapse in a trading policy means it stops varying its
     positions, which is exactly the near-zero-turnover failure mode
     this project has already hit once.

  2. DYNAMIC SAMPLING - drop groups whose rewards are all (near-)equal.
     Their advantages are all ~0, so they contribute no gradient but do
     contribute to the denominator, silently shrinking the effective
     learning rate. In trading these are common: on a quiet bar, every
     sampled portfolio earns roughly the same nothing.

  3. ASSET-LEVEL LOSS - DAPO's token-level aggregation, transposed. Sum
     the loss over assets and normalise by TOTAL asset count across the
     batch, rather than averaging per-sample then across samples. Keeps
     every asset's contribution equally weighted regardless of how many
     assets a given sample has.

The policy is a diagonal Gaussian over target portfolio weights, squashed
through tanh, so an action is a full portfolio in [-1, 1]^n_assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DAPOConfig:
    """GRPO/DAPO hyperparameters.

    group_size: G. Larger = lower-variance baseline but linearly more
        forward passes. 16 is a reasonable floor for a usable std.
    eps_low / eps_high: the DAPO decoupled clip. eps_high > eps_low is
        the whole point (clip-higher); setting them equal reduces this
        to standard GRPO and is supported for ablation.
    min_reward_std: dynamic-sampling threshold. Groups flatter than this
        are dropped entirely.
    entropy_coef: explicit entropy bonus. Belt-and-braces alongside
        clip-higher, because policy collapse is the failure mode that
        matters most here.
    """

    group_size: int = 16
    eps_low: float = 0.2
    eps_high: float = 0.28
    min_reward_std: float = 1e-6
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    init_log_std: float = -1.0


class GaussianPortfolioPolicy(nn.Module):
    """State -> diagonal Gaussian over pre-squash portfolio weights.

    Actions are tanh(z), so every sampled portfolio lies in [-1, 1] per
    asset before position sizing. log_std is a free parameter per asset
    (state-independent): a state-dependent std is easy to add but tends
    to collapse early in low-signal regimes, and a fixed-but-learned
    scale is the more honest starting point.
    """

    def __init__(self, state_dim: int, n_assets: int, hidden: int = 64,
                 init_log_std: float = -1.0):
        super().__init__()
        self.n_assets = n_assets
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_assets),
        )
        self.log_std = nn.Parameter(torch.full((n_assets,), float(init_log_std)))

    def distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        mean = self.net(state)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Deterministic action (the squashed mean) - used at inference."""
        return torch.tanh(self.net(state))

    def sample_group(self, state: torch.Tensor, group_size: int
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample G actions per state.

        Returns (actions [B, G, n_assets] in [-1,1],
                 logprobs [B, G, n_assets] pre-squash, summed later).

        Note logprobs are of the PRE-squash z, kept per-asset so the
        asset-level loss aggregation (DAPO fix 3) can operate on them.
        The tanh Jacobian is deliberately NOT included: it is identical
        between the old and new policy for the same z, so it cancels
        exactly in the importance ratio.
        """
        dist = self.distribution(state)
        z = dist.rsample((group_size,))              # [G, B, n_assets]
        logp = dist.log_prob(z)                      # [G, B, n_assets]
        actions = torch.tanh(z)
        return actions.permute(1, 0, 2), logp.permute(1, 0, 2)

    def log_prob_of(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Recompute per-asset log-probs of given actions. [B,G,A] -> [B,G,A]."""
        dist = self.distribution(state)              # batch [B, A]
        # Invert the tanh squash to recover z. atanh explodes at +-1, so
        # clamp inside the open interval first.
        z = torch.atanh(actions.clamp(-0.999999, 0.999999))
        return dist.log_prob(z.permute(1, 0, 2)).permute(1, 0, 2)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).entropy().sum(dim=-1)


class GroupRelativePolicy:
    """GRPO/DAPO trainer around a GaussianPortfolioPolicy."""

    def __init__(self, policy: GaussianPortfolioPolicy,
                 config: Optional[DAPOConfig] = None, lr: float = 3e-4):
        self.policy = policy
        self.cfg = config or DAPOConfig()
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    @staticmethod
    def group_advantages(rewards: torch.Tensor, min_std: float = 1e-6
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Group-relative advantages - THE critic-free step.

        rewards: [B, G] -> (advantages [B, G], keep_mask [B] bool)

        keep_mask implements DAPO dynamic sampling: a group whose rewards
        are effectively identical carries no comparative information, so
        it is excluded rather than allowed to dilute the batch.
        """
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True)
        keep = (std.squeeze(1) > min_std)
        adv = (rewards - mean) / (std + 1e-8)
        return adv, keep

    def step(self, state: torch.Tensor, actions: torch.Tensor,
             old_logp: torch.Tensor, rewards: torch.Tensor) -> dict:
        """One DAPO update.

        state:     [B, state_dim]
        actions:   [B, G, n_assets]   (sampled under the old policy)
        old_logp:  [B, G, n_assets]   (per-asset log-probs at sample time)
        rewards:   [B, G]
        """
        cfg = self.cfg
        adv, keep = self.group_advantages(rewards, cfg.min_reward_std)

        n_kept = int(keep.sum().item())
        if n_kept == 0:
            # Every group was flat. Nothing to learn from - and reporting
            # this honestly matters, because a run where most batches are
            # skipped is telling you the reward signal is degenerate.
            return {"loss": 0.0, "kept_groups": 0, "total_groups": int(keep.numel()),
                    "mean_abs_adv": 0.0, "entropy": 0.0, "clip_frac": 0.0}

        state_k = state[keep]
        actions_k = actions[keep]
        old_logp_k = old_logp[keep].detach()
        adv_k = adv[keep].detach()

        new_logp = self.policy.log_prob_of(state_k, actions_k)      # [b,G,A]
        ratio = (new_logp - old_logp_k).exp()

        adv_exp = adv_k.unsqueeze(-1).expand_as(ratio)
        unclipped = ratio * adv_exp
        clipped = torch.clamp(ratio, 1.0 - cfg.eps_low, 1.0 + cfg.eps_high) * adv_exp

        # DAPO asset-level aggregation: sum over (group, asset) and divide
        # by the total count, NOT a mean-of-means. With equal asset counts
        # these coincide; the form is kept because it is the one that
        # stays correct if asset counts ever vary per sample.
        per_elem = -torch.min(unclipped, clipped)
        policy_loss = per_elem.sum() / per_elem.numel()

        entropy = self.policy.entropy(state_k).mean()
        loss = policy_loss - cfg.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            clip_frac = ((ratio < 1 - cfg.eps_low) | (ratio > 1 + cfg.eps_high)).float().mean()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.item()),
            "kept_groups": n_kept,
            "total_groups": int(keep.numel()),
            "mean_abs_adv": float(adv_k.abs().mean().item()),
            "clip_frac": float(clip_frac.item()),
            "grad_norm": float(grad_norm),
        }

    def collect_and_step(self, state: torch.Tensor,
                         reward_fn: Callable[[torch.Tensor], torch.Tensor]) -> dict:
        """Sample a group, score it, and take one update.

        reward_fn: [B, G, n_assets] actions -> [B, G] rewards. Kept as a
        callback so the reward definition (raw PnL, cost-adjusted,
        risk-adjusted, drawdown-penalised) lives with the caller and can
        be swapped without touching the optimiser.
        """
        with torch.no_grad():
            actions, old_logp = self.policy.sample_group(state, self.cfg.group_size)
            rewards = reward_fn(actions)
        return self.step(state, actions, old_logp, rewards)


def realised_pnl_reward(actions: torch.Tensor, next_returns: torch.Tensor,
                        cost_bps: float = 25.0, prev_weights: Optional[torch.Tensor] = None,
                        risk_aversion: float = 0.0) -> torch.Tensor:
    """Cost-aware realised reward for a group of candidate portfolios.

    actions:      [B, G, A] target weights in [-1, 1]
    next_returns: [B, A]    the realised next-bar return (train window only)
    cost_bps:     round-trip transaction cost on turnover, in basis points.
                  25bp is the realistic NSE delivery figure (STT 0.1% on
                  the sell side dominates, plus exchange/GST/stamp) - and
                  charging it INSIDE the reward is deliberate: a policy
                  trained on gross PnL learns to churn, then dies on
                  costs in the backtest.
    risk_aversion: penalty on squared portfolio return, a
                  mean-variance-style shrink toward smaller bets.
    """
    nr = next_returns.unsqueeze(1)                       # [B,1,A]
    gross = (actions * nr).sum(dim=-1)                   # [B,G]

    if prev_weights is None:
        turnover = actions.abs().sum(dim=-1)
    else:
        turnover = (actions - prev_weights.unsqueeze(1)).abs().sum(dim=-1)
    cost = turnover * (cost_bps / 10_000.0)

    reward = gross - cost
    if risk_aversion > 0:
        reward = reward - risk_aversion * gross.pow(2)
    return reward
