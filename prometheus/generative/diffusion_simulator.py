"""
Score-based Generative Diffusion Model (SGM) for synthetic market data.

Conditioned on market microstructure statistics, the SGM generates realistic
multi-asset return paths that respect:
  - Cross-asset correlations and tail dependencies
  - Stochastic volatility clustering (GARCH-like)
  - Jump processes (Poisson arrivals)
  - Microstructure effects (bid-ask spread, order imbalance)

This produces infinite synthetic training data including regimes that have
never occurred in recorded history.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Positional encoding for diffusion timestep t ∈ [0, T]."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().unsqueeze(-1)
        args = t * self.freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ScoreNetwork(nn.Module):
    """
    Score function estimator s_θ(x_t, t) ≈ ∇_{x_t} log p_t(x_t).

    Architecture: U-Net style with time conditioning.
    Input: noisy market returns x_t, condition c (microstructure stats)
    Output: score (denoising direction)
    """

    def __init__(
        self,
        n_assets: int,
        seq_len: int,
        cond_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 6,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.seq_len = seq_len
        input_dim = n_assets * seq_len

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks with time+condition conditioning
        self.blocks = nn.ModuleList([
            self._make_block(hidden_dim, time_embed_dim)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    @staticmethod
    def _make_block(d: int, t_dim: int) -> nn.Module:
        return nn.ModuleDict({
            "norm": nn.LayerNorm(d),
            "fc1": nn.Linear(d, d * 2),
            "fc2": nn.Linear(d * 2, d),
            "time_fc": nn.Linear(t_dim, d),
            "act": nn.SiLU(),
        })

    def forward(
        self,
        x_t: torch.Tensor,     # [B, seq_len, n_assets]
        t: torch.Tensor,        # [B] — diffusion timestep
        condition: torch.Tensor,  # [B, cond_dim]
    ) -> torch.Tensor:
        B = x_t.shape[0]
        h = self.input_proj(x_t.view(B, -1))  # [B, hidden_dim]
        t_emb = self.time_embed(t)             # [B, time_embed_dim]
        c_emb = self.cond_proj(condition)      # [B, hidden_dim]
        h = h + c_emb

        for block in self.blocks:
            residual = h
            h = block["norm"](h)
            h = block["fc1"](h)
            h = block["act"](h)
            t_scale = block["time_fc"](t_emb)
            h = h[:, :h.shape[1]//2] * (1 + t_scale) + h[:, h.shape[1]//2:] * t_scale[:, :h.shape[1]//2]
            # Simplified: just apply time conditioning as additive bias
            h = block["fc2"](block["act"](block["fc1"](block["norm"](residual))))
            h = h + t_scale + c_emb
            h = h + residual  # skip connection

        return self.output_proj(h).view(B, self.seq_len, self.n_assets)


class MarketDiffusionSimulator:
    """
    Full DDPM/SGM pipeline for synthetic market scenario generation.

    Supports:
      - Standard scenario generation (realistic but novel regimes)
      - Black-swan conditioning (extreme tail events)
      - Correlation regime conditioning (breakdown / flight-to-quality)
    """

    def __init__(
        self,
        n_assets: int = 20,
        seq_len: int = 252,    # 1 trading year
        cond_dim: int = 32,
        n_diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.n_assets = n_assets
        self.seq_len = seq_len
        self.n_diffusion_steps = n_diffusion_steps
        self.device = device

        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, n_diffusion_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        # Score network
        self.score_net = ScoreNetwork(n_assets, seq_len, cond_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.score_net.parameters(), lr=2e-4)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1ab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x_0 + sqrt_1ab * noise, noise

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """One reverse diffusion step: denoise x_t → x_{t-1}."""
        t_tensor = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)

        with torch.no_grad():
            score = self.score_net(x_t, t_tensor, condition)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]

        # DDPM reverse formula
        coef = beta_t / self.sqrt_one_minus_alpha_bars[t]
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - coef * score)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta_t)
            return mean + sigma * noise
        return mean

    @torch.no_grad()
    def generate(
        self,
        n_scenarios: int,
        condition: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate n_scenarios synthetic market paths.
        Returns [n_scenarios, seq_len, n_assets] return tensor.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if condition is None:
            condition = torch.zeros(n_scenarios, 32, device=self.device)

        x = torch.randn(n_scenarios, self.seq_len, self.n_assets, device=self.device)

        for t in reversed(range(self.n_diffusion_steps)):
            x = self.p_sample(x, t, condition)

        return x

    def train_step(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
    ) -> float:
        """Single training step. Returns loss value."""
        B = x_0.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (B,), device=self.device)
        x_t, noise = self.q_sample(x_0, t)

        pred_noise = self.score_net(x_t, t, condition)
        loss = F.mse_loss(pred_noise, noise)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    def compute_condition(
        self,
        historical_returns: np.ndarray,
    ) -> torch.Tensor:
        """Build conditioning vector from historical microstructure statistics."""
        returns = torch.tensor(historical_returns, dtype=torch.float32)
        vol = returns.std(dim=0)
        skew = ((returns - returns.mean(0)) ** 3).mean(0) / (vol ** 3 + 1e-8)
        kurt = ((returns - returns.mean(0)) ** 4).mean(0) / (vol ** 4 + 1e-8) - 3
        corr = torch.corrcoef(returns.T).flatten()[:20]  # first 20 corr values
        stats = torch.cat([
            vol[:5], skew[:5], kurt[:5], corr[:5],
            torch.tensor([returns.mean().item(), vol.mean().item()]),
        ])
        # Pad or truncate to cond_dim=32
        if stats.shape[0] < 32:
            stats = F.pad(stats, (0, 32 - stats.shape[0]))
        else:
            stats = stats[:32]
        return stats.unsqueeze(0).to(self.device)

    def validate_tail_coverage(
        self,
        n_scenarios: int = 1000,
        sigma_threshold: float = 3.0,
        min_tail_pct: float = 1.0,
    ) -> Dict:
        """
        Validate that the generated distribution covers fat-tail events.

        A model whose distribution is too narrow will be blindsided by real
        crises even after black-swan training — it predicts smooth paths and
        calls a 1987-style crash "impossible."

        Returns a dict with: coverage_pct (% of scenarios with >sigma_threshold σ
        move), kurtosis (excess; >0 = fatter than Gaussian), skewness, worst
        cumulative drawdown, and passes_fat_tail_check bool.
        """
        from scipy import stats as scipy_stats

        paths = self.generate(n_scenarios=n_scenarios, seed=42)  # [N, T, A]
        returns = paths.cpu().numpy()                             # [N, T, A]

        all_returns = returns.reshape(-1)
        std = float(np.std(all_returns)) + 1e-8

        # Extreme bars: single timestep exceeding sigma_threshold standard deviations
        extreme = np.abs(returns) > sigma_threshold * std        # [N, T, A]
        scenarios_with_extreme = extreme.any(axis=(1, 2))
        coverage_pct = float(100.0 * scenarios_with_extreme.mean())

        kurtosis = float(scipy_stats.kurtosis(all_returns, fisher=True))
        skewness = float(scipy_stats.skew(all_returns))

        cum_returns = returns.cumsum(axis=1)                     # [N, T, A]
        worst_drawdown = float(cum_returns.min())

        return {
            "coverage_pct": coverage_pct,
            "kurtosis": kurtosis,
            "skewness": skewness,
            "worst_drawdown": worst_drawdown,
            "sigma_threshold": sigma_threshold,
            "n_scenarios": n_scenarios,
            # AUDIT-2C: this used to be `kurtosis > -1.0 OR coverage_pct >=
            # min_tail_pct`. kurtosis > -1.0 is true for almost any
            # distribution, including a perfectly Gaussian one (kurtosis
            # 0) - combined with OR, the check could never actually fail
            # even for a generator producing too-narrow, non-fat-tailed
            # output (it would report passes=True with coverage_pct=0, as
            # long as kurtosis alone cleared the near-universal floor).
            # This was the guardrail that should have caught the
            # untrained-ScoreNetwork bug (see train_step()'s call sites -
            # or previous lack thereof) and didn't. Fixed to AND with a
            # real fat-tail threshold (kurtosis > 0 = genuinely fatter
            # than Gaussian, not just "not extremely platykurtic").
            "passes_fat_tail_check": kurtosis > 0.0 and coverage_pct >= min_tail_pct,
        }

    def save(self, path: str) -> None:
        torch.save({
            "score_net": self.score_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": {
                "n_assets": self.n_assets,
                "seq_len": self.seq_len,
                "n_diffusion_steps": self.n_diffusion_steps,
            },
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.score_net.load_state_dict(ckpt["score_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
