"""
Kronos Generative Nightmare - phase 2 of the daily cycle (02:00 - 04:00).

Seeds the existing MarketDiffusionSimulator with the current market state
(from DailyMemory) and generates thousands of adversarial futures biased
towards the worst case for the CURRENT portfolio. These futures become the
evaluation gauntlet for the NEAT evolution phase: tomorrow's model must
survive tonight's nightmares.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from prometheus.generative.diffusion_simulator import MarketDiffusionSimulator

logger = logging.getLogger(__name__)


@dataclass
class NightmareBuffer:
    """Temporary buffer of adversarial futures for the evolution phase."""

    futures: torch.Tensor          # [n_futures, horizon, n_assets] returns
    portfolio_pnl: torch.Tensor    # [n_futures] simulated portfolio P&L per future
    condition_vector: torch.Tensor # the market-state seed used

    @property
    def n_futures(self) -> int:
        return int(self.futures.shape[0])

    @property
    def variance(self) -> float:
        return float(self.futures.var().item())

    def worst(self, k: int) -> torch.Tensor:
        """The k futures with the worst portfolio P&L."""
        idx = torch.argsort(self.portfolio_pnl)[:k]
        return self.futures[idx]


class NightmareGenerator:
    """
    Conditional adversarial future generation.

    worst_case_bias fraction of the futures are re-ranked and kept from an
    oversampled pool so the buffer skews toward scenarios that hurt the
    current portfolio; the rest are unconditioned so evolution never
    overfits to doom alone.
    """

    def __init__(self, config, simulator: Optional[MarketDiffusionSimulator] = None):
        self.cfg = config
        n_assets = len(config.data.tickers)
        self.simulator = simulator or MarketDiffusionSimulator(
            n_assets=n_assets,
            seq_len=config.nightmare.horizon_days,
            n_diffusion_steps=config.nightmare.diffusion_steps,
        )

    # -- conditioning -------------------------------------------------------

    def condition_from_memory(self, memory) -> torch.Tensor:
        """Build the diffusion conditioning vector from today's DailyMemory."""
        returns = memory.returns_window(self.cfg.data.lookback_days)
        return self.simulator.compute_condition(returns)

    # -- generation ---------------------------------------------------------

    def generate(
        self,
        memory,
        portfolio_weights: Optional[np.ndarray] = None,
        n_futures: Optional[int] = None,
    ) -> NightmareBuffer:
        """
        Generate the nightly adversarial buffer.

        portfolio_weights: current position weights per asset. Defaults to
        equal-weight long if the book is flat (worst case still meaningful).
        """
        n_total = n_futures or self.cfg.nightmare.n_futures
        batch = min(self.cfg.nightmare.batch_size, n_total)
        n_assets = self.simulator.n_assets

        if portfolio_weights is None:
            portfolio_weights = np.full(n_assets, 1.0 / n_assets)
        weights = torch.tensor(portfolio_weights[:n_assets], dtype=torch.float32)

        condition = self.condition_from_memory(memory)
        seed = self.cfg.nightmare.get("seed", None)

        # Oversample so the adversarial re-ranking has candidates to discard.
        bias = float(self.cfg.nightmare.worst_case_bias)
        n_oversample = int(n_total * (1.0 + bias))

        chunks = []
        generated = 0
        while generated < n_oversample:
            n = min(batch, n_oversample - generated)
            cond = condition.repeat(n, 1)
            paths = self.simulator.generate(
                n_scenarios=n, condition=cond,
                seed=(seed + generated) if seed is not None else None,
            )
            chunks.append(paths)
            generated += n
        pool = torch.cat(chunks, dim=0)                    # [pool, T, A]

        # Portfolio P&L per future: sum over time of weighted returns
        pnl = (pool * weights.view(1, 1, -1)).sum(dim=(1, 2))   # [pool]

        # Keep the worst n_adversarial + a random slice of the rest
        n_adversarial = int(n_total * bias)
        n_random = n_total - n_adversarial
        order = torch.argsort(pnl)                          # ascending = worst first
        worst_idx = order[:n_adversarial]
        rest_idx = order[n_adversarial:]
        keep_rand = rest_idx[torch.randperm(len(rest_idx))[:n_random]]
        keep = torch.cat([worst_idx, keep_rand])

        buffer = NightmareBuffer(
            futures=pool[keep],
            portfolio_pnl=pnl[keep],
            condition_vector=condition,
        )
        logger.info(
            "[nightmare] %d futures generated (%d adversarial, %d random), "
            "variance=%.6f, worst_pnl=%.4f",
            buffer.n_futures, n_adversarial, n_random,
            buffer.variance, float(buffer.portfolio_pnl.min()),
        )
        if buffer.variance <= 0:
            raise RuntimeError(
                "Nightmare buffer has zero variance - diffusion collapse. "
                "Refusing to evolve against identical futures."
            )
        return buffer
