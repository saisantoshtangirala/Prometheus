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

COLLAPSE_VARIANCE_THRESHOLD = 1e-5
INF_CLIP = 1e6
MIN_PRICE = 0.01


class NightmareCollapseError(RuntimeError):
    """Diffusion mode collapse: futures are (near-)identical even after reseed."""


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

    def to_price_paths(self, initial_prices: torch.Tensor) -> torch.Tensor:
        """
        Convert return futures to price paths, clamped at MIN_PRICE ($0.01).
        A simulated black swan may take returns below -100%; prices must
        never go negative (NIG-04).
        """
        cum = (1.0 + self.futures).cumprod(dim=1)          # [N, T, A]
        prices = initial_prices.view(1, 1, -1) * cum
        return prices.clamp(min=MIN_PRICE)


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

        pool = self._generate_pool(n_oversample, batch, condition, seed)
        pool = self._sanitize(pool)

        # NIG-02: mode-collapse detection -> reseed once -> bootstrap fallback
        if float(pool.var().item()) < COLLAPSE_VARIANCE_THRESHOLD:
            logger.error(
                "[nightmare] diffusion collapse (var=%.2e) - reseeding",
                float(pool.var().item()),
            )
            reseed = (seed or 0) + 7919
            pool = self._sanitize(
                self._generate_pool(n_oversample, batch, condition, reseed)
            )
            if float(pool.var().item()) < COLLAPSE_VARIANCE_THRESHOLD:
                logger.critical(
                    "[nightmare] collapse persists after reseed - "
                    "falling back to historical bootstrap"
                )
                return self._bootstrap_from_history(memory, n_total, weights)

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
        return buffer

    # -- resilience helpers -------------------------------------------------

    def _generate_pool(
        self, n_total: int, batch: int, condition: torch.Tensor,
        seed: Optional[int],
    ) -> torch.Tensor:
        """
        Batched generation with OOM back-off (NIG-01): a MemoryError or CUDA
        OOM halves the batch size and retries instead of dying.
        """
        chunks = []
        generated = 0
        current_batch = batch
        while generated < n_total:
            n = min(current_batch, n_total - generated)
            cond = condition.repeat(n, 1)
            try:
                paths = self.simulator.generate(
                    n_scenarios=n, condition=cond,
                    seed=(seed + generated) if seed is not None else None,
                )
            except (MemoryError, torch.cuda.OutOfMemoryError) as e:
                if current_batch <= 1:
                    raise
                current_batch = max(1, current_batch // 2)
                logger.warning(
                    "[nightmare] OOM at batch=%d - reducing to %d (%s)",
                    n, current_batch, type(e).__name__,
                )
                continue
            chunks.append(paths)
            generated += n
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _sanitize(pool: torch.Tensor) -> torch.Tensor:
        """NIG-03: NaN -> 0.0, +/-inf -> +/-INF_CLIP, with a critical log."""
        n_nan = int(torch.isnan(pool).sum().item())
        n_inf = int(torch.isinf(pool).sum().item())
        if n_nan or n_inf:
            logger.critical(
                "[nightmare] sanitizing generated futures: %d NaN, %d inf",
                n_nan, n_inf,
            )
            pool = torch.nan_to_num(pool, nan=0.0, posinf=INF_CLIP, neginf=-INF_CLIP)
        return pool

    def _bootstrap_from_history(
        self, memory, n_total: int, weights: torch.Tensor
    ) -> NightmareBuffer:
        """
        Last-resort futures: block-bootstrap resampling of historical returns.
        Not adversarial, but statistically honest - and never degenerate.

        Each future is a real horizon-length run of CONSECUTIVE historical
        days (from a random start point), not `horizon` independently
        resampled days - the previous implementation resampled each day
        independently, which despite this docstring's "block-bootstrap"
        label was not actually block-bootstrapping: it destroyed all
        temporal structure (autocorrelation, momentum, vol clustering) in
        what it fed the evolution phase whenever this fallback triggered.
        Verified on a synthetic AR(1) series (kronos/backtest.py's
        _bootstrap_futures fix carries the same measurement in its
        docstring): independent per-bar resampling took a true lag-1
        autocorr of 0.293 down to a pooled 0.002 (destroyed); contiguous
        block resampling preserves it at 0.287.
        """
        horizon = self.cfg.nightmare.horizon_days
        history = torch.tensor(
            memory.returns_window(self.cfg.data.lookback_days),
            dtype=torch.float32,
        )                                                   # [H, A]
        n_hist = history.shape[0]
        if n_hist < 2:
            raise NightmareCollapseError(
                "Diffusion collapsed and history is too short to bootstrap."
            )
        if n_hist <= horizon:
            starts = torch.zeros(n_total, dtype=torch.long)
        else:
            starts = torch.randint(0, n_hist - horizon, (n_total,))
        idx = starts.unsqueeze(1) + torch.arange(horizon).unsqueeze(0)  # [N, horizon] contiguous
        futures = history[idx]                              # [N, T, A]
        # Jitter so resampled rows are never bit-identical
        futures = futures + torch.randn_like(futures) * history.std() * 0.1
        pnl = (futures * weights.view(1, 1, -1)).sum(dim=(1, 2))
        logger.warning(
            "[nightmare] bootstrap fallback produced %d futures (var=%.6f)",
            n_total, float(futures.var().item()),
        )
        return NightmareBuffer(
            futures=futures, portfolio_pnl=pnl,
            condition_vector=torch.zeros(1),
        )
