"""
Hetzner-side executor: turns a verified checkpoint into live signals.

The contract with the RunPod side is deliberately narrow - a genome plus
the metadata needed to prove it decodes the same way here as it did
there. Everything downstream (indicator computation, scoring, position
rules) is the SAME CODE the GA optimised against, imported from the
same modules. There is no second implementation of the strategy to
drift out of sync, which is the specific failure that made the previous
project's backtest measure a different model from the one that traded.

Latency: `signal()` computes 20 indicators over a short trailing window
and takes a weighted vote. The acceptance criterion is <100ms/tick;
measured cost is on the order of a few milliseconds for a 10-name book,
because the work is a handful of pandas rolling ops on ~60 bars.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nightevolver.data_loader import WARMUP_BARS, build_market_data
from nightevolver.genome import DecodedStrategy, decode, score_matrix
from nightevolver.saver import load_checkpoint

logger = logging.getLogger("nightevolver.decoder")


@dataclass
class LiveSignal:
    """One bar's decision for one asset."""

    ticker: str
    score: float            # weighted indicator vote, [-1, +1]
    direction: int          # -1 short, 0 flat, +1 long
    target_weight: float    # after conviction floor, Kelly and the cap
    latency_ms: float = 0.0


class EvolvedStrategy:
    """A loaded, verified genome ready to produce live signals."""

    def __init__(self, strategy: DecodedStrategy, tickers: Sequence[str],
                 max_position: float = 0.10, metrics: Optional[Dict] = None):
        self.strategy = strategy
        self.tickers = tuple(tickers)
        self.max_position = float(max_position)
        self.metrics = metrics or {}

    @classmethod
    def from_checkpoint(cls, path: Optional[Path] = None, require_gate: bool = True,
                        max_position: float = 0.10) -> "EvolvedStrategy":
        """Load from disk, refusing anything that fails verification."""
        ck = load_checkpoint(path, require_gate=require_gate)
        logger.info(
            "[nightevolver] loaded checkpoint trained %s | OOS Sharpe %s | "
            "deflated P(SR>0) %.3f",
            ck.get("trained_at"),
            (ck.get("metrics", {}).get("out_of_sample") or {}).get("sharpe"),
            float(ck.get("metrics", {}).get("deflated_sharpe_prob", 0.0)),
        )
        return cls(ck["decoded"], ck.get("tickers", ()), max_position,
                   ck.get("metrics"))

    def signal(self, close: pd.DataFrame, high: Optional[pd.DataFrame] = None,
               low: Optional[pd.DataFrame] = None,
               volume: Optional[pd.DataFrame] = None) -> List[LiveSignal]:
        """Signals for the MOST RECENT bar of the supplied history.

        `close` must contain at least WARMUP_BARS + a few rows so every
        indicator has its full lookback; short history yields a
        zero-signal (flat) result rather than a guess computed from
        half-warmed indicators.
        """
        t0 = time.perf_counter()
        if len(close) < WARMUP_BARS + 2:
            logger.warning("[nightevolver] only %d bars, need >= %d - staying flat",
                           len(close), WARMUP_BARS + 2)
            return [LiveSignal(t, 0.0, 0, 0.0) for t in close.columns]

        md = build_market_data(close, high, low, volume)
        if md.n_bars == 0:
            return [LiveSignal(t, 0.0, 0, 0.0) for t in close.columns]

        scores = score_matrix(md.indicators[-1:], self.strategy)[0]     # [A]
        latency = (time.perf_counter() - t0) * 1000.0

        out: List[LiveSignal] = []
        for i, ticker in enumerate(md.tickers):
            s = float(scores[i])
            if abs(s) > self.strategy.conviction_floor:
                direction = int(np.sign(s))
                weight = float(np.clip(abs(s) * self.strategy.kelly_fraction,
                                       0.0, self.max_position)) * direction
            else:
                direction, weight = 0, 0.0
            out.append(LiveSignal(ticker, s, direction, weight, latency))
        return out

    def describe(self) -> str:
        return self.strategy.describe()
