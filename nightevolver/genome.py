"""
Strategy DNA: a fixed-length real vector encoding an interpretable
rule-based trading strategy.

Everything the GA searches over lives in the unit hypercube [0,1]^N.
Genetic operators (crossover, Gaussian mutation, clipping) therefore
need no special-casing per gene, and `decode()` is the single place
where a gene becomes a meaningful quantity. That separation matters:
a bug in the mapping shows up in one function, not scattered through
the evolutionary loop.

THE STRATEGY THIS ENCODES

Each of N_INDICATORS classical indicators casts a directional vote:

    vote_i = +1  if normalised_i >  entry_i     (bullish trigger)
             -1  if normalised_i < -exit_i      (bearish trigger)
              0  otherwise                      (no opinion)

    score  = sum_i (w_i * vote_i) / sum_i w_i          in [-1, +1]

A position opens when |score| clears `conviction_floor`, sized by
`kelly_fraction`, and is held for `hold_days` bars or until the
trailing stop triggers. That is a completely ordinary technical
strategy - which is the point. The search space is small,
interpretable, and constrained to shapes that have some prior claim to
being real, rather than an unconstrained function approximator free to
fit noise.

WHY INTERPRETABILITY IS A RISK CONTROL, NOT A NICETY
When the previous model produced a 48% hit rate there was no way to ask
it "what do you believe and why". A decoded genome answers that in one
printed table, so a strategy that has latched onto something absurd is
visible before it trades rather than after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Indicator ordering is FIXED and shared by training and execution.
# If this list ever changes, previously-saved genomes become
# mis-decoded - a silent, catastrophic failure where gene 7 stops
# meaning what it meant when it was evolved. `GENOME_VERSION` exists so
# that mismatch is detected at load time instead of at trade time.
TECHNICAL_INDICATOR_NAMES: Tuple[str, ...] = (
    "rsi_14", "rsi_28",
    "macd_hist", "macd_signal_cross",
    "bb_position", "bb_width",
    "ema_9_21_cross", "ema_21_50_cross", "price_vs_ema50",
    "sma_20_slope",
    "adx_14", "di_spread",
    "stoch_k", "stoch_d_cross",
    "atr_pct",
    "vwap_gap",
    "mom_5", "mom_21",
    "vol_ratio", "ret_zscore",
)

# FII/DII participant-positioning channels, appended AFTER the technical
# indicators so that the first 20 gene positions keep the meaning they
# had in GENOME_VERSION 1.
#
# These are market-wide: one value per session, identical across every
# asset. That is worth stating where the genome can see it, because the
# information audit measured what they carry, and at market level - their
# honest sample size - it was nothing: no flow channel survived FDR
# correction against any of the four targets, and the best (fii_stk_fut_net
# -> rel_strength_1d) landed at p=0.60 once the T*A inflation was removed.
#
# They are wired in because adding them was asked for and because the
# measurement should be reproducible, not because they are expected to
# help. Enabling them widens the search space by 30% at no information
# gain, which raises the best-of-N noise ceiling the deflated Sharpe has
# to clear. Expect the gate to get HARDER to pass, not easier.
FLOW_CHANNEL_NAMES: Tuple[str, ...] = (
    "fii_idx_fut_net", "fii_idx_fut_net_chg", "fii_stk_fut_net",
    "dii_idx_fut_net", "client_idx_fut_net", "fii_opt_directional",
)

N_TECHNICAL = len(TECHNICAL_INDICATOR_NAMES)
N_FLOW = len(FLOW_CHANNEL_NAMES)

# One STATIC layout rather than a runtime toggle. A toggle would have to
# rebind these module globals, and every module that does
# `from nightevolver.genome import INDICATOR_NAMES` captures the binding
# at import time - so the layout could differ between two modules in the
# same process. That is precisely the "gene 7 means something else now"
# failure this ordering comment warns about, so the flow channels are
# always present and are filled with ZEROS when flow data is unavailable.
# A zero channel casts a zero vote, so it is inert rather than absent.
INDICATOR_NAMES: Tuple[str, ...] = TECHNICAL_INDICATOR_NAMES + FLOW_CHANNEL_NAMES
N_INDICATORS = len(INDICATOR_NAMES)

# 2: flow channels appended. Version 1 genomes decode incorrectly against
# this layout and load_checkpoint must reject them - which is the point
# of the field.
GENOME_VERSION = 2

# Gene layout in the flat [0,1] vector.
_W0, _W1 = 0, N_INDICATORS                              # indicator weights
_E0, _E1 = _W1, _W1 + N_INDICATORS                      # entry thresholds
_X0, _X1 = _E1, _E1 + N_INDICATORS                      # exit thresholds
IDX_HOLD_DAYS = _X1
IDX_TRAILING_STOP = _X1 + 1
IDX_KELLY = _X1 + 2
IDX_REGIME = _X1 + 3
IDX_CONVICTION = _X1 + 4
GENOME_LENGTH = _X1 + 5                                  # 65

# Decoded parameter ranges.
HOLD_DAYS_RANGE = (2, 60)
TRAILING_STOP_RANGE = (0.02, 0.20)      # floor at 2%: a 0% trailing stop
                                        # would exit on any tick against
                                        # the position, which is not a
                                        # strategy, it is a cost machine
KELLY_RANGE = (0.0, 1.0)
REGIME_RANGE = (0.0, 1.0)
CONVICTION_RANGE = (0.05, 0.60)         # floor at 0.05 so a genome cannot
                                        # trade on numerical dust


@dataclass
class DecodedStrategy:
    """Human-readable form of a genome - what actually gets traded."""

    indicator_weights: np.ndarray        # [N_INDICATORS], sums to 1
    entry_thresholds: np.ndarray         # [N_INDICATORS] in [0, 1]
    exit_thresholds: np.ndarray          # [N_INDICATORS] in [0, 1]
    hold_days: int
    trailing_stop: float
    kelly_fraction: float
    regime_sensitivity: float
    conviction_floor: float

    def top_indicators(self, n: int = 5) -> List[Tuple[str, float]]:
        order = np.argsort(-self.indicator_weights)[:n]
        return [(INDICATOR_NAMES[i], float(self.indicator_weights[i])) for i in order]

    def describe(self) -> str:
        """One-screen summary. This is the 'what does it believe' answer."""
        lines = [
            f"hold_days={self.hold_days}  trailing_stop={self.trailing_stop:.1%}  "
            f"kelly={self.kelly_fraction:.2f}  conviction_floor={self.conviction_floor:.2f}  "
            f"regime_sensitivity={self.regime_sensitivity:.2f}",
            "top indicators by weight:",
        ]
        for name, w in self.top_indicators(6):
            i = INDICATOR_NAMES.index(name)
            lines.append(f"    {name:20s} w={w:.3f}  entry>{self.entry_thresholds[i]:+.2f}"
                         f"  exit<{-self.exit_thresholds[i]:+.2f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "genome_version": GENOME_VERSION,
            "indicator_names": list(INDICATOR_NAMES),
            "indicator_weights": self.indicator_weights.tolist(),
            "entry_thresholds": self.entry_thresholds.tolist(),
            "exit_thresholds": self.exit_thresholds.tolist(),
            "hold_days": int(self.hold_days),
            "trailing_stop": float(self.trailing_stop),
            "kelly_fraction": float(self.kelly_fraction),
            "regime_sensitivity": float(self.regime_sensitivity),
            "conviction_floor": float(self.conviction_floor),
        }


def _lerp(gene: float, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * float(np.clip(gene, 0.0, 1.0)))


def decode(genome: np.ndarray) -> DecodedStrategy:
    """[0,1]^GENOME_LENGTH -> DecodedStrategy.

    THE single mapping from genes to meaning. Training and execution
    both call this, so a strategy cannot mean one thing on RunPod and
    another on Hetzner - which is the exact class of failure that made
    the previous project's backtest measure a different model from the
    one that traded.
    """
    g = np.asarray(genome, dtype=np.float64).ravel()
    if g.size != GENOME_LENGTH:
        raise ValueError(f"genome length {g.size}, expected {GENOME_LENGTH}")
    g = np.clip(g, 0.0, 1.0)

    w = g[_W0:_W1].copy()
    total = w.sum()
    # A genome that zeroed every weight would divide by zero downstream;
    # fall back to equal weighting rather than raising inside a
    # 1000-evaluation GA loop.
    w = w / total if total > 1e-9 else np.full(N_INDICATORS, 1.0 / N_INDICATORS)

    return DecodedStrategy(
        indicator_weights=w,
        entry_thresholds=g[_E0:_E1].copy(),
        exit_thresholds=g[_X0:_X1].copy(),
        hold_days=int(round(_lerp(g[IDX_HOLD_DAYS], *HOLD_DAYS_RANGE))),
        trailing_stop=_lerp(g[IDX_TRAILING_STOP], *TRAILING_STOP_RANGE),
        kelly_fraction=_lerp(g[IDX_KELLY], *KELLY_RANGE),
        regime_sensitivity=_lerp(g[IDX_REGIME], *REGIME_RANGE),
        conviction_floor=_lerp(g[IDX_CONVICTION], *CONVICTION_RANGE),
    )


def random_genome(rng: np.random.Generator) -> np.ndarray:
    return rng.random(GENOME_LENGTH)


def score_matrix(normalised: np.ndarray, strat: DecodedStrategy) -> np.ndarray:
    """Weighted indicator vote -> conviction score in [-1, +1].

    normalised: [T, n_assets, N_INDICATORS], each channel squashed to
                roughly [-1, 1] by the data loader.
    returns:    [T, n_assets] score

    Fully vectorised over time AND assets. This is the GA's inner loop -
    it is called once per genome per generation, so a Python loop here
    would dominate the entire training run.
    """
    X = np.asarray(normalised, dtype=np.float64)
    if X.shape[-1] != N_INDICATORS:
        raise ValueError(f"expected {N_INDICATORS} indicators, got {X.shape[-1]}")

    long_vote = (X > strat.entry_thresholds[None, None, :])
    short_vote = (X < -strat.exit_thresholds[None, None, :])
    votes = long_vote.astype(np.float64) - short_vote.astype(np.float64)
    return votes @ strat.indicator_weights          # weights already sum to 1


def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
              p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform crossover - each gene independently swapped with prob p."""
    mask = rng.random(a.size) < p
    c1, c2 = a.copy(), b.copy()
    c1[mask], c2[mask] = b[mask], a[mask]
    return c1, c2


def mutate(g: np.ndarray, rng: np.random.Generator, rate: float = 0.1,
           sigma: float = 0.05) -> np.ndarray:
    """Gaussian perturbation on a random subset of genes, clipped to [0,1].

    Clipping rather than reflecting: a gene pinned at a bound is a
    meaningful outcome (e.g. "maximum trailing stop"), and reflection
    would inject spurious movement away from bounds the search may
    legitimately want to sit on.
    """
    out = g.copy()
    mask = rng.random(g.size) < rate
    out[mask] += rng.normal(0.0, sigma, size=int(mask.sum()))
    return np.clip(out, 0.0, 1.0)
