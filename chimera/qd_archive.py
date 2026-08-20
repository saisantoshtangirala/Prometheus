"""
Component 6/6 - Quality-Diversity evolution (MAP-Elites) over strategies.

Standard evolution converges: run it long enough and the population is N
near-copies of one local optimum. For trading that is the worst possible
outcome - an "ensemble" of correlated strategies has the risk profile of
a single strategy while looking diversified.

MAP-Elites (Mouret & Clune) fixes this by refusing to compare strategies
that BEHAVE differently. The archive is a grid over behaviour
descriptors; each cell keeps only the fittest strategy exhibiting that
behaviour. A mediocre strategy in an empty cell is kept; an excellent
strategy only displaces the incumbent of its OWN cell. The output is a
map of "the best way to trade, for each style of trading".

Behaviour descriptors here are chosen to be things a risk manager would
actually name, and to be causally independent of fitness (a BD that
correlates with fitness collapses the archive):

  turnover       mean |w_t - w_{t-1}| - patient vs frenetic
  net_exposure   mean sum(w)          - directional vs market-neutral
  concentration  mean HHI of |w|      - punchy vs diversified

Fitness is Sharpe on the train window. Two strategies with Sharpe 0.8
occupy different cells if one is a concentrated long-biased holder and
the other a diversified market-neutral churner, and the ensemble of the
two is genuinely diversified in a way an ensemble of top-2-by-fitness
would not be.

The payoff at inference: `ensemble_weights` blends elites across cells
with a fitness softmax, so the deployed signal is an average over
behaviourally distinct strategies. That is a real robustness argument -
if one behavioural regime stops working, the others are not
automatically correlated with its failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Behaviour descriptor bounds. Fixed rather than adaptive so archive
# cells mean the same thing across walk-forward windows - otherwise
# "cell (2,3,1)" silently changes definition between windows and the
# archive cannot be compared or carried forward.
BD_BOUNDS = {
    "turnover": (0.0, 2.0),
    "net_exposure": (-1.0, 1.0),
    "concentration": (0.0, 1.0),
}
BD_NAMES = tuple(BD_BOUNDS.keys())


@dataclass
class StrategyGenome:
    """Parameters of one strategy in the archive.

    Deliberately a small, interpretable vector rather than raw network
    weights: MAP-Elites needs many cheap evaluations, and mutating a
    genome that maps to comprehensible behaviour is what lets the
    behaviour descriptors spread. The genome modulates a FIXED trained
    policy rather than replacing it - evolution here is searching over
    how to deploy the learned signal, not re-learning the signal.
    """

    signal_gain: float = 1.0        # scales raw signal before squashing
    signal_threshold: float = 0.0   # deadband: |signal| below this -> 0
    max_weight: float = 0.25        # per-asset cap
    long_bias: float = 0.0          # additive tilt applied to every asset
    smoothing: float = 0.0          # EMA on weights: 0 = none, ->1 = frozen
    top_k_frac: float = 1.0         # keep only the strongest fraction of names

    def to_array(self) -> np.ndarray:
        return np.array([self.signal_gain, self.signal_threshold, self.max_weight,
                         self.long_bias, self.smoothing, self.top_k_frac], dtype=np.float64)

    @staticmethod
    def from_array(a: np.ndarray) -> "StrategyGenome":
        return StrategyGenome(
            signal_gain=float(np.clip(a[0], 0.05, 10.0)),
            signal_threshold=float(np.clip(a[1], 0.0, 0.9)),
            max_weight=float(np.clip(a[2], 0.02, 1.0)),
            long_bias=float(np.clip(a[3], -0.5, 0.5)),
            smoothing=float(np.clip(a[4], 0.0, 0.95)),
            top_k_frac=float(np.clip(a[5], 0.1, 1.0)),
        )

    @staticmethod
    def random(rng: np.random.Generator) -> "StrategyGenome":
        return StrategyGenome.from_array(np.array([
            rng.uniform(0.2, 4.0), rng.uniform(0.0, 0.5), rng.uniform(0.05, 0.5),
            rng.uniform(-0.3, 0.3), rng.uniform(0.0, 0.8), rng.uniform(0.2, 1.0),
        ]))

    def mutate(self, rng: np.random.Generator, sigma: float = 0.15) -> "StrategyGenome":
        """Gaussian mutation, scaled per-gene to each gene's own range."""
        a = self.to_array()
        scale = np.array([2.0, 0.3, 0.25, 0.25, 0.3, 0.3])
        return StrategyGenome.from_array(a + rng.normal(0.0, sigma, size=a.shape) * scale)


def apply_genome(signals: np.ndarray, genome: StrategyGenome,
                 prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Turn a raw signal vector into portfolio weights under a genome.

    signals: [n_assets] raw model output (roughly [-1, 1])
    returns: [n_assets] target weights
    """
    s = np.asarray(signals, dtype=np.float64).copy()
    s = np.tanh(s * genome.signal_gain)

    # Deadband: forces the strategy to actually abstain rather than hold
    # a permanent dust position in every name (which is pure cost).
    s[np.abs(s) < genome.signal_threshold] = 0.0

    # Keep only the strongest names - the concentration lever.
    if genome.top_k_frac < 1.0:
        k = max(1, int(round(genome.top_k_frac * s.size)))
        if k < s.size:
            cutoff = np.partition(np.abs(s), s.size - k)[s.size - k]
            s[np.abs(s) < cutoff] = 0.0

    w = s + genome.long_bias
    w = np.clip(w, -genome.max_weight, genome.max_weight)

    if prev_weights is not None and genome.smoothing > 0:
        w = genome.smoothing * np.asarray(prev_weights, dtype=np.float64) \
            + (1.0 - genome.smoothing) * w
    return w


def behaviour_descriptors(weight_path: np.ndarray) -> Dict[str, float]:
    """Behaviour of a strategy from its realised weight path [T, n_assets]."""
    W = np.asarray(weight_path, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] < 2:
        return {"turnover": 0.0, "net_exposure": 0.0, "concentration": 0.0}

    turnover = float(np.abs(np.diff(W, axis=0)).sum(axis=1).mean())
    net_exposure = float(W.sum(axis=1).mean())

    absW = np.abs(W)
    tot = absW.sum(axis=1, keepdims=True)
    ok = tot.squeeze(1) > 1e-12
    if np.any(ok):
        shares = absW[ok] / tot[ok]
        concentration = float((shares ** 2).sum(axis=1).mean())
    else:
        concentration = 0.0
    return {"turnover": turnover, "net_exposure": net_exposure,
            "concentration": concentration}


@dataclass
class Elite:
    genome: StrategyGenome
    fitness: float
    bd: Dict[str, float]
    cell: Tuple[int, ...]


class MapElitesArchive:
    """MAP-Elites archive over (turnover, net_exposure, concentration).

    bins: cells per behaviour dimension. 6 gives 216 cells - enough for
    real behavioural spread, few enough that a few hundred evaluations
    can populate a meaningful fraction of it.
    """

    def __init__(self, bins: int = 6, seed: int = 0):
        self.bins = bins
        self.rng = np.random.default_rng(seed)
        self.archive: Dict[Tuple[int, ...], Elite] = {}
        self.history: List[dict] = []

    def _cell(self, bd: Dict[str, float]) -> Tuple[int, ...]:
        idx = []
        for name in BD_NAMES:
            lo, hi = BD_BOUNDS[name]
            frac = (bd[name] - lo) / (hi - lo) if hi > lo else 0.0
            idx.append(int(np.clip(int(frac * self.bins), 0, self.bins - 1)))
        return tuple(idx)

    def add(self, genome: StrategyGenome, fitness: float, bd: Dict[str, float]) -> bool:
        """Insert if the cell is empty or this genome beats its incumbent.

        The elitism-per-cell rule is the entire mechanism: a globally
        mediocre strategy is retained if nothing else behaves like it.
        """
        if not np.isfinite(fitness):
            return False
        cell = self._cell(bd)
        cur = self.archive.get(cell)
        if cur is None or fitness > cur.fitness:
            self.archive[cell] = Elite(genome=genome, fitness=float(fitness), bd=dict(bd),
                                       cell=cell)
            return True
        return False

    def illuminate(
        self,
        evaluate: Callable[[StrategyGenome], Tuple[float, Dict[str, float]]],
        n_iterations: int = 200,
        n_initial: int = 40,
        mutation_sigma: float = 0.15,
    ) -> "MapElitesArchive":
        """Run the MAP-Elites loop.

        evaluate: genome -> (fitness, behaviour_descriptors)
        """
        for _ in range(n_initial):
            g = StrategyGenome.random(self.rng)
            fit, bd = evaluate(g)
            self.add(g, fit, bd)

        for it in range(n_iterations):
            if not self.archive:
                g = StrategyGenome.random(self.rng)
            else:
                # Uniform selection over CELLS, not over fitness. Biasing
                # selection toward high fitness would re-introduce the
                # convergence that MAP-Elites exists to prevent.
                cells = list(self.archive.keys())
                parent = self.archive[cells[self.rng.integers(len(cells))]]
                g = parent.genome.mutate(self.rng, mutation_sigma)
            fit, bd = evaluate(g)
            self.add(g, fit, bd)
            if (it + 1) % 25 == 0:
                self.history.append({"iteration": it + 1, "coverage": self.coverage,
                                     "best_fitness": self.best_fitness,
                                     "qd_score": self.qd_score})
        return self

    @property
    def coverage(self) -> float:
        """Fraction of behaviour space occupied - the diversity measure."""
        return len(self.archive) / float(self.bins ** len(BD_NAMES))

    @property
    def best_fitness(self) -> float:
        return max((e.fitness for e in self.archive.values()), default=float("-nan"))

    @property
    def qd_score(self) -> float:
        """Sum of elite fitnesses - the standard QD metric.

        Rewards being good AND being everywhere, which is exactly the
        property that distinguishes this from plain evolution.
        """
        return float(sum(max(e.fitness, 0.0) for e in self.archive.values()))

    def elites(self, top_n: Optional[int] = None) -> List[Elite]:
        out = sorted(self.archive.values(), key=lambda e: e.fitness, reverse=True)
        return out[:top_n] if top_n else out

    def ensemble_weights(self, signals: np.ndarray, top_n: int = 8,
                         temperature: float = 1.0,
                         prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Blend the top elites' portfolios - the QD payoff.

        Softmax over fitness with `temperature`, applied to elites drawn
        from DIFFERENT behaviour cells. The result is an average over
        strategies that trade differently from each other, not an average
        over N variants of the same one.
        """
        elites = self.elites(top_n)
        if not elites:
            return np.zeros(np.asarray(signals).shape[-1])

        fits = np.array([e.fitness for e in elites], dtype=np.float64)
        fits = np.nan_to_num(fits, nan=0.0, posinf=0.0, neginf=0.0)
        z = (fits - fits.max()) / max(temperature, 1e-6)
        wts = np.exp(z)
        wts /= wts.sum()

        acc = np.zeros(np.asarray(signals).shape[-1], dtype=np.float64)
        for w, e in zip(wts, elites):
            acc += w * apply_genome(signals, e.genome, prev_weights)
        return acc

    def summary(self) -> dict:
        return {
            "n_elites": len(self.archive),
            "coverage": self.coverage,
            "best_fitness": self.best_fitness,
            "qd_score": self.qd_score,
            "bins": self.bins,
            "n_cells": self.bins ** len(BD_NAMES),
        }
