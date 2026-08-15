"""
Kronos NEAT Evolution - phase 3 of the daily cycle (04:00 - 05:00).

Spawns architectural variants via the existing NEATArchitectureEvolver,
evaluates every variant against the nightly NightmareBuffer, and combines
the top-k survivors into a fitness-weighted ensemble - the day's new
"Master Model".

Note on "weighted average": distinct NEAT genomes decode to distinct
topologies, so parameter-space averaging is undefined. The weighted average
is therefore taken in PREDICTION space: the master model is an ensemble
whose output is the fitness-weighted mean of the top-k variants' outputs.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from prometheus.meta.neat_evolver import (
    GenomeDecoder,
    NEATArchitectureEvolver,
    NetworkGenome,
)

logger = logging.getLogger(__name__)


class WeightedEnsemble(nn.Module):
    """Fitness-weighted prediction-space average of top-k NEAT variants."""

    def __init__(self, models: List[nn.Module], weights: List[float]):
        super().__init__()
        if len(models) != len(weights) or not models:
            raise ValueError("models and weights must be same non-zero length")
        self.members = nn.ModuleList(models)
        w = torch.tensor(weights, dtype=torch.float32)
        # Shift so the minimum fitness contributes epsilon, then normalize.
        w = w - w.min() + 1e-6
        self.register_buffer("weights", w / w.sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([m(x) for m in self.members], dim=0)  # [k, ...]
        w = self.weights.view(-1, *([1] * (outputs.dim() - 1)))
        return (outputs * w).sum(dim=0)


@dataclass
class EvolutionResult:
    master_model: nn.Module
    top_genomes: List[NetworkGenome]
    top_fitness: List[float]
    population_size: int
    degraded: bool = False
    history: List[dict] = field(default_factory=list)


class KronosEvolver:
    """
    Nightly architecture search hardened against the nightmare buffer.

    Graceful degradation (non-negotiable #2): if evolve() is invoked with
    degraded=True (e.g. the AWS spot instance was preempted and we are on
    the fallback machine), population and generations shrink per config.
    """

    def __init__(self, config):
        self.cfg = config
        n_assets = len(config.data.tickers)
        horizon = config.nightmare.horizon_days
        self.input_dim = n_assets * horizon
        self.output_dim = n_assets

    def _make_evolver(self, degraded: bool) -> NEATArchitectureEvolver:
        evo_cfg = self.cfg.evolution
        pop = evo_cfg.fallback_population_size if degraded else evo_cfg.population_size
        gens = evo_cfg.fallback_generations if degraded else evo_cfg.n_generations
        return NEATArchitectureEvolver(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            population_size=pop,
            n_generations=gens,
            mutation_rate=evo_cfg.mutation_rate,
            crossover_rate=evo_cfg.crossover_rate,
            elitism=evo_cfg.elitism,
        )

    # -- fitness against nightmares -----------------------------------------

    def _nightmare_val_data(
        self, buffer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the NightmareBuffer into (X, y) validation tensors.

        X: flattened future path per scenario  [N, horizon * n_assets]
        y: next-step returns (last bar)        [N, n_assets]

        A variant scores well only if its directional calls survive the
        adversarial futures - exactly the gauntlet Kronos wants.
        """
        futures = buffer.futures                         # [N, T, A]
        X = futures.reshape(futures.shape[0], -1)
        y = futures[:, -1, :]                            # final-bar returns
        return X.float(), y.float()

    # -- main entry ---------------------------------------------------------

    def evolve(self, buffer, degraded: bool = False) -> EvolutionResult:
        """Run one nightly evolution cycle against the nightmare buffer."""
        evolver = self._make_evolver(degraded)
        evolver.initialize_population()
        val_data = self._nightmare_val_data(buffer)
        loss_fn = nn.functional.mse_loss

        # Evaluate + evolve. NEATArchitectureEvolver.evolve() handles the loop.
        evolver.evolve(val_data, loss_fn)

        # Final fitness pass so the ranking reflects the LAST generation
        for genome in evolver.population:
            genome.fitness = evolver.evaluate_fitness(genome, val_data, loss_fn)
        ranked = sorted(evolver.population, key=lambda g: g.fitness, reverse=True)

        top_k = int(self.cfg.evolution.top_k)
        top = [copy.deepcopy(g) for g in ranked[:top_k]]
        models = [
            GenomeDecoder.decode(g, self.input_dim, self.output_dim) for g in top
        ]
        master = WeightedEnsemble(models, [g.fitness for g in top])

        result = EvolutionResult(
            master_model=master,
            top_genomes=top,
            top_fitness=[float(g.fitness) for g in top],
            population_size=evolver.population_size,
            degraded=degraded,
            history=list(evolver.history),
        )
        logger.info(
            "[evolution] population=%d degraded=%s top_fitness=%s",
            result.population_size, degraded,
            [f"{f:.3f}" for f in result.top_fitness],
        )
        return result

    def spawn_variants(self, degraded: bool = False) -> List[NetworkGenome]:
        """Spawn (but do not evaluate) the nightly variant population."""
        evolver = self._make_evolver(degraded)
        evolver.initialize_population()
        return evolver.population
