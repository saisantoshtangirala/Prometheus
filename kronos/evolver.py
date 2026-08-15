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
    """
    Fitness-weighted prediction-space average of top-k NEAT variants.

    NEA-03: extreme fitness values (|w| > 1e3) are softmax-scaled after
    normalizing by the largest magnitude, so no single variant can dominate
    the ensemble through a runaway fitness score, and every stored weight
    stays in [0, 1] summing to 1.
    """

    DOMINANCE_THRESHOLD = 1e3

    def __init__(self, models: List[nn.Module], weights: List[float]):
        super().__init__()
        if len(models) != len(weights) or not models:
            raise ValueError("models and weights must be same non-zero length")
        self.members = nn.ModuleList(models)
        w = torch.tensor(weights, dtype=torch.float32)
        w = torch.nan_to_num(w, nan=0.0, posinf=self.DOMINANCE_THRESHOLD,
                             neginf=-self.DOMINANCE_THRESHOLD)
        if w.abs().max() > self.DOMINANCE_THRESHOLD:
            w = torch.softmax(w / w.abs().max(), dim=0)
        else:
            # Shift so the minimum fitness contributes epsilon, then normalize.
            w = w - w.min() + 1e-6
            w = w / w.sum()
        self.register_buffer("weights", w)

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

    def evolve(
        self,
        buffer,
        degraded: bool = False,
        time_budget_seconds: Optional[float] = None,
        resume_population: Optional[list] = None,
        resume_generation: int = 0,
        on_generation: Optional[callable] = None,
    ) -> EvolutionResult:
        """
        Run one nightly evolution cycle against the nightmare buffer.

        Generation-at-a-time loop enables:
          - NEA-04: a wall-clock budget that stops early (fewer generations)
            rather than blowing through the 06:00 deadline
          - ORC-07/E2E-02: checkpoint/resume at generation granularity via
            resume_population/resume_generation and the on_generation hook
        """
        import time as _time
        evolver = self._make_evolver(degraded)
        target_gens = evolver.n_generations
        evolver.n_generations = 1          # we drive the loop ourselves

        if resume_population:
            evolver.population = resume_population
            logger.info(
                "[evolution] resuming from generation %d (population=%d)",
                resume_generation, len(resume_population),
            )
        else:
            evolver.initialize_population()

        val_data = self._nightmare_val_data(buffer)
        loss_fn = nn.functional.mse_loss
        start = _time.monotonic()
        completed = resume_generation

        for gen in range(resume_generation, target_gens):
            if time_budget_seconds is not None and \
                    _time.monotonic() - start > time_budget_seconds:
                logger.warning(
                    "[evolution] time budget %.0fs exhausted at generation "
                    "%d/%d - degrading gracefully", time_budget_seconds,
                    gen, target_gens,
                )
                break
            evolver.evolve(val_data, loss_fn)
            completed = gen + 1
            if on_generation is not None:
                on_generation(completed, evolver.population)

        # Final fitness pass so the ranking reflects the LAST generation
        for genome in evolver.population:
            genome.fitness = evolver.evaluate_fitness(genome, val_data, loss_fn)

        self._replace_broken_variants(evolver, val_data, loss_fn)
        self._break_stagnation(evolver, val_data, loss_fn)

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
            "[evolution] population=%d degraded=%s generations=%d top_fitness=%s",
            result.population_size, degraded, completed,
            [f"{f:.3f}" for f in result.top_fitness],
        )
        return result

    # -- population health (NEA-01, NEA-02) ---------------------------------

    def _replace_broken_variants(self, evolver, val_data, loss_fn) -> None:
        """
        NEA-02: a variant whose fitness is non-finite (inf/nan loss during
        evaluation) is discarded and replaced with a mutated copy of the
        current best genome - the generation survives.
        """
        finite = [g for g in evolver.population if np.isfinite(g.fitness)]
        broken = [g for g in evolver.population if not np.isfinite(g.fitness)]
        if not broken:
            return
        if not finite:
            evolver.initialize_population()
            for g in evolver.population:
                g.fitness = evolver.evaluate_fitness(g, val_data, loss_fn)
            return
        best = max(finite, key=lambda g: g.fitness)
        logger.warning(
            "[evolution] replacing %d broken variant(s) with copies of "
            "genome %d", len(broken), best.genome_id,
        )
        for g in broken:
            replacement = evolver._mutate(copy.deepcopy(best))
            replacement.fitness = evolver.evaluate_fitness(
                replacement, val_data, loss_fn
            )
            idx = evolver.population.index(g)
            evolver.population[idx] = replacement

    def _break_stagnation(self, evolver, val_data, loss_fn) -> None:
        """
        NEA-01: if every variant has an identical fitness score (total tie),
        mutate 5 of them to reintroduce diversity and re-evaluate. Prevents
        the selection loop from spinning on a degenerate population.
        """
        fitnesses = [g.fitness for g in evolver.population]
        if len(set(np.round(fitnesses, 12))) > 1:
            return
        logger.warning(
            "[evolution] population stagnation (all fitness=%.6f) - "
            "mutating 5 variants to break the tie", fitnesses[0],
        )
        n_shake = min(5, len(evolver.population))
        for i in range(n_shake):
            mutated = evolver._mutate(copy.deepcopy(evolver.population[i]))
            mutated.fitness = evolver.evaluate_fitness(mutated, val_data, loss_fn)
            evolver.population[i] = mutated

    def spawn_variants(self, degraded: bool = False) -> List[NetworkGenome]:
        """Spawn (but do not evaluate) the nightly variant population."""
        evolver = self._make_evolver(degraded)
        evolver.initialize_population()
        return evolver.population
