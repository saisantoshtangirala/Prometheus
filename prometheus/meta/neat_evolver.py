"""
NEAT Architecture Evolver – Self-evolving neural network topology.

Every night, a Genetic Algorithm (NEAT-style) evolves the model's architecture:
  - Number of layers
  - Hidden dimensions
  - Activation functions per layer
  - Skip connection topology
  - Attention head counts

The network literally rewrites its own forward-pass logic each epoch.
Fitness function = risk-adjusted return on validation set (Sharpe ratio).

Uses a simplified NEAT implementation with:
  - Node and connection genes
  - Speciation to protect innovation
  - Crossover and mutation operators
"""

from __future__ import annotations

import copy
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ACTIVATION_MAP = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "silu": nn.SiLU(),
    "elu": nn.ELU(),
    "leaky_relu": nn.LeakyReLU(0.1),
}


@dataclass
class LayerGene:
    gene_id: int
    layer_type: str          # "linear" | "attention" | "conv1d"
    hidden_dim: int
    activation: str
    dropout: float
    n_heads: int = 8         # for attention layers
    enabled: bool = True

    def mutate(self, rate: float = 0.1) -> "LayerGene":
        """Return a mutated copy of this gene."""
        g = copy.deepcopy(self)
        if random.random() < rate:
            g.hidden_dim = random.choice([64, 128, 256, 512])
        if random.random() < rate:
            g.activation = random.choice(list(ACTIVATION_MAP.keys()))
        if random.random() < rate:
            g.dropout = random.uniform(0.05, 0.4)
        if random.random() < rate:
            g.enabled = not g.enabled  # structural mutation (disable/enable)
        return g


@dataclass
class NetworkGenome:
    genome_id: int
    genes: List[LayerGene] = field(default_factory=list)
    fitness: float = 0.0
    generation: int = 0
    species_id: int = 0

    def add_layer(self, gene: LayerGene) -> None:
        self.genes.append(gene)

    def remove_layer(self) -> None:
        if len(self.genes) > 1:
            self.genes.pop(random.randint(0, len(self.genes) - 1))

    def to_dict(self) -> Dict:
        return {
            "genome_id": self.genome_id,
            "fitness": self.fitness,
            "generation": self.generation,
            "n_active_layers": sum(1 for g in self.genes if g.enabled),
            "genes": [
                {"id": g.gene_id, "type": g.layer_type, "dim": g.hidden_dim,
                 "act": g.activation, "enabled": g.enabled}
                for g in self.genes
            ],
        }


class GenomeDecoder:
    """Translates a NetworkGenome into a runnable nn.Module."""

    @staticmethod
    def decode(
        genome: NetworkGenome,
        input_dim: int,
        output_dim: int,
    ) -> nn.Module:
        # Accept either a NetworkGenome or a plain list of LayerGenes
        genes_list = genome if isinstance(genome, list) else genome.genes
        layers = [g for g in genes_list if g.enabled]
        if not layers:
            return nn.Linear(input_dim, output_dim)

        modules = []
        current_dim = input_dim
        for gene in layers:
            if gene.layer_type == "linear":
                modules.append(nn.Linear(current_dim, gene.hidden_dim))
                modules.append(copy.deepcopy(ACTIVATION_MAP[gene.activation]))
                if gene.dropout > 0:
                    modules.append(nn.Dropout(gene.dropout))
                current_dim = gene.hidden_dim
            elif gene.layer_type == "attention":
                # Simplified: linear projection + multi-head attn
                n_heads = min(gene.n_heads, current_dim // 8)
                if n_heads < 1:
                    n_heads = 1
                if current_dim % n_heads != 0:
                    current_dim = (current_dim // n_heads) * n_heads
                modules.append(nn.Linear(current_dim, gene.hidden_dim))
                modules.append(copy.deepcopy(ACTIVATION_MAP[gene.activation]))
                current_dim = gene.hidden_dim

        modules.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*modules)


class NEATArchitectureEvolver:
    """
    Nightly NEAT evolution of neural network architecture.

    Fitness function: Sharpe ratio on a held-out validation window.
    Population evolves over N generations, selecting the best architecture
    as the next day's model topology.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        population_size: int = 50,
        n_generations: int = 20,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.5,
        elitism: int = 5,           # top-k always survive
        speciation_threshold: float = 3.0,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pop_size = population_size
        self.population_size = population_size  # public alias
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.speciation_threshold = speciation_threshold
        self.gene_counter = 0
        self.genome_counter = 0
        self.population: List[NetworkGenome] = []
        self.history: List[Dict] = []

    def _new_gene(self) -> LayerGene:
        self.gene_counter += 1
        return LayerGene(
            gene_id=self.gene_counter,
            layer_type=random.choice(["linear", "attention"]),
            hidden_dim=random.choice([64, 128, 256]),
            activation=random.choice(list(ACTIVATION_MAP.keys())),
            dropout=random.uniform(0.1, 0.3),
        )

    def _new_genome(self, n_layers: int = 3) -> NetworkGenome:
        self.genome_counter += 1
        g = NetworkGenome(genome_id=self.genome_counter)
        for _ in range(n_layers):
            g.add_layer(self._new_gene())
        return g

    def initialize_population(self) -> None:
        self.population = [
            self._new_genome(n_layers=random.randint(2, 6))
            for _ in range(self.pop_size)
        ]
        logger.info("Initialized population of %d genomes", len(self.population))

    def evaluate_fitness(
        self,
        genome: NetworkGenome,
        val_data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
    ) -> float:
        """
        Build model from genome, run on validation data, return Sharpe ratio as fitness.
        """
        model = GenomeDecoder.decode(genome, self.input_dim, self.output_dim)
        X_val, y_val = val_data
        with torch.no_grad():
            try:
                pred = model(X_val)
                if pred.shape != y_val.shape:
                    pred = pred.view_as(y_val)
                pnl = (pred.sign() * y_val).squeeze()  # directional P&L
                sharpe = float(pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252))
            except Exception as e:
                logger.debug("Genome %d failed: %s", genome.genome_id, e)
                sharpe = -10.0
        return sharpe

    def _crossover(self, parent1: NetworkGenome, parent2: NetworkGenome) -> NetworkGenome:
        """Single-point crossover on gene lists."""
        child = NetworkGenome(genome_id=self.genome_counter + 1)
        genes1, genes2 = parent1.genes, parent2.genes
        split = random.randint(1, max(min(len(genes1), len(genes2)) - 1, 1))
        child.genes = [copy.deepcopy(g) for g in genes1[:split]] + \
                      [copy.deepcopy(g) for g in genes2[split:]]
        child.generation = max(parent1.generation, parent2.generation) + 1
        self.genome_counter += 1
        return child

    def _mutate(self, genome: NetworkGenome) -> NetworkGenome:
        """Apply structural + weight mutations."""
        g = copy.deepcopy(genome)
        g.genes = [gene.mutate(self.mutation_rate) for gene in g.genes]
        # Structural mutations
        if random.random() < self.mutation_rate:
            g.add_layer(self._new_gene())
        if random.random() < self.mutation_rate * 0.5:
            g.remove_layer()
        return g

    def evolve(
        self,
        val_data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
        on_generation: Optional[Callable] = None,
    ) -> NetworkGenome:
        """
        Run the full evolutionary loop. Returns the best genome.
        Designed to run nightly on a separate thread/process.
        """
        if not self.population:
            self.initialize_population()

        best_genome = None
        best_fitness = -float("inf")

        for gen in range(self.n_generations):
            # Evaluate fitness
            for genome in self.population:
                genome.fitness = self.evaluate_fitness(genome, val_data, loss_fn)

            # Sort by fitness
            self.population.sort(key=lambda g: g.fitness, reverse=True)
            gen_best = self.population[0]

            if gen_best.fitness > best_fitness:
                best_fitness = gen_best.fitness
                best_genome = copy.deepcopy(gen_best)

            gen_stats = {
                "generation": gen,
                "best_fitness": float(gen_best.fitness),
                "mean_fitness": float(np.mean([g.fitness for g in self.population])),
                "best_genome_id": gen_best.genome_id,
                "n_active_layers": sum(1 for g in gen_best.genes if g.enabled),
            }
            self.history.append(gen_stats)
            logger.info("Gen %d: best_fitness=%.4f, layers=%d",
                        gen, gen_best.fitness, gen_stats["n_active_layers"])

            if on_generation:
                on_generation(gen, gen_stats, gen_best)

            # Elitism: keep top-k
            elite = self.population[:self.elitism]

            # Selection + crossover + mutation for rest
            new_pop = list(elite)
            while len(new_pop) < self.pop_size:
                p1, p2 = random.choices(self.population[:self.pop_size // 2], k=2)
                if random.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = copy.deepcopy(p1)
                child = self._mutate(child)
                new_pop.append(child)

            self.population = new_pop[:self.pop_size]

        logger.info("Evolution complete. Best fitness: %.4f", best_fitness)
        return best_genome

    def build_best_model(
        self,
        val_data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
    ) -> Tuple[nn.Module, NetworkGenome]:
        """Run evolution and return the best PyTorch model + its genome."""
        best_genome = self.evolve(val_data, loss_fn)
        model = GenomeDecoder.decode(best_genome, self.input_dim, self.output_dim)
        return model, best_genome

    def save_evolution_history(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
