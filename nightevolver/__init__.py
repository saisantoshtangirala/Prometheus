"""
NightEvolver-Hybrid: genetic-algorithm strategy evolution with the
multiple-testing controls a GA search requires.

ONE package, not two (nightevolver_runpod/ + a kronos/ copy), and that
is deliberate. Genome encode/decode, indicator computation and the
position rules MUST be byte-identical between the machine that trains
and the machine that trades. Two copies of that logic is precisely the
"backtest tested the wrong model" failure this redesign exists to fix,
so training and execution import the same modules and there is nothing
to drift.

  data_loader.py        NSE OHLCV -> 20 causal indicators, ~[-1,1]
  genome.py             strategy DNA (65 genes) + decode + operators
  ga_engine.py          tournament GA, vectorised sim, noise benchmark
  rl_trainer.py         optional tabular Q-learning
  saver.py              verifiable JSON checkpoint + deflated-Sharpe gate
  strategy_decoder.py   Hetzner-side live signal generation
  backtest_evolved.py   125-window walk-forward, GA re-run per window

RunPod runs train_nightevolver.py; Hetzner runs strategy_decoder against
the checkpoint it produced. See README_NIGHTEVOLVER.md.
"""

from nightevolver.data_loader import MarketData, build_market_data, fetch_nse_data
from nightevolver.ga_engine import (
    GAConfig, GeneticEvolver, EvolutionResult, simulate, fitness,
    expected_max_sharpe_from_noise,
)
from nightevolver.genome import (
    DecodedStrategy, GENOME_LENGTH, INDICATOR_NAMES, decode, random_genome,
)
from nightevolver.saver import load_checkpoint, save_checkpoint
from nightevolver.strategy_decoder import EvolvedStrategy, LiveSignal

__all__ = [
    "MarketData", "build_market_data", "fetch_nse_data",
    "GAConfig", "GeneticEvolver", "EvolutionResult", "simulate", "fitness",
    "expected_max_sharpe_from_noise",
    "DecodedStrategy", "GENOME_LENGTH", "INDICATOR_NAMES", "decode", "random_genome",
    "load_checkpoint", "save_checkpoint",
    "EvolvedStrategy", "LiveSignal",
]
