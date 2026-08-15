"""
Project Kronos - the self-evolving daily lifecycle layer on top of Prometheus.

Kronos does not reimplement anything Prometheus already does. It imports
the causal engine, diffusion simulator, NEAT evolver, MAML learner, SNN
encoder, Kalman filter, and sentiment analyzer, and drives them through a
365-day paper-trading loop:

    digestion -> nightmare -> evolution -> adaptation -> report
        -> reflex (market hours) -> logging -> repeat
"""

from kronos.config import KronosConfig, load_config
from kronos.data_pipeline import DailyMemory, DataPipeline
from kronos.evolver import EvolutionResult, KronosEvolver, WeightedEnsemble
from kronos.nightmare_generator import NightmareBuffer, NightmareGenerator
from kronos.orchestrator import DayState, KronosOrchestrator, Phase
from kronos.paper_trader import PaperTrader
from kronos.reflex import OrderBookSimulator, ReflexArc, RegimeSwitchGate
from kronos.reporter import GodsEyeReporter
from kronos.warmer import KronosWarmer, WarmupResult

__all__ = [
    "KronosConfig", "load_config",
    "DailyMemory", "DataPipeline",
    "NightmareBuffer", "NightmareGenerator",
    "EvolutionResult", "KronosEvolver", "WeightedEnsemble",
    "KronosWarmer", "WarmupResult",
    "ReflexArc", "RegimeSwitchGate", "OrderBookSimulator",
    "PaperTrader",
    "GodsEyeReporter",
    "KronosOrchestrator", "DayState", "Phase",
]
