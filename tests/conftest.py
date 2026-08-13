"""
Shared pytest fixtures for Project Prometheus test suite.

All external APIs (yfinance, boto3, SEC EDGAR) are mocked here
so tests run fully offline in a Docker container.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(
    tickers: List[str],
    n_bars: int = 100,
    seed: int = 42,
    include_zero_volume: bool = False,
) -> pd.DataFrame:
    """Synthetic multi-level OHLCV DataFrame (yfinance format)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_bars, freq="B")
    dfs = {}
    for ticker in tickers:
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n_bars)))
        vol = rng.integers(1_000_000, 50_000_000, n_bars).astype(float)
        if include_zero_volume:
            vol[10:13] = 0  # 3 consecutive zero-volume bars at index 10-12
        dfs[ticker] = pd.DataFrame(
            {
                "Open": prices * (1 + rng.uniform(-0.002, 0.002, n_bars)),
                "High": prices * (1 + np.abs(rng.normal(0, 0.005, n_bars))),
                "Low": prices * (1 - np.abs(rng.normal(0, 0.005, n_bars))),
                "Close": prices,
                "Volume": vol,
            },
            index=dates,
        )
    # Produce (field, ticker) MultiIndex — standard yfinance format
    df = pd.concat(dfs, axis=1)  # (ticker, field) initially
    df.columns = df.columns.swaplevel(0, 1)
    df = df.sort_index(axis=1, level=0)
    return df


def make_returns(n_bars: int = 100, n_assets: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, (n_bars, n_assets)).astype(np.float32)


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_returns():
    return make_returns(n_bars=80, n_assets=5)


@pytest.fixture
def sample_ohlcv():
    return make_ohlcv(["SPY", "QQQ", "GLD"], n_bars=100)


@pytest.fixture
def small_engine():
    """Minimal PrometheusEngine for fast unit tests (CPU, d_model=32)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from prometheus.engine import PrometheusEngine, PrometheusConfig

    cfg = PrometheusConfig(
        n_assets=5,
        seq_len=16,
        horizon=3,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        device="cpu",
    )
    return PrometheusEngine(cfg)


@pytest.fixture
def mock_yfinance(sample_ohlcv):
    """Patch yfinance.download to return synthetic data."""
    with patch("yfinance.download", return_value=sample_ohlcv) as mock:
        yield mock


@pytest.fixture
def mock_boto3():
    """Patch boto3 so no AWS calls are made."""
    with patch("boto3.client") as mock_client:
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3
        yield mock_s3


# ---------------------------------------------------------------------------
# Neural module fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def neuromod():
    from prometheus.neuro.neuromodulation import NeuromodulationSystem
    return NeuromodulationSystem(hidden_size=32)


@pytest.fixture
def cortisol():
    from prometheus.neuro.neuromodulation import CortisolSystem
    return CortisolSystem(hidden_size=32, fear_threshold=0.7)


@pytest.fixture
def dopamine():
    from prometheus.neuro.neuromodulation import DopamineSystem
    return DopamineSystem(hidden_size=32)


@pytest.fixture
def ltc_net():
    from prometheus.neuro.ltc_network import LiquidTimeConstantNetwork
    return LiquidTimeConstantNetwork(
        input_size=5, hidden_sizes=[16, 8], output_size=5
    )


@pytest.fixture
def snn_encoder():
    from prometheus.neuro.spiking_network import SpikingMarketEncoder
    return SpikingMarketEncoder(input_size=5, layer_sizes=[16, 8], output_size=5, tau_mem=20.0)


@pytest.fixture
def causal_transformer():
    from prometheus.causal.causal_transformer import CausalTransformer
    return CausalTransformer(
        n_features=5, n_targets=5, horizon=3,
        d_model=32, n_heads=2, n_layers=2,
    )


@pytest.fixture
def dag_engine():
    from prometheus.causal.dag_engine import CausalDAGEngine
    return CausalDAGEngine(max_nodes=50)


@pytest.fixture
def hive_mind():
    from prometheus.graph.hive_mind import HiveMindGraphEngine
    return HiveMindGraphEngine(
        asset_names=["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"],
        device="cpu",
    )


@pytest.fixture
def loss_fn():
    from prometheus.loss.asymmetric_loss import AsymmetricUtilityLoss
    return AsymmetricUtilityLoss(alpha=0.5, beta=2.0, gamma=3.0)


@pytest.fixture
def kelly():
    from prometheus.loss.kelly_optimizer import KellyCriterionOptimizer
    return KellyCriterionOptimizer(n_assets=5, kelly_fraction=0.5, max_position=0.25)


@pytest.fixture
def maml_learner(causal_transformer):
    from prometheus.meta.maml_engine import MAMLMetaLearner
    return MAMLMetaLearner(model=causal_transformer, inner_lr=0.01, n_inner_steps=3)


@pytest.fixture
def neat_evolver():
    from prometheus.meta.neat_evolver import NEATArchitectureEvolver
    return NEATArchitectureEvolver(
        input_dim=5, output_dim=5,
        population_size=10, n_generations=1, mutation_rate=0.3,
    )


@pytest.fixture
def diffusion_sim():
    from prometheus.generative.diffusion_simulator import MarketDiffusionSimulator
    return MarketDiffusionSimulator(n_assets=5, seq_len=20, n_diffusion_steps=10)


@pytest.fixture
def black_swan_gen():
    from prometheus.generative.black_swan_generator import BlackSwanGenerator
    from prometheus.generative.diffusion_simulator import MarketDiffusionSimulator
    sim = MarketDiffusionSimulator(n_assets=5, seq_len=20, n_diffusion_steps=10)
    return BlackSwanGenerator(simulator=sim, n_assets=5)
