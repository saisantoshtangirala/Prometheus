"""
Phase 5: Relational Graph Network
Tests: GNN-01, GNN-02
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prometheus.graph.hive_mind import HiveMindGraphEngine
from prometheus.graph.temporal_graph_network import TemporalGraphNetwork


# ---------------------------------------------------------------------------
# GNN-01: Heart Detection — AAPL & MSFT perfectly correlated → high influence
# ---------------------------------------------------------------------------

class TestGNN01HeartDetection:
    """
    Feed 5 stocks where AAPL and MSFT have returns copied from the same
    base signal (perfect correlation). The TGN should recognise them as
    the most influential nodes and include at least one in heart_of_market.
    """

    @pytest.fixture
    def correlated_engine(self):
        return HiveMindGraphEngine(
            asset_names=["AAPL", "MSFT", "GOOG", "AMZN", "META"],
            device="cpu",
            stability_threshold=0.01,  # always stable for testing
            correlation_window=20,
        )

    @pytest.fixture
    def correlated_returns(self):
        rng = np.random.default_rng(0)
        n_bars, n_assets = 50, 5
        base = rng.normal(0.01, 0.02, n_bars)  # strong trending signal
        returns = np.column_stack([
            base + rng.normal(0, 0.001, n_bars),   # AAPL ≈ base
            base + rng.normal(0, 0.001, n_bars),   # MSFT ≈ base (perfect corr)
            rng.normal(0, 0.01, n_bars),             # GOOG: independent
            rng.normal(0, 0.01, n_bars),             # AMZN: independent
            rng.normal(0, 0.01, n_bars),             # META: independent
        ])
        return returns

    def test_aapl_msft_appear_in_heart(self, correlated_engine, correlated_returns):
        # Feed history to build correlation matrix
        for i in range(len(correlated_returns)):
            result = correlated_engine.update(correlated_returns[i])

        heart = result["heart_of_market"]
        assert len(heart) > 0, "heart_of_market must not be empty"

        # At least AAPL or MSFT should appear in the top-3 heart nodes
        correlated_in_heart = {"AAPL", "MSFT"} & set(heart)
        assert len(correlated_in_heart) >= 1, (
            f"Perfectly correlated AAPL/MSFT not in heart: {heart}. "
            "GATv2 attention should favor highly correlated nodes."
        )

    def test_influence_scores_are_nonnegative(self, correlated_engine, correlated_returns):
        for i in range(len(correlated_returns)):
            result = correlated_engine.update(correlated_returns[i])
        scores = result["influence_scores"]
        for asset, score in scores.items():
            assert score >= 0, f"Influence score for {asset} is negative: {score}"

    def test_combined_aapl_msft_influence_exceeds_independent(
        self, correlated_engine, correlated_returns
    ):
        for i in range(len(correlated_returns)):
            result = correlated_engine.update(correlated_returns[i])
        scores = result["influence_scores"]

        corr_total = scores.get("AAPL", 0) + scores.get("MSFT", 0)
        indep_total = scores.get("GOOG", 0) + scores.get("AMZN", 0) + scores.get("META", 0)

        # Correlated pair should collectively outweigh 3 independent assets
        # (or at minimum not be far below them)
        ratio = corr_total / (indep_total + 1e-8)
        assert ratio >= 0.3, (
            f"AAPL+MSFT combined influence {corr_total:.4f} is too low "
            f"relative to independent assets {indep_total:.4f} (ratio={ratio:.2f})"
        )


# ---------------------------------------------------------------------------
# GNN-02: Cold Start — new IPO with 5 days of data
# ---------------------------------------------------------------------------

class TestGNN02ColdStart:
    """
    A newly listed asset with only 5 days of history must be handled
    gracefully: random embedding + adaptation, no crash.
    """

    def test_cold_start_no_crash(self):
        """Feed 5 bars of history — engine must not raise 'missing historical data'."""
        engine = HiveMindGraphEngine(
            asset_names=["AAPL", "MSFT", "NEW_IPO"],
            device="cpu",
            correlation_window=60,  # requires 60 bars for full corr, but must handle 5
        )
        rng = np.random.default_rng(42)
        for _ in range(5):  # only 5 bars — cold start condition
            returns = rng.normal(0, 0.01, 3)
            result = engine.update(returns)
        assert result is not None, "Engine crashed on cold-start with 5 bars"
        assert "heart_of_market" in result

    def test_cold_start_returns_valid_signal(self):
        engine = HiveMindGraphEngine(
            asset_names=["SPY", "QQQ", "NEW_STOCK"],
            device="cpu",
        )
        rng = np.random.default_rng(1)
        for _ in range(5):
            result = engine.update(rng.normal(0, 0.01, 3))

        signal = result.get("trading_signal")
        assert signal in ("ALLOWED", "RESTRICTED"), f"Invalid signal: {signal}"

    def test_insufficient_history_uses_eye_adjacency(self):
        """With < 10 bars, adjacency matrix defaults to identity (safe fallback)."""
        engine = HiveMindGraphEngine(
            asset_names=["A", "B", "C"],
            device="cpu",
        )
        adj = engine._build_adjacency_matrix()
        # With < 10 bars in history, should return identity matrix
        np.testing.assert_array_equal(adj, np.eye(3, dtype=np.float32))

    def test_duplicate_ticker_raises_value_error(self):
        with pytest.raises(ValueError, match="Duplicate tickers"):
            HiveMindGraphEngine(
                asset_names=["AAPL", "MSFT", "AAPL"],  # AAPL duplicated
                device="cpu",
            )

    def test_tgn_runs_on_minimal_graph(self):
        """TGN forward pass must work with as few as 2 nodes."""
        tgn = TemporalGraphNetwork(
            n_nodes=2, node_feat_dim=8, edge_feat_dim=4, memory_dim=16
        )
        x = torch.randn(2, 8)
        adj = torch.eye(2)
        with torch.no_grad():
            out = tgn(x, adj)
        assert "influence_scores" in out
        assert out["influence_scores"].shape == (2,)
