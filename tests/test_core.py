"""
Core unit tests for Prometheus subsystems.
Run: pytest tests/ -v
"""

import numpy as np
import pytest
import torch


# ── Causal subsystem ──────────────────────────────────────────────────────────

class TestCausalDAGEngine:
    def setup_method(self):
        from prometheus.causal import CausalDAGEngine
        self.dag = CausalDAGEngine()

    def test_dag_is_acyclic(self):
        import networkx as nx
        assert nx.is_directed_acyclic_graph(self.dag.dag)

    def test_do_intervention_returns_dict(self):
        result = self.dag.do_intervention("FED_FUNDS_RATE", 0.5, "SPX")
        assert "causal_effect" in result
        assert "method" in result
        assert isinstance(result["causal_effect"], float)

    def test_counterfactual(self):
        result = self.dag.counterfactual("FED_FUNDS_RATE", 0.0, 0.5, "GOLD")
        assert "delta" in result
        assert "direction" in result

    def test_causal_attribution(self):
        attributions = self.dag.causal_attribution("SPX", -0.05, top_k=5)
        assert len(attributions) <= 5
        assert all("lever" in a for a in attributions)
        assert all("attributed_effect" in a for a in attributions)

    def test_graph_summary(self):
        summary = self.dag.get_graph_summary()
        assert summary["nodes"] > 0
        assert summary["edges"] > 0
        assert summary["is_dag"] is True


# ── Causal Transformer ────────────────────────────────────────────────────────

class TestCausalTransformer:
    def setup_method(self):
        from prometheus.causal import CausalTransformer
        self.model = CausalTransformer(
            n_features=5, n_targets=5, horizon=3,
            d_model=32, n_heads=4, n_layers=2, d_ff=64,
        )

    def test_forward_shape(self):
        x = torch.randn(2, 10, 5)
        out = self.model(x)
        assert "predictions" in out
        assert out["predictions"].shape == (2, 3, 5)

    def test_attribution_shape(self):
        x = torch.randn(2, 10, 5)
        out = self.model(x, return_attributions=True)
        assert "causal_attributions" in out
        assert out["causal_attributions"].shape == (2, 5)

    def test_attributions_sum_to_one(self):
        x = torch.randn(1, 10, 5)
        out = self.model(x, return_attributions=True)
        attn_sum = out["causal_attributions"].sum(dim=-1)
        assert torch.allclose(attn_sum, torch.ones(1), atol=1e-5)


# ── LTC Network ───────────────────────────────────────────────────────────────

class TestLTCNetwork:
    def setup_method(self):
        from prometheus.neuro import LiquidTimeConstantNetwork
        self.ltc = LiquidTimeConstantNetwork(
            input_size=5, hidden_sizes=[16, 8], output_size=5
        )

    def test_forward_shape(self):
        x = torch.randn(2, 20, 5)
        out, hidden, meta = self.ltc(x)
        assert out.shape == (2, 20, 5)
        assert len(hidden) == 2

    def test_regime_detection(self):
        x = torch.randn(1, 30, 5)
        regime = self.ltc.get_regime(x)
        assert regime in ["HIGH_VOLATILITY", "TRENDING", "MEAN_REVERTING"]

    def test_tau_in_valid_range(self):
        x = torch.randn(1, 10, 5)
        _, _, meta = self.ltc(x)
        assert 0.0 <= meta["mean_tau"] <= 1.0


# ── Spiking Neural Network ────────────────────────────────────────────────────

class TestSpikingMarketEncoder:
    def setup_method(self):
        from prometheus.neuro import SpikingMarketEncoder
        self.snn = SpikingMarketEncoder(
            input_size=5, layer_sizes=[16, 8], output_size=4, n_timesteps=20
        )

    def test_forward_output_shape(self):
        x = torch.randn(2, 20, 5)
        out, meta = self.snn(x)
        assert out.shape == (2, 4)

    def test_meta_keys(self):
        x = torch.randn(1, 20, 5)
        _, meta = self.snn(x)
        assert "firing_rate" in meta
        assert "synchrony" in meta
        assert "stress_signal" in meta

    def test_firing_rate_in_range(self):
        x = torch.randn(2, 20, 5)
        _, meta = self.snn(x)
        assert 0.0 <= meta["firing_rate"] <= 1.0


# ── Neuromodulation ───────────────────────────────────────────────────────────

class TestNeuromodulationSystem:
    def setup_method(self):
        from prometheus.neuro import NeuromodulationSystem
        self.neuromod = NeuromodulationSystem()

    def test_step_returns_dict(self):
        result = self.neuromod.step(
            predicted_return=0.01,
            actual_return=0.02,
            market_entropy=0.3,
            drawdown=0.05,
        )
        assert "dopamine" in result
        assert "cortisol" in result
        assert "position_multiplier" in result

    def test_fear_mode_on_high_entropy(self):
        # Simulate stress: high entropy, large drawdown
        result = None
        for _ in range(10):
            result = self.neuromod.step(0.01, -0.05, 0.95, 0.30)
        # After repeated stress, cortisol should be elevated
        assert result["cortisol"] > 0.2

    def test_position_multiplier_in_range(self):
        result = self.neuromod.step(0.0, 0.0, 0.5, 0.1)
        assert 0.0 <= result["position_multiplier"] <= 3.0


# ── HTM ───────────────────────────────────────────────────────────────────────

class TestHTM:
    def setup_method(self):
        from prometheus.neuro import HierarchicalTemporalMemory
        self.htm = HierarchicalTemporalMemory(n_input_features=3, n_columns=64, n_active_cols=5)

    def test_forward_step_returns_anomaly(self):
        vec = np.random.randn(3).astype(np.float32)
        result = self.htm.forward_step(vec, learn=True)
        assert "anomaly_score" in result
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_sequence_processing(self):
        seq = np.random.randn(20, 3).astype(np.float32)
        result = self.htm.process_sequence(seq, learn=True)
        assert "mean_anomaly" in result
        assert len(result["anomaly_scores"]) == 20


# ── Asymmetric Loss ───────────────────────────────────────────────────────────

class TestAsymmetricUtilityLoss:
    def setup_method(self):
        from prometheus.loss import AsymmetricUtilityLoss
        self.loss_fn = AsymmetricUtilityLoss()

    def test_zero_loss_on_bullish_error(self):
        """If we predict $100 and it goes to $101, loss should be lower than directional error."""
        pred = torch.tensor([[0.05]])  # predicted +5%
        target = torch.tensor([[0.06]])  # actual +6% (we underestimated upside)
        loss_mild = self.loss_fn(pred, target)

        pred_wrong = torch.tensor([[0.05]])  # predicted +5%
        target_wrong = torch.tensor([[-0.05]])  # actual -5% (directional error)
        loss_severe = self.loss_fn(pred_wrong, target_wrong)

        assert loss_severe > loss_mild

    def test_directional_error_penalized_more(self):
        pred_correct = torch.tensor([[0.01, -0.01]])
        pred_wrong = torch.tensor([[0.01, -0.01]])
        target = torch.tensor([[0.02, 0.02]])  # both positive actual
        loss_wrong = self.loss_fn(pred_wrong, target)

        pred_same_dir = torch.tensor([[0.01, 0.01]])
        loss_right = self.loss_fn(pred_same_dir, target)

        assert loss_wrong > loss_right


# ── Kelly Optimizer ───────────────────────────────────────────────────────────

class TestKellyCriterionOptimizer:
    def setup_method(self):
        from prometheus.loss import KellyCriterionOptimizer
        self.kelly = KellyCriterionOptimizer(n_assets=4, kelly_fraction=0.5, max_position=0.20)

    def test_returns_valid_fractions(self):
        predictions = np.array([0.02, -0.01, 0.03, -0.02])
        confidence = np.array([0.8, 0.7, 0.9, 0.6])
        result = self.kelly.compute_kelly_fractions(predictions, confidence)
        fractions = np.array(result["kelly_fractions"])
        assert len(fractions) == 4
        assert np.all(np.abs(fractions) <= 0.20 + 1e-6)

    def test_position_direction_matches_prediction(self):
        predictions = np.array([0.05, -0.05])
        confidence = np.array([0.9, 0.9])
        self.kelly = __import__("prometheus.loss", fromlist=["KellyCriterionOptimizer"]).KellyCriterionOptimizer(n_assets=2)
        result = self.kelly.compute_kelly_fractions(predictions, confidence)
        fractions = result["kelly_fractions"]
        assert fractions[0] > 0  # long for positive prediction
        assert fractions[1] < 0  # short for negative prediction


# ── Graph Engine ──────────────────────────────────────────────────────────────

class TestHiveMindGraphEngine:
    def setup_method(self):
        from prometheus.graph import HiveMindGraphEngine
        self.engine = HiveMindGraphEngine(
            asset_names=["SPY", "QQQ", "GLD", "TLT"],
            node_feat_dim=32,
            device="cpu",
        )

    def test_update_returns_dict(self):
        returns = np.random.randn(4) * 0.01
        result = self.engine.update(returns)
        assert "heart_of_market" in result
        assert "market_stability" in result
        assert "trading_signal" in result

    def test_trading_permission(self):
        returns = np.random.randn(4) * 0.01
        self.engine.update(returns)
        allowed, reason = self.engine.get_trading_permission("SPY")
        assert isinstance(allowed, bool)
        assert isinstance(reason, str)


# ── Probability Volcano ───────────────────────────────────────────────────────

class TestProbabilityVolcano:
    def setup_method(self):
        from prometheus.output import ProbabilityVolcano
        self.volcano = ProbabilityVolcano()

    def test_compute_distribution(self):
        paths = np.random.randn(100, 5, 3) * 0.01
        dist = self.volcano.compute_distribution(paths)
        assert "percentile_surface" in dist
        assert "var_95" in dist
        assert "p_positive" in dist

    def test_summary_statistics(self):
        paths = np.random.randn(50, 5, 3) * 0.01
        stats = self.volcano.summary_statistics(paths, ["A", "B", "C"])
        assert len(stats) == 3
        assert all("expected_return" in s for s in stats)
        assert all("var_95" in s for s in stats)


# ── Integration test ──────────────────────────────────────────────────────────

class TestPrometheusIntegration:
    def setup_method(self):
        from prometheus.engine import PrometheusEngine, PrometheusConfig
        cfg = PrometheusConfig(
            n_assets=4,
            seq_len=16,
            horizon=2,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
        )
        self.engine = PrometheusEngine(cfg)

    def test_forward_pass(self):
        x = torch.randn(1, 16, 4)
        result = self.engine.forward(x, return_monte_carlo=False)
        assert "predictions" in result
        assert result["predictions"].shape == (1, 2, 4)
        assert "regime" in result

    def test_analyze_returns_report(self):
        data = np.random.randn(16, 4) * 0.01
        report = self.engine.analyze(
            market_data=data,
            asset_names=["A", "B", "C", "D"],
        )
        assert "primary_recommendation" in report
        assert "trade_recommendations" in report
        assert "system_state" in report
        assert "formatted_text" in report

    def test_adapt_to_regime(self):
        data = np.random.randn(20, 4) * 0.01
        result = self.engine.adapt_to_regime(data)
        assert result["status"] == "adapted"
        assert result["steps"] == 3
