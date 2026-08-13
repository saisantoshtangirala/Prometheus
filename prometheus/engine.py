"""
Prometheus Engine – Master orchestrator for all subsystems.

Wires together:
  CausalDAGEngine → CausalTransformer
  LTCNetwork + SNN → temporal encoding
  NeuromodulationSystem → position sizing
  HiveMindGraphEngine → market structure
  MarketDiffusionSimulator + BlackSwanGenerator → synthetic training data
  MAMLMetaLearner → fast regime adaptation
  NEATArchitectureEvolver → nightly self-evolution
  AsymmetricUtilityLoss + KellyCriterionOptimizer → decision layer
  ProbabilityVolcano + GodsEyeReportGenerator → output
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .causal import CausalDAGEngine, CausalTransformer
from .data import MarketDataFetcher, SentimentAnalyzer, OrderBookSimulator
from .generative import MarketDiffusionSimulator, BlackSwanGenerator, ScenarioLibrary
from .graph import HiveMindGraphEngine
from .loss import AsymmetricUtilityLoss, KellyCriterionOptimizer
from .meta import MAMLMetaLearner, NEATArchitectureEvolver
from .neuro import (
    LiquidTimeConstantNetwork,
    SpikingMarketEncoder,
    NeuromodulationSystem,
    HierarchicalTemporalMemory,
)
from .output import ProbabilityVolcano, GodsEyeReportGenerator

logger = logging.getLogger(__name__)


class PrometheusConfig:
    """Central configuration for the Prometheus engine."""

    def __init__(
        self,
        n_assets: int = 20,
        seq_len: int = 128,
        horizon: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        ltc_hidden: List[int] = None,
        snn_layer_sizes: List[int] = None,
        memory_dim: int = 64,
        device: str = "cpu",
        output_dir: str = "output",
        log_level: str = "INFO",
    ):
        self.n_assets = n_assets
        self.seq_len = seq_len
        self.horizon = horizon
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.ltc_hidden = ltc_hidden or [128, 64]
        self.snn_layer_sizes = snn_layer_sizes or [128, 64]
        self.memory_dim = memory_dim
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)


class PrometheusEngine:
    """
    Master system orchestrator. Connects all subsystems into a unified
    forward pass and provides high-level API for:
      - Training (synthetic black-swan data + real market data)
      - Inference (God's Eye report generation)
      - MAML adaptation (instant regime calibration in 3 gradient steps)
      - Nightly evolution (NEAT architecture optimization)
    """

    def __init__(self, config: Optional[PrometheusConfig] = None):
        self.config = config or PrometheusConfig()
        cfg = self.config
        self.device = cfg.device

        logger.info("Initializing Prometheus Engine (device=%s)", self.device)

        # Causal subsystem
        self.dag = CausalDAGEngine(max_nodes=500)
        self.causal_transformer = CausalTransformer(
            n_features=cfg.n_assets,
            n_targets=cfg.n_assets,
            horizon=cfg.horizon,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
        ).to(self.device)

        # Neural subsystems
        self.ltc = LiquidTimeConstantNetwork(
            input_size=cfg.n_assets,
            hidden_sizes=cfg.ltc_hidden,
            output_size=cfg.n_assets,
        ).to(self.device)

        self.snn = SpikingMarketEncoder(
            input_size=cfg.n_assets,
            layer_sizes=cfg.snn_layer_sizes,
            output_size=cfg.n_assets // 2,
        ).to(self.device)

        self.htm = HierarchicalTemporalMemory(
            n_input_features=cfg.n_assets,
            n_columns=1024,
            n_active_cols=20,
        )

        self.neuromod = NeuromodulationSystem(hidden_size=64)

        # Graph subsystem
        asset_names = [f"ASSET_{i}" for i in range(cfg.n_assets)]
        self.hive_mind = HiveMindGraphEngine(
            asset_names=asset_names,
            node_feat_dim=32,
            edge_feat_dim=8,
            memory_dim=cfg.memory_dim,
            device=self.device,
        )

        # Generative subsystem
        self.diffusion = MarketDiffusionSimulator(
            n_assets=cfg.n_assets,
            seq_len=cfg.seq_len,
            device=self.device,
        )
        self.black_swan_gen = BlackSwanGenerator(
            simulator=self.diffusion,
            n_assets=cfg.n_assets,
            device=self.device,
        )
        self.scenario_library = ScenarioLibrary(
            storage_dir=os.path.join(cfg.output_dir, "scenarios")
        )

        # Loss and optimization
        self.loss_fn = AsymmetricUtilityLoss(alpha=0.5, beta=2.0, gamma=3.0)
        self.kelly = KellyCriterionOptimizer(n_assets=cfg.n_assets)

        # Meta-learning
        self.maml = MAMLMetaLearner(
            model=self.causal_transformer,
            inner_lr=0.01,
            outer_lr=1e-3,
            n_inner_steps=3,
        )
        self.neat = NEATArchitectureEvolver(
            input_dim=cfg.n_assets * cfg.seq_len,
            output_dim=cfg.n_assets * cfg.horizon,
            population_size=30,
            n_generations=10,
        )

        # Data
        self.data_fetcher = MarketDataFetcher()
        self.sentiment = SentimentAnalyzer()

        # Output
        self.volcano = ProbabilityVolcano()
        self.report_gen = GodsEyeReportGenerator()

        # Main optimizer (for causal transformer)
        self.optimizer = torch.optim.AdamW(
            list(self.causal_transformer.parameters()) +
            list(self.ltc.parameters()) +
            list(self.loss_fn.parameters()),
            lr=1e-4,
            weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,        # [batch, seq_len, n_assets]
        return_monte_carlo: bool = True,
        n_mc_samples: int = 500,
    ) -> Dict:
        """
        Full Prometheus forward pass.

        Returns:
          predictions, causal_attributions, mc_paths, regime, neuromod_state
        """
        B, T, F = x.shape

        # 1. LTC temporal encoding (regime-aware)
        ltc_out, _, ltc_meta = self.ltc(x)
        regime = ltc_meta["regime"]

        # 2. Causal transformer (with DAG adjacency bias)
        causal_out = self.causal_transformer(ltc_out, return_attributions=True)
        predictions = causal_out["predictions"]
        attributions = causal_out.get("causal_attributions")

        # 3. Monte Carlo paths via diffusion (dropout-based uncertainty)
        mc_paths = None
        if return_monte_carlo:
            mc_paths = self._monte_carlo_paths(x, n_mc_samples)

        return {
            "predictions": predictions,
            "causal_attributions": attributions,
            "regime": regime,
            "ltc_tau_mean": ltc_meta.get("mean_tau", 0.5),
            "mc_paths": mc_paths,
        }

    def _monte_carlo_paths(
        self, x: torch.Tensor, n_samples: int
    ) -> np.ndarray:
        """Generate MC future paths using dropout-enabled forward passes."""
        self.causal_transformer.train()  # enable dropout
        paths = []
        with torch.no_grad():
            for _ in range(n_samples):
                ltc_out, _, _ = self.ltc(x)
                out = self.causal_transformer(ltc_out, return_attributions=False)
                pred = out["predictions"].squeeze(0).cpu().numpy()
                paths.append(pred)
        self.causal_transformer.eval()
        return np.stack(paths)  # [n_samples, horizon, n_assets]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        market_state: Optional[Dict] = None,
    ) -> Dict:
        """Single training step. Returns loss breakdown."""
        self.causal_transformer.train()
        self.ltc.train()

        self.optimizer.zero_grad()

        # Normalise input: numpy→tensor, 2D→3D (add batch dim)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # Forward pass
        ltc_out, _, _ = self.ltc(x)
        causal_out = self.causal_transformer(ltc_out, return_attributions=True)
        preds = causal_out["predictions"]

        # Compute asymmetric utility loss
        # preds: [B, horizon, n_assets], y: [B, horizon, n_assets]
        if preds.shape != y.shape:
            if y.dim() == 2:
                y = y.unsqueeze(1).expand_as(preds)
        loss = self.loss_fn(preds, y)

        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.causal_transformer.parameters()) + list(self.ltc.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()
        self.scheduler.step()

        loss_detail = self.loss_fn.get_loss_breakdown(preds.detach(), y.detach())
        return {
            "loss": float(loss.item()),
            "directional_accuracy": loss_detail["directional_accuracy"],
            "n_wrong_direction": loss_detail["n_wrong_direction"],
        }

    def train_on_black_swans(
        self,
        n_scenarios: int = 2000,
        n_epochs: int = 10,
        batch_size: int = 32,
        on_epoch_end: Optional[callable] = None,
    ) -> List[Dict]:
        """
        Train the model on synthetic black-swan scenarios.
        This is the core of the 'synthetic chaos pre-training' approach.
        """
        logger.info("Pre-training on %d black-swan scenarios...", n_scenarios)

        # Generate or load scenarios
        if not self.scenario_library.scenarios:
            logger.info("Generating black-swan library (this may take a while)...")
            scenarios = self.black_swan_gen.generate_doomsday_library(
                n_per_template=200,
                n_pure_random=500,
            )
            self.scenario_library.add_scenarios(scenarios)
            self.scenario_library.save()

        history = []
        loader = self.scenario_library.get_curriculum_loader(
            n_assets=self.config.n_assets,
            seq_len=self.config.seq_len,
            batch_size=batch_size,
            stage="all",
        )

        for epoch in range(n_epochs):
            epoch_losses = []
            for batch in loader:
                returns = batch["returns"].to(self.device)  # [B, seq_len, n_assets]
                # Input: first seq_len-horizon bars; Target: last horizon bars
                x = returns[:, :-self.config.horizon, :]
                y = returns[:, -self.config.horizon:, :]

                if x.shape[1] < 2:
                    continue

                step_result = self.train_step(x, y)
                epoch_losses.append(step_result["loss"])

            epoch_stats = {
                "epoch": epoch,
                "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "stage": "black_swan_pretrain",
            }
            history.append(epoch_stats)
            logger.info("Epoch %d: loss=%.4f", epoch, epoch_stats["mean_loss"])
            if on_epoch_end:
                on_epoch_end(epoch, epoch_stats)

        return history

    # ------------------------------------------------------------------
    # Inference / Reporting
    # ------------------------------------------------------------------

    def analyze(
        self,
        market_data: np.ndarray,           # [seq_len, n_assets]
        asset_names: Optional[List[str]] = None,
        sentiment_signals: Optional[Dict] = None,
        ofi: Optional[np.ndarray] = None,
        news_sentiment: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Full Prometheus analysis pipeline.
        Returns a complete God's Eye report.
        """
        self.causal_transformer.eval()
        self.ltc.eval()

        x = torch.tensor(market_data, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward pass with Monte Carlo
        result = self.forward(x, return_monte_carlo=True, n_mc_samples=200)
        mc_paths = result["mc_paths"]  # [200, horizon, n_assets]

        # HTM anomaly detection
        htm_result = self.htm.process_sequence(market_data, learn=False)

        # Graph intelligence
        latest_returns = market_data[-1]
        graph_result = self.hive_mind.update(latest_returns, ofi, news_sentiment)

        # Causal attribution
        pred = result["predictions"].squeeze(0)  # [horizon, n_assets]
        expected_move = pred.mean(0).detach().cpu().numpy()
        biggest_mover_idx = int(np.argmax(np.abs(expected_move)))
        asset_name = (asset_names or [f"ASSET_{i}" for i in range(self.config.n_assets)])[biggest_mover_idx]

        causal_attribution_result = {
            "target_asset": asset_name,
            "expected_move": float(expected_move[biggest_mover_idx]),
            "attributions": self.dag.causal_attribution(
                "SPX",  # default to SPX causal chain
                float(expected_move[biggest_mover_idx]),
                top_k=10,
            ),
        }

        # Neuromodulation update (using HTM anomaly as stress proxy)
        neuromod_state = self.neuromod.step(
            predicted_return=float(expected_move.mean()),
            actual_return=float(market_data[-1].mean()),
            market_entropy=float(htm_result["mean_anomaly"]),
            drawdown=0.0,  # would come from live portfolio tracker
            corr_breakdown=1.0 - float(graph_result.get("market_stability", 0.8)),
        )

        # Kelly position sizing
        asset_names_list = asset_names or [f"ASSET_{i}" for i in range(self.config.n_assets)]
        confidence = np.ones(self.config.n_assets) * 0.7
        kelly_result = self.kelly.compute_kelly_fractions(
            predictions=expected_move,
            confidence=confidence,
            neuromod_multiplier=neuromod_state["position_multiplier"],
        )

        # Probability volcano statistics
        if asset_names:
            volcano_stats = self.volcano.summary_statistics(mc_paths, asset_names)
        else:
            volcano_stats = self.volcano.summary_statistics(
                mc_paths, [f"ASSET_{i}" for i in range(self.config.n_assets)]
            )

        # Sentiment aggregation
        if sentiment_signals:
            sentiment_result = self.sentiment.aggregate_social_sentiment(sentiment_signals)
        else:
            sentiment_result = {}

        # God's Eye report
        report = self.report_gen.generate(
            causal_result=causal_attribution_result,
            graph_result=graph_result,
            neuromod_result=neuromod_state,
            kelly_result=kelly_result,
            volcano_stats=volcano_stats,
            htm_result=htm_result,
            sentiment_result=sentiment_result,
        )

        report["regime"] = result["regime"]
        report["mc_paths"] = mc_paths
        return report

    def adapt_to_regime(
        self,
        regime_data: np.ndarray,   # [n_bars, n_assets] — recent data from new regime
        n_steps: Optional[int] = None,
    ) -> Dict:
        """
        MAML fast adaptation: calibrate to new regime in 3 gradient steps.
        Call this when HTM anomaly score spikes or cortisol hits fear threshold.
        """
        n = n_steps or self.maml.n_inner_steps
        logger.info("MAML adaptation: %d gradient steps for new regime", n)

        x = torch.tensor(regime_data[:-self.config.horizon], dtype=torch.float32).unsqueeze(0).to(self.device)
        y = torch.tensor(regime_data[-self.config.horizon:], dtype=torch.float32).unsqueeze(0).to(self.device)

        if x.shape[1] < 2:
            return {"status": "insufficient_data", "steps": 0}

        adapted, losses = self.maml.adapt(
            support_data=(x, y),
            loss_fn=lambda p, t: self.loss_fn(p, t),
            return_adapted_model=False,
        )
        return {
            "status": "adapted",
            "steps": n,
            "inner_losses": losses,
            "final_loss": losses[-1] if losses else None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all model states."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.causal_transformer.state_dict(), f"{path}/causal_transformer.pt")
        torch.save(self.ltc.state_dict(), f"{path}/ltc.pt")
        torch.save(self.snn.state_dict(), f"{path}/snn.pt")
        torch.save(self.loss_fn.state_dict(), f"{path}/loss_fn.pt")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
        logger.info("Saved Prometheus engine to %s", path)

    def load(self, path: str) -> None:
        """Load all model states."""
        self.causal_transformer.load_state_dict(
            torch.load(f"{path}/causal_transformer.pt", map_location=self.device)
        )
        self.ltc.load_state_dict(
            torch.load(f"{path}/ltc.pt", map_location=self.device)
        )
        self.loss_fn.load_state_dict(
            torch.load(f"{path}/loss_fn.pt", map_location=self.device)
        )
        logger.info("Loaded Prometheus engine from %s", path)
