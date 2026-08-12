"""
Black-Swan Event Generator.

Uses the diffusion model with extreme conditioning vectors to generate
10,000 synthetic "doomsday" scenarios — simultaneous multi-sigma moves
across asset classes that have statistically never occurred in history.

Scenarios are calibrated to stress-test the main model and build resilience
to tail events before they happen in the real world.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BlackSwanScenario:
    """Defines the conditioning parameters for one extreme scenario."""
    name: str
    description: str
    category: str  # geopolitical, financial, natural, pandemic, cyber
    sigma_multiplier: float  # how many σ above normal
    asset_shocks: Dict[str, float]  # asset → expected return
    correlation_regime: str  # "crash", "flight_to_quality", "decoupled"
    vix_target: float
    duration_bars: int
    probability_estimate: float  # subjective annual probability


# Pre-defined historical black-swan archetypes as seeds
HISTORICAL_ARCHETYPES = [
    BlackSwanScenario(
        "GFC_2008", "Global Financial Crisis - Lehman Brothers collapse",
        "financial", 8.0,
        {"SPX": -0.55, "VIX": 2.5, "GOLD": 0.25, "CREDIT_SPREAD": 5.0, "USDJPY": -0.15},
        "crash", 80.0, 252, 0.03,
    ),
    BlackSwanScenario(
        "COVID_CRASH", "COVID-19 pandemic market crash",
        "pandemic", 10.0,
        {"SPX": -0.34, "VIX": 4.2, "OIL": -1.50, "GOLD": 0.12, "NDX": -0.27},
        "crash", 85.0, 40, 0.02,
    ),
    BlackSwanScenario(
        "FLASH_CRASH_2010", "May 6 2010 Flash Crash",
        "technical", 20.0,
        {"SPX": -0.09, "VIX": 1.5, "GOLD": 0.02},
        "crash", 45.0, 1, 0.15,
    ),
    BlackSwanScenario(
        "SOVEREIGN_DEBT_CRISIS", "Eurozone sovereign debt crisis",
        "geopolitical", 5.0,
        {"EURUSD": -0.25, "SPX": -0.20, "GOLD": 0.30, "CREDIT_SPREAD": 3.0},
        "flight_to_quality", 45.0, 180, 0.05,
    ),
]

# Novel synthetic scenarios — never occurred in recorded history
SYNTHETIC_DOOMSDAY_TEMPLATES = [
    {
        "name": "SIMULTANEOUS_20SIGMA",
        "description": "Simultaneous 20-sigma moves in bonds, FX, and equities",
        "asset_shocks": {"SPX": -0.60, "US_10Y_YIELD": -0.50, "DXY": 0.30, "VIX": 5.0},
        "sigma_mult": 20.0, "vix": 150.0, "duration": 5,
    },
    {
        "name": "DOLLAR_HYPERINFLATION",
        "description": "USD loses reserve currency status overnight",
        "asset_shocks": {"DXY": -0.40, "GOLD": 3.0, "BTC": 5.0, "EURUSD": 0.60},
        "sigma_mult": 15.0, "vix": 120.0, "duration": 30,
    },
    {
        "name": "AI_SINGULARITY_SHOCK",
        "description": "AGI announcement causes complete market repricing",
        "asset_shocks": {"SEMICONDUCTOR_INDEX": 3.0, "SPX": 0.30, "GOLD": -0.30, "VIX": 3.0},
        "sigma_mult": 12.0, "vix": 60.0, "duration": 10,
    },
    {
        "name": "CYBER_EXCHANGE_ATTACK",
        "description": "Major exchange infrastructure destroyed by cyberattack",
        "asset_shocks": {"SPX": -0.30, "VIX": 3.0, "BTC": -0.70, "GOLD": 0.50},
        "sigma_mult": 18.0, "vix": 130.0, "duration": 3,
    },
    {
        "name": "CHINA_TAIWAN_KINETIC",
        "description": "Full-scale China-Taiwan military conflict",
        "asset_shocks": {"SEMICONDUCTOR_INDEX": -0.80, "CHINA_PMI": -0.50, "WTI_OIL": 3.0},
        "sigma_mult": 14.0, "vix": 100.0, "duration": 60,
    },
    {
        "name": "BRAZIL_COFFEE_CASCADE",
        "description": "Complete Brazilian coffee crop failure → EM currency crisis → JPY carry unwind",
        "asset_shocks": {"BRAZIL_COFFEE": 2.0, "USDJPY": -0.20, "JAPAN_TOPIX": -0.25, "NDX": -0.15},
        "sigma_mult": 6.0, "vix": 50.0, "duration": 45,
    },
    {
        "name": "NEGATIVE_OIL_2",
        "description": "Second WTI negative price event — storage capacity crisis",
        "asset_shocks": {"WTI_OIL": -2.0, "SPX": -0.20, "DXY": 0.10, "GOLD": 0.05},
        "sigma_mult": 25.0, "vix": 80.0, "duration": 2,
    },
    {
        "name": "RATES_SHOCK_8PCT",
        "description": "Federal Reserve emergency hike to 8% in one meeting",
        "asset_shocks": {"FED_FUNDS_RATE": 3.0, "US_10Y_YIELD": 2.5, "SPX": -0.35, "GOLD": -0.15},
        "sigma_mult": 16.0, "vix": 90.0, "duration": 20,
    },
]


class BlackSwanGenerator:
    """
    Generates 10,000+ synthetic extreme market scenarios using the diffusion
    model conditioned on black-swan parameter vectors.

    The generated scenarios are used to train the main model to be resilient
    to tail events — real market volatility will feel boring by comparison.
    """

    def __init__(self, simulator, n_assets: int = 20, device: str = "cpu"):
        self.simulator = simulator
        self.n_assets = n_assets
        self.device = device
        self.generated_scenarios: List[Dict] = []

    def build_condition_vector(self, template: Dict) -> torch.Tensor:
        """
        Convert a black-swan template into a diffusion conditioning vector.
        The vector encodes: sigma_multiplier, VIX target, correlation regime,
        and per-asset shock magnitudes.
        """
        cond = torch.zeros(32, device=self.device)
        cond[0] = float(template.get("sigma_mult", 10.0)) / 25.0  # normalized
        cond[1] = float(template.get("vix", 80.0)) / 150.0
        cond[2] = float(template.get("duration", 10)) / 252.0

        # Asset shock encoding (first 5 shocks → positions 3-7)
        shocks = list(template.get("asset_shocks", {}).values())
        for i, shock in enumerate(shocks[:10]):
            cond[3 + i] = float(np.tanh(shock))  # bounded shock amplitude

        # Correlation regime: 1=crash (high corr), 0=decoupled
        regime = template.get("correlation_regime", "crash")
        cond[15] = 1.0 if regime == "crash" else (0.5 if regime == "flight_to_quality" else 0.0)

        return cond.unsqueeze(0)

    def generate_doomsday_library(
        self,
        n_per_template: int = 500,
        n_pure_random: int = 5000,
        seed: int = 42,
    ) -> List[Dict]:
        """
        Generate the full black-swan scenario library.

        Args:
            n_per_template: scenarios generated per template (with variation)
            n_pure_random: completely novel scenarios from extreme noise
            seed: random seed for reproducibility

        Returns list of scenario dicts with generated return paths.
        """
        logger.info("Generating black-swan scenario library...")
        scenarios = []
        np.random.seed(seed)

        # 1. Template-conditioned scenarios
        all_templates = SYNTHETIC_DOOMSDAY_TEMPLATES
        for i, template in enumerate(all_templates):
            logger.info("  Template %d/%d: %s", i + 1, len(all_templates), template["name"])
            cond = self.build_condition_vector(template)
            cond_batch = cond.expand(n_per_template, -1)

            paths = self.simulator.generate(
                n_scenarios=n_per_template,
                condition=cond_batch,
                seed=seed + i,
            )

            for j in range(n_per_template):
                path = paths[j].cpu().numpy()
                scenario = {
                    "id": f"{template['name']}_{j:04d}",
                    "template": template["name"],
                    "description": template["description"],
                    "sigma_multiplier": template.get("sigma_mult", 10.0),
                    "return_path": path,
                    "max_drawdown": self._compute_max_drawdown(path),
                    "peak_sigma": float(np.abs(path).max() / (path.std() + 1e-8)),
                    "correlation_crunch": float(np.corrcoef(path.T).mean()),
                    "severity_score": self._severity_score(path, template),
                }
                scenarios.append(scenario)

        # 2. Pure extreme-tail scenarios (far tails of diffusion prior)
        logger.info("  Generating %d pure extreme-tail scenarios...", n_pure_random)
        extreme_cond = torch.ones(n_pure_random, 32, device=self.device) * 0.9
        # Randomize to cover the full extreme space
        extreme_cond += torch.randn_like(extreme_cond) * 0.2
        extreme_cond = extreme_cond.clamp(0.0, 1.0)

        extreme_paths = self.simulator.generate(
            n_scenarios=n_pure_random,
            condition=extreme_cond,
            seed=seed + 99,
        )
        for j in range(n_pure_random):
            path = extreme_paths[j].cpu().numpy()
            if np.abs(path).max() > 3.0:  # only keep truly extreme paths
                scenarios.append({
                    "id": f"EXTREME_RANDOM_{j:05d}",
                    "template": "PURE_EXTREME",
                    "description": "Statistically novel extreme event",
                    "sigma_multiplier": float(np.abs(path).max()),
                    "return_path": path,
                    "max_drawdown": self._compute_max_drawdown(path),
                    "peak_sigma": float(np.abs(path).max() / (path.std() + 1e-8)),
                    "correlation_crunch": float(np.corrcoef(path.T).mean()),
                    "severity_score": float(np.abs(path).max()),
                })

        self.generated_scenarios = scenarios
        logger.info("Generated %d black-swan scenarios total", len(scenarios))
        return scenarios

    def get_training_dataset(self, n_scenarios: Optional[int] = None) -> torch.Tensor:
        """Return scenario return paths as a tensor for model training."""
        if not self.generated_scenarios:
            raise RuntimeError("Call generate_doomsday_library() first")
        paths = [s["return_path"] for s in self.generated_scenarios]
        if n_scenarios:
            paths = paths[:n_scenarios]
        return torch.tensor(np.stack(paths), dtype=torch.float32)

    def get_severity_ranked(self, top_k: int = 100) -> List[Dict]:
        """Return top-k most severe scenarios for red-team testing."""
        return sorted(
            self.generated_scenarios,
            key=lambda s: s["severity_score"],
            reverse=True,
        )[:top_k]

    def export_summary(self) -> Dict:
        """Summary statistics of the generated library."""
        if not self.generated_scenarios:
            return {}
        severities = [s["severity_score"] for s in self.generated_scenarios]
        drawdowns = [s["max_drawdown"] for s in self.generated_scenarios]
        return {
            "total_scenarios": len(self.generated_scenarios),
            "template_based": sum(1 for s in self.generated_scenarios if s["template"] != "PURE_EXTREME"),
            "pure_extreme": sum(1 for s in self.generated_scenarios if s["template"] == "PURE_EXTREME"),
            "severity": {
                "mean": float(np.mean(severities)),
                "p99": float(np.percentile(severities, 99)),
                "max": float(np.max(severities)),
            },
            "max_drawdown": {
                "mean": float(np.mean(drawdowns)),
                "p99": float(np.percentile(drawdowns, 99)),
                "worst": float(np.min(drawdowns)),
            },
        }

    @staticmethod
    def _compute_max_drawdown(path: np.ndarray) -> float:
        """Compute max drawdown from cumulative return path."""
        cum = np.cumprod(1 + path[:, 0].clip(-0.99, 10))
        roll_max = np.maximum.accumulate(cum)
        drawdowns = (cum - roll_max) / (roll_max + 1e-8)
        return float(drawdowns.min())

    @staticmethod
    def _severity_score(path: np.ndarray, template: Dict) -> float:
        sigma_mult = template.get("sigma_mult", 1.0)
        peak_move = float(np.abs(path).max())
        return sigma_mult * peak_move
