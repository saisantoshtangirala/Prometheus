"""
Kronos configuration loader.

Single source of truth: kronos/config.yaml. Every module receives a
KronosConfig instance - no hardcoded hyperparameters anywhere in kronos/.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class KronosConfig:
    """Dot-accessible, dict-backed configuration."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(f"No config key '{name}'") from None
        if isinstance(value, dict):
            return KronosConfig(value)
        return value

    def get(self, name: str, default: Any = None) -> Any:
        value = self._data.get(name, default)
        if isinstance(value, dict):
            return KronosConfig(value)
        return value

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def override(self, dotted_key: str, value: Any) -> None:
        """Set e.g. 'evolution.population_size' = 6 (graceful degradation)."""
        keys = dotted_key.split(".")
        node = self._data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value


def load_config(path: Optional[str] = None) -> KronosConfig:
    """Load config.yaml. Falls back to a minimal built-in default if missing."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        import yaml
        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        logger.info("Loaded Kronos config from %s", cfg_path)
        return KronosConfig(data)
    except FileNotFoundError:
        logger.warning("Config file %s not found - using built-in defaults", cfg_path)
    except ImportError:
        logger.warning("pyyaml not installed - using built-in defaults")
    return KronosConfig(_builtin_defaults())


def _builtin_defaults() -> Dict[str, Any]:
    """Minimal defaults so tests run without yaml installed."""
    return {
        "run": {"total_days": 365, "timezone": "America/New_York",
                "mode": "paper", "log_dir": "logs", "seed": 42},
        "schedule": {"digestion_start": "00:00", "nightmare_start": "02:00",
                     "evolution_start": "04:00", "adaptation_start": "05:00",
                     "report_time": "06:00", "market_open": "09:30",
                     "market_close": "16:00"},
        "data": {"tickers": ["SPY", "QQQ", "GLD", "TLT", "AAPL"],
                 "vix_ticker": "^VIX", "lookback_days": 30,
                 "sources": ["yfinance", "polygon", "alphavantage"],
                 "cross_validation_tolerance_pct": 1.0,
                 "max_missing_pct": 20.0,
                 "kalman_process_noise": 0.001, "kalman_obs_noise": 0.1},
        "nightmare": {"n_futures": 10000, "batch_size": 500, "horizon_days": 5,
                      "worst_case_bias": 0.7, "diffusion_steps": 50, "seed": None},
        "evolution": {"population_size": 20, "n_generations": 3, "top_k": 5,
                      "mutation_rate": 0.25, "crossover_rate": 0.5, "elitism": 2,
                      "fallback_population_size": 6, "fallback_generations": 1},
        "adaptation": {"inner_lr": 0.01, "n_inner_steps": 3, "support_days": 3},
        "reflex": {"inference_budget_ms": 1000, "vix_spike_sigma": 2.0,
                   "vix_window": 20, "lockout_minutes": 30,
                   "imbalance_threshold": 0.65},
        "trading": {"initial_capital": 100000.0, "max_position_pct": 0.25,
                    "kelly_fraction": 0.5,
                    "slippage": {"high_liquidity_pct": 0.05,
                                 "mid_liquidity_pct": 0.15,
                                 "low_liquidity_pct": 0.50,
                                 "high_liquidity_min_volume": 10_000_000,
                                 "mid_liquidity_min_volume": 1_000_000},
                    "commission_per_trade": 0.0, "db_path": "logs/trades.db"},
        "orchestrator": {"max_retries_per_phase": 1, "veto_file": "veto.txt",
                         "veto_delay_hours": 24, "heartbeat_minutes": 60,
                         "checkpoint_dir": "logs/models",
                         "report_dir": "logs/reports"},
        "risk": {"enabled": True, "max_daily_loss_pct": 0.05,
                 "max_drawdown_pct": 0.20, "max_single_order_pct": 0.30,
                 "max_price_deviation_pct": 0.20,
                 "halt_file": "logs/risk_halt.flag",
                 "kill_switch_file": "KILL_SWITCH"},
        "reporting": {"top_movers": 3, "volatility_windows": [5, 20]},
        "notifications": {"enabled": False, "send_daily_digest": True},
    }
