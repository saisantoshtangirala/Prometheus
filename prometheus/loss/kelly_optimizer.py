"""
Kelly Criterion Position Size Optimizer.

The Kelly Criterion maximizes the long-run geometric growth rate of capital.
For a bet with:
  - win probability p
  - win/loss ratio b = avg_win / avg_loss

The optimal fraction to bet is: f* = (bp - q) / b  where q = 1 - p

Prometheus extends this with:
  - Fractional Kelly (f_k = 0.5 * f*) for safety
  - Confidence-adjusted Kelly (uses model uncertainty as probability adjustment)
  - Multi-asset Kelly with correlation matrix (portfolio Kelly)
  - Dynamic Kelly that adjusts to cortisol and market entropy signals
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class KellyCriterionOptimizer:
    """
    Dynamic Kelly Criterion position sizer integrated with Prometheus signals.

    Takes:
      - Model predictions + confidence
      - Neuromodulation state (dopamine/cortisol)
      - Historical win/loss rates

    Outputs:
      - Per-asset Kelly fraction
      - Portfolio-level position sizes
      - Risk budget allocation
    """

    def __init__(
        self,
        n_assets: int,
        kelly_fraction: float = 0.5,   # fractional Kelly (0.5 = half-Kelly)
        max_position: float = 0.20,    # max 20% in any single position
        max_portfolio_exposure: float = 1.0,  # fully invested at max
        risk_free_rate: float = 0.05,  # annual, for Sharpe adjustment
        lookback: int = 252,
    ):
        self.n_assets = n_assets
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.max_portfolio_exposure = max_portfolio_exposure
        self.risk_free_rate = risk_free_rate / 252  # per bar
        self.lookback = lookback

        self._return_history: List[List[float]] = [[] for _ in range(n_assets)]
        self._prediction_history: List[List[float]] = [[] for _ in range(n_assets)]

    def record(self, asset_idx: int, predicted: float, actual: float) -> None:
        """Record a prediction/actual pair for calibration."""
        self._prediction_history[asset_idx].append(predicted)
        self._return_history[asset_idx].append(actual)
        if len(self._return_history[asset_idx]) > self.lookback:
            self._return_history[asset_idx].pop(0)
            self._prediction_history[asset_idx].pop(0)

    def compute_kelly_fractions(
        self,
        predictions: np.ndarray,         # [n_assets] predicted returns
        confidence: np.ndarray,           # [n_assets] model confidence [0,1]
        neuromod_multiplier: float = 1.0, # from NeuromodulationSystem
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute per-asset Kelly fractions, adjusted for model confidence
        and neuromodulatory state.
        """
        fractions = np.zeros(self.n_assets)
        details = []

        for i in range(self.n_assets):
            hist = self._return_history[i]
            if len(hist) < 30:
                # Not enough history: use conservative flat estimate
                p_win = 0.5 + 0.1 * confidence[i]
                avg_win = max(abs(predictions[i]) * 1.2, 0.001)
                avg_loss = max(abs(predictions[i]) * 0.8, 0.001)
            else:
                returns = np.array(hist)
                preds = np.array(self._prediction_history[i])
                correct = (np.sign(preds) == np.sign(returns))
                p_win = float(correct.mean())
                wins = returns[correct & (returns > 0)]
                losses = returns[~correct & (returns < 0)]
                avg_win = float(wins.mean()) if len(wins) > 0 else 0.01
                avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.01

            # Confidence-adjusted win probability
            adjusted_p_win = 0.5 + (p_win - 0.5) * confidence[i]
            q = 1.0 - adjusted_p_win

            if avg_loss < 1e-8:
                f_kelly = 0.0
            else:
                b = avg_win / avg_loss  # win/loss ratio
                f_kelly = (b * adjusted_p_win - q) / b

            # Apply fractional Kelly + neuromodulation
            f_kelly = f_kelly * self.kelly_fraction * neuromod_multiplier

            # Clip to max position
            f_kelly = float(np.clip(f_kelly, -self.max_position, self.max_position))

            # Only take position in direction of prediction
            if predictions[i] < 0:
                f_kelly = -abs(f_kelly)
            else:
                f_kelly = abs(f_kelly)

            fractions[i] = f_kelly
            details.append({
                "asset_idx": i,
                "p_win": float(adjusted_p_win),
                "win_loss_ratio": float(avg_win / max(avg_loss, 1e-8)),
                "raw_kelly": float(f_kelly / self.kelly_fraction),
                "final_fraction": float(f_kelly),
            })

        # Portfolio-level: scale down if total exposure exceeds budget
        total_exposure = np.abs(fractions).sum()
        if total_exposure > self.max_portfolio_exposure:
            scale = self.max_portfolio_exposure / total_exposure
            fractions *= scale

        # Multi-asset portfolio Kelly with correlation adjustment
        if correlation_matrix is not None:
            fractions = self._corr_adjust(fractions, correlation_matrix)

        return {
            "kelly_fractions": fractions.tolist(),
            "total_long_exposure": float(fractions[fractions > 0].sum()),
            "total_short_exposure": float(abs(fractions[fractions < 0]).sum()),
            "net_exposure": float(fractions.sum()),
            "n_active_positions": int((fractions != 0).sum()),
            "per_asset_details": details,
            "neuromod_multiplier_applied": neuromod_multiplier,
        }

    def _corr_adjust(
        self, fractions: np.ndarray, corr: np.ndarray
    ) -> np.ndarray:
        """
        Reduce positions in highly correlated pairs to avoid concentration.
        If two assets have correlation > 0.8, scale both down by correlation factor.
        """
        n = len(fractions)
        adjusted = fractions.copy()
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) > 0.8 and fractions[i] * fractions[j] > 0:
                    # Same-direction positions in correlated assets: reduce both
                    factor = 1.0 - (abs(corr[i, j]) - 0.8) * 2.0
                    adjusted[i] *= factor
                    adjusted[j] *= factor
        return adjusted

    def compute_portfolio_sharpe(self) -> Dict:
        """Compute realized Sharpe ratio across all assets."""
        all_returns = []
        for hist in self._return_history:
            if len(hist) >= 10:
                all_returns.extend(hist[-252:])
        if not all_returns:
            return {"sharpe": 0.0, "n_obs": 0}
        arr = np.array(all_returns)
        excess = arr - self.risk_free_rate
        sharpe = float(excess.mean() / (arr.std() + 1e-8) * np.sqrt(252))
        return {
            "sharpe": sharpe,
            "mean_return": float(arr.mean()),
            "volatility": float(arr.std()),
            "n_obs": len(all_returns),
        }

    def get_risk_budget(self, volatility_targets: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute per-asset risk budget (volatility parity weights).
        Each asset contributes equal volatility to the portfolio.
        """
        vols = []
        for hist in self._return_history:
            if len(hist) >= 20:
                vols.append(float(np.std(hist[-20:])))
            else:
                vols.append(0.01)
        vols = np.array(vols)
        inv_vol = 1.0 / (vols + 1e-8)
        return inv_vol / inv_vol.sum()
