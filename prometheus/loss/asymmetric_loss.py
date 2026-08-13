"""
Asymmetric Utility-Based Loss Function.

Standard losses (MSE, MAE) punish all prediction errors equally.
This loss function ONLY penalizes errors that lose money, with
exponentially increasing penalty for directional mistakes.

The loss surface is:
  - prediction > target (bullish error on upside): ZERO penalty
  - prediction < target (bearish error when we predicted gains): SMALL penalty
  - prediction positive but market goes down (wrong direction): LARGE penalty
  - magnitude of mistake matters: bigger directional error = exponential penalty

This is connected to the Kelly Criterion optimizer for bet-sizing.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricUtilityLoss(nn.Module):
    """
    Direction-aware, asymmetric prediction loss for financial returns.

    Args:
        alpha:       penalty for small bearish errors (prediction > actual but both positive)
        beta:        penalty for directional errors (wrong sign)
        gamma:       exponential magnification for large directional errors
        confidence:  model's self-assessed confidence [0,1] — scales penalty
    """

    def __init__(
        self,
        alpha: float = 0.5,     # penalty for conservative errors
        beta: float = 2.0,      # base multiplier for directional errors
        gamma: float = 3.0,     # exponential magnifier for large errors
        use_confidence: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_confidence = use_confidence
        # Learnable penalty shaping (adapts to market regime)
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha)))
        self.log_beta = nn.Parameter(torch.log(torch.tensor(beta)))

    def forward(
        self,
        pred: torch.Tensor,           # [B, horizon] predicted returns
        target: torch.Tensor,         # [B, horizon] actual returns
        confidence: Optional[torch.Tensor] = None,  # [B] model confidence
    ) -> torch.Tensor:
        """
        Compute asymmetric utility loss.

        Cases:
          1. Same sign, pred magnitude >= actual: zero loss (conservative, correct direction)
          2. Same sign, pred magnitude < actual: small alpha loss (missed magnitude)
          3. Different sign: exponentially scaled beta loss (directional error = money lost)
        """
        alpha = torch.exp(self.log_alpha)
        beta = torch.exp(self.log_beta)

        # Directional correctness: +1 if same sign, -1 if opposite
        same_sign = (pred.sign() == target.sign()).float()
        wrong_direction = 1.0 - same_sign

        # Magnitude error
        mag_error = torch.abs(pred - target)

        # Case 1 & 2: correct direction
        # If pred is more conservative (abs(pred) < abs(target)): mild loss
        underestimated = (torch.abs(pred) < torch.abs(target)).float() * same_sign
        # Tiny calibration floor for correct-direction overestimates prevents the
        # model from collapsing to always-zero predictions (which would be "safe"
        # under pure zero-loss design).  1e-4 is ~200x below alpha so it never
        # dominates; it only ensures a nonzero gradient survives to the optimiser.
        calibration = 1e-4 * same_sign * mag_error
        correct_dir_loss = alpha * underestimated * mag_error + calibration

        # Case 3: wrong direction — exponentially penalized
        # Larger miss = exponentially worse (losing more money)
        directional_loss = wrong_direction * beta * (
            mag_error ** self.gamma
        ).clamp(max=100.0)  # cap to avoid explosions

        total_loss = correct_dir_loss + directional_loss

        # Scale by confidence: model should be heavily penalized when confident but wrong
        if self.use_confidence and confidence is not None:
            conf = confidence.unsqueeze(-1).expand_as(total_loss)
            confidence_penalty = 1.0 + conf * wrong_direction * 2.0
            total_loss = total_loss * confidence_penalty

        return total_loss.mean()

    def get_loss_breakdown(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """Return detailed breakdown for diagnostics."""
        with torch.no_grad():
            same_sign = (pred.sign() == target.sign()).float()
            wrong = 1.0 - same_sign
            mag_error = torch.abs(pred - target)

            return {
                "directional_accuracy": float(same_sign.mean().item()),
                "mean_directional_error": float((wrong * mag_error).mean().item()),
                "mean_mag_error_correct_dir": float(((1 - wrong) * mag_error).mean().item()),
                "n_wrong_direction": int(wrong.sum().item()),
                "total_samples": int(pred.numel()),
            }


class ProfitWeightedMSE(nn.Module):
    """
    Simple profit-weighted MSE: errors on high-magnitude moves are penalized more.
    Use this as a lightweight alternative to AsymmetricUtilityLoss.
    """

    def __init__(self, magnitude_power: float = 1.5):
        super().__init__()
        self.power = magnitude_power

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = torch.abs(target).pow(self.power) + 1.0
        return (weights * (pred - target).pow(2)).mean()
