"""
Neuromodulation System – Dopamine & Cortisol as trainable hyper-parameters.

Simulates two neuromodulatory systems:
  - Dopamine (DA): reward prediction error signal → drives position sizing up
    when the model is consistently correct.
  - Cortisol (CORT): risk/stress hormone → reduces position sizing aggressively
    when market entropy spikes, mimicking the amygdala's fight-or-flight response.

These are not static hyperparameters; they are trained through the loss
landscape and evolve with market conditions.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DopamineSystem(nn.Module):
    """
    Reward Prediction Error (RPE) system.

    DA level = f(actual_reward - predicted_reward).
    High DA → confident, upsize position.
    Low/negative DA → prediction failed, reduce exposure.

    Uses a learned eligibility trace over recent prediction history.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        trace_len: int = 20,          # how many recent RPEs to integrate
        baseline_da: float = 0.5,
    ):
        super().__init__()
        self.trace_len = trace_len
        self.baseline_da = baseline_da

        # Learned RPE processor
        self.rpe_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        # Eligibility trace decay (trainable)
        self.trace_decay = nn.Parameter(torch.tensor(0.9))
        self._rpe_buffer: deque = deque(maxlen=trace_len)
        self._da_level: float = baseline_da

    DA_RPE_CLAMP: float = 5.0  # biological saturation limit for raw RPE signal

    def update(self, predicted_return: float, actual_return: float) -> float:
        """Compute RPE and update dopamine level."""
        rpe = actual_return - predicted_return
        # Clamp RPE to biological saturation range [-5, +5]
        self._raw_rpe: float = float(max(-self.DA_RPE_CLAMP, min(self.DA_RPE_CLAMP, rpe)))
        rpe = self._raw_rpe
        self._rpe_buffer.append(rpe)

        # Exponentially weighted trace
        rpe_tensor = torch.tensor([[rpe]], dtype=torch.float32)
        raw_da = self.rpe_net(rpe_tensor).item()

        # Decay old signal + incorporate new
        decay = torch.sigmoid(self.trace_decay).item()
        self._da_level = decay * self._da_level + (1 - decay) * raw_da
        self._da_level = max(0.05, min(1.0, self._da_level))
        return self._da_level

    def get_level(self) -> float:
        return self._da_level

    def get_position_multiplier(self) -> float:
        """Convert DA level to position sizing multiplier [0.2, 2.0]."""
        return 0.2 + 1.8 * self._da_level


class CortisolSystem(nn.Module):
    """
    Stress / Risk Hormone System (amygdala model).

    CORT level rises when:
      - Market entropy (VIX-like) spikes
      - Recent P&L drawdown exceeds threshold
      - Correlation breakdown detected (crisis regime)
      - Causal confidence collapses

    High CORT → aggressively reduce all position sizing.
    The response is asymmetric: spikes fast, recovers slowly (half-life = 48 bars).
    """

    # Hard-wired biological constraint: 70% position reduction in fear mode.
    FEAR_POSITION_CAP: float = 0.30

    def __init__(
        self,
        hidden_size: int = 64,
        fast_decay: float = 0.95,    # fast rise
        slow_decay: float = 0.98,    # slow recovery (asymmetric)
        fear_threshold: float = 0.7, # CORT level triggering full defensive mode
        lockout_duration: int = 30,  # steps to stay in fear after flash crash
    ):
        super().__init__()
        self.fast_decay = nn.Parameter(torch.tensor(fast_decay))
        self.slow_decay = nn.Parameter(torch.tensor(slow_decay))
        self.fear_threshold = fear_threshold
        self.lockout_duration = lockout_duration

        # Stress detector: takes [entropy, drawdown, corr_break, causal_conf]
        self.stress_net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        self._cort_level: float = 0.1
        self._in_fear_mode: bool = False
        self._lockout_remaining: int = 0
        # Per-asset fear states: a single asset's crash doesn't paralyse the whole book
        self._asset_fear: Dict[str, bool] = {}
        self._asset_lockout: Dict[str, int] = {}

    def update(
        self,
        market_entropy: float,   # normalized volatility [0, 1]
        drawdown: float,         # current drawdown fraction [0, 1]
        corr_breakdown: float,   # correlation collapse score [0, 1]
        causal_confidence: float, # avg causal confidence from DAG [0, 1]
    ) -> Tuple[float, bool]:
        """Update cortisol level. Returns (cort_level, fear_mode_active)."""
        x = torch.tensor(
            [[market_entropy, drawdown, corr_breakdown, 1.0 - causal_confidence]],
            dtype=torch.float32,
        )
        stress_signal = self.stress_net(x).item()

        # Hard-wired amygdala floor: extreme market conditions bypass neural uncertainty
        rule_stress = 0.5 * market_entropy + 0.3 * drawdown + 0.2 * corr_breakdown
        stress_signal = max(stress_signal, rule_stress)

        fast = torch.sigmoid(self.fast_decay).item()
        slow = torch.sigmoid(self.slow_decay).item()

        if stress_signal > self._cort_level:
            # Fast rise: threat detected
            self._cort_level = fast * self._cort_level + (1 - fast) * stress_signal
        else:
            # Slow decay: recovery is gradual (trauma persists)
            self._cort_level = slow * self._cort_level + (1 - slow) * stress_signal

        self._cort_level = max(0.0, min(1.0, self._cort_level))

        # Lockout takes priority: forced fear state for N steps
        if self._lockout_remaining > 0:
            self._lockout_remaining -= 1
            self._in_fear_mode = True
        else:
            self._in_fear_mode = self._cort_level >= self.fear_threshold

        return self._cort_level, self._in_fear_mode

    def trigger_flash_crash_lockout(self, asset: Optional[str] = None) -> None:
        """Force fear state for lockout_duration steps (amygdala hijack on crash).

        If `asset` is given, only that ticker enters lockout — the rest of the
        portfolio is unaffected (per-asset cortisol).  Calling without `asset`
        triggers a market-wide lockout as before.
        """
        self._cort_level = 1.0
        self._in_fear_mode = True
        self._lockout_remaining = self.lockout_duration
        if asset is not None:
            self._asset_fear[asset] = True
            self._asset_lockout[asset] = self.lockout_duration

    def step_asset_lockouts(self) -> None:
        """Decrement per-asset lockout counters each bar."""
        for ticker in list(self._asset_lockout):
            if self._asset_lockout[ticker] > 0:
                self._asset_lockout[ticker] -= 1
            if self._asset_lockout[ticker] == 0:
                self._asset_fear[ticker] = False

    def get_position_cap(self, asset: Optional[str] = None) -> float:
        """Max position size multiplier based on CORT. Ranges [0.0, 1.0].

        If `asset` is given, checks per-asset lockout first; other assets are
        unaffected by a single-asset flash crash.
        """
        if asset is not None and self._asset_fear.get(asset, False):
            return self.FEAR_POSITION_CAP
        if self._in_fear_mode:
            return self.FEAR_POSITION_CAP  # hard-wired 70% reduction
        return max(0.1, 1.0 - self._cort_level)

    def is_fear_mode(self) -> bool:
        return self._in_fear_mode


class NeuromodulationSystem(nn.Module):
    """
    Integrated neuromodulation: combines Dopamine and Cortisol into a single
    position-sizing signal that mimics the prefrontal cortex's risk-adjusted
    decision making.

    Final multiplier = DA_multiplier * CORT_cap
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.dopamine = DopamineSystem(hidden_size)
        self.cortisol = CortisolSystem(hidden_size)

    def step(
        self,
        predicted_return: float,
        actual_return: float,
        market_entropy: float,
        drawdown: float,
        corr_breakdown: float = 0.0,
        causal_confidence: float = 0.8,
    ) -> Dict:
        da_level = self.dopamine.update(predicted_return, actual_return)
        cort_level, fear_mode = self.cortisol.update(
            market_entropy, drawdown, corr_breakdown, causal_confidence
        )

        da_mult = self.dopamine.get_position_multiplier()
        cort_cap = self.cortisol.get_position_cap()
        final_multiplier = da_mult * cort_cap

        return {
            "dopamine": da_level,
            "cortisol": cort_level,
            "fear_mode": fear_mode,
            "da_multiplier": da_mult,
            "cort_cap": cort_cap,
            "position_multiplier": float(final_multiplier),
            "recommendation": self._recommend(final_multiplier, fear_mode),
        }

    def _recommend(self, mult: float, fear: bool) -> str:
        if fear:
            return "FULL_DEFENSE: exit most positions, hold only hedges"
        if mult < 0.3:
            return "CAUTIOUS: reduce to 30% normal size"
        if mult < 0.7:
            return "MODERATE: trade at 50-70% normal size"
        if mult < 1.2:
            return "NORMAL: standard position sizing"
        return f"AGGRESSIVE: upsize to {mult:.1f}x (high dopamine, low stress)"
