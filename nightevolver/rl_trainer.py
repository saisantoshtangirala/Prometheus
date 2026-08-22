"""
Optional tabular Q-learning trainer.

Deliberately TABULAR, not deep. The state space is discretised to a few
hundred cells, so Q-values are counted rather than approximated, the
policy is a table you can print and audit, and there is no network to
silently no-op (which is exactly what happened to this project's MAML).
If a deep RL agent and a Q-table disagree on data this small, the
Q-table is more likely to be right.

State  = (regime, RSI quantile, MACD quantile)   - 3 x 5 x 5 = 75 cells
Action = {short, flat, long}
Reward = log return of the position - turnover penalty

THE SAME MULTIPLE-TESTING PROBLEM APPLIES. Q-learning over 1000 episodes
on one window is also a search, and its greedy policy is also a
maximum-over-many. `QLearningResult` therefore reports out-of-sample
performance alongside in-sample, on the same footing as the GA, and the
checkpoint gate applies identically. A Q-table that looks brilliant on
its training window and mediocre after it has learned the window, not
the market.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nightevolver.data_loader import MarketData
from nightevolver.genome import INDICATOR_NAMES

logger = logging.getLogger("nightevolver.rl")

N_REGIMES = 3          # low / normal / high volatility
N_QUANTILES = 5
N_ACTIONS = 3          # 0 = short, 1 = flat, 2 = long
ACTION_POSITION = np.array([-1.0, 0.0, 1.0])

# Indicator channels used to build the state.
_RSI_CHANNEL = INDICATOR_NAMES.index("rsi_14")
_MACD_CHANNEL = INDICATOR_NAMES.index("macd_hist")
_ATR_CHANNEL = INDICATOR_NAMES.index("atr_pct")


@dataclass
class QConfig:
    episodes: int = 1000
    alpha: float = 0.1          # learning rate
    gamma: float = 0.95         # discount
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    turnover_penalty: float = 0.01
    cost_bps: float = 22.0
    max_position: float = 0.10
    seed: int = 42


@dataclass
class QLearningResult:
    q_table: np.ndarray                  # [N_REGIMES, N_Q, N_Q, N_ACTIONS]
    in_sample_sharpe: float
    out_of_sample_sharpe: Optional[float]
    episodes: int
    state_coverage: float                # fraction of states ever visited
    history: List[Dict] = field(default_factory=list)

    @property
    def overfitting_gap(self) -> float:
        if self.out_of_sample_sharpe is None:
            return float("nan")
        return self.in_sample_sharpe - self.out_of_sample_sharpe

    def summary(self) -> str:
        oos = ("%+.2f" % self.out_of_sample_sharpe) if self.out_of_sample_sharpe is not None else "n/a"
        return (f"Q-learning: {self.episodes} episodes, state coverage "
                f"{self.state_coverage:.1%}\n"
                f"  IN-SAMPLE Sharpe {self.in_sample_sharpe:+.2f}  "
                f"OUT-SAMPLE Sharpe {oos}  gap {self.overfitting_gap:+.2f}")


def _quantise(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Map ~[-1,1] indicator values to bin indices [0, n_bins).

    Fixed bin edges, NOT data-derived quantiles: quantile edges fitted on
    the whole series would be look-ahead, and edges fitted per-window
    would make a state mean different things in training and execution.
    """
    edges = np.linspace(-1.0, 1.0, n_bins + 1)[1:-1]
    return np.clip(np.digitize(x, edges), 0, n_bins - 1)


def build_states(md: MarketData) -> np.ndarray:
    """MarketData -> [T, n_assets, 3] integer state indices."""
    ind = md.indicators
    atr = ind[:, :, _ATR_CHANNEL]
    regime = _quantise(atr, N_REGIMES)
    rsi = _quantise(ind[:, :, _RSI_CHANNEL], N_QUANTILES)
    macd = _quantise(ind[:, :, _MACD_CHANNEL], N_QUANTILES)
    return np.stack([regime, rsi, macd], axis=2).astype(np.int64)


def _evaluate_policy(q: np.ndarray, states: np.ndarray, fwd: np.ndarray,
                     cfg: QConfig) -> float:
    """Greedy-policy Sharpe over a window (annualised, net of costs)."""
    T, A, _ = states.shape
    prev = np.zeros(A)
    daily = np.zeros(T)
    for t in range(T):
        s = states[t]
        actions = np.argmax(q[s[:, 0], s[:, 1], s[:, 2]], axis=-1)
        pos = ACTION_POSITION[actions] * cfg.max_position
        turnover = np.abs(pos - prev).sum()
        # cost_bps is the full ROUND TRIP (see ga_engine.simulate for the
        # measurement); a trip is two turnover units, so the per-unit
        # charge is cost_bps/2. This mirrors the same fix there - this
        # function carried an identical double-count.
        daily[t] = float((pos * fwd[t]).sum() - turnover * cfg.cost_bps / 2.0 / 10_000.0)
        prev = pos
    sd = float(daily.std())
    return float(daily.mean() / sd * np.sqrt(252)) if sd > 1e-12 else 0.0


def train_q_learning(train: MarketData, validation: Optional[MarketData] = None,
                     config: Optional[QConfig] = None) -> QLearningResult:
    """Tabular Q-learning on the training window."""
    cfg = config or QConfig()
    rng = np.random.default_rng(cfg.seed)

    states = build_states(train)
    fwd = train.forward_returns
    T, A, _ = states.shape

    q = np.zeros((N_REGIMES, N_QUANTILES, N_QUANTILES, N_ACTIONS))
    visits = np.zeros_like(q, dtype=np.int64)
    history: List[Dict] = []

    for ep in range(cfg.episodes):
        # Linear epsilon decay: broad exploration early, exploitation late.
        eps = cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * \
            (ep / max(cfg.episodes - 1, 1))
        prev_action = np.ones(A, dtype=np.int64)          # start flat

        for t in range(T - 1):
            s = states[t]
            sn = states[t + 1]
            qsa = q[s[:, 0], s[:, 1], s[:, 2]]                # [A, N_ACTIONS]
            greedy = np.argmax(qsa, axis=-1)
            explore = rng.random(A) < eps
            actions = np.where(explore, rng.integers(0, N_ACTIONS, size=A), greedy)

            pos = ACTION_POSITION[actions]
            reward = pos * fwd[t] - cfg.turnover_penalty * np.abs(
                ACTION_POSITION[actions] - ACTION_POSITION[prev_action])

            best_next = q[sn[:, 0], sn[:, 1], sn[:, 2]].max(axis=-1)
            target = reward + cfg.gamma * best_next
            idx = (s[:, 0], s[:, 1], s[:, 2], actions)
            q[idx] += cfg.alpha * (target - q[idx])
            visits[idx] += 1
            prev_action = actions

        if (ep + 1) % max(cfg.episodes // 10, 1) == 0:
            sr = _evaluate_policy(q, states, fwd, cfg)
            history.append({"episode": ep + 1, "epsilon": eps, "in_sample_sharpe": sr})
            logger.info("[rl] episode %d/%d eps=%.2f sharpe=%.3f",
                        ep + 1, cfg.episodes, eps, sr)

    in_sr = _evaluate_policy(q, states, fwd, cfg)
    oos_sr = None
    if validation is not None and validation.n_bars > 1:
        oos_sr = _evaluate_policy(q, build_states(validation),
                                  validation.forward_returns, cfg)

    return QLearningResult(
        q_table=q, in_sample_sharpe=in_sr, out_of_sample_sharpe=oos_sr,
        episodes=cfg.episodes, state_coverage=float((visits > 0).mean()),
        history=history,
    )
