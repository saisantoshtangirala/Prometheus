"""
125-window walk-forward for NightEvolver.

The one rule this harness enforces above all others: **the GA is re-run
inside every window, on that window's training data only.** What is
validated is therefore exactly what gets deployed - the same evolution,
on the same kind of data, scored purely out-of-sample.

That directly fixes the failure the spec's own comparison table names
("Backtest tested the wrong model"), and it is why this is slower than
scoring one pre-evolved genome across all windows. Scoring a single
genome that was evolved on the full history would be fast, and would be
look-ahead of the worst kind: the strategy would have seen every test
window before trading it.

REPORTED METRICS, and why each is here:

  raw hit rate + p-value   the model's directional call, ties excluded -
                           the same SignalDiagnostic every other strategy
                           in this repo is measured with.
  net Sharpe               after 22bp NSE round-trip costs.
  deflated P(SR>0)         with n_trials = windows x GA search budget,
                           because a 125-window walk-forward that re-runs
                           a 1000-evaluation search is 125,000 trials, not
                           one.
  mean overfitting gap     average (in-sample - out-of-sample) Sharpe
                           across windows. Near zero = the GA is finding
                           something durable. Large and positive = it is
                           fitting each window's noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kronos.backtest import (
    SignalDiagnostic, WalkForwardConfig, _compute_signal_diagnostic,
    deflated_sharpe, max_drawdown, sharpe,
)
from nightevolver.data_loader import MarketData, build_market_data
from nightevolver.ga_engine import GAConfig, GeneticEvolver, simulate
from nightevolver.genome import decode, score_matrix

logger = logging.getLogger("nightevolver.backtest")

TRADING_DAYS = 252


@dataclass
class EvolvedBacktestResult:
    signal: SignalDiagnostic
    daily_returns: pd.Series
    total_return: float
    cagr: float
    sharpe: float
    deflated_sharpe_prob: float
    max_drawdown: float
    win_rate: float
    avg_turnover: float
    n_windows: int
    total_trials: int
    mean_overfitting_gap: float
    mean_in_sample_sharpe: float
    windows: List[Dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"NightEvolver walk-forward ({self.n_windows} windows, GA re-run per window)\n"
            f"  RAW SIGNAL   hit {self.signal.hit_rate:.1%} "
            f"(p={self.signal.hit_rate_p_value:.4f}, n={self.signal.n_calls})  "
            f"pearson {self.signal.pearson_r:+.4f} (p={self.signal.pearson_p:.4f})\n"
            f"  NET          total {self.total_return:+.2%}  CAGR {self.cagr:+.2%}  "
            f"Sharpe {self.sharpe:+.2f}  maxDD {self.max_drawdown:.2%}  "
            f"win {self.win_rate:.1%}  turnover {self.avg_turnover:.3f}/bar\n"
            f"  OVERFITTING  mean in-sample Sharpe {self.mean_in_sample_sharpe:+.2f} "
            f"vs realised OOS - mean gap {self.mean_overfitting_gap:+.2f}\n"
            f"  DEFLATED     P(SR>0) = {self.deflated_sharpe_prob:.3f} "
            f"(n_trials={self.total_trials:,} = windows x GA budget; gate 0.95)"
        )


class EvolvedWalkForward:
    """Walk-forward that re-evolves the strategy inside every window."""

    def __init__(self, md: MarketData, config: Optional[WalkForwardConfig] = None,
                 ga_config: Optional[GAConfig] = None):
        self.md = md
        self.cfg = config or WalkForwardConfig()
        self.ga_cfg = ga_config or GAConfig()

    def windows(self) -> List[Tuple[int, int, int]]:
        """Same index arithmetic as kronos.WalkForwardBacktester."""
        out: List[Tuple[int, int, int]] = []
        n = self.md.n_bars
        start = 0
        while start + self.cfg.train_window + 1 < n:
            train_end = start + self.cfg.train_window
            test_end = min(train_end + self.cfg.test_window, n)
            if test_end - train_end < 1:
                break
            out.append((start, train_end, test_end))
            start += self.cfg.test_window
        return out

    def run(self, max_windows: Optional[int] = None) -> EvolvedBacktestResult:
        spans = self.windows()
        if max_windows:
            spans = spans[:max_windows]

        preds: List[float] = []
        actuals: List[float] = []
        tp: Dict[str, List[float]] = {t: [] for t in self.md.tickers}
        ta: Dict[str, List[float]] = {t: [] for t in self.md.tickers}

        daily: List[float] = []
        dates: List[pd.Timestamp] = []
        turnovers: List[float] = []
        gaps: List[float] = []
        in_sample_sharpes: List[float] = []
        win_rows: List[Dict] = []

        for wi, (s, e, te) in enumerate(spans, 1):
            train = self.md.slice(s, e)
            test = self.md.slice(e, te)
            logger.info("[nightevolver] window %d/%d train=[%d,%d) test=[%d,%d)",
                        wi, len(spans), s, e, e, te)

            # Re-evolve on THIS window's training data only.
            cfg = GAConfig(**{**self.ga_cfg.__dict__, "seed": self.ga_cfg.seed + wi})
            res = GeneticEvolver(cfg).evolve(train, validation=None)
            strat = res.best_strategy

            # Score the evolved strategy strictly out-of-sample.
            oos = simulate(test, strat, cfg.cost_bps, cfg.max_position)
            in_sample_sharpes.append(res.in_sample.sharpe)
            gaps.append(res.in_sample.sharpe - oos.sharpe)

            scores = score_matrix(test.indicators, strat)          # [T, A]
            for t in range(test.n_bars):
                for i, ticker in enumerate(test.tickers):
                    preds.append(float(scores[t, i]))
                    actuals.append(float(test.forward_returns[t, i]))
                    tp[ticker].append(float(scores[t, i]))
                    ta[ticker].append(float(test.forward_returns[t, i]))

            daily.extend(oos.daily_returns.tolist())
            dates.extend(list(test.dates[: len(oos.daily_returns)]))
            turnovers.append(oos.avg_turnover)
            win_rows.append({
                "window": wi, "in_sample_sharpe": res.in_sample.sharpe,
                "oos_sharpe": oos.sharpe, "oos_win_rate": oos.win_rate,
                "oos_trades": oos.n_trades,
                "top_indicators": strat.top_indicators(3),
            })

        diag = _compute_signal_diagnostic("nightevolver", preds, actuals, tp, ta)

        dr = pd.Series(daily, index=pd.DatetimeIndex(dates)) if daily else pd.Series(dtype=float)
        if len(dr) > 1:
            equity = (1.0 + dr).cumprod()
            total_return = float(equity.iloc[-1] - 1.0)
            years = len(dr) / TRADING_DAYS
            cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0
            sr = sharpe(dr.values)
            mdd = max_drawdown(equity.values)
            wr = float((dr > 0).mean())
        else:
            total_return = cagr = sr = mdd = wr = 0.0

        # The honest trial count: every window ran a full GA search.
        total_trials = max(len(spans) * self.ga_cfg.search_budget, 1)
        dsr = deflated_sharpe(sr, max(len(dr), 3), n_trials=total_trials)

        return EvolvedBacktestResult(
            signal=diag, daily_returns=dr, total_return=total_return, cagr=cagr,
            sharpe=sr, deflated_sharpe_prob=dsr, max_drawdown=mdd, win_rate=wr,
            avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
            n_windows=len(spans), total_trials=total_trials,
            mean_overfitting_gap=float(np.mean(gaps)) if gaps else 0.0,
            mean_in_sample_sharpe=float(np.mean(in_sample_sharpes)) if in_sample_sharpes else 0.0,
            windows=win_rows,
        )
