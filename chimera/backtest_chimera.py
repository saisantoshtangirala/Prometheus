"""
125-window walk-forward harness for CHIMERA.

Deliberately reuses kronos/backtest.py's machinery rather than
reimplementing it: WalkForwardConfig, the window index arithmetic,
SignalDiagnostic, deflated_sharpe. A new strategy that is scored by a
new, friendlier yardstick has proven nothing, and this repo has enough
history of near-identical harnesses drifting apart.

Two things are measured, and keeping them separate is the point:

  RAW SIGNAL   - the model's directional call, stripped of genome,
                 sizing and costs. Answers "does the model know
                 anything?"
  NET PORTFOLIO - the deployed strategy after QD ensembling, integer
                 NSE share sizing and real costs. Answers "would this
                 have made money in a ₹10,000 account?"

A system can pass the first and fail the second (edge too small to clear
22bp round trips) and that is a completely different problem from
failing both. Reporting one number would hide which.
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
from chimera.sizing import IntegerShareSizer, NSECostModel
from chimera.strategy import ChimeraStrategy, ChimeraStrategyConfig

logger = logging.getLogger("chimera.backtest")

TRADING_DAYS = 252


@dataclass
class ChimeraBacktestResult:
    """Both views of performance, plus the small-account diagnostics."""

    signal: SignalDiagnostic
    daily_returns: pd.Series
    equity: pd.Series
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    deflated_sharpe_prob: float
    max_drawdown: float
    day_hit_rate: float
    avg_turnover: float
    total_costs: float
    mean_quantisation_error: float
    mean_untradeable_names: float
    n_windows: int
    fit_reports: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"CHIMERA walk-forward ({self.n_windows} windows)\n"
            f"  RAW SIGNAL   hit {self.signal.hit_rate:.1%} "
            f"(p={self.signal.hit_rate_p_value:.4f}, n={self.signal.n_calls}) "
            f"pearson {self.signal.pearson_r:+.4f} (p={self.signal.pearson_p:.4f})\n"
            f"  NET PORTFOLIO  total {self.total_return:+.2%}  CAGR {self.cagr:+.2%}  "
            f"Sharpe {self.sharpe:+.2f}  deflated P(SR>0) {self.deflated_sharpe_prob:.3f}\n"
            f"  maxDD {self.max_drawdown:.2%}  day-hit {self.day_hit_rate:.1%}  "
            f"turnover {self.avg_turnover:.3f}/bar  costs ₹{self.total_costs:,.2f}\n"
            f"  SMALL-ACCOUNT  quantisation err {self.mean_quantisation_error:.3f} "
            f"L1/bar  untradeable names {self.mean_untradeable_names:.2f}/bar"
        )


class ChimeraWalkForward:
    """Walk-forward backtester for ChimeraStrategy on an NSE book."""

    def __init__(
        self,
        closes: pd.DataFrame,
        tickers: Optional[List[str]] = None,
        config: Optional[WalkForwardConfig] = None,
        strategy_config: Optional[ChimeraStrategyConfig] = None,
        capital: float = 10_000.0,
        allow_short: bool = False,
    ):
        self.closes = closes.dropna(how="any")
        self.returns = self.closes.pct_change().dropna()
        self.tickers = tickers or list(self.closes.columns)
        self.cfg = config or WalkForwardConfig()
        self.scfg = strategy_config or ChimeraStrategyConfig()
        self.capital = float(capital)
        self.costs = NSECostModel()
        self.allow_short = allow_short

    def windows(self) -> List[Tuple[int, int, int]]:
        """Same index arithmetic as kronos.WalkForwardBacktester.windows().

        Reimplemented rather than imported because that method is bound
        to a class that owns its own returns frame - but the semantics
        are identical and a test asserts they agree exactly.
        """
        out: List[Tuple[int, int, int]] = []
        start = 0
        while start + self.cfg.train_window + 1 < len(self.returns):
            train_end = start + self.cfg.train_window
            test_end = min(train_end + self.cfg.test_window, len(self.returns))
            if test_end - train_end < 1:
                break
            out.append((start, train_end, test_end))
            start += self.cfg.test_window
        return out

    def run(self, max_windows: Optional[int] = None) -> ChimeraBacktestResult:
        rets = self.returns.values
        prices = self.closes.iloc[1:].values          # aligned with returns
        dates = self.returns.index

        spans = self.windows()
        if max_windows:
            spans = spans[:max_windows]

        preds: List[float] = []
        actuals: List[float] = []
        tick_preds: Dict[str, List[float]] = {t: [] for t in self.tickers}
        tick_actuals: Dict[str, List[float]] = {t: [] for t in self.tickers}

        daily: List[float] = []
        daily_dates: List[pd.Timestamp] = []
        turnovers: List[float] = []
        quant_errs: List[float] = []
        untradeable: List[float] = []
        fit_reports: List[dict] = []
        total_costs = 0.0

        equity = self.capital
        shares = np.zeros(len(self.tickers), dtype=np.int64)

        for wi, (s, e, te) in enumerate(spans, 1):
            logger.info("[chimera] window %d/%d (train=[%d,%d) test=[%d,%d))",
                        wi, len(spans), s, e, e, te)
            strat = ChimeraStrategy(self.scfg)
            try:
                strat.fit(rets[s:e])
                fit_reports.append({"window": wi, **strat.fit_report})
            except Exception as exc:
                # A single bad window must not abort a multi-hour run; it
                # is recorded and skipped, exactly as the SNN harness does.
                logger.warning("[chimera] window %d fit failed (%s) - skipping", wi, exc)
                fit_reports.append({"window": wi, "error": str(exc)})
                continue

            sizer = IntegerShareSizer(
                capital=self.capital, cost_model=self.costs,
                max_position_pct=self.scfg.max_weight, allow_short=self.allow_short,
            )

            for t in range(e, te):
                hist = rets[:t]                        # strictly < t: no look-ahead
                realised = rets[t]

                sig = strat.raw_signal(hist)
                for i, tk in enumerate(self.tickers):
                    preds.append(float(sig[i])); actuals.append(float(realised[i]))
                    tick_preds[tk].append(float(sig[i]))
                    tick_actuals[tk].append(float(realised[i]))

                target_w = strat.weights_for(hist)
                res = sizer.size(target_w, prices[t], current_shares=shares, equity=equity)

                pnl = float((res.shares * prices[t] * realised).sum())
                equity = equity + pnl - res.cost
                total_costs += res.cost

                daily.append(pnl / max(equity, 1e-9))
                daily_dates.append(dates[t])
                turnovers.append(float(np.abs(res.trade_shares * prices[t]).sum()
                                       / max(equity, 1e-9)))
                quant_errs.append(res.quantisation_error)
                untradeable.append(float(res.n_untradeable))
                shares = res.shares

        diag = _compute_signal_diagnostic("chimera", preds, actuals,
                                          tick_preds, tick_actuals)

        dr = pd.Series(daily, index=pd.DatetimeIndex(daily_dates)) if daily \
            else pd.Series(dtype=float)
        eq = (1.0 + dr).cumprod() * self.capital if len(dr) else pd.Series(dtype=float)

        if len(dr) > 1:
            total_return = float(eq.iloc[-1] / self.capital - 1.0)
            years = len(dr) / TRADING_DAYS
            cagr = float((eq.iloc[-1] / self.capital) ** (1 / years) - 1.0) if years > 0 else 0.0
            ann_vol = float(dr.std() * np.sqrt(TRADING_DAYS))
            sr = sharpe(dr.values)
            dsr = deflated_sharpe(sr, len(dr), n_trials=self.cfg.n_trials)
            mdd = max_drawdown(eq.values)
            day_hit = float((dr > 0).mean())
        else:
            total_return = cagr = ann_vol = sr = dsr = mdd = day_hit = 0.0

        return ChimeraBacktestResult(
            signal=diag, daily_returns=dr, equity=eq,
            total_return=total_return, cagr=cagr, ann_vol=ann_vol, sharpe=sr,
            deflated_sharpe_prob=dsr, max_drawdown=mdd, day_hit_rate=day_hit,
            avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
            total_costs=total_costs,
            mean_quantisation_error=float(np.mean(quant_errs)) if quant_errs else 0.0,
            mean_untradeable_names=float(np.mean(untradeable)) if untradeable else 0.0,
            n_windows=len(spans), fit_reports=fit_reports,
        )
