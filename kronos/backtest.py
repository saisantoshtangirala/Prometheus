"""
Walk-forward backtest harness - the honesty layer of Kronos.

Rolls a train window across real historical data, trades the following
test window out-of-sample, and repeats to the end of history. Every
strategy pays transaction costs on turnover. The Kronos stack is judged
against two controls it must beat to claim any edge:

  - momentum: a deliberately dumb 20-day trailing-sign baseline
  - buy_hold: equal-weight buy-and-hold of the same universe

Outputs: a comparison table (markdown), per-strategy equity curves, and
a deflated Sharpe ratio that punishes multiple testing.

NO LOOK-AHEAD: the weight held during day t is computed from data up to
and including day t-1. The walk-forward split is enforced by index
arithmetic and covered by tests.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Data loading ladder: CSV -> yfinance
# ---------------------------------------------------------------------------

def load_history(
    tickers: List[str],
    start: str,
    end: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return a [date x ticker] close-price DataFrame.

    csv_path: a CSV with a Date column/index and one column per ticker
    (what `save_history` writes). If absent, yfinance is tried.
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        cols = [t for t in tickers if t in df.columns]
        if not cols:
            raise ValueError(f"CSV {csv_path} has none of {tickers}")
        out = df[cols].loc[start:end].dropna(how="all")
        logger.info("[backtest] loaded %d rows x %d tickers from %s",
                    len(out), len(cols), csv_path)
        return out

    import yfinance as yf
    data = yf.download(tickers, start=start, end=end,
                       interval="1d", progress=False, auto_adjust=True)
    if data is None or data.empty:
        raise RuntimeError(
            "yfinance returned no data and no csv_path was given. "
            "Provide --csv with historical closes to run offline."
        )
    closes = data["Close"] if "Close" in data else data
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    return closes.dropna(how="all")


def save_history(closes: pd.DataFrame, csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    closes.to_csv(csv_path)


# ---------------------------------------------------------------------------
# Strategies - fit on the train window, predict weights one day at a time
# ---------------------------------------------------------------------------

class Strategy:
    """fit() sees ONLY the train window; weights_for() sees data <= t-1."""

    name = "base"

    def fit(self, train_returns: np.ndarray) -> None:   # [T, A]
        pass

    def weights_for(self, recent_returns: np.ndarray) -> np.ndarray:  # [W, A]
        raise NotImplementedError


class BuyHoldStrategy(Strategy):
    """Equal-weight long the whole universe, always."""

    name = "buy_hold"

    def weights_for(self, recent_returns: np.ndarray) -> np.ndarray:
        n = recent_returns.shape[1]
        return np.full(n, 1.0 / n)


class MomentumStrategy(Strategy):
    """The dumb control: sign of the trailing 20-day mean, equal risk."""

    name = "momentum"

    def __init__(self, lookback: int = 20, max_weight: float = 0.25):
        self.lookback = lookback
        self.max_weight = max_weight

    def weights_for(self, recent_returns: np.ndarray) -> np.ndarray:
        window = recent_returns[-self.lookback:]
        signal = np.sign(window.mean(axis=0))
        n_active = max(int(np.abs(signal).sum()), 1)
        w = signal / n_active
        return np.clip(w, -self.max_weight, self.max_weight)


class KronosStrategy(Strategy):
    """
    The compressed Kronos daily stack, refit every walk-forward window:
      1. bootstrap adversarial futures from the train window (nightmare)
      2. evolve a tiny NEAT population against them (evolution)
      3. 3-step ClippedMAML warm-up on the last days of the train window
      4. predict next-bar returns -> tanh -> Kelly-capped weights
    """

    name = "kronos"

    def __init__(
        self,
        horizon: int = 5,
        population: int = 8,
        generations: int = 2,
        top_k: int = 3,
        n_futures: int = 64,
        max_weight: float = 0.25,
        seed: int = 42,
    ):
        self.horizon = horizon
        self.population = population
        self.generations = generations
        self.top_k = top_k
        self.n_futures = n_futures
        self.max_weight = max_weight
        self.seed = seed
        self.model: Optional[torch.nn.Module] = None
        self._n_assets: int = 0
        # See _calibrate_size_scale() below - mirrors kronos/reflex.py's
        # ReflexArc.calibrate_size_scale() exactly, so this harness's
        # sizing behavior matches what production actually does.
        self._size_scale: float = 1.0

    # -- internal: nightmare bootstrap (block resample of train returns) ----

    def _bootstrap_futures(self, train: np.ndarray) -> torch.Tensor:
        rng = np.random.default_rng(self.seed)
        T = train.shape[0]
        idx = rng.integers(0, T, size=(self.n_futures, self.horizon))
        futures = torch.tensor(train[idx], dtype=torch.float32)
        jitter = torch.randn_like(futures) * float(train.std()) * 0.1
        return futures + jitter

    def fit(self, train_returns: np.ndarray) -> None:
        from kronos.evolver import WeightedEnsemble
        from kronos.warmer import ClippedMAML
        from prometheus.meta.neat_evolver import (
            GenomeDecoder, NEATArchitectureEvolver,
        )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        import random as _random
        _random.seed(self.seed)

        self._n_assets = train_returns.shape[1]
        input_dim = self._n_assets * self.horizon

        # 1. Nightmare: bootstrap futures from the train window only
        futures = self._bootstrap_futures(train_returns)     # [N, H, A]
        X_val = futures.reshape(futures.shape[0], -1)
        y_val = futures[:, -1, :]

        # 2. Evolution: tiny NEAT against the futures
        evolver = NEATArchitectureEvolver(
            input_dim=input_dim, output_dim=self._n_assets,
            population_size=self.population, n_generations=self.generations,
            mutation_rate=0.25, elitism=2,
        )
        evolver.evolve((X_val, y_val), torch.nn.functional.mse_loss)
        for g in evolver.population:
            g.fitness = evolver.evaluate_fitness(
                g, (X_val, y_val), torch.nn.functional.mse_loss
            )
        ranked = sorted(evolver.population, key=lambda g: g.fitness, reverse=True)
        top = ranked[: self.top_k]
        models = [
            GenomeDecoder.decode(g, input_dim, self._n_assets) for g in top
        ]
        master = WeightedEnsemble(models, [g.fitness for g in top])

        # 3. MAML warm-up on the tail of the train window (still no test data)
        tail = train_returns[-(self.horizon + 5):]
        xs, ys = [], []
        for s in range(len(tail) - self.horizon):
            xs.append(tail[s:s + self.horizon].reshape(-1))
            ys.append(tail[s + self.horizon])
        if xs:
            learner = ClippedMAML(master, inner_lr=0.01, n_inner_steps=3)
            master, _ = learner.adapt(
                (torch.tensor(np.array(xs), dtype=torch.float32),
                 torch.tensor(np.array(ys), dtype=torch.float32)),
                torch.nn.functional.mse_loss,
                return_adapted_model=True,
            )
        self.model = master
        self.model.eval()
        self._size_scale = self._calibrate_size_scale(train_returns)

    def _calibrate_size_scale(self, train_returns: np.ndarray) -> float:
        """Mirrors kronos/reflex.py's ReflexArc.calibrate_size_scale() -
        see its docstring for the full rationale (a normalize-by-its-own-
        volatility scaler was tested and rejected: it's scale-invariant
        and can't tell noise from signal). This fits the same pooled OLS
        slope of actual next-bar return on raw prediction, using ONLY the
        train window fit() already has (no look-ahead - test data is
        never touched here)."""
        from kronos.reflex import MAX_SIZE_SCALE, MIN_CALIBRATION_SAMPLES
        T = train_returns.shape[0]
        preds, actuals = [], []
        with torch.no_grad():
            for t in range(self.horizon, T - 1):
                window = train_returns[t - self.horizon:t]
                x = torch.tensor(window.reshape(1, -1), dtype=torch.float32)
                pred = self.model(x).squeeze(0).numpy()
                preds.append(pred)
                actuals.append(train_returns[t + 1])
        if not preds:
            return 1.0
        p_arr = np.asarray(preds).ravel()
        a_arr = np.asarray(actuals).ravel()
        if p_arr.size < MIN_CALIBRATION_SAMPLES:
            return 1.0
        var_p = float(p_arr.var())
        if var_p < 1e-12:
            return 1.0
        scale = float(np.cov(p_arr, a_arr, bias=True)[0, 1] / var_p)
        scale = max(0.0, min(scale, MAX_SIZE_SCALE))
        return scale if scale > 0.0 else 1.0

    def weights_for(self, recent_returns: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(recent_returns.shape[1])
        window = recent_returns[-self.horizon:]
        if window.shape[0] < self.horizon:
            pad = np.zeros((self.horizon - window.shape[0], window.shape[1]))
            window = np.vstack([pad, window])
        x = torch.tensor(window.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            pred = self.model(x).squeeze(0).numpy()
        # size_scale: see _calibrate_size_scale() - fit once per
        # walk-forward window from train-only data, mirrors
        # ReflexArc.infer()'s use of calibrate_size_scale() in production.
        # A flat "* 50.0" here previously saturated tanh on pure noise
        # (verified: 68% of max_weight on average, 40% of calls landing
        # above 90% of cap) regardless of actual model confidence; the
        # calibrated scale only grows when the model's raw predictions
        # actually correlate with realized outcomes.
        w = np.tanh(pred * self._size_scale) * self.max_weight
        return np.clip(w, -self.max_weight, self.max_weight)


STRATEGIES: Dict[str, Callable[[], Strategy]] = {
    "buy_hold": BuyHoldStrategy,
    "momentum": MomentumStrategy,
    "kronos": KronosStrategy,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    return float(((equity - peaks) / peaks).min())


def sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS))


def deflated_sharpe(
    observed_sharpe: float,
    n_returns: int,
    n_trials: int = 3,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    Bailey & Lopez de Prado deflated Sharpe probability.

    Discounts the observed Sharpe for the number of strategy variants
    tried (n_trials) and for non-normal returns. Returns P(true SR > 0);
    values below ~0.95 mean the Sharpe is not statistically distinguishable
    from selection luck.
    """
    from scipy.stats import norm
    if n_returns < 3:
        return 0.0
    sr = observed_sharpe / np.sqrt(TRADING_DAYS)     # de-annualize per-bar
    # Expected max Sharpe of n_trials pure-noise strategies
    e = 0.5772156649
    z1 = norm.ppf(1 - 1.0 / n_trials) if n_trials > 1 else 0.0
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e)) if n_trials > 1 else 0.0
    sr_benchmark = z1 * (1 - e) + z2 * e
    sr_benchmark /= np.sqrt(max(n_returns - 1, 1))
    denom = np.sqrt(
        max(1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2, 1e-12)
        / max(n_returns - 1, 1)
    )
    return float(norm.cdf((sr - sr_benchmark) / denom))


# ---------------------------------------------------------------------------
# The walk-forward engine
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    strategy: str
    daily_returns: pd.Series
    equity: pd.Series
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    deflated_sharpe_prob: float
    max_drawdown: float
    hit_rate: float
    avg_turnover: float
    n_windows: int
    cost_bps_per_turnover: float


@dataclass
class WalkForwardConfig:
    train_window: int = 252         # 1 year of training data
    test_window: int = 21           # trade 1 month out-of-sample
    cost_bps: float = 10.0          # round-trip cost per unit turnover (10 bps)
    n_trials: int = 3               # strategies tried, for Sharpe deflation


class WalkForwardBacktester:
    """
    Rolls (train_window -> test_window) across the price history.

    For each window: strategy.fit(train) once, then for every day t in the
    test window, weights are computed from returns up to t-1 and applied to
    the return of day t. Turnover between consecutive days pays cost_bps.
    """

    def __init__(self, closes: pd.DataFrame, config: Optional[WalkForwardConfig] = None):
        self.closes = closes.dropna(how="any")
        self.returns = self.closes.pct_change().dropna()
        self.cfg = config or WalkForwardConfig()
        if len(self.returns) < self.cfg.train_window + self.cfg.test_window:
            raise ValueError(
                f"Need >= {self.cfg.train_window + self.cfg.test_window} bars, "
                f"got {len(self.returns)}"
            )

    def windows(self) -> List[Tuple[int, int, int]]:
        """[(train_start, train_end, test_end)] index triples, no overlap of
        train and test: train = [s, e), test = [e, te)."""
        out = []
        r = self.cfg
        start = 0
        while start + r.train_window + 1 < len(self.returns):
            train_end = start + r.train_window
            test_end = min(train_end + r.test_window, len(self.returns))
            if test_end - train_end < 1:
                break
            out.append((start, train_end, test_end))
            start += r.test_window
        return out

    def run(self, strategy: Strategy) -> BacktestResult:
        rets = self.returns.values                     # [T, A]
        dates = self.returns.index
        daily: List[float] = []
        daily_dates: List = []
        turnovers: List[float] = []
        prev_w = np.zeros(rets.shape[1])
        cost = self.cfg.cost_bps / 10_000.0

        spans = self.windows()
        for (s, e, te) in spans:
            strategy.fit(rets[s:e])
            for t in range(e, te):
                w = np.asarray(strategy.weights_for(rets[:t]), dtype=float)
                w = np.nan_to_num(w, nan=0.0)
                turnover = float(np.abs(w - prev_w).sum())
                gross = float((w * rets[t]).sum())
                net = gross - turnover * cost
                daily.append(net)
                daily_dates.append(dates[t])
                turnovers.append(turnover)
                prev_w = w

        r = pd.Series(daily, index=daily_dates, name=strategy.name)
        equity = (1 + r).cumprod()
        years = max(len(r) / TRADING_DAYS, 1e-9)
        total = float(equity.iloc[-1] - 1) if len(equity) else 0.0
        cagr = float((1 + total) ** (1 / years) - 1) if len(equity) else 0.0
        sr = sharpe(r.values)
        from scipy.stats import kurtosis as _k, skew as _s
        dsp = deflated_sharpe(
            sr, len(r), n_trials=self.cfg.n_trials,
            skew=float(_s(r.values)) if len(r) > 2 else 0.0,
            kurt=float(_k(r.values, fisher=False)) if len(r) > 2 else 3.0,
        )
        return BacktestResult(
            strategy=strategy.name,
            daily_returns=r,
            equity=equity,
            total_return=total,
            cagr=cagr,
            ann_vol=float(r.std() * np.sqrt(TRADING_DAYS)),
            sharpe=sr,
            deflated_sharpe_prob=dsp,
            max_drawdown=max_drawdown(equity.values) if len(equity) else 0.0,
            hit_rate=float((r > 0).mean()) if len(r) else 0.0,
            avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
            n_windows=len(spans),
            cost_bps_per_turnover=self.cfg.cost_bps,
        )

    # -- full comparison ----------------------------------------------------

    def compare(
        self, strategy_names: Optional[List[str]] = None
    ) -> Dict[str, BacktestResult]:
        names = strategy_names or list(STRATEGIES)
        results = {}
        for name in names:
            logger.info("[backtest] running strategy: %s", name)
            results[name] = self.run(STRATEGIES[name]())
        return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_report(
    results: Dict[str, BacktestResult],
    data_label: str,
    out_dir: str = "logs/backtests",
) -> str:
    """Write a markdown comparison report + JSON. Returns the md path."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    header = (
        "| Strategy | Total Return | CAGR | Ann.Vol | Sharpe | "
        "Deflated SR prob | Max DD | Hit Rate | Avg Turnover |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for name, res in results.items():
        rows.append(
            f"| {name} | {res.total_return:+.1%} | {res.cagr:+.1%} | "
            f"{res.ann_vol:.1%} | {res.sharpe:.2f} | "
            f"{res.deflated_sharpe_prob:.2f} | {res.max_drawdown:.1%} | "
            f"{res.hit_rate:.1%} | {res.avg_turnover:.2f} |"
        )

    verdict = _verdict(results)
    any_res = next(iter(results.values()))
    md = f"""# Walk-Forward Backtest Report

_Generated {ts} | data: {data_label} | windows: {any_res.n_windows} \
(train {any_res.n_windows and 'rolling'}) | costs: \
{any_res.cost_bps_per_turnover:.0f} bps per unit turnover_

{header}{chr(10).join(rows)}

## Verdict

{verdict}

## Reading this honestly

- **Deflated SR prob** is P(true Sharpe > 0) after discounting for the
  {len(results)} strategy variants tried. Below 0.95 = statistically
  indistinguishable from selection luck.
- Every strategy paid the same transaction costs on turnover. High-turnover
  strategies must clear a higher bar - that is realistic, not unfair.
- One backtest is one draw. Rerun across different periods and universes
  before believing anything.
"""

    md_path = os.path.join(out_dir, f"backtest_{ts}.md")
    with open(md_path, "w") as f:
        f.write(md)
    json_path = os.path.join(out_dir, f"backtest_{ts}.json")
    with open(json_path, "w") as f:
        json.dump({
            name: {
                "total_return": r.total_return, "cagr": r.cagr,
                "ann_vol": r.ann_vol, "sharpe": r.sharpe,
                "deflated_sharpe_prob": r.deflated_sharpe_prob,
                "max_drawdown": r.max_drawdown, "hit_rate": r.hit_rate,
                "avg_turnover": r.avg_turnover, "n_windows": r.n_windows,
            } for name, r in results.items()
        }, f, indent=2)
    logger.info("[backtest] report written: %s", md_path)
    return md_path


def _verdict(results: Dict[str, BacktestResult]) -> str:
    if "kronos" not in results:
        return "_Kronos not included in this run._"
    k = results["kronos"]
    lines = []
    bench = results.get("buy_hold")
    mom = results.get("momentum")
    if bench:
        beat = k.sharpe > bench.sharpe
        lines.append(
            f"- Kronos {'BEAT' if beat else 'LOST TO'} buy-and-hold on "
            f"risk-adjusted return (Sharpe {k.sharpe:.2f} vs {bench.sharpe:.2f})."
        )
    if mom:
        beat = k.sharpe > mom.sharpe
        lines.append(
            f"- Kronos {'BEAT' if beat else 'LOST TO'} the dumb momentum "
            f"control (Sharpe {k.sharpe:.2f} vs {mom.sharpe:.2f})."
        )
    if k.deflated_sharpe_prob < 0.95:
        lines.append(
            f"- Deflated Sharpe probability {k.deflated_sharpe_prob:.2f} < 0.95: "
            "the result is NOT statistically distinguishable from luck. "
            "Do not deploy capital on this evidence."
        )
    else:
        lines.append(
            f"- Deflated Sharpe probability {k.deflated_sharpe_prob:.2f} >= 0.95: "
            "the edge is statistically significant in THIS sample. Verify on "
            "other periods/universes before trusting it."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic fixture for offline environments (clearly labeled)
# ---------------------------------------------------------------------------

def synthetic_history(
    tickers: List[str], n_days: int = 1500, seed: int = 11
) -> pd.DataFrame:
    """
    Regime-switching GBM: bull / bear / choppy segments with correlated
    assets. For harness validation ONLY - results on this data say nothing
    about real markets and reports are labeled accordingly.
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)
    corr_base = rng.normal(0, 0.01, (n_days, 1))
    prices = np.zeros((n_days, n))
    regimes = []
    t = 0
    while t < n_days:
        length = int(rng.integers(60, 250))
        regimes.append((t, min(t + length, n_days),
                        rng.choice(["bull", "bear", "chop"])))
        t += length
    rets = np.zeros((n_days, n))
    for (s, e, kind) in regimes:
        drift = {"bull": 0.0006, "bear": -0.0008, "chop": 0.0}[kind]
        vol = {"bull": 0.009, "bear": 0.018, "chop": 0.012}[kind]
        seg = drift + vol * (
            0.6 * corr_base[s:e] + 0.4 * rng.normal(0, 1, (e - s, n))
        )
        rets[s:e] = seg
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)
