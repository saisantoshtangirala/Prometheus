"""
Walk-forward backtest of the model that ACTUALLY trades - ReflexArc's
SpikingMarketEncoder (kronos/reflex.py), trained via scripts/train.py's
real pretrain -> finetune -> meta procedure - not kronos/backtest.py's
separate, much simpler KronosStrategy (a from-scratch-every-window
NEAT+MAML model that orchestrator.py's run_reflex_tick() never calls;
its own NEAT ensemble is used only for the daily markdown report, see
kronos/orchestrator.py's _master_signals()/run_report()).

Every backtest number this project has produced before this module
(Sharpe -0.43 to -1.02, hit rate 47-49%) is about that OTHER model. This
closes the gap: it is the first walk-forward test of the model whose
weights KronosOrchestrator.maybe_adopt_runpod_checkpoint() actually loads
into ReflexArc.snn and trades with every day.

Design, grounded in real measurement, not guessed:
  - pretrain and meta both train on synthetic data with no dependency on
    any specific walk-forward window (black-swan scenarios / synthetic
    regime tasks), so they run ONCE, shared across every window - not
    per-window like a naive walk-forward port of
    `scripts/train.py --mode full` would do. Measured: pretrain's
    black-swan library generation + 1 epoch = 206s on CPU, n_assets=10
    (PrometheusEngine.train_on_black_swans() caches its scenario library
    in memory - `if not self.scenario_library.scenarios` - so this is a
    one-time cost within a run, not per-epoch or per-window).
  - finetune - the only stage that trains directly on REAL market data
    (via engine.train_snn_step(), which is what actually shapes
    ReflexArc.snn's weights) - is the only stage that depends on the
    specific window, and repeats per window, always starting fresh from
    the shared pretrain+meta baseline (matching kronos/backtest.py's own
    KronosStrategy.fit() convention: refit every window, no state carried
    across windows). Measured: ~2.5s/epoch on a 10-ticker, ~500-bar real
    slice; ~50-65s for the default 20 epochs including overhead. A full
    125-window run is therefore on the order of 2-2.5 hours, not the
    10-15 min/window (20-30 hour) figure guessed - not measured - by two
    external AI reviews of this project.
  - Per window, a fresh PrometheusEngine is constructed (fresh, zero-state
    optimizers) and immediately engine.load()'s the shared pretrain+meta
    checkpoint - PrometheusEngine.load() deliberately does NOT restore
    optimizer state (see its own docstring/code), so this reuses existing,
    tested serialization to get exactly "fresh optimizer, pretrained
    weights" with no manual state-dict surgery.

Evaluation: after each window's finetune, engine.snn's state_dict is
loaded into a REAL ReflexArc (kronos/reflex.py) - the exact class
KronosOrchestrator.run_reflex_tick() trades with - and its own infer()
is called on every test-window day, recording (predicted signal,
realized return) pairs exactly like kronos/backtest.py's
diagnose_signal_direction() does for KronosStrategy, for a directly
comparable SignalDiagnostic. This works because scripts/train.py's own
PrometheusConfig construction (snn_layer_sizes=[32,16],
snn_output_size=n_assets) is deliberately shape-matched to
ReflexArc.snn's hardcoded architecture - see
kronos/runpod_trigger.py's load_runpod_checkpoint() docstring, which
documents this exact compatibility requirement for the real nightly
RunPod pipeline this harness mirrors. calibrate_size_scale() is called
on each window's train-only data before evaluation, exactly matching
what KronosOrchestrator.maybe_adopt_runpod_checkpoint() does after every
real checkpoint adoption.

Also faithfully reproduces an existing mismatch rather than silently
"fixing" it: engine.train_snn_step() finetunes on seq_len=64-bar windows,
but ReflexArc.infer() in production is only ever called with
nightmare.horizon_days=5-bar windows (orchestrator.py's
run_reflex_tick(): `self.state.memory.returns_window(self.cfg.nightmare.
horizon_days)`). This harness evaluates with the same 5-bar window
production actually uses, train/test-length mismatch and all - the goal
is testing the real system as built, not an idealized version of it.
"""

from __future__ import annotations

import copy
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from kronos.backtest import (
    SignalDiagnostic,
    WalkForwardConfig,
    _compute_signal_diagnostic,
)

logger = logging.getLogger(__name__)


@dataclass
class SNNTrainConfig:
    """Kept separate from kronos/config.yaml's own hyperparameters -
    this harness trains prometheus.engine.PrometheusEngine, a different
    model with its own knobs, mirroring scripts/train.py's CLI defaults
    unless overridden (tests use smaller values for speed, same
    convention as KronosStrategy's population/generations args)."""
    seq_len: int = 64
    horizon: int = 5              # matches kronos/config.yaml's nightmare.horizon_days -
                                   # NOT scripts/train.py's own --horizon default of 10,
                                   # since evaluation must match what ReflexArc.infer()
                                   # is actually called with in production.
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    pretrain_epochs: int = 5
    finetune_epochs: int = 20
    meta_epochs: int = 10
    n_black_swans: int = 1000     # STILL not the library size - PrometheusEngine's
                                   # n_scenarios only reaches a log line. Kept to
                                   # mirror scripts/train.py's CLI surface. The two
                                   # fields below are the ones that size the library.
    # THE ACTUAL COST KNOBS. The scenario library is 8 templates x
    # n_per_template + n_pure_random, and EVERY scenario runs a reverse
    # diffusion loop of n_diffusion_steps. At the production defaults
    # (200/500/1000) that is 2.1 MILLION network forward passes per
    # baseline build - roughly an hour of CPU, which is why this file's
    # end-to-end tests had to be excluded from the suite despite their
    # docstring promising "reasonable CI time". Defaults here match
    # production; the tests set them small.
    n_per_template: int = 200
    n_pure_random: int = 500
    n_diffusion_steps: int = 1000
    batch_size: int = 16
    device: str = "cpu"
    seed: int = 42


class SNNWalkForwardBacktester:
    """
    Mirrors kronos/backtest.py's WalkForwardBacktester's window logic
    exactly (same train_window/test_window/no-look-ahead index
    arithmetic), but drives prometheus.engine.PrometheusEngine's real
    training procedure and kronos/reflex.py's real ReflexArc for
    evaluation, instead of KronosStrategy's NEAT+MAML stand-in.
    """

    def __init__(
        self,
        closes: pd.DataFrame,
        tickers: List[str],
        config: Optional[WalkForwardConfig] = None,
        train_cfg: Optional[SNNTrainConfig] = None,
    ):
        self.closes = closes.dropna(how="any")
        self.returns = self.closes.pct_change().dropna()
        self.tickers = tickers
        self.cfg = config or WalkForwardConfig()
        self.train_cfg = train_cfg or SNNTrainConfig()
        if len(self.returns) < self.cfg.train_window + self.cfg.test_window:
            raise ValueError(
                f"Need >= {self.cfg.train_window + self.cfg.test_window} bars, "
                f"got {len(self.returns)}"
            )

    def windows(self) -> List[Tuple[int, int, int]]:
        """Identical logic to kronos/backtest.py's
        WalkForwardBacktester.windows() - duplicated rather than shared
        because that class is keyed on a Strategy interface this harness
        deliberately does not use."""
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

    # -- one-time shared baseline (pretrain + meta) ------------------------

    def _build_baseline_checkpoint(self, checkpoint_dir: str) -> None:
        """Runs pretrain then meta ONCE and saves the result - both stages
        train on synthetic data independent of any walk-forward window,
        see this module's docstring for why that makes sharing correct
        (not a shortcut)."""
        from prometheus.engine import PrometheusEngine, PrometheusConfig

        tc = self.train_cfg
        n_assets = len(self.tickers)
        torch.manual_seed(tc.seed)
        np.random.seed(tc.seed)

        cfg = PrometheusConfig(
            n_assets=n_assets, seq_len=tc.seq_len, horizon=tc.horizon,
            d_model=tc.d_model, n_heads=tc.n_heads, n_layers=tc.n_layers,
            device=tc.device, output_dir=tempfile.mkdtemp(prefix="snn_bt_pretrain_"),
            snn_layer_sizes=[32, 16], snn_output_size=n_assets,
            n_diffusion_steps=tc.n_diffusion_steps,
        )
        engine = PrometheusEngine(cfg)
        logger.info("[backtest_snn] pretrain: %d epochs (one-time, shared)", tc.pretrain_epochs)

        # LOOK-AHEAD, FIXED. train_on_black_swans fits the diffusion
        # ScoreNetwork on real returns, and when none are supplied it
        # falls back to MarketDataFetcher.fetch_all() - which pulls
        # `datetime.now() - lookback_days` through `datetime.now()`.
        #
        # In a walk-forward that is future data. This baseline is built
        # ONCE and shared by every window, so a score net fitted on bars
        # up to today shapes the black-swan library that pretrains the
        # SNN that then trades 2024 test windows. Indirect, but it is
        # future information reaching the model that trades - the exact
        # class of defect this harness exists to detect.
        #
        # Windows advance forward from 0, so bars [0, train_window) sit
        # before EVERY test window and are the only causally safe source
        # for a shared artifact. Passing them explicitly also stops the
        # network fetch, which made these tests non-hermetic (they hung
        # on yfinance behind a proxy) and non-deterministic (the fitted
        # data changed with the calendar).
        warmup = self.returns.iloc[: self.cfg.train_window]
        engine.train_on_black_swans(
            n_scenarios=tc.n_black_swans, n_epochs=tc.pretrain_epochs,
            batch_size=tc.batch_size,
            n_per_template=tc.n_per_template, n_pure_random=tc.n_pure_random,
            real_returns=warmup.values.astype(float),
        )
        logger.info("[backtest_snn] meta: %d epochs/regime (one-time, shared)", tc.meta_epochs)
        self._run_meta(engine, tc)
        engine.save(checkpoint_dir)

    @staticmethod
    def _run_meta(engine, tc: SNNTrainConfig) -> None:
        """Reproduces scripts/train.py's run_meta_training() exactly -
        synthetic regime tasks (trending/mean_reverting/volatile/crash/
        recovery), no real market data, hence safe to share across every
        walk-forward window."""
        n, seq, h = engine.config.n_assets, tc.seq_len, tc.horizon
        tasks = []
        rng = np.random.default_rng(tc.seed)
        for regime_type in ["trending", "mean_reverting", "volatile", "crash", "recovery"]:
            for _ in range(tc.meta_epochs):
                if regime_type == "trending":
                    drift = rng.uniform(0.001, 0.003)
                    data = np.cumsum(rng.normal(drift, 0.01, (seq + h, n)), axis=0)
                elif regime_type == "mean_reverting":
                    data = rng.normal(0, 0.01, (seq + h, n))
                    for t in range(1, seq + h):
                        data[t] = 0.9 * data[t - 1] + rng.normal(0, 0.005, n)
                elif regime_type == "volatile":
                    data = rng.normal(0, 0.05, (seq + h, n))
                elif regime_type == "crash":
                    data = rng.normal(-0.02, 0.03, (seq + h, n))
                else:
                    data = rng.normal(0.01, 0.02, (seq + h, n))
                x = torch.tensor(data[:seq], dtype=torch.float32).unsqueeze(0).to(tc.device)
                y = torch.tensor(data[seq:], dtype=torch.float32).unsqueeze(0).to(tc.device)
                tasks.append(((x, y), (x, y)))

        for i in range(0, len(tasks), 5):
            batch_tasks = tasks[i:i + 5]
            engine.maml.meta_train_step(
                tasks=batch_tasks, loss_fn=lambda p, t: engine.loss_fn(p, t),
            )

    # -- per-window finetune -------------------------------------------------

    def _finetune_window(self, checkpoint_dir: str, train_returns: np.ndarray):
        """Fresh engine (fresh, zero-state optimizers) loaded from the
        shared pretrain+meta baseline, finetuned ONLY on this window's
        real training data - no look-ahead, no cross-window state."""
        from prometheus.engine import PrometheusEngine, PrometheusConfig

        tc = self.train_cfg
        n_assets = len(self.tickers)
        cfg = PrometheusConfig(
            n_assets=n_assets, seq_len=tc.seq_len, horizon=tc.horizon,
            d_model=tc.d_model, n_heads=tc.n_heads, n_layers=tc.n_layers,
            device=tc.device, output_dir=tempfile.mkdtemp(prefix="snn_bt_window_"),
            snn_layer_sizes=[32, 16], snn_output_size=n_assets,
            n_diffusion_steps=tc.n_diffusion_steps,
        )
        engine = PrometheusEngine(cfg)
        engine.load(checkpoint_dir)   # weights only - optimizers stay fresh

        n_bars = train_returns.shape[0]
        for _epoch in range(tc.finetune_epochs):
            for start in range(0, n_bars - tc.seq_len - tc.horizon, tc.seq_len):
                end = start + tc.seq_len
                x_np = train_returns[start:end]
                y_np = train_returns[end:end + tc.horizon]
                if x_np.shape[0] < 4 or y_np.shape[0] < 1:
                    continue
                x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(tc.device)
                y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(0).to(tc.device)
                if y.dim() == 2:
                    y = y.unsqueeze(0)
                engine.train_step(x, y)
                engine.train_snn_step(x, y[:, 0, :])

        return copy.deepcopy(engine.snn.state_dict())

    # -- per-window evaluation via the REAL ReflexArc ------------------------

    def _evaluate_window(
        self, snn_state_dict: dict, train_returns: np.ndarray,
        rets: np.ndarray, e: int, te: int,
    ) -> Tuple[List[float], List[float], Dict[str, List[float]], Dict[str, List[float]]]:
        from kronos.config import load_config
        from kronos.reflex import ReflexArc

        cfg = load_config()
        cfg.override("data.tickers", self.tickers)
        cfg.override("nightmare.horizon_days", self.train_cfg.horizon)

        reflex = ReflexArc(cfg)
        reflex.snn.load_state_dict(snn_state_dict)
        reflex.snn.eval()
        # Matches KronosOrchestrator.maybe_adopt_runpod_checkpoint()'s real
        # post-adoption call - train-only data, no look-ahead.
        reflex.calibrate_size_scale(train_returns)

        preds: List[float] = []
        actuals: List[float] = []
        ticker_preds: Dict[str, List[float]] = {t: [] for t in self.tickers}
        ticker_actuals: Dict[str, List[float]] = {t: [] for t in self.tickers}

        h = self.train_cfg.horizon
        for t in range(e, te):
            window = rets[t - h:t]
            if window.shape[0] < h:
                pad = np.zeros((h - window.shape[0], window.shape[1]))
                window = np.vstack([pad, window])
            decision = reflex.infer(window, vix_value=20.0)
            for i, ticker in enumerate(self.tickers):
                if i >= len(decision.signals):
                    continue
                preds.append(float(decision.signals[i]))
                actuals.append(float(rets[t, i]))
                ticker_preds[ticker].append(float(decision.signals[i]))
                ticker_actuals[ticker].append(float(rets[t, i]))

        return preds, actuals, ticker_preds, ticker_actuals

    # -- full run --------------------------------------------------------

    def run_signal_diagnostic(
        self, max_windows: Optional[int] = None,
    ) -> SignalDiagnostic:
        """max_windows: for a cheap sanity check on a subset before
        committing to the full walk-forward run (e.g. 10 of 125)."""
        with tempfile.TemporaryDirectory(prefix="snn_bt_baseline_") as baseline_dir:
            self._build_baseline_checkpoint(baseline_dir)

            all_preds: List[float] = []
            all_actuals: List[float] = []
            ticker_preds: Dict[str, List[float]] = {t: [] for t in self.tickers}
            ticker_actuals: Dict[str, List[float]] = {t: [] for t in self.tickers}

            rets = self.returns.values
            spans = self.windows()
            if max_windows is not None:
                spans = spans[:max_windows]

            for i, (s, e, te) in enumerate(spans):
                logger.info("[backtest_snn] window %d/%d (train=[%d,%d) test=[%d,%d))",
                           i + 1, len(spans), s, e, e, te)
                train_returns = rets[s:e]
                snn_state = self._finetune_window(baseline_dir, train_returns)
                p, a, tp, ta = self._evaluate_window(snn_state, train_returns, rets, e, te)
                all_preds.extend(p)
                all_actuals.extend(a)
                for tkr in self.tickers:
                    ticker_preds[tkr].extend(tp[tkr])
                    ticker_actuals[tkr].extend(ta[tkr])

        return _compute_signal_diagnostic(
            "snn", all_preds, all_actuals, ticker_preds, ticker_actuals,
        )
