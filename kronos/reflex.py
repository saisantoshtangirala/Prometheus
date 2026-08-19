"""
Kronos Reflex Arc - market-hours mode (09:30 - 16:00 EST).

During trading hours the heavy transformer stack is bypassed entirely.
A lightweight Spiking Neural Network (existing SpikingMarketEncoder)
handles sub-second inference, an order-book imbalance simulator watches
microstructure, and a rolling-statistics regime gate (HMM-style two-state
switch on VIX) instantly kills new long exposure when volatility spikes
beyond `vix_spike_sigma` standard deviations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Optional
from collections import deque

import numpy as np
import torch

from prometheus.neuro.neuromodulation import CortisolSystem
from prometheus.neuro.spiking_network import SpikingMarketEncoder

logger = logging.getLogger(__name__)

FLASH_CRASH_DROP_PCT = 0.20    # single-tick drop that triggers per-asset lockout

# -- Size calibration (see calibrate_size_scale below) ----------------------
DEFAULT_SIZE_SCALE = 1.0       # safe fallback: today's plain tanh(pred), no scaling
MIN_CALIBRATION_SAMPLES = 100  # pooled (bars * assets) - below this, trust nothing
MAX_SIZE_SCALE = 40.0          # hard ceiling regardless of what the fit says


# ---------------------------------------------------------------------------
# Regime Switch Gate (two-state volatility HMM approximation)
# ---------------------------------------------------------------------------

@dataclass
class GateState:
    regime: str = "calm"                  # "calm" | "panic"
    position_cap: float = 1.0
    lockout_until: Optional[datetime] = None
    last_vix: float = 0.0
    zscore: float = 0.0


class RegimeSwitchGate:
    """
    Monitors VIX with a rolling window. A print more than `spike_sigma`
    standard deviations above the rolling mean flips the gate to "panic":
    position_cap -> 0.0 for `lockout_minutes`. New longs are impossible
    while the cap is zero; existing positions may still be closed.
    """

    def __init__(self, config):
        self.cfg = config
        self.window: Deque[float] = deque(maxlen=int(config.reflex.vix_window))
        self.state = GateState()

    def update(self, vix_value: float, now: Optional[datetime] = None) -> GateState:
        now = now or datetime.now(timezone.utc)
        self.state.last_vix = float(vix_value)

        # Expire an elapsed lockout
        if (
            self.state.lockout_until is not None
            and now >= self.state.lockout_until
        ):
            logger.info("[reflex] lockout expired - gate back to calm")
            self.state.regime = "calm"
            self.state.position_cap = 1.0
            self.state.lockout_until = None

        if len(self.window) >= 2:
            mean = float(np.mean(self.window))
            std = float(np.std(self.window)) + 1e-9
            z = (vix_value - mean) / std
            self.state.zscore = z
            if z > float(self.cfg.reflex.vix_spike_sigma):
                lockout = timedelta(minutes=int(self.cfg.reflex.lockout_minutes))
                self.state.regime = "panic"
                self.state.position_cap = 0.0
                self.state.lockout_until = now + lockout
                logger.warning(
                    "[reflex] VIX spike z=%.2f > %.1f sigma - "
                    "position_cap=0.0 for %d minutes",
                    z, self.cfg.reflex.vix_spike_sigma,
                    self.cfg.reflex.lockout_minutes,
                )

        self.window.append(float(vix_value))
        return self.state

    @property
    def position_cap(self) -> float:
        return self.state.position_cap

    def allows_new_longs(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if self.state.lockout_until is not None and now < self.state.lockout_until:
            return False
        return self.state.position_cap > 0.0


# ---------------------------------------------------------------------------
# Order-book imbalance simulator
# ---------------------------------------------------------------------------

@dataclass
class MicrostructureSignal:
    imbalance: float          # [-1, 1]: bid pressure minus ask pressure
    vwap_deviation: float     # last price deviation from session VWAP
    spread_bps: float
    alert: bool


class OrderBookSimulator:
    """
    Simulates Level-2 imbalance from OHLCV bars: VWAP tracking plus a
    volume-signed spread proxy. Real L2 data would slot in here unchanged.
    """

    def __init__(self, config):
        self.cfg = config
        self._cum_pv: Dict[str, float] = {}
        self._cum_v: Dict[str, float] = {}

    def reset_session(self) -> None:
        self._cum_pv.clear()
        self._cum_v.clear()

    def update(
        self, ticker: str, price: float, volume: float,
        prev_price: Optional[float] = None,
    ) -> MicrostructureSignal:
        self._cum_pv[ticker] = self._cum_pv.get(ticker, 0.0) + price * volume
        self._cum_v[ticker] = self._cum_v.get(ticker, 0.0) + volume
        vwap = self._cum_pv[ticker] / (self._cum_v[ticker] + 1e-9)

        vwap_dev = (price - vwap) / (vwap + 1e-9)
        # Signed-volume imbalance proxy: direction of the last move scaled
        # by how unusual the print volume is for the session so far.
        n_prints = max(len(self._cum_v), 1)
        avg_vol = self._cum_v[ticker] / n_prints
        direction = 0.0
        if prev_price is not None and prev_price > 0:
            direction = float(np.sign(price - prev_price))
        imbalance = float(np.tanh(direction * volume / (avg_vol + 1e-9)))

        spread_bps = float(min(50.0, 10000.0 / (volume ** 0.5 + 1.0)))
        alert = abs(imbalance) > float(self.cfg.reflex.imbalance_threshold)
        return MicrostructureSignal(
            imbalance=imbalance,
            vwap_deviation=float(vwap_dev),
            spread_bps=spread_bps,
            alert=alert,
        )


# ---------------------------------------------------------------------------
# The Reflex Arc
# ---------------------------------------------------------------------------

@dataclass
class ReflexDecision:
    signals: np.ndarray            # per-asset signal in [-1, 1]
    position_cap: float
    regime: str
    latency_ms: float
    microstructure_alerts: Dict[str, MicrostructureSignal] = field(
        default_factory=dict
    )
    asset_caps: Dict[str, float] = field(default_factory=dict)
    fallback_mode: bool = False    # REF-04: SNN OOM -> lookup-table signals
    confidence_blended: bool = False   # daily_bias was available and applied


class ReflexArc:
    """
    Market-hours inference path. NO training happens here - the SNN weights
    were frozen at 06:00 when the daily cycle finished.

    Two independent kill-switches:
      - RegimeSwitchGate: market-wide VIX 2-sigma spike -> global cap 0.0
      - Per-asset cortisol: a single ticker crashing >FLASH_CRASH_DROP_PCT in
        one tick locks THAT asset only (existing per-asset CortisolSystem),
        so one stock's flash crash does not freeze the whole book.
    """

    def __init__(self, config, snn: Optional[SpikingMarketEncoder] = None):
        self.cfg = config
        n_assets = len(config.data.tickers)
        self.snn = snn or SpikingMarketEncoder(
            input_size=n_assets,
            layer_sizes=[32, 16],
            output_size=n_assets,
        )
        self.snn.eval()
        self.gate = RegimeSwitchGate(config)
        self.order_book = OrderBookSimulator(config)
        # Per-asset flash-crash lockout, bar-based (existing Prometheus system)
        lockout_bars = int(config.reflex.lockout_minutes)  # 1 bar = 1 minute
        self.cortisol = CortisolSystem(
            hidden_size=16, lockout_duration=lockout_bars
        )
        self._prev_prices: Dict[str, float] = {}
        # Set by KronosOrchestrator once per checkpoint adoption via
        # set_daily_bias() - kronos/bias_estimator.py's once-per-day
        # causal_transformer forecast, checked against the SNN's own
        # per-tick signal below as a confidence modifier. None until the
        # first adoption computes one; infer() degrades to SNN-only when
        # unset, same behavior as before this existed.
        self._daily_bias: Optional[np.ndarray] = None
        # Set by KronosOrchestrator once per checkpoint adoption via
        # calibrate_size_scale() - see that method for why this exists.
        # 1.0 (no scaling, today's already-verified-non-saturating
        # tanh(pred)) until the first successful calibration.
        self._size_scale: float = DEFAULT_SIZE_SCALE

    # -- per-asset flash-crash handling (REF-01 / REF-02) -------------------

    def _check_flash_crashes(self, bar_prices: Dict[str, float]) -> None:
        for ticker, price in bar_prices.items():
            prev = self._prev_prices.get(ticker)
            if prev is not None and prev > 0:
                change = (price - prev) / prev
                if change < -FLASH_CRASH_DROP_PCT:
                    logger.warning(
                        "[reflex] FLASH CRASH %s: %.1f%% in one tick - "
                        "asset locked for %d bars",
                        ticker, change * 100.0,
                        self.cortisol.lockout_duration,
                    )
                    self.cortisol.trigger_flash_crash_lockout(asset=ticker)
            self._prev_prices[ticker] = price

    def set_daily_bias(self, bias: Optional[np.ndarray]) -> None:
        """Called by KronosOrchestrator after each checkpoint adoption
        with kronos/bias_estimator.py's output (or None if it couldn't
        be computed this time - clears any stale prior-day bias rather
        than silently keeping it)."""
        self._daily_bias = bias

    def calibrate_size_scale(self, recent_returns: np.ndarray) -> None:
        """Once-per-checkpoint-adoption recalibration of the raw SNN
        prediction -> position-size scale.

        Earlier approach considered and REJECTED after empirical testing:
        normalizing each prediction by the rolling std of its OWN recent
        history. That's mathematically scale-invariant - fed 400 ticks of
        pure white noise (zero real information by construction), it still
        produced avg 57% of cap conviction, 15% of calls >90% of cap,
        identical regardless of the noise's absolute scale (x1 vs x10).
        It can't distinguish a genuine signal from noise because it only
        ever compares the signal to itself.

        This version instead fits against something external: the SNN's
        OWN realized track record. Pools (raw_pred_t, actual_return_{t+1})
        pairs across every asset and the trailing lookback window, and
        fits scale = cov(pred, actual) / var(pred) - a pooled OLS slope.
        Simulated (see scratchpad probe, 500 replicates per case, pooled
        n=250 matching this project's real lookback_days=30/horizon_days=5/
        10-ticker universe): a pred stream with zero genuine correlation to
        outcomes fits scale with median ~0.4 (never saturates, and does NOT
        reproduce the old bug when pred's absolute scale is 10x bigger -
        it's anchored to realized outcomes, not to itself). A pred stream
        with genuine correlation (~0.15) fits scale ~3.0, and stronger
        correlation (~0.40) fits ~8.0 - it responds monotonically to real
        signal strength instead of being flat/saturated like the rejected
        approach.

        Floors at 0.0 (a confirmed non-positive relationship falls back to
        the safe default rather than inverting signal direction on a live
        system), caps at MAX_SIZE_SCALE, and falls back to
        DEFAULT_SIZE_SCALE whenever there isn't enough history or pred has
        ~zero variance to regress against. Never raises.
        """
        try:
            horizon = int(self.cfg.nightmare.horizon_days)
            T, n_assets = recent_returns.shape
            preds, actuals = [], []
            with torch.no_grad():
                for t in range(horizon, T - 1):
                    window = recent_returns[t - horizon:t]
                    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                    out = self.snn(x)
                    pred = out[0] if isinstance(out, tuple) else out
                    p = pred.squeeze(0).numpy()
                    if p.ndim > 1:
                        p = p[-1]
                    preds.append(p)
                    actuals.append(recent_returns[t + 1])
            if not preds:
                self._size_scale = DEFAULT_SIZE_SCALE
                return
            p_arr = np.asarray(preds).ravel()
            a_arr = np.asarray(actuals).ravel()
            if p_arr.size < MIN_CALIBRATION_SAMPLES:
                self._size_scale = DEFAULT_SIZE_SCALE
                return
            var_p = float(p_arr.var())
            if var_p < 1e-12:
                self._size_scale = DEFAULT_SIZE_SCALE
                return
            scale = float(np.cov(p_arr, a_arr, bias=True)[0, 1] / var_p)
            scale = max(0.0, min(scale, MAX_SIZE_SCALE))
            self._size_scale = scale if scale > 0.0 else DEFAULT_SIZE_SCALE
            logger.info("[reflex] recalibrated size scale: %.3f (n=%d)", self._size_scale, p_arr.size)
        except Exception as e:
            logger.warning("[reflex] size-scale calibration failed (%s) - keeping %.3f", e, self._size_scale)

    def asset_position_cap(self, ticker: str) -> float:
        """0.0 while the asset's flash-crash lockout is active, else gate cap."""
        if self.cortisol._asset_fear.get(ticker, False):
            return 0.0
        return self.gate.position_cap

    def infer(
        self,
        recent_returns: np.ndarray,     # [T, n_assets] most recent bars
        vix_value: float,
        bar_prices: Optional[Dict[str, float]] = None,
        bar_volumes: Optional[Dict[str, float]] = None,
        now: Optional[datetime] = None,
    ) -> ReflexDecision:
        """One low-latency inference tick. Budget: reflex.inference_budget_ms."""
        t0 = time.perf_counter()

        # 1. Kill-switches first - a panic print must never wait on the SNN.
        gate_state = self.gate.update(vix_value, now=now)
        if bar_prices:
            self._check_flash_crashes(bar_prices)
        self.cortisol.step_asset_lockouts()      # REF-02: decrement per bar

        # 2. SNN forward (frozen weights); REF-04: OOM falls back to a
        #    momentum lookup instead of killing market-hour operations.
        fallback_mode = False
        try:
            x = torch.tensor(recent_returns, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = self.snn(x)
            pred = out[0] if isinstance(out, tuple) else out
            # size_scale: see calibrate_size_scale() - defaults to 1.0
            # (identical to plain tanh(pred)) until the first successful
            # calibration.
            signals = torch.tanh(pred.squeeze(0) * self._size_scale).numpy()
            if signals.ndim > 1:
                signals = signals[-1]
        except (MemoryError, torch.cuda.OutOfMemoryError) as e:
            logger.critical(
                "[reflex] SNN OOM (%s) - switching to momentum lookup "
                "fallback", type(e).__name__,
            )
            fallback_mode = True
            # Pre-computed lookup: sign of recent mean return, half strength
            signals = np.tanh(
                np.asarray(recent_returns).mean(axis=0) * 50.0
            ) * 0.5

        # 2b. Confidence blend against the daily bias (kronos/bias_estimator.py's
        #     once-per-adoption causal_transformer forecast) - the "second
        #     opinion" ReflexArc had no way to check itself against before.
        #     Skipped in fallback_mode (the OOM lookup-table signal isn't
        #     really comparable) and whenever no bias is available yet
        #     (before the first adoption computes one, or the checkpoint's
        #     arch.json/weights were missing - see compute_daily_bias).
        #     Disagreement dampens rather than vetoes: the SNN is still the
        #     faster, primary signal, and a slower end-of-day forecast
        #     disagreeing with a fresh tick isn't grounds to silence it.
        confidence_blended = False
        if (
            not fallback_mode
            and self._daily_bias is not None
            and len(self._daily_bias) == len(signals)
        ):
            agree = np.sign(signals) == np.sign(self._daily_bias)
            scale = np.where(agree, 1.15, 0.7)
            signals = np.clip(signals * scale, -1.0, 1.0)
            confidence_blended = True

        # 3. Microstructure scan
        alerts: Dict[str, MicrostructureSignal] = {}
        if bar_prices:
            for ticker, price in bar_prices.items():
                vol = (bar_volumes or {}).get(ticker, 0.0)
                sig = self.order_book.update(ticker, price, vol)
                if sig.alert:
                    alerts[ticker] = sig

        # 4. Apply the global gate: cap of 0 zeroes any NEW long signal
        if gate_state.position_cap == 0.0:
            signals = np.minimum(signals, 0.0)   # longs killed, exits allowed

        # 5. Per-asset caps (flash-crashed tickers get 0.0)
        tickers = list(self.cfg.data.tickers)
        asset_caps = {t: self.asset_position_cap(t) for t in tickers}
        for i, ticker in enumerate(tickers):
            if i < len(signals) and asset_caps.get(ticker, 1.0) == 0.0:
                signals[i] = min(signals[i], 0.0)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        if latency_ms > float(self.cfg.reflex.inference_budget_ms):
            logger.warning(
                "[reflex] inference took %.1f ms > %.0f ms budget",
                latency_ms, self.cfg.reflex.inference_budget_ms,
            )

        return ReflexDecision(
            signals=signals,
            position_cap=gate_state.position_cap,
            regime=gate_state.regime,
            latency_ms=latency_ms,
            microstructure_alerts=alerts,
            asset_caps=asset_caps,
            fallback_mode=fallback_mode,
            confidence_blended=confidence_blended,
        )
