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

from prometheus.neuro.spiking_network import SpikingMarketEncoder

logger = logging.getLogger(__name__)


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


class ReflexArc:
    """
    Market-hours inference path. NO training happens here - the SNN weights
    were frozen at 06:00 when the daily cycle finished.
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

        # 1. Regime gate first - a panic print must never wait on the SNN.
        gate_state = self.gate.update(vix_value, now=now)

        # 2. SNN forward (frozen weights)
        x = torch.tensor(recent_returns, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = self.snn(x)
        pred = out[0] if isinstance(out, tuple) else out
        signals = torch.tanh(pred.squeeze(0)).numpy()
        if signals.ndim > 1:
            signals = signals[-1]

        # 3. Microstructure scan
        alerts: Dict[str, MicrostructureSignal] = {}
        if bar_prices:
            for ticker, price in bar_prices.items():
                vol = (bar_volumes or {}).get(ticker, 0.0)
                sig = self.order_book.update(ticker, price, vol)
                if sig.alert:
                    alerts[ticker] = sig

        # 4. Apply the gate: cap of 0 zeroes any NEW long signal
        if gate_state.position_cap == 0.0:
            signals = np.minimum(signals, 0.0)   # longs killed, exits allowed

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
        )
