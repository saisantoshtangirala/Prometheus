"""
Order Book Simulator – Level-2 data from trade flow reconstruction.

Since Level-2 order book data is expensive, this module reconstructs
approximate order book imbalance from tick data using:
  - Trade size distribution analysis
  - Bid-ask spread inference from OHLCV
  - Volume-weighted direction inference (VWAP-based)
  - Toxic flow detection (adverse selection)

The Order Flow Imbalance (OFI) is the strongest microstructure predictor
of short-term price moves (Cont et al., 2014).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class OrderFlowImbalanceCalculator:
    """
    Computes Order Flow Imbalance (OFI) from trade data.

    OFI = (bid_size_delta - ask_size_delta) / (bid_size_delta + ask_size_delta)
    Positive OFI → buy pressure → price tends to rise
    Negative OFI → sell pressure → price tends to fall

    Uses Lee-Ready algorithm to classify trades as buyer/seller initiated.
    """

    def __init__(self, window: int = 20, decay: float = 0.95):
        self.window = window
        self.decay = decay
        self._trade_buffer: deque = deque(maxlen=window)
        self._ofi_history: List[float] = []

    def classify_trade(
        self,
        price: float,
        size: float,
        midpoint: float,
        prev_midpoint: float,
    ) -> int:
        """
        Lee-Ready (1991) trade classification.
        Returns: +1 (buyer initiated), -1 (seller initiated), 0 (unknown).
        """
        if price > midpoint:
            return 1   # buyer initiated: price above mid
        if price < midpoint:
            return -1  # seller initiated: price below mid
        # Tick rule when at midpoint
        if midpoint > prev_midpoint:
            return 1
        if midpoint < prev_midpoint:
            return -1
        return 0

    def update(
        self,
        trades: pd.DataFrame,  # columns: price, size, bid, ask
    ) -> float:
        """
        Process a batch of trades and compute current OFI.
        Returns OFI in [-1, 1].
        """
        if trades.empty:
            return 0.0

        signed_volumes = []
        prev_mid = (trades["bid"].iloc[0] + trades["ask"].iloc[0]) / 2 if "bid" in trades.columns else trades["price"].iloc[0]

        for _, row in trades.iterrows():
            mid = (row.get("bid", row["price"]) + row.get("ask", row["price"])) / 2
            direction = self.classify_trade(row["price"], row["size"], mid, prev_mid)
            signed_volumes.append(direction * row["size"])
            prev_mid = mid

        buy_vol = sum(v for v in signed_volumes if v > 0)
        sell_vol = abs(sum(v for v in signed_volumes if v < 0))
        total_vol = buy_vol + sell_vol

        ofi = (buy_vol - sell_vol) / (total_vol + 1e-8)
        self._trade_buffer.append(ofi)
        self._ofi_history.append(ofi)
        return ofi

    def get_cumulative_ofi(self, window: Optional[int] = None) -> float:
        """Exponentially weighted cumulative OFI."""
        buf = list(self._trade_buffer)
        if not buf:
            return 0.0
        w = self.window if window is None else window
        weights = np.array([self.decay ** (w - 1 - i) for i in range(min(len(buf), w))])
        recent = np.array(buf[-w:])
        return float(np.average(recent, weights=weights[:len(recent)]))


class ToxicFlowDetector:
    """
    Detects adverse selection / toxic order flow — institutional informed trading.
    High toxicity = smart money is taking liquidity aggressively → follow them.
    """

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self._price_impact: List[float] = []
        self._ofi: List[float] = []

    def compute_toxicity(
        self,
        ofi_sequence: List[float],
        price_returns: List[float],
        window: int = 20,
    ) -> Dict:
        """
        Estimate probability of informed trading (PIN) via:
        - Price impact of unit order flow (high = toxic/informed)
        - Contemporaneous OFI-return correlation

        High PIN → institutions are moving; follow the flow, not the price.
        """
        if len(ofi_sequence) < window or len(price_returns) < window:
            return {"pin_estimate": 0.5, "toxicity_level": "UNKNOWN"}

        ofi = np.array(ofi_sequence[-window:])
        ret = np.array(price_returns[-window:])

        # Price impact = regression slope of return on OFI
        if ofi.std() < 1e-8:
            price_impact = 0.0
        else:
            price_impact = float(np.corrcoef(ofi, ret)[0, 1])

        # Amihud illiquidity-like measure
        if len(ret) > 0:
            illiquidity = float(np.mean(np.abs(ret) / (np.abs(ofi) + 1e-4)))
        else:
            illiquidity = 0.0

        # Simple PIN estimate: corr between signed flow and future returns
        if len(ret) > 1:
            flow_predictability = float(np.corrcoef(ofi[:-1], ret[1:])[0, 1])
        else:
            flow_predictability = 0.0

        pin = abs(flow_predictability)

        return {
            "pin_estimate": float(pin),
            "price_impact": float(price_impact),
            "illiquidity": float(illiquidity),
            "flow_predictability": float(flow_predictability),
            "toxicity_level": (
                "HIGH" if pin > 0.5 else ("MODERATE" if pin > 0.25 else "LOW")
            ),
            "follow_signal": float(np.sign(flow_predictability) * pin),
        }


class OrderBookSimulator:
    """
    Master order book interface: combines OFI, toxicity, and spread estimation.
    Provides the ORDER_FLOW_IMBALANCE feature for the causal DAG.
    """

    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.calculators = [OrderFlowImbalanceCalculator() for _ in range(n_assets)]
        self.toxicity_detectors = [ToxicFlowDetector() for _ in range(n_assets)]
        self._ofi_history: List[np.ndarray] = []
        self._return_history: List[np.ndarray] = []

    def update_from_ohlcv(
        self,
        ohlcv: pd.DataFrame,  # one bar: [n_assets] with Open/High/Low/Close/Volume
    ) -> np.ndarray:
        """
        Reconstruct approximate OFI from OHLCV bar.
        Uses: close vs open direction, wick asymmetry, and volume.
        """
        n = self.n_assets
        ofi = np.zeros(n)

        for i in range(min(n, len(ohlcv))):
            row = ohlcv.iloc[i] if isinstance(ohlcv, pd.DataFrame) else {}
            o = float(row.get("Open", 100))
            h = float(row.get("High", 101))
            l = float(row.get("Low", 99))
            c = float(row.get("Close", 100))
            v = float(row.get("Volume", 1_000_000))

            # Direction: close vs open
            direction = np.sign(c - o)

            # Wick asymmetry: long upper wick = selling pressure
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            wick_imbalance = (lower_wick - upper_wick) / (upper_wick + lower_wick + 1e-8)

            # Volume-weighted OFI approximation
            ofi[i] = float(direction * 0.6 + wick_imbalance * 0.4)

        self._ofi_history.append(ofi)
        return ofi

    def get_feature_vector(self, asset_idx: int) -> np.ndarray:
        """Return [ofi, cum_ofi, toxicity_pin] for one asset."""
        if len(self._ofi_history) < 5:
            return np.zeros(3)
        recent_ofi = [h[asset_idx] for h in self._ofi_history[-20:]]
        ofi_now = recent_ofi[-1]
        cum_ofi = np.average(recent_ofi, weights=np.exp(np.linspace(-2, 0, len(recent_ofi))))
        if len(self._return_history) >= 5:
            recent_ret = [h[asset_idx] for h in self._return_history[-20:]]
            tox = self.toxicity_detectors[asset_idx].compute_toxicity(recent_ofi, recent_ret)
            pin = tox["pin_estimate"]
        else:
            pin = 0.5
        return np.array([ofi_now, cum_ofi, pin], dtype=np.float32)
