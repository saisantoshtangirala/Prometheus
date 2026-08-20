"""
NSE-realistic position sizing for a small account.

This module exists because of one number in the brief: ₹10,000 of
starting capital. That is roughly $120, and it invalidates the
fractional-weight arithmetic every backtest in this repository currently
uses. Concretely, on a typical NSE large-cap board:

    TCS        ~ ₹3,000/share  -> a 10% weight (₹1,000) buys ZERO shares
    RELIANCE   ~ ₹1,300/share  -> a 10% weight buys ZERO shares
    ITC        ~ ₹  270/share  -> a 10% weight buys 3 shares (₹810),
                                  i.e. 8.1% actual, not 10%

NSE's cash segment has no fractional shares. So a backtest that assumes
w=0.10 is executable is reporting the performance of a portfolio that
cannot be held. At ₹10,000 across 10 names the quantisation error is not
a rounding detail - it dominates. `IntegerShareSizer` makes it explicit
rather than silently optimistic.

COSTS (delivery / CNC, the realistic mode for a daily-rebalanced book;
figures are the standard discount-broker schedule):

    brokerage        ₹0 on delivery at most discount brokers
    STT              0.1% on BUY + 0.1% on SELL
    exchange txn     ~0.00297% both sides
    SEBI charges     ~0.0001% both sides
    stamp duty       0.015% on BUY only
    GST              18% on (brokerage + exchange txn + SEBI)

    => round trip ~ 0.22%, i.e. ~22bp, dominated by STT.

That is the single most important number for this system: a strategy
rebalancing daily at 100% turnover pays ~22bp/day ~ 55%/year in costs.
No 55%-win-rate edge survives that. The sizer therefore also exposes
`min_edge_bps_to_trade`, so the caller can refuse trades whose expected
edge does not clear their own cost - which is the only sane way to run a
₹10,000 account.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# Cost components as fractions of traded notional.
STT_BUY = 0.001
STT_SELL = 0.001
EXCHANGE_TXN = 0.0000297
SEBI_CHARGES = 0.000001
STAMP_DUTY_BUY = 0.00015
GST_RATE = 0.18
BROKERAGE_DELIVERY = 0.0


@dataclass
class NSECostModel:
    """Per-side transaction cost as a fraction of notional."""

    brokerage: float = BROKERAGE_DELIVERY
    stt_buy: float = STT_BUY
    stt_sell: float = STT_SELL
    exchange_txn: float = EXCHANGE_TXN
    sebi: float = SEBI_CHARGES
    stamp_buy: float = STAMP_DUTY_BUY
    gst_rate: float = GST_RATE

    def cost_fraction(self, side: str) -> float:
        """Total cost as a fraction of notional for one side."""
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        stt = self.stt_buy if side == "buy" else self.stt_sell
        stamp = self.stamp_buy if side == "buy" else 0.0
        taxable = self.brokerage + self.exchange_txn + self.sebi
        return stt + stamp + taxable + self.gst_rate * taxable

    def round_trip_fraction(self) -> float:
        return self.cost_fraction("buy") + self.cost_fraction("sell")

    def round_trip_bps(self) -> float:
        return self.round_trip_fraction() * 10_000.0

    def cost_of(self, notional: float, side: str) -> float:
        return abs(float(notional)) * self.cost_fraction(side)


@dataclass
class SizingResult:
    """Outcome of turning target weights into executable share counts."""

    shares: np.ndarray             # integer share counts (signed)
    realised_weights: np.ndarray   # weights ACTUALLY achievable
    target_weights: np.ndarray     # what was asked for
    trade_shares: np.ndarray       # shares to trade this bar
    cost: float                    # total cost of this rebalance, in rupees
    deployed_capital: float
    n_untradeable: int             # names whose target rounded to zero

    @property
    def quantisation_error(self) -> float:
        """L1 gap between intended and achievable weights.

        The headline number for a small account: if this is large, the
        backtest's weight-space PnL is fiction.
        """
        return float(np.abs(self.realised_weights - self.target_weights).sum())


class IntegerShareSizer:
    """Converts target weights to whole NSE shares under a capital cap.

    capital:            account equity in rupees
    max_position_pct:   per-name cap (Kelly cap lives upstream)
    allow_short:        NSE cash segment does not permit overnight shorts.
                        Default False, which is the honest setting for a
                        delivery-based ₹10,000 account - and it materially
                        changes achievable returns, so it must not be
                        quietly assumed away.
    min_edge_bps_to_trade: skip a rebalance whose expected edge does not
                        clear the round-trip cost.
    """

    def __init__(
        self,
        capital: float,
        cost_model: Optional[NSECostModel] = None,
        max_position_pct: float = 0.25,
        allow_short: bool = False,
        min_edge_bps_to_trade: float = 0.0,
        lot_size: int = 1,
    ):
        self.capital = float(capital)
        self.costs = cost_model or NSECostModel()
        self.max_position_pct = float(max_position_pct)
        self.allow_short = bool(allow_short)
        self.min_edge_bps_to_trade = float(min_edge_bps_to_trade)
        self.lot_size = int(lot_size)

    def size(
        self,
        target_weights: np.ndarray,
        prices: np.ndarray,
        current_shares: Optional[np.ndarray] = None,
        equity: Optional[float] = None,
        expected_edge_bps: Optional[np.ndarray] = None,
    ) -> SizingResult:
        """Round target weights to executable integer share counts."""
        w = np.asarray(target_weights, dtype=np.float64).copy()
        px = np.asarray(prices, dtype=np.float64)
        n = w.size
        if px.shape != w.shape:
            raise ValueError(f"weights {w.shape} vs prices {px.shape}")
        eq = float(equity) if equity is not None else self.capital
        cur = np.zeros(n, dtype=np.int64) if current_shares is None \
            else np.asarray(current_shares, dtype=np.int64).copy()

        if not self.allow_short:
            w = np.clip(w, 0.0, None)
        w = np.clip(w, -self.max_position_pct, self.max_position_pct)

        # Refuse names whose expected edge cannot pay for the round trip.
        if expected_edge_bps is not None:
            rt = self.costs.round_trip_bps()
            too_thin = np.abs(np.asarray(expected_edge_bps, dtype=np.float64)) < \
                max(self.min_edge_bps_to_trade, rt)
            w[too_thin] = 0.0

        # Truncate toward zero, never round up: rounding up can demand
        # more cash than the account holds, which is not a real portfolio.
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.where(px > 0, w * eq / px, 0.0)
        target_shares = np.trunc(np.nan_to_num(raw)).astype(np.int64)
        if self.lot_size > 1:
            target_shares = (target_shares // self.lot_size) * self.lot_size

        # Cash feasibility: if the buy leg exceeds available capital, drop
        # the least-conviction names until it fits. Real accounts cannot
        # be levered by accident.
        for _ in range(n):
            gross = float(np.abs(target_shares * px).sum())
            if gross <= eq or gross <= 0:
                break
            live = np.flatnonzero(target_shares != 0)
            if live.size == 0:
                break
            weakest = live[np.argmin(np.abs(w[live]))]
            target_shares[weakest] = 0

        trade = target_shares - cur
        cost = 0.0
        for i in range(n):
            if trade[i] == 0:
                continue
            notional = abs(float(trade[i]) * px[i])
            cost += self.costs.cost_of(notional, "buy" if trade[i] > 0 else "sell")

        realised = np.where(eq > 0, target_shares * px / eq, 0.0)
        n_untradeable = int(np.sum((np.abs(w) > 1e-9) & (target_shares == 0)))

        return SizingResult(
            shares=target_shares,
            realised_weights=realised,
            target_weights=w,
            trade_shares=trade,
            cost=float(cost),
            deployed_capital=float(np.abs(target_shares * px).sum()),
            n_untradeable=n_untradeable,
        )


def kelly_fraction(win_rate: float, win_loss_ratio: float,
                   kelly_multiplier: float = 0.25) -> float:
    """Fractional Kelly.

        f* = (b*p - q) / b,   b = win/loss ratio, p = win rate, q = 1-p

    `kelly_multiplier` defaults to quarter-Kelly, and that default is a
    deliberate position: full Kelly is growth-optimal only if p and b are
    KNOWN. They are estimated here, from a small sample, on a system whose
    measured edge is currently indistinguishable from zero - and Kelly is
    violently sensitive to overestimating p. Quarter-Kelly is the
    standard practitioner haircut for exactly this parameter uncertainty.

    Returns 0.0 when the edge is non-positive - i.e. it refuses to size a
    strategy that does not have one, rather than returning a negative
    fraction and inverting the bet.
    """
    p = float(np.clip(win_rate, 0.0, 1.0))
    b = float(win_loss_ratio)
    if b <= 0:
        return 0.0
    f = (b * p - (1.0 - p)) / b
    return float(max(0.0, f) * kelly_multiplier)


def edge_required_bps(cost_model: Optional[NSECostModel] = None,
                      turnover_per_bar: float = 1.0) -> float:
    """Per-bar edge (bps) needed just to break even on costs.

    The reality check for any target win rate: at 100% daily turnover on
    NSE delivery, break-even is ~22bp/bar. A 55% win rate with symmetric
    win/loss magnitudes yields an expected edge of (0.55-0.45) = 10% of
    average move size - so the average absolute move must exceed ~220bp
    for that win rate to be profitable at full turnover. NSE large-caps
    move ~1.2%/day, so this is tight but not impossible - and it is
    precisely why turnover control (component 6's deadband and smoothing
    genes) is load-bearing rather than cosmetic.
    """
    cm = cost_model or NSECostModel()
    return cm.round_trip_bps() * float(turnover_per_bar)
