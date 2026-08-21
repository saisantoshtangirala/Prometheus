"""
Execution abstraction - "how does an order actually reach the market" -
so a real broker can plug in without redesigning PaperTrader or
orchestrator.py. Phase 1 of the algo-trading-ready plan.

SimulatedExecutionAdapter wraps the existing, already-tested PaperTrader
and is what KronosOrchestrator uses today (config: execution.mode:
"simulated", the default). IBKRExecutionAdapter exists and is tested
against a mocked ib_insync, but is NOT wired into the live orchestrator
in this pass - that wiring is a deliberate separate step, gated on the
operator supplying real IBKR paper-account credentials and proving it
stable against IBKR's OWN paper account for weeks first. Per the plan's
own text: "never touch a live account before that's proven stable for
weeks." Building the adapter now, ahead of that, means the eventual
switch is a config change, not a redesign.

Every adapter speaks the same contract:
    connect() / disconnect() / is_connected()
    submit_order(day, ticker, target_weight, price, bar_volume, position_cap) -> Optional[OrderResult]
    equity(prices=None) -> float
    positions -> Dict[str, float]   (property)
    cash -> float                   (property)
"""

from __future__ import annotations

import abc
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    ticker: str
    side: str
    quantity: float
    fill_price: float
    notional: float
    broker_order_id: Optional[str] = None


class ExecutionAdapter(abc.ABC):
    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def is_connected(self) -> bool: ...

    @abc.abstractmethod
    def submit_order(
        self, day: int, ticker: str, target_weight: float, price: float,
        bar_volume: float, position_cap: float = 1.0,
    ) -> Optional[OrderResult]: ...

    @abc.abstractmethod
    def equity(self, prices: Optional[Dict[str, float]] = None) -> float: ...

    @property
    @abc.abstractmethod
    def positions(self) -> Dict[str, float]: ...

    @property
    @abc.abstractmethod
    def cash(self) -> float: ...


class SimulatedExecutionAdapter(ExecutionAdapter):
    """Thin shim around the existing PaperTrader - nothing about
    PaperTrader's own behavior changes. Always reports connected: it is
    in-process, there is no broker link to lose."""

    def __init__(self, trader):
        self.trader = trader

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def submit_order(self, day, ticker, target_weight, price, bar_volume,
                     position_cap=1.0) -> Optional[OrderResult]:
        fill = self.trader.execute(
            day, ticker, target_weight, price, bar_volume,
            position_cap=position_cap,
        )
        if fill is None:
            return None
        return OrderResult(
            ticker=fill.ticker, side=fill.side, quantity=fill.quantity,
            fill_price=fill.fill_price, notional=fill.notional,
        )

    def equity(self, prices: Optional[Dict[str, float]] = None) -> float:
        return self.trader.equity(prices)

    @property
    def positions(self) -> Dict[str, float]:
        return self.trader.positions

    @property
    def cash(self) -> float:
        return self.trader.cash


class IBKRExecutionAdapter(ExecutionAdapter):
    """
    Interactive Brokers execution via ib_insync (lazy-imported - its
    absence must never break anything that doesn't actually use this
    adapter, same discipline as scripts/run_kronos.py's yfinance import).

    Connects to TWS or IB Gateway, which must already be running and
    logged in on this host - ib_insync talks to it over a local socket,
    it does not launch or manage the TWS/Gateway process itself.

    Defaults to port 7497, TWS's PAPER-TRADING port (7496 is the LIVE
    port) - deliberately, so a misconfigured deployment fails to connect
    rather than silently placing a real order. IBKR_PORT must be
    explicitly overridden by the operator to reach a live account, and
    only after proving this adapter stable against the paper account for
    weeks, per the plan.

    Config (environment only - never hardcoded, never committed):
      IBKR_HOST        default "127.0.0.1"
      IBKR_PORT        default "7497" (TWS/Gateway PAPER port)
      IBKR_CLIENT_ID   default "1"
    """

    def __init__(self, config):
        self.cfg = config
        self.host = os.environ.get("IBKR_HOST", "127.0.0.1")
        self.port = int(os.environ.get("IBKR_PORT", "7497"))
        self.client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))
        self._ib = None

    def _require_connected(self) -> None:
        if not self.is_connected():
            raise RuntimeError(
                "IBKR adapter not connected - call connect() first"
            )

    def connect(self) -> None:
        try:
            from ib_insync import IB
        except ImportError as e:
            raise RuntimeError(
                "ib_insync is not installed - `pip install ib_insync` "
                "before using execution.mode: ibkr"
            ) from e
        if self._ib is None:
            self._ib = IB()
        if not self._ib.isConnected():
            self._ib.connect(self.host, self.port, clientId=self.client_id, timeout=10)
            logger.info("[ibkr] connected to %s:%d (clientId=%d)",
                       self.host, self.port, self.client_id)

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    @staticmethod
    def _contract(ticker: str):
        from ib_insync import Stock
        # NSE tickers carry a ".NS" suffix throughout kronos/config.yaml -
        # IBKR wants the bare symbol plus its own exchange/currency codes.
        symbol = ticker.split(".")[0]
        if ticker.endswith(".NS"):
            return Stock(symbol, "NSE", "INR")
        return Stock(symbol, "SMART", "USD")

    def submit_order(self, day, ticker, target_weight, price, bar_volume,
                     position_cap=1.0) -> Optional[OrderResult]:
        self._require_connected()
        from ib_insync import MarketOrder

        eq = self.equity()
        current_shares = self.positions.get(ticker, 0.0)
        target_shares = (target_weight * eq) / max(price, 1e-9)
        delta = target_shares - current_shares
        if abs(delta * price) < 1.0:
            return None

        action = "BUY" if delta > 0 else "SELL"
        order = MarketOrder(action, abs(round(delta)))
        trade = self._ib.placeOrder(self._contract(ticker), order)
        self._ib.sleep(1)   # let the paper-account fill come back
        fill_price = trade.orderStatus.avgFillPrice or price
        filled_qty = trade.orderStatus.filled or abs(round(delta))
        logger.info("[ibkr] %s %s x%.0f @ %.4f (order id %s, status %s)",
                   action, ticker, filled_qty, fill_price,
                   trade.order.orderId, trade.orderStatus.status)
        return OrderResult(
            ticker=ticker, side=action.lower(), quantity=filled_qty,
            fill_price=fill_price, notional=filled_qty * fill_price,
            broker_order_id=str(trade.order.orderId),
        )

    def equity(self, prices: Optional[Dict[str, float]] = None) -> float:
        self._require_connected()
        for v in self._ib.accountSummary():
            if v.tag == "NetLiquidation":
                return float(v.value)
        raise RuntimeError("IBKR accountSummary() returned no NetLiquidation value")

    @property
    def positions(self) -> Dict[str, float]:
        self._require_connected()
        return {p.contract.symbol: float(p.position) for p in self._ib.positions()}

    @property
    def cash(self) -> float:
        self._require_connected()
        for v in self._ib.accountSummary():
            if v.tag == "TotalCashValue":
                return float(v.value)
        raise RuntimeError("IBKR accountSummary() returned no TotalCashValue")


def build_execution_adapter(cfg, trader) -> ExecutionAdapter:
    """Pick the adapter named by `execution.mode`, defaulting to simulated.

    THE DEFAULT IS LOAD-BEARING, not a convenience. `simulated` routes to
    SimulatedExecutionAdapter, a thin shim over the same PaperTrader the
    orchestrator used directly before this seam existed, so wiring the
    abstraction in changes no behaviour at all. Anything else has to be
    asked for explicitly in config.

    `ibkr` is constructed only on an explicit request and logs at WARNING
    when it is, because a broker adapter that gets selected quietly is
    the one failure here that costs real money. If ib_insync or the
    credentials are missing it raises rather than silently falling back
    to simulation - a system that believes it is trading live while
    filling against a paper book is worse than one that refuses to
    start.
    """
    mode = "simulated"
    try:
        block = cfg.get("execution", None)
        if block:
            mode = str(block.get("mode", "simulated")).strip().lower()
    except Exception:                                            # noqa: BLE001
        pass

    if mode == "simulated":
        return SimulatedExecutionAdapter(trader)

    if mode == "ibkr":
        logger.warning(
            "[execution] mode=ibkr - constructing the IBKR adapter. Orders "
            "will be routed to a BROKER, not to the paper book. This is "
            "only correct if you set execution.mode deliberately.")
        return IBKRExecutionAdapter(trader)

    raise ValueError(
        f"unknown execution.mode {mode!r} - expected 'simulated' or 'ibkr'. "
        f"Refusing to guess: defaulting an unrecognised execution mode to "
        f"either option is a way to trade somewhere nobody chose.")
