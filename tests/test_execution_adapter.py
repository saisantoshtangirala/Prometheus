"""
Tests for kronos/execution_adapter.py - Phase 1 of the algo-trading-ready
plan (broker abstraction). Two adapters:

  - SimulatedExecutionAdapter: a thin shim around the existing, already
    heavily-tested PaperTrader. These tests just confirm delegation is
    correct, not re-test PaperTrader's own trading math.

  - IBKRExecutionAdapter: talks to Interactive Brokers via ib_insync.
    Nothing here touches a real network or a real IBKR account - every
    test installs a fake ib_insync module into sys.modules so the
    adapter's OWN logic (port defaults, contract mapping, delta sizing,
    connection-state guards) is exercised without any live dependency.
    This adapter is not wired into KronosOrchestrator yet (see the
    module's docstring) - these tests exist so it's provably correct
    before that wiring ever happens.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos.config import load_config
from kronos.execution_adapter import (
    IBKRExecutionAdapter,
    OrderResult,
    SimulatedExecutionAdapter,
)
from kronos.paper_trader import PaperTrader


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg.override("trading.db_path", str(tmp_path / "trades.db"))
    return cfg


# ---------------------------------------------------------------------------
# SimulatedExecutionAdapter
# ---------------------------------------------------------------------------

class TestSimulatedExecutionAdapter:
    def test_always_connected(self, config):
        trader = PaperTrader(config)
        adapter = SimulatedExecutionAdapter(trader)
        assert adapter.is_connected()
        adapter.connect()      # no-op, must not raise
        adapter.disconnect()   # no-op, must not raise
        assert adapter.is_connected()
        trader.close()

    def test_submit_order_delegates_to_trader_execute(self, config):
        trader = PaperTrader(config)
        adapter = SimulatedExecutionAdapter(trader)
        result = adapter.submit_order(
            day=1, ticker="AAA", target_weight=0.20,
            price=100.0, bar_volume=50_000_000,
        )
        assert isinstance(result, OrderResult)
        assert result.ticker == "AAA"
        assert trader.positions["AAA"] != 0.0
        trader.close()

    def test_submit_order_returns_none_for_a_no_op_trade(self, config):
        trader = PaperTrader(config)
        adapter = SimulatedExecutionAdapter(trader)
        # target_weight=0 with no existing position -> nothing to do
        result = adapter.submit_order(
            day=1, ticker="AAA", target_weight=0.0,
            price=100.0, bar_volume=50_000_000,
        )
        assert result is None
        trader.close()

    def test_equity_positions_cash_delegate_to_trader(self, config):
        trader = PaperTrader(config)
        adapter = SimulatedExecutionAdapter(trader)
        adapter.submit_order(day=1, ticker="AAA", target_weight=0.20,
                             price=100.0, bar_volume=50_000_000)
        assert adapter.equity() == trader.equity()
        assert adapter.positions == trader.positions
        assert adapter.cash == trader.cash
        trader.close()


# ---------------------------------------------------------------------------
# IBKRExecutionAdapter - fully mocked ib_insync, no network
# ---------------------------------------------------------------------------

class FakeContract:
    def __init__(self, symbol, exchange, currency):
        self.symbol = symbol
        self.exchange = exchange
        self.currency = currency


class FakeAccountValue:
    def __init__(self, tag, value):
        self.tag = tag
        self.value = value


class FakePosition:
    def __init__(self, symbol, position):
        self.contract = FakeContract(symbol, "SMART", "USD")
        self.position = position


class FakeOrderStatus:
    def __init__(self, status="Filled", filled=0.0, avgFillPrice=0.0):
        self.status = status
        self.filled = filled
        self.avgFillPrice = avgFillPrice


class FakeOrder:
    def __init__(self, orderId):
        self.orderId = orderId


class FakeTrade:
    def __init__(self, order_id, filled, avg_fill_price):
        self.order = FakeOrder(order_id)
        self.orderStatus = FakeOrderStatus(filled=filled, avgFillPrice=avg_fill_price)


class FakeIB:
    """Stands in for ib_insync.IB - tracks connect calls, serves
    configurable account state, and records every placed order."""

    def __init__(self):
        self.connected = False
        self.connect_args = None
        self.net_liquidation = 100_000.0
        self.total_cash = 100_000.0
        self._positions: Dict[str, float] = {}
        self.placed_orders: List[tuple] = []
        self._next_order_id = 1

    def connect(self, host, port, clientId, timeout=10):
        self.connect_args = (host, port, clientId)
        self.connected = True

    def disconnect(self):
        self.connected = False

    def isConnected(self):
        return self.connected

    def accountSummary(self):
        return [
            FakeAccountValue("NetLiquidation", self.net_liquidation),
            FakeAccountValue("TotalCashValue", self.total_cash),
        ]

    def positions(self):
        return [FakePosition(sym, qty) for sym, qty in self._positions.items()]

    def placeOrder(self, contract, order):
        self.placed_orders.append((contract, order))
        oid = self._next_order_id
        self._next_order_id += 1
        return FakeTrade(oid, filled=order.totalQuantity, avg_fill_price=100.0)

    def sleep(self, seconds):
        pass


def _install_fake_ib_insync(monkeypatch, fake_ib: FakeIB):
    fake_module = MagicMock()
    fake_module.IB = MagicMock(return_value=fake_ib)

    class _Stock:
        def __init__(self, symbol, exchange, currency):
            self.symbol = symbol
            self.exchange = exchange
            self.currency = currency

    class _MarketOrder:
        def __init__(self, action, totalQuantity):
            self.action = action
            self.totalQuantity = totalQuantity

    fake_module.Stock = _Stock
    fake_module.MarketOrder = _MarketOrder
    monkeypatch.setitem(sys.modules, "ib_insync", fake_module)
    return fake_module


@pytest.fixture
def fake_ib(monkeypatch):
    ib = FakeIB()
    _install_fake_ib_insync(monkeypatch, ib)
    return ib


class TestIBKRConnection:
    def test_defaults_to_paper_trading_port(self, fake_ib, monkeypatch):
        monkeypatch.delenv("IBKR_HOST", raising=False)
        monkeypatch.delenv("IBKR_PORT", raising=False)
        monkeypatch.delenv("IBKR_CLIENT_ID", raising=False)
        adapter = IBKRExecutionAdapter(config=None)
        assert adapter.port == 7497, "must default to TWS's PAPER port, never the live port"
        adapter.connect()
        assert fake_ib.connect_args == ("127.0.0.1", 7497, 1)

    def test_env_vars_override_connection_params(self, fake_ib, monkeypatch):
        monkeypatch.setenv("IBKR_HOST", "10.0.0.5")
        monkeypatch.setenv("IBKR_PORT", "4002")
        monkeypatch.setenv("IBKR_CLIENT_ID", "7")
        adapter = IBKRExecutionAdapter(config=None)
        adapter.connect()
        assert fake_ib.connect_args == ("10.0.0.5", 4002, 7)

    def test_is_connected_false_before_connect(self, fake_ib):
        adapter = IBKRExecutionAdapter(config=None)
        assert not adapter.is_connected()

    def test_is_connected_true_after_connect(self, fake_ib):
        adapter = IBKRExecutionAdapter(config=None)
        adapter.connect()
        assert adapter.is_connected()

    def test_disconnect_reflected_in_is_connected(self, fake_ib):
        adapter = IBKRExecutionAdapter(config=None)
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected()

    def test_missing_ib_insync_raises_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ib_insync", None)
        adapter = IBKRExecutionAdapter(config=None)
        with pytest.raises(RuntimeError, match="ib_insync is not installed"):
            adapter.connect()

    def test_connect_is_idempotent(self, fake_ib):
        adapter = IBKRExecutionAdapter(config=None)
        adapter.connect()
        first_args = fake_ib.connect_args
        fake_ib.connect_args = None
        adapter.connect()   # second call while already connected
        assert fake_ib.connect_args is None, "must not reconnect when already connected"


class TestIBKRGuardsBeforeConnection:
    """Every account-touching call must fail loudly, not silently return
    garbage, if used before connect()."""

    def test_equity_raises_before_connect(self, fake_ib):
        adapter = IBKRExecutionAdapter(config=None)
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.equity()

    def test_cash_raises_before_connect(self, fake_ib):
        adapter = IBKRExecutionAdapter(config=None)
        with pytest.raises(RuntimeError, match="not connected"):
            _ = adapter.cash

    def test_positions_raises_before_connect(self, fake_ib):
        adapter = IBKRExecutionAdapter(config=None)
        with pytest.raises(RuntimeError, match="not connected"):
            _ = adapter.positions

    def test_submit_order_raises_before_connect(self, fake_ib):
        adapter = IBKRExecutionAdapter(config=None)
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.submit_order(day=1, ticker="AAPL", target_weight=0.1,
                                 price=100.0, bar_volume=1_000_000)


class TestIBKRAccountState:
    def test_equity_reads_net_liquidation(self, fake_ib):
        fake_ib.net_liquidation = 123_456.78
        adapter = IBKRExecutionAdapter(config=None)
        adapter.connect()
        assert adapter.equity() == pytest.approx(123_456.78)

    def test_cash_reads_total_cash_value(self, fake_ib):
        fake_ib.total_cash = 54_321.0
        adapter = IBKRExecutionAdapter(config=None)
        adapter.connect()
        assert adapter.cash == pytest.approx(54_321.0)

    def test_positions_reads_from_ib_positions(self, fake_ib):
        fake_ib._positions = {"AAPL": 10.0, "MSFT": -5.0}
        adapter = IBKRExecutionAdapter(config=None)
        adapter.connect()
        assert adapter.positions == {"AAPL": 10.0, "MSFT": -5.0}


class TestIBKRContractMapping:
    def test_nse_ticker_maps_to_nse_inr(self, fake_ib):
        adapter = IBKRExecutionAdapter(config=None)
        contract = adapter._contract("RELIANCE.NS")
        assert contract.symbol == "RELIANCE"
        assert contract.exchange == "NSE"
        assert contract.currency == "INR"

    def test_us_ticker_maps_to_smart_usd(self, fake_ib):
        adapter = IBKRExecutionAdapter(config=None)
        contract = adapter._contract("AAPL")
        assert contract.symbol == "AAPL"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"


class TestIBKRSubmitOrder:
    def test_buy_order_placed_for_positive_target(self, fake_ib):
        adapter = IBKRExecutionAdapter(config=None)
        adapter.connect()
        result = adapter.submit_order(
            day=1, ticker="AAPL", target_weight=0.5,
            price=100.0, bar_volume=1_000_000,
        )
        assert result is not None
        assert result.side == "buy"
        assert len(fake_ib.placed_orders) == 1
        _, order = fake_ib.placed_orders[0]
        assert order.action == "BUY"
        assert order.totalQuantity == pytest.approx(500.0)   # 50% of $100k @ $100

    def test_sell_order_placed_to_reduce_existing_long(self, fake_ib):
        fake_ib._positions = {"AAPL": 1000.0}   # already long more than target
        adapter = IBKRExecutionAdapter(config=None)
        adapter.connect()
        result = adapter.submit_order(
            day=1, ticker="AAPL", target_weight=0.1,
            price=100.0, bar_volume=1_000_000,
        )
        assert result is not None
        assert result.side == "sell"
        _, order = fake_ib.placed_orders[0]
        assert order.action == "SELL"

    def test_sub_dollar_rebalance_is_a_no_op(self, fake_ib):
        fake_ib._positions = {"AAPL": 500.0}   # already exactly at target
        adapter = IBKRExecutionAdapter(config=None)
        adapter.connect()
        result = adapter.submit_order(
            day=1, ticker="AAPL", target_weight=0.5,
            price=100.0, bar_volume=1_000_000,
        )
        assert result is None
        assert fake_ib.placed_orders == []

    def test_result_carries_broker_order_id(self, fake_ib):
        adapter = IBKRExecutionAdapter(config=None)
        adapter.connect()
        result = adapter.submit_order(
            day=1, ticker="AAPL", target_weight=0.5,
            price=100.0, bar_volume=1_000_000,
        )
        assert result.broker_order_id == "1"
