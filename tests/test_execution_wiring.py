"""
The execution seam is actually wired, and routes where config says.

WHY THIS FILE EXISTS. ExecutionAdapter was written, tested and exported,
and the orchestrator called self.trader.execute() directly in four
places - so the abstraction routed nothing. It was dead code that looked
alive, the same shape as the NightEvolver checkpoint that was delivered
to Hetzner and never consumed.

The tests here are about ROUTING, not about fills: that every order path
goes through the adapter, and that the adapter chosen is the one config
names. The second half matters more than it looks - a broker adapter
selected by accident is the only failure in this file that costs real
money.
"""

from __future__ import annotations

import pytest

from kronos.execution_adapter import (
    ExecutionAdapter, SimulatedExecutionAdapter, build_execution_adapter,
)


class FakeCfg:
    def __init__(self, block=None):
        self._block = block

    def get(self, name, default=None):
        return self._block if name == "execution" else default


class FakeTrader:
    """Must carry every attribute the adapter's properties read.

    The first version omitted `cash`, and the interface test failed
    claiming the ADAPTER lacked a cash property. It does not - the
    property exists and forwards to trader.cash, which raised
    AttributeError on this fake, and hasattr() reports False for a
    property that raises. The defect was in the double, and it accused
    the code under test.
    """

    def __init__(self):
        self.calls = []
        self.last_prices = {}
        self.positions = {}
        self.cash = 100000.0

    def execute(self, *a, **kw):
        self.calls.append((a, kw))
        return None

    def equity(self, prices=None):
        return self.cash


class TestAdapterSelection:
    def test_absent_block_defaults_to_simulated(self):
        """A missing config section must not be a reason to reach a
        broker, and must not be a reason to fail either."""
        a = build_execution_adapter(FakeCfg(None), FakeTrader())
        assert isinstance(a, SimulatedExecutionAdapter)

    def test_explicit_simulated(self):
        a = build_execution_adapter(FakeCfg({"mode": "simulated"}), FakeTrader())
        assert isinstance(a, SimulatedExecutionAdapter)

    def test_case_and_whitespace_tolerant(self):
        a = build_execution_adapter(FakeCfg({"mode": "  SIMULATED "}), FakeTrader())
        assert isinstance(a, SimulatedExecutionAdapter)

    def test_an_unknown_mode_raises_rather_than_guessing(self):
        """Defaulting an unrecognised mode to EITHER option is a way to
        trade somewhere nobody chose. Simulated would silently ignore a
        typo'd 'ibkr'; broker would be catastrophic."""
        with pytest.raises(ValueError, match="unknown execution.mode"):
            build_execution_adapter(FakeCfg({"mode": "ibkrr"}), FakeTrader())

    def test_the_shipped_config_is_simulated(self):
        """The default in kronos/config.yaml is load-bearing, not a
        convenience - wiring the seam must change no behaviour."""
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load(Path("kronos/config.yaml").read_text())
        assert cfg["execution"]["mode"] == "simulated"

    def test_ibkr_selection_is_loud(self, caplog):
        """It may fail to construct without ib_insync - that is fine and
        expected here. What must NOT happen is quiet selection."""
        import logging
        caplog.set_level(logging.WARNING)
        try:
            build_execution_adapter(FakeCfg({"mode": "ibkr"}), FakeTrader())
        except Exception:
            pass                      # missing ib_insync / credentials
        assert any("ibkr" in r.message.lower() or "BROKER" in r.message
                   for r in caplog.records), "ibkr was selected silently"


class TestOrchestratorRoutesThroughTheAdapter:
    def test_the_orchestrator_holds_an_adapter(self):
        import inspect
        from kronos import orchestrator as o
        src = inspect.getsource(o.KronosOrchestrator.__init__)
        assert "build_execution_adapter" in src

    def test_no_order_path_bypasses_the_adapter(self):
        """The regression guard. Every order must go through
        submit_order; a direct trader.execute() in the orchestrator is
        the seam being bypassed again.

        Counted from source rather than exercised, because the four call
        sites sit behind risk halts, veto directives and market-hours
        gating that a unit test would have to defeat one at a time - and
        a test that defeats the guards is not testing the live path.
        """
        import inspect
        from kronos import orchestrator as o
        src = inspect.getsource(o)
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.strip().startswith("#"))
        assert "self.trader.execute(" not in code, (
            "an order path bypasses the execution adapter")
        assert code.count("self.execution.submit_order(") >= 4

    def test_the_adapter_satisfies_the_interface(self):
        a = build_execution_adapter(FakeCfg(None), FakeTrader())
        assert isinstance(a, ExecutionAdapter)
        for m in ("connect", "disconnect", "is_connected", "submit_order",
                  "equity", "positions", "cash"):
            assert hasattr(a, m), m

    def test_simulated_forwards_to_the_same_trader(self):
        """The shim must not become a second implementation of fills -
        that is the drift that made a previous project's backtest measure
        a different model from the one that traded."""
        t = FakeTrader()
        build_execution_adapter(FakeCfg(None), t).submit_order(
            1, "RELIANCE", 0.1, 100.0, 1000.0, position_cap=0.5)
        assert len(t.calls) == 1
        assert t.calls[0][0][1] == "RELIANCE"
