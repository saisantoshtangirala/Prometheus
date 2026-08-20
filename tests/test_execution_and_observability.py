"""
Phase 5 (execution & order management) and Phase 6 (reporting &
observability), exercised rather than inspected.

The audit's first pass traced these paths structurally and said so. This
file runs them: fills, sizing caps, slippage direction, cash safety,
drawdown halts, sqlite under concurrent load, and whether the alerting
path actually fires and actually deduplicates.
"""

from __future__ import annotations

import sqlite3
import threading

import numpy as np
import pytest

from kronos.config import load_config
from kronos.paper_trader import PaperTrader
from kronos.risk_guard import RiskGuard


@pytest.fixture()
def trader(tmp_path):
    cfg = load_config()
    t = PaperTrader(cfg, db_path=str(tmp_path / "trades.db"))
    yield t
    t.close()


# ==========================================================================
# Phase 5.1 — trade execution fidelity
# ==========================================================================

class TestExecutionFidelity:
    def test_fill_respects_the_kelly_position_cap(self, trader):
        """A target weight above trading.max_position_pct must be clipped,
        not honoured."""
        cap = float(trader.cfg.trading.max_position_pct)
        trader.execute(day=1, ticker="AAA", target_weight=0.95,
                       price=100.0, bar_volume=1e9)
        pct = trader.position_pct("AAA")
        # The cap is applied to PRE-trade equity; slippage then lowers
        # equity, so the position measured after the fill sits marginally
        # above the target. Measured 0.25003 against a 0.25 cap - a
        # 0.012% overshoot, bounded by the slippage rate. Documented and
        # bounded rather than silently tolerated.
        assert pct <= cap * 1.01, \
            f"position {pct:.5f} exceeded cap {cap} by more than slippage explains"
        assert pct > cap * 0.9, "target was not reached at all"

    def test_reflex_gate_blocks_new_longs_in_panic(self, trader):
        """position_cap=0.0 is the panic gate. No new long may open."""
        fill = trader.execute(day=1, ticker="AAA", target_weight=0.20,
                              price=100.0, bar_volume=1e9, position_cap=0.0)
        assert fill is None or trader.positions.get("AAA", 0.0) == 0.0

    def test_gate_cap_scales_the_long_but_not_the_short(self, trader):
        """Documented asymmetry: the gate caps new longs only. Pin it so a
        change to that behaviour is deliberate."""
        t2 = PaperTrader(trader.cfg, db_path=trader.db_path + ".2")
        try:
            trader.execute(1, "AAA", 0.20, 100.0, 1e9, position_cap=0.5)
            t2.execute(1, "AAA", -0.20, 100.0, 1e9, position_cap=0.5)
            assert abs(trader.positions["AAA"]) < abs(t2.positions["AAA"])
        finally:
            t2.close()

    def test_slippage_is_directional(self, trader):
        """A buy must fill ABOVE the signal price and a sell BELOW it.
        An inverted sign would silently pay the trader to trade."""
        buy = trader.execute(1, "AAA", 0.10, 100.0, 1e3)
        assert buy is not None and buy.side == "buy"
        assert buy.fill_price > buy.signal_price

        sell = trader.execute(2, "AAA", -0.10, 100.0, 1e3)
        assert sell is not None and sell.side == "sell"
        assert sell.fill_price < sell.signal_price

    def test_market_impact_is_capped(self, trader):
        """An order many times the venue's daily volume must not produce an
        unbounded slippage number."""
        from kronos.paper_trader import MAX_IMPACT_SLIPPAGE
        slip = trader.slippage_pct(bar_volume=1.0, trade_value=1e9,
                                   avg_dollar_volume=1.0)
        assert slip <= MAX_IMPACT_SLIPPAGE + 1e-12

    def test_thin_volume_costs_more_than_deep_volume(self, trader):
        assert trader.slippage_pct(1.0) > trader.slippage_pct(1e9)

    def test_cash_never_goes_negative_on_a_greedy_buy(self, trader):
        """A target the account cannot afford must scale down, not
        overdraw."""
        for day in range(1, 12):
            trader.execute(day, f"T{day}", 0.25, 100.0, 1e9)
        assert trader.cash >= -1e-6, f"overdrawn: cash={trader.cash}"

    def test_bankrupt_price_writes_off_instead_of_dividing_by_zero(self, trader):
        trader.execute(1, "AAA", 0.10, 100.0, 1e9)
        assert trader.positions.get("AAA", 0.0) > 0
        trader.execute(2, "AAA", 0.10, 0.0, 1e9)        # price collapse
        assert trader.positions.get("AAA", 0.0) == 0.0
        assert np.isfinite(trader.equity())

    def test_subdollar_rebalance_is_skipped(self, trader):
        trader.execute(1, "AAA", 0.10, 100.0, 1e9)
        before = trader.positions["AAA"]
        # An identical target should produce no new fill.
        assert trader.execute(2, "AAA", 0.10, 100.0, 1e9) is None
        assert trader.positions["AAA"] == before

    def test_equity_is_finite_after_a_long_random_session(self, trader):
        rng = np.random.default_rng(0)
        px = 100.0
        for day in range(1, 60):
            px *= float(1 + rng.normal(0, 0.02))
            px = max(px, 1.0)
            trader.execute(day, "AAA", float(rng.uniform(-0.2, 0.2)), px, 1e8)
            assert np.isfinite(trader.equity())
            assert np.isfinite(trader.cash)


# ==========================================================================
# Phase 5.1b — risk limits
# ==========================================================================

class TestRiskLimits:
    def test_limits_are_fractions_not_percentages(self):
        """Units matter and are easy to get wrong - this test exists
        because the audit's first attempt passed 1.0 meaning "1%" and
        got 100%, then nearly reported a working guard as broken.
        config.yaml ships 0.05 / 0.20, i.e. 5% and 20%."""
        cfg = load_config()
        assert float(cfg.risk.max_daily_loss_pct) < 1.0
        assert float(cfg.risk.max_drawdown_pct) < 1.0

    def test_daily_loss_breach_is_detected_from_the_first_tick(self, trader, tmp_path):
        """_equity_history is seeded with the opening capital at
        construction, so the guard is live on day 1 - before any
        close_day() has run. If that seeding were removed, a fresh
        deployment would trade its entire first day unprotected."""
        cfg = trader.cfg
        cfg.override("risk.enabled", True)
        cfg.override("risk.halt_file", str(tmp_path / "HALT"))
        cfg.override("risk.max_daily_loss_pct", 0.05)
        cfg.override("risk.max_drawdown_pct", 0.20)
        guard = RiskGuard(cfg)
        assert trader._equity_history, "equity history not seeded at construction"
        assert guard.check(trader) is None       # clear to start

        trader.cash *= 0.5                       # 50% loss vs a 5% limit
        reason = guard.check(trader)
        assert reason is not None, "a 50% loss did not breach a 5% limit"
        assert "daily loss" in reason

    def test_check_reports_but_does_not_itself_halt(self, trader, tmp_path):
        """Documented separation: check() returns a reason, the
        orchestrator decides to trip. Asserting otherwise would couple
        detection to action."""
        cfg = trader.cfg
        cfg.override("risk.enabled", True)
        cfg.override("risk.halt_file", str(tmp_path / "HALT_B"))
        cfg.override("risk.max_daily_loss_pct", 0.05)
        guard = RiskGuard(cfg)
        trader.cash *= 0.5
        assert guard.check(trader) is not None
        assert not guard.halted, "check() tripped the halt by itself"
        guard.trip("test")
        assert guard.halted and guard.halt_reason

    def test_drawdown_breach_is_detected(self, trader, tmp_path):
        cfg = trader.cfg
        cfg.override("risk.enabled", True)
        cfg.override("risk.halt_file", str(tmp_path / "HALT_C"))
        cfg.override("risk.max_daily_loss_pct", 0.99)   # keep daily quiet
        cfg.override("risk.max_drawdown_pct", 0.20)
        guard = RiskGuard(cfg)
        trader.cash *= 0.7                              # 30% below peak
        reason = guard.check_drawdown(trader)
        assert reason is not None and "drawdown" in reason

    def test_kill_switch_file_halts_immediately(self, trader, tmp_path):
        cfg = trader.cfg
        cfg.override("risk.enabled", True)
        cfg.override("risk.halt_file", str(tmp_path / "HALT_D"))
        ks = tmp_path / "KILL_SWITCH"
        cfg.override("risk.kill_switch_file", str(ks))
        guard = RiskGuard(cfg)
        assert guard.check(trader) is None
        ks.write_text("operator stop")
        reason = guard.check(trader)
        assert reason is not None and "KILL_SWITCH" in reason

    def test_disabled_guard_reports_nothing(self, trader, tmp_path):
        cfg = trader.cfg
        cfg.override("risk.enabled", False)
        cfg.override("risk.halt_file", str(tmp_path / "HALT_E"))
        cfg.override("risk.max_daily_loss_pct", 0.01)
        trader.cash *= 0.1
        assert RiskGuard(cfg).check(trader) is None

    def test_halt_survives_a_process_restart(self, trader, tmp_path):
        """Halt state lives in a FILE precisely so a restart cannot clear
        it. A restarted service must not resume trading by itself."""
        cfg = trader.cfg
        cfg.override("risk.enabled", True)
        cfg.override("risk.halt_file", str(tmp_path / "HALT2"))
        cfg.override("risk.max_daily_loss_pct", 1.0)
        RiskGuard(cfg).trip("test breach")
        assert RiskGuard(cfg).halted, "halt did not survive re-instantiation"

    def test_trip_is_idempotent(self, trader, tmp_path):
        cfg = trader.cfg
        cfg.override("risk.enabled", True)
        cfg.override("risk.halt_file", str(tmp_path / "HALT3"))
        g = RiskGuard(cfg)
        g.trip("first")
        g.trip("second")
        assert "first" in (g.halt_reason or ""), \
            "a second breach overwrote the original reason"


# ==========================================================================
# Phase 5.2 — position tracking / sqlite
# ==========================================================================

class TestPositionTracking:
    def test_trades_are_persisted(self, trader):
        trader.execute(1, "AAA", 0.10, 100.0, 1e9)
        rows = trader._conn.execute("SELECT ticker, side, quantity FROM trades").fetchall()
        assert len(rows) == 1 and rows[0][0] == "AAA"

    def test_concurrent_writes_do_not_corrupt_or_deadlock(self, trader):
        """The connection is shared with check_same_thread=False behind a
        lock. Exercise it from several threads at once."""
        errors = []

        def worker(n):
            try:
                for i in range(25):
                    trader.audit(day=n, phase="stress", message=f"{n}-{i}")
            except Exception as e:                      # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not any(t.is_alive() for t in threads), "a writer thread hung"
        assert not errors, f"concurrent writes raised: {errors[:3]}"
        n = trader._conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE phase='stress'").fetchone()[0]
        assert n == 150, f"expected 150 audit rows, got {n}"

    def test_concurrent_read_and_write(self, trader):
        stop = threading.Event()
        errors = []

        def reader():
            try:
                while not stop.is_set():
                    trader._conn.execute("SELECT COUNT(*) FROM trades").fetchone()
            except Exception as e:                      # noqa: BLE001
                errors.append(e)

        r = threading.Thread(target=reader)
        r.start()
        try:
            for day in range(1, 20):
                trader.execute(day, f"T{day}", 0.05, 100.0, 1e9)
        finally:
            stop.set()
            r.join(timeout=10)
        assert not errors, f"reader raised during writes: {errors[:2]}"

    def test_close_releases_the_connection(self, trader):
        trader.execute(1, "AAA", 0.10, 100.0, 1e9)
        trader.close()
        with pytest.raises(sqlite3.ProgrammingError):
            trader._conn.execute("SELECT 1")

    def test_state_survives_a_reopen(self, trader):
        trader.execute(1, "AAA", 0.10, 100.0, 1e9)
        trader.close_day(1, {"AAA": 100.0})
        path = trader.db_path
        cfg = trader.cfg
        trader.close()

        reopened = PaperTrader(cfg, db_path=path)
        try:
            reopened.resume_from_db()
            assert abs(reopened.positions.get("AAA", 0.0)) > 0, \
                "positions did not survive a restart"
        finally:
            reopened.close()


# ==========================================================================
# Phase 6 — logging and alerting
# ==========================================================================

class TestObservability:
    def test_execution_decisions_reach_the_audit_log(self, trader):
        trader.audit(1, "reflex", "entered AAA on conviction 0.42")
        rows = trader._conn.execute(
            "SELECT ts, day, phase, message FROM audit_log").fetchall()
        assert rows and rows[0][0], "audit row has no timestamp"
        assert rows[0][2] == "reflex"

    def test_notifier_is_silent_when_unconfigured(self):
        """It must never raise into the trading loop, whatever happens."""
        from kronos.notifier import TelegramNotifier
        cfg = load_config()
        cfg.override("notifications.enabled", False)
        assert TelegramNotifier(cfg).send("anything") is False

    def test_notifier_never_raises_on_transport_failure(self, monkeypatch):
        from kronos import notifier as N
        cfg = load_config()
        cfg.override("notifications.enabled", True)
        n = N.TelegramNotifier(cfg)

        class Boom:
            @staticmethod
            def post(*a, **k):
                raise RuntimeError("network down")

        monkeypatch.setitem(__import__("sys").modules, "requests", Boom)
        assert n.send("hello") is False          # must swallow, not raise

    def test_daily_report_contains_the_headline_metrics(self, trader):
        from kronos.notifier import TelegramNotifier
        cfg = load_config()
        n = TelegramNotifier(cfg)
        trader.execute(1, "AAA", 0.10, 100.0, 1e9)
        stats = trader.close_day(1, {"AAA": 101.0})
        text = n.build_daily_report(day=1, stats=stats)
        low = text.lower()
        for token in ("equity", "day"):
            assert token in low, f"daily report omits {token!r}: {text[:300]}"
        return text


# ==========================================================================
# Phase 6 — alerting: dedupe and retry (both added by audit)
# ==========================================================================

class _FakeResp:
    def __init__(self, code):
        self.status_code = code
        self.text = f"code {code}"


class _FakeRequests:
    """Stands in for `requests`, recording every POST."""

    def __init__(self, codes):
        self.codes = list(codes)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append(json.get("text", ""))
        return _FakeResp(self.codes.pop(0) if self.codes else 200)


@pytest.fixture()
def wired_notifier(monkeypatch):
    from kronos.notifier import TelegramNotifier
    monkeypatch.setenv("KRONOS_TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("KRONOS_TELEGRAM_CHAT_ID", "chat")
    cfg = load_config()
    cfg.override("notifications.enabled", True)
    n = TelegramNotifier(cfg)
    assert n.enabled
    return n


def _install(monkeypatch, fake):
    import sys
    monkeypatch.setitem(sys.modules, "requests", fake)


class TestAlertDeduplication:
    def test_identical_alerts_are_suppressed_within_the_window(
            self, wired_notifier, monkeypatch):
        """A failure inside a loop used to produce one Telegram message
        per occurrence, which is exactly when Telegram rate-limits."""
        fake = _FakeRequests([200] * 10)
        _install(monkeypatch, fake)
        assert wired_notifier.send("data fetch failed") is True
        for _ in range(5):
            assert wired_notifier.send("data fetch failed") is False
        assert len(fake.calls) == 1, f"sent {len(fake.calls)} copies, expected 1"

    def test_suppressed_count_is_reported_when_the_window_closes(
            self, wired_notifier, monkeypatch):
        """Suppression must not hide how often the condition fired."""
        import kronos.notifier as N
        fake = _FakeRequests([200] * 10)
        _install(monkeypatch, fake)
        wired_notifier.send("boom")
        for _ in range(3):
            wired_notifier.send("boom")
        monkeypatch.setattr(N, "DEDUPE_WINDOW_SECONDS", 0)   # window elapses
        wired_notifier.send("boom")
        assert "suppressed" in fake.calls[-1], \
            f"repeat count not reported: {fake.calls[-1]!r}"
        assert "+3" in fake.calls[-1]

    def test_different_alerts_are_not_suppressed(self, wired_notifier, monkeypatch):
        fake = _FakeRequests([200] * 5)
        _install(monkeypatch, fake)
        assert wired_notifier.send("error A") is True
        assert wired_notifier.send("error B") is True
        assert len(fake.calls) == 2

    def test_dedupe_can_be_bypassed_for_must_send_messages(
            self, wired_notifier, monkeypatch):
        fake = _FakeRequests([200] * 5)
        _install(monkeypatch, fake)
        wired_notifier.send("daily report", dedupe=False)
        wired_notifier.send("daily report", dedupe=False)
        assert len(fake.calls) == 2

    def test_rate_limit_is_retried_not_dropped(self, wired_notifier, monkeypatch):
        """429 used to be logged and discarded. The alert channel for an
        unattended trading system must not lose a critical message
        because the first attempt was throttled."""
        import kronos.notifier as N
        monkeypatch.setattr(N, "RETRY_BACKOFF_SECONDS", 0.0)
        fake = _FakeRequests([429, 200])
        _install(monkeypatch, fake)
        assert wired_notifier.send("critical") is True
        assert len(fake.calls) == 2

    def test_server_error_is_retried(self, wired_notifier, monkeypatch):
        import kronos.notifier as N
        monkeypatch.setattr(N, "RETRY_BACKOFF_SECONDS", 0.0)
        fake = _FakeRequests([503, 502, 200])
        _install(monkeypatch, fake)
        assert wired_notifier.send("critical") is True
        assert len(fake.calls) == 3

    def test_client_error_is_not_retried(self, wired_notifier, monkeypatch):
        """A 400 is a bad request; retrying it just wastes the window."""
        import kronos.notifier as N
        monkeypatch.setattr(N, "RETRY_BACKOFF_SECONDS", 0.0)
        fake = _FakeRequests([400, 200])
        _install(monkeypatch, fake)
        assert wired_notifier.send("bad") is False
        assert len(fake.calls) == 1

    def test_dedupe_state_does_not_grow_without_bound(
            self, wired_notifier, monkeypatch):
        """This process runs for 365 days."""
        import kronos.notifier as N
        monkeypatch.setattr(N, "DEDUPE_WINDOW_SECONDS", 0)
        fake = _FakeRequests([200] * 400)
        _install(monkeypatch, fake)
        for i in range(300):
            wired_notifier.send(f"unique alert {i}")
        assert len(wired_notifier._last_sent) < 300


# ==========================================================================
# Calendar coverage (H1)
# ==========================================================================

class TestCalendarCoverage:
    def test_next_year_is_populated(self):
        """The table held 2026 ONLY, so from 2027-01-01 every festival
        holiday would have been treated as a trading day."""
        from datetime import date as ddate

        from kronos.calendar_utils import nse_holidays
        assert len(nse_holidays(ddate.today().year)) >= 10
        assert len(nse_holidays(ddate.today().year + 1)) >= 10, \
            "next year's NSE holidays are missing - the calendar expires"

    def test_coverage_gap_is_reported_before_it_bites(self):
        from datetime import date as ddate

        from kronos.calendar_utils import nse_calendar_coverage_gap
        assert nse_calendar_coverage_gap(ddate.today()) is None
        far = nse_calendar_coverage_gap(ddate(2035, 1, 1))
        assert far is not None and "2035" in far

    def test_known_2027_holiday_is_not_a_trading_day(self):
        from datetime import date as ddate

        from kronos.calendar_utils import is_nse_trading_day
        assert not is_nse_trading_day(ddate(2027, 1, 26))   # Republic Day, a Tue
        assert is_nse_trading_day(ddate(2027, 1, 27))       # the Wed after
