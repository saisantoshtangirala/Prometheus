"""
Kronos Paper Trading Engine.

Simulates execution with liquidity-tiered slippage, enforces the Kelly cap,
and persists every trade plus daily performance to SQLite (logs/trades.db).
Uses the stdlib sqlite3 driver - zero external dependencies, zero daemons,
survives 365 days of appends.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MAX_IMPACT_SLIPPAGE = 0.05      # 5% hard cap on market-impact slippage
BANKRUPTCY_PRICE = 0.001        # below this, the position is written off

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    day INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    signal_price REAL NOT NULL,
    fill_price REAL NOT NULL,
    slippage_pct REAL NOT NULL,
    notional REAL NOT NULL,
    position_after REAL NOT NULL,
    cash_after REAL
);
CREATE TABLE IF NOT EXISTS daily_performance (
    day INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    equity REAL NOT NULL,
    pnl REAL NOT NULL,
    sharpe REAL,
    directional_accuracy REAL,
    n_trades INTEGER NOT NULL,
    max_position_pct REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    day INTEGER,
    phase TEXT NOT NULL,
    message TEXT NOT NULL
);
"""


@dataclass
class Fill:
    ticker: str
    side: str              # "buy" | "sell"
    quantity: float
    signal_price: float
    fill_price: float
    slippage_pct: float
    notional: float


class PaperTrader:
    """
    Cash-account paper trader with Kelly cap enforcement.

    Positions are tracked in shares; equity marks to the latest price map.
    Slippage tiers by bar volume (config: trading.slippage).
    """

    def __init__(self, config, db_path: Optional[str] = None):
        self.cfg = config
        self.capital = float(config.trading.initial_capital)
        self.cash = self.capital
        self.positions: Dict[str, float] = {}       # ticker -> shares
        self.last_prices: Dict[str, float] = {}
        self._daily_predictions: List[tuple] = []   # (predicted_dir, actual_dir)
        self._equity_history: List[float] = [self.capital]

        self.db_path = db_path or config.trading.db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        # PAP-05: allow use from worker threads; a lock serializes writes so
        # concurrent audit/trade inserts never hit "database is locked".
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level="IMMEDIATE",
        )
        self._db_lock = threading.Lock()
        with self._db_lock:
            self._conn.executescript(SCHEMA)
            self._migrate_add_cash_after()
            self._conn.commit()

        # Restore cash/positions/equity history from any prior campaign
        # already in this DB, instead of always starting a fresh
        # initial_capital account - see resume_from_db()'s docstring for
        # why this matters (every service restart, including every
        # deploy, used to silently reset the whole paper-trading campaign
        # to day 1 with a brand-new account, discarding all prior history
        # even though trades.db itself survives restarts on disk).
        self.resume_from_db()

    def _migrate_add_cash_after(self) -> None:
        """trades.db files created before cash_after existed need the
        column added in place - CREATE TABLE IF NOT EXISTS is a no-op
        against an existing table with an older schema, it does not add
        missing columns. Safe to call every startup: checks first."""
        cols = [row[1] for row in self._conn.execute("PRAGMA table_info(trades)")]
        if "cash_after" not in cols:
            self._conn.execute("ALTER TABLE trades ADD COLUMN cash_after REAL")

    def resume_from_db(self) -> None:
        """
        Reconstruct cash/positions/equity history from this DB's existing
        trades/daily_performance rows, if any - otherwise leaves the
        fresh-account defaults __init__ already set untouched.

        cash: the cash_after of the single most recently recorded trade
        (by id) across all tickers - the ground truth for "how much cash
        do I actually have right now." Rows written before cash_after
        existed have it as NULL; if the most recent trade predates the
        migration, cash cannot be reconstructed and this falls back to a
        fresh account rather than guessing.

        positions: per ticker, the position_after of THAT ticker's own
        most recent trade (not the single most recent trade overall -
        a ticker untouched by the latest trade can still hold a real
        position from an earlier one). write_off() also writes a
        position_after=0 trade-like row precisely so a written-off
        position is never incorrectly revived here.

        equity history: daily_performance.equity in day order - keeps
        close_day()'s rolling Sharpe reflecting the whole campaign, not
        just whatever ran since the last restart.
        """
        last_cash_row = self._conn.execute(
            "SELECT cash_after FROM trades ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if last_cash_row is None or last_cash_row[0] is None:
            return   # no trades yet, or they predate cash_after - stay fresh

        self.cash = float(last_cash_row[0])
        rows = self._conn.execute(
            """
            SELECT ticker, position_after FROM trades t1
            WHERE id = (
                SELECT MAX(id) FROM trades t2 WHERE t2.ticker = t1.ticker
            )
            """
        ).fetchall()
        self.positions = {
            ticker: float(shares) for ticker, shares in rows if shares != 0.0
        }

        equity_rows = self._conn.execute(
            "SELECT equity FROM daily_performance ORDER BY day ASC"
        ).fetchall()
        if equity_rows:
            self._equity_history = [float(r[0]) for r in equity_rows]

        logger.info(
            "[trader] resumed from %s: cash=%.2f positions=%d equity_history=%d days",
            self.db_path, self.cash, len(self.positions), len(self._equity_history),
        )

    # -- slippage -----------------------------------------------------------

    def slippage_pct(
        self, bar_volume: float, trade_value: Optional[float] = None,
        avg_dollar_volume: Optional[float] = None,
    ) -> float:
        """
        Slippage model.

        Base: liquidity-tiered percentage from config thresholds.
        PAP-01 market impact: when avg_dollar_volume is known, an impact term
        trade_value / avg_dollar_volume * 0.01 is added, and the total is
        capped at MAX_IMPACT_SLIPPAGE (5%). A $100k order against $50k of
        average daily dollar volume therefore pays 2% impact on top of the
        liquidity tier, never more than 5% total.
        """
        s = self.cfg.trading.slippage
        if bar_volume >= float(s.high_liquidity_min_volume):
            base = float(s.high_liquidity_pct) / 100.0
        elif bar_volume >= float(s.mid_liquidity_min_volume):
            base = float(s.mid_liquidity_pct) / 100.0
        else:
            base = float(s.low_liquidity_pct) / 100.0

        if trade_value is not None and avg_dollar_volume:
            impact = (trade_value / avg_dollar_volume) * 0.01
            return min(MAX_IMPACT_SLIPPAGE, base + impact)
        return base

    # -- portfolio math -----------------------------------------------------

    def equity(self, prices: Optional[Dict[str, float]] = None) -> float:
        prices = prices or self.last_prices
        mark = sum(
            shares * prices.get(t, self.last_prices.get(t, 0.0))
            for t, shares in self.positions.items()
        )
        return self.cash + mark

    def position_pct(self, ticker: str) -> float:
        price = self.last_prices.get(ticker, 0.0)
        eq = self.equity()
        if eq <= 0:
            return 0.0
        return abs(self.positions.get(ticker, 0.0) * price) / eq

    # -- execution ----------------------------------------------------------

    def execute(
        self,
        day: int,
        ticker: str,
        target_weight: float,          # desired signed portfolio weight
        price: float,
        bar_volume: float,
        position_cap: float = 1.0,     # reflex gate cap (0.0 in panic)
        avg_dollar_volume: Optional[float] = None,  # enables impact slippage
    ) -> Optional[Fill]:
        """
        Move the position toward target_weight, respecting BOTH the Kelly cap
        (trading.max_position_pct) and the reflex gate cap. Short sales are
        supported: a negative target_weight holds negative shares (PAP-03).
        """
        if price <= BANKRUPTCY_PRICE:
            self.write_off(day, ticker, price)
            return None

        self.last_prices[ticker] = price
        kelly_cap = float(self.cfg.trading.max_position_pct)
        effective_cap = min(kelly_cap, kelly_cap * position_cap) if target_weight > 0 \
            else kelly_cap
        target_weight = float(np.clip(target_weight, -effective_cap, effective_cap))
        if position_cap == 0.0 and target_weight > 0:
            target_weight = 0.0        # gate: no new longs in panic

        eq = self.equity()
        target_shares = (target_weight * eq) / max(price, 1e-9)
        delta = target_shares - self.positions.get(ticker, 0.0)
        if abs(delta * price) < 1.0:   # ignore sub-dollar rebalances
            return None

        side = "buy" if delta > 0 else "sell"
        slip = self.slippage_pct(
            bar_volume, trade_value=abs(delta) * price,
            avg_dollar_volume=avg_dollar_volume,
        )
        fill_price = price * (1 + slip) if side == "buy" else price * (1 - slip)
        notional = abs(delta) * fill_price

        commission = float(self.cfg.trading.commission_per_trade)
        if side == "buy" and notional + commission > self.cash:
            # scale down to available cash
            delta = (self.cash - commission) / fill_price
            if delta <= 0:
                return None
            notional = abs(delta) * fill_price

        self.cash -= delta * fill_price + commission
        self.positions[ticker] = self.positions.get(ticker, 0.0) + delta

        fill = Fill(
            ticker=ticker, side=side, quantity=abs(delta),
            signal_price=price, fill_price=fill_price,
            slippage_pct=slip * 100.0, notional=notional,
        )
        self._record_trade(day, fill)
        return fill

    def write_off(self, day: int, ticker: str, price: float = 0.0) -> None:
        """
        PAP-04: bankruptcy/delisting. The position's value is set to zero,
        the loss is realized in equity, and the event is audit-logged.
        No division by zero anywhere downstream.
        """
        shares = self.positions.pop(ticker, 0.0)
        if shares == 0.0:
            self.last_prices[ticker] = max(price, 0.0)
            return
        loss = shares * self.last_prices.get(ticker, 0.0)
        self.last_prices[ticker] = 0.0
        logger.warning(
            "[trader] WRITE-OFF %s: %.2f shares, realized loss ~$%.2f",
            ticker, shares, loss,
        )
        self.audit(day, "write-off",
                   f"{ticker}: {shares:.4f} shares written off at "
                   f"price={price:.6f}, est. loss ${loss:.2f}")
        # Record a position_after=0 trade-like row - resume_from_db() reads
        # each ticker's own most recent trade row to reconstruct positions
        # on restart; without this, a written-off position (cash unaffected,
        # since the loss is a mark-to-market wipeout, not a sale) would
        # look untouched since its last real trade and get incorrectly
        # revived as if it still existed.
        self._record_trade(day, Fill(
            ticker=ticker, side="writeoff", quantity=shares,
            signal_price=price, fill_price=0.0, slippage_pct=0.0, notional=0.0,
        ))

    # -- daily bookkeeping --------------------------------------------------

    def record_prediction(self, predicted_return: float, actual_return: float) -> None:
        self._daily_predictions.append(
            (float(np.sign(predicted_return)), float(np.sign(actual_return)))
        )

    def close_day(self, day: int, prices: Dict[str, float]) -> Dict:
        """Mark to market, compute daily stats, persist, reset daily state."""
        self.last_prices.update(prices)
        eq = self.equity(prices)
        prev_eq = self._equity_history[-1]
        pnl = eq - prev_eq
        self._equity_history.append(eq)

        rets = np.diff(self._equity_history) / (
            np.array(self._equity_history[:-1]) + 1e-9
        )
        sharpe = None
        if len(rets) >= 2 and rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * np.sqrt(252))

        dir_acc = None
        if self._daily_predictions:
            hits = sum(1 for p, a in self._daily_predictions if p == a and p != 0)
            total = sum(1 for p, _ in self._daily_predictions if p != 0)
            dir_acc = hits / total if total else None
        self._daily_predictions = []

        n_trades = self._conn.execute(
            "SELECT COUNT(*) FROM trades WHERE day = ?", (day,)
        ).fetchone()[0]
        max_pos = max(
            (self.position_pct(t) for t in self.positions), default=0.0
        )

        stats = {
            "day": day, "equity": eq, "pnl": pnl, "sharpe": sharpe,
            "directional_accuracy": dir_acc, "n_trades": n_trades,
            "max_position_pct": max_pos,
        }
        with self._db_lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO daily_performance VALUES (?,?,?,?,?,?,?,?)",
                (day, datetime.now(timezone.utc).isoformat(), eq, pnl, sharpe,
                 dir_acc, n_trades, max_pos),
            )
            self._conn.commit()
        logger.info(
            "[trader] day %d closed: equity=%.2f pnl=%+.2f sharpe=%s trades=%d",
            day, eq, pnl, f"{sharpe:.2f}" if sharpe else "n/a", n_trades,
        )
        return stats

    def audit(self, day: Optional[int], phase: str, message: str) -> None:
        """Infinite logging (non-negotiable #4): every decision on the record."""
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO audit_log (ts, day, phase, message) VALUES (?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), day, phase, message),
            )
            self._conn.commit()

    # -- internals ----------------------------------------------------------

    def _record_trade(self, day: int, fill: Fill) -> None:
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO trades (ts, day, ticker, side, quantity, signal_price,"
                " fill_price, slippage_pct, notional, position_after, cash_after)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), day, fill.ticker, fill.side,
                 fill.quantity, fill.signal_price, fill.fill_price, fill.slippage_pct,
                 fill.notional, self.positions.get(fill.ticker, 0.0), self.cash),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
