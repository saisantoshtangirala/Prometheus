"""
Independent risk layer for Kronos.

Every limit here is enforced in code the ML pipeline can never reach, so a
bad signal, a NaN/inf edge case in reflex or evolution, or a corrupted
checkpoint can never itself defeat these limits - the whole point of a
"second, independent risk layer that doesn't depend on the ML pipeline
being correct."

This is deliberately the opposite of orchestrator.py's veto.txt mechanism:
veto.txt is a 24h-delayed human override, built specifically so intraday
panic cannot touch the book. RiskGuard is the other direction - immediate
and fully automatic, because catching a bad signal or a fat-finger price
fast is exactly the point.

Halt state lives in a file (risk.halt_file), never in memory - a service
restart must never silently clear a real risk trip. Resuming trading after
an automatic halt is a deliberate operator action: delete the halt file.
A separate kill_switch_file lets an operator trigger the same immediate
flatten-and-halt manually, with no ML and no 24h delay involved.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class RiskGuard:
    def __init__(self, config):
        self.cfg = config.risk
        self.halt_file = str(self.cfg.halt_file)
        self.kill_switch_file = str(self.cfg.kill_switch_file)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True))

    @property
    def halted(self) -> bool:
        return os.path.exists(self.halt_file)

    @property
    def halt_reason(self) -> Optional[str]:
        if not self.halted:
            return None
        try:
            with open(self.halt_file) as f:
                return f.read().strip() or None
        except OSError:
            return None

    def trip(self, reason: str) -> None:
        """Idempotent: a second breach while already halted does not
        overwrite the original reason - the first cause is what matters."""
        if self.halted:
            return
        os.makedirs(os.path.dirname(self.halt_file) or ".", exist_ok=True)
        with open(self.halt_file, "w") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {reason}\n")
        logger.critical("[risk] HALT TRIPPED: %s", reason)

    def kill_switch_active(self) -> bool:
        return os.path.exists(self.kill_switch_file)

    # -- automatic checks (act on the trader's actual state) --------------

    def check_daily_loss(self, trader) -> Optional[str]:
        """Loss from the day's opening equity - trader._equity_history[-1]
        is exactly that: close_day() only appends to it once per day, at
        the LOGGING phase, so the last entry is always today's starting
        point regardless of whether today has closed yet."""
        if not trader._equity_history:
            return None
        day_start = trader._equity_history[-1]
        if day_start <= 0:
            return None
        equity = trader.equity()
        loss_pct = (day_start - equity) / day_start
        limit = float(self.cfg.max_daily_loss_pct)
        if loss_pct >= limit:
            return (f"daily loss {loss_pct:.1%} >= limit {limit:.1%} "
                    f"(day start ${day_start:,.2f} -> now ${equity:,.2f})")
        return None

    def check_drawdown(self, trader) -> Optional[str]:
        """Loss from the campaign's peak-ever equity (all persisted daily
        closes plus the current live mark). Daily-close granularity, same
        as the rest of the system's Sharpe/PnL accounting - intraday
        peaks between closes aren't captured, only the closes themselves."""
        equity = trader.equity()
        peak = max(trader._equity_history + [equity])
        if peak <= 0:
            return None
        dd_pct = (peak - equity) / peak
        limit = float(self.cfg.max_drawdown_pct)
        if dd_pct >= limit:
            return (f"drawdown {dd_pct:.1%} >= limit {limit:.1%} "
                    f"(peak ${peak:,.2f} -> now ${equity:,.2f})")
        return None

    def check(self, trader) -> Optional[str]:
        """Run every automatic check in priority order; returns the first
        breach reason, or None if clear. Does not itself trip - the caller
        (orchestrator._enforce_risk) decides when and how to act."""
        if not self.enabled:
            return None
        if self.kill_switch_active():
            return f"KILL_SWITCH file present ({self.kill_switch_file})"
        reason = self.check_daily_loss(trader)
        if reason:
            return reason
        return self.check_drawdown(trader)

    # -- pre-trade sanity check (independent of execute()'s own clipping) -

    def sanity_check_order(
        self, ticker: str, price: float, last_price: Optional[float],
        target_weight: float,
    ) -> Optional[str]:
        """Rejects an order BEFORE it reaches PaperTrader.execute(), on
        grounds execute() itself doesn't check: a price that jumped too far
        from the last known quote (bad data / fat-finger), or a target
        weight too large in magnitude to be a sane signal (protects
        against a NaN/inf/scaling bug upstream, independent of execute()'s
        own Kelly-cap clip - this check runs whether or not that clip is
        working correctly)."""
        max_dev = float(self.cfg.max_price_deviation_pct)
        if last_price and last_price > 0 and price > 0:
            dev = abs(price - last_price) / last_price
            if dev > max_dev:
                return (f"{ticker}: price {price:.4f} deviates {dev:.1%} from "
                        f"last known {last_price:.4f} (limit {max_dev:.1%}) - "
                        f"looks like bad data, rejecting")
        max_w = float(self.cfg.max_single_order_pct)
        if abs(target_weight) > max_w:
            return (f"{ticker}: target weight {target_weight:.1%} exceeds "
                    f"max_single_order_pct {max_w:.1%} - rejecting as a "
                    f"signal-computation anomaly")
        return None
