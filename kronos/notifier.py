"""
Kronos daily progress notifier - a WhatsApp summary sent to your own
phone at the end of each trading day, via CallMeBot.

CallMeBot (https://www.callmebot.com/blog/free-api-whatsapp-messages/) is
a free, unofficial third-party service built specifically for "let my
script message my own WhatsApp" automation - no business account, no
paid API, no signup beyond opting your own number in. It is NOT
affiliated with WhatsApp/Meta, is rate-limited, and could change or go
offline without notice - acceptable for a personal daily digest, not
something to depend on for anything time-critical. Twilio's WhatsApp
Business API is the official, paid upgrade path if you ever need one.

One-time setup (on your phone, not this server):
  1. Save this contact: +34 644 84 71 64
  2. WhatsApp it exactly: "I allow callmebot to send me messages"
  3. CallMeBot replies with your personal API key.

Then set two environment variables wherever Kronos runs (see
hetzner_bootstrap.sh - it creates /etc/kronos.env for exactly this,
kept out of git):
  KRONOS_WHATSAPP_PHONE=<your number with country code, digits only>
  KRONOS_WHATSAPP_APIKEY=<the key CallMeBot sent you>

Notifications are fully optional: with either variable unset, or
notifications.enabled: false in config.yaml, send() is a silent no-op -
nothing else in Kronos depends on this working.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CALLMEBOT_URL = "https://api.callmebot.com/whatsapp.php"
MAX_MESSAGE_CHARS = 2000   # generous WhatsApp-safe cap; our reports are short


class WhatsAppNotifier:
    """Sends the daily progress digest. Every failure is caught and logged -
    a notification problem must never take down the trading loop."""

    def __init__(self, config):
        self.cfg = config
        self.phone = os.environ.get("KRONOS_WHATSAPP_PHONE", "").strip()
        self.apikey = os.environ.get("KRONOS_WHATSAPP_APIKEY", "").strip()

    @property
    def enabled(self) -> bool:
        notif_cfg = self.cfg.get("notifications", None)
        cfg_on = bool(notif_cfg.get("enabled", False)) if notif_cfg else False
        return cfg_on and bool(self.phone) and bool(self.apikey)

    def send(self, text: str) -> bool:
        """Best-effort send. Returns True on success, never raises."""
        if not self.enabled:
            logger.debug("[notifier] disabled or unconfigured - skipping send")
            return False
        text = text[:MAX_MESSAGE_CHARS]
        try:
            import requests
            resp = requests.get(
                CALLMEBOT_URL,
                params={"phone": self.phone, "text": text, "apikey": self.apikey},
                timeout=15,
            )
            ok = resp.status_code == 200
            if ok:
                logger.info("[notifier] WhatsApp report sent")
            else:
                logger.warning(
                    "[notifier] WhatsApp send returned status %d: %s",
                    resp.status_code, resp.text[:200],
                )
            return ok
        except Exception as e:
            logger.warning("[notifier] WhatsApp send failed: %s", e)
            return False

    # -- report formatting ---------------------------------------------

    def build_daily_report(
        self,
        day: int,
        stats: Dict,
        total_days: Optional[int] = None,
        regime: Optional[str] = None,
        top_fitness: Optional[List[float]] = None,
        source_used: Optional[str] = None,
        quality_flags: Optional[List[str]] = None,
        phase_failures: Optional[Dict[str, str]] = None,
        reflex_regime: Optional[str] = None,
    ) -> str:
        """Build the plain-text WhatsApp message for one day's close."""
        total = total_days or int(self.cfg.run.total_days)
        pct = (day / total * 100.0) if total else 0.0

        lines = [
            f"Kronos - Day {day}/{total} ({pct:.1f}% complete)",
            "",
            f"Equity: ${stats.get('equity', 0.0):,.2f}",
            f"Today's PnL: {stats.get('pnl', 0.0):+,.2f}",
        ]
        sharpe = stats.get("sharpe")
        if sharpe is not None:
            lines.append(f"Sharpe (running): {sharpe:.2f}")
        dir_acc = stats.get("directional_accuracy")
        if dir_acc is not None:
            lines.append(f"Directional accuracy: {dir_acc:.0%}")
        lines.append(f"Trades today: {stats.get('n_trades', 0)}")
        max_pos = stats.get("max_position_pct")
        if max_pos is not None:
            lines.append(f"Largest position: {max_pos:.1%} (cap enforced)")

        if regime:
            lines.append(f"Market regime: {regime}")
        if reflex_regime and reflex_regime != "calm":
            lines.append(f"Reflex gate: {reflex_regime.upper()}")
        if top_fitness:
            lines.append(f"NEAT best fitness: {top_fitness[0]:.3f}")
        if source_used:
            lines.append(f"Data source: {source_used}")

        if quality_flags:
            notable = [f for f in quality_flags if not f.startswith("kalman_repaired")]
            if notable:
                lines.append("")
                lines.append(f"Data flags: {', '.join(notable[:5])}")

        if phase_failures:
            lines.append("")
            lines.append(f"WARNINGS ({len(phase_failures)}): " +
                        ", ".join(phase_failures.keys()))

        return "\n".join(lines)
