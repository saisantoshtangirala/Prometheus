"""
Kronos daily progress notifier - a Telegram message sent to your own
phone at the end of each trading day.

Uses Telegram's official Bot API (https://core.telegram.org/bots/api) -
a first-party feature of Telegram itself, not a third-party workaround.
You create your own bot via Telegram's own @BotFather, so no unaccountable
third party is ever in the loop - just you, your bot, and Telegram.

One-time setup (on your phone):
  1. Open Telegram, search for @BotFather (Telegram's official bot for
     creating bots), start a chat with it.
  2. Send: /newbot
     Follow the prompts - pick a display name, then a username ending
     in "bot" (e.g. "kronos_yourname_bot").
  3. BotFather replies with a token that looks like:
       123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
     That's KRONOS_TELEGRAM_BOT_TOKEN.
  4. Search for YOUR new bot by its username and send it any message
     (e.g. "hi") - Telegram requires this before a bot can message you
     back, as an anti-spam measure.
  5. In a browser, visit (with your real token):
       https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
     Find "chat":{"id": ...} in the JSON response - that number is
     KRONOS_TELEGRAM_CHAT_ID.

Set both as environment variables wherever Kronos runs (see
hetzner_bootstrap.sh - it creates /etc/kronos.env for exactly this,
kept out of git):
  KRONOS_TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
  KRONOS_TELEGRAM_CHAT_ID=987654321

Notifications are fully optional: with either variable unset, or
notifications.enabled: false in config.yaml, send() is a silent no-op -
nothing else in Kronos depends on this working.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
MAX_MESSAGE_CHARS = 4000   # Telegram's real limit is 4096; leave margin


class TelegramNotifier:
    """Sends the daily progress digest via Telegram. Every failure is
    caught and logged - a notification problem must never take down the
    trading loop."""

    def __init__(self, config):
        self.cfg = config
        self.bot_token = os.environ.get("KRONOS_TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.environ.get("KRONOS_TELEGRAM_CHAT_ID", "").strip()

    @property
    def enabled(self) -> bool:
        notif_cfg = self.cfg.get("notifications", None)
        cfg_on = bool(notif_cfg.get("enabled", False)) if notif_cfg else False
        return cfg_on and bool(self.bot_token) and bool(self.chat_id)

    def send(self, text: str) -> bool:
        """Best-effort send. Returns True on success, never raises."""
        if not self.enabled:
            logger.debug("[notifier] disabled or unconfigured - skipping send")
            return False
        text = text[:MAX_MESSAGE_CHARS]
        try:
            import requests
            url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage"
            resp = requests.post(
                url,
                json={"chat_id": self.chat_id, "text": text},
                timeout=15,
            )
            ok = resp.status_code == 200
            if ok:
                logger.info("[notifier] Telegram report sent")
            else:
                logger.warning(
                    "[notifier] Telegram send returned status %d: %s",
                    resp.status_code, resp.text[:200],
                )
            return ok
        except Exception as e:
            logger.warning("[notifier] Telegram send failed: %s", e)
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
        runpod_status: Optional[str] = None,
    ) -> str:
        """Build the plain-text Telegram message for one day's close."""
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
        if runpod_status == "adopted":
            lines.append("RunPod: adopted fresh checkpoint")
        elif runpod_status == "unchanged":
            lines.append("RunPod: none today (kept yesterday's)")

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
