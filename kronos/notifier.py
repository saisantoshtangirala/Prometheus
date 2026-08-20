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

import hashlib
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
MAX_MESSAGE_CHARS = 4000   # Telegram's real limit is 4096; leave margin

# Identical alerts inside this window are suppressed and counted. 15
# minutes is long enough to collapse a failure storm into one message,
# short enough that a genuinely recurring condition is re-reported.
DEDUPE_WINDOW_SECONDS = 900
# One bounded retry on 429/5xx. The alert channel for an unattended
# trading system must not drop a critical message because the first
# attempt was throttled.
SEND_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.5


class TelegramNotifier:
    """Sends the daily progress digest via Telegram. Every failure is
    caught and logged - a notification problem must never take down the
    trading loop."""

    def __init__(self, config):
        self.cfg = config
        self.bot_token = os.environ.get("KRONOS_TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.environ.get("KRONOS_TELEGRAM_CHAT_ID", "").strip()
        # message-hash -> monotonic time of last successful send / count
        self._last_sent: Dict[str, float] = {}
        self._suppressed: Dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        notif_cfg = self.cfg.get("notifications", None)
        cfg_on = bool(notif_cfg.get("enabled", False)) if notif_cfg else False
        return cfg_on and bool(self.bot_token) and bool(self.chat_id)

    def send(self, text: str, dedupe: bool = True) -> bool:
        """Best-effort send. Returns True on success, never raises.

        DEDUPLICATION AND RETRY, both added by audit.

        Before: `send()` posted once and returned False on any failure.
        Two consequences, and the second is worse than the first.

        1. SPAM. A data-fetch failure inside a loop produced one Telegram
           message per occurrence. Telegram then rate-limits, and the
           alerts most worth reading are the ones dropped - because a
           storm is exactly when something is wrong.
        2. SILENT LOSS. An HTTP 429 or a 5xx was logged at WARNING and
           discarded. The primary human interface for a system that
           trades unattended must not drop a critical alert because the
           first attempt was throttled.

        Identical messages inside DEDUPE_WINDOW_SECONDS are suppressed
        (counted, and the count is reported when the window closes), and
        429/5xx get one bounded retry. Pass `dedupe=False` for a message
        that must always go out, such as a daily report.
        """
        if not self.enabled:
            logger.debug("[notifier] disabled or unconfigured - skipping send")
            return False
        text = text[:MAX_MESSAGE_CHARS]

        if dedupe:
            now = time.monotonic()
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            last = self._last_sent.get(key)
            if last is not None and (now - last) < DEDUPE_WINDOW_SECONDS:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
                logger.debug("[notifier] suppressed duplicate (%d in window)",
                             self._suppressed[key])
                return False
            repeats = self._suppressed.pop(key, 0)
            if repeats:
                text = (f"{text}\n\n(+{repeats} identical alert(s) suppressed "
                        f"in the last {DEDUPE_WINDOW_SECONDS // 60}m)")[:MAX_MESSAGE_CHARS]
            self._last_sent[key] = now
            self._prune_dedupe_state(now)

        url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage"
        for attempt in range(SEND_ATTEMPTS):
            try:
                import requests
                resp = requests.post(
                    url,
                    json={"chat_id": self.chat_id, "text": text},
                    timeout=15,
                )
                if resp.status_code == 200:
                    logger.info("[notifier] Telegram report sent")
                    return True
                retryable = resp.status_code == 429 or resp.status_code >= 500
                logger.warning(
                    "[notifier] Telegram send returned status %d: %s%s",
                    resp.status_code, resp.text[:200],
                    " (retrying)" if retryable and attempt < SEND_ATTEMPTS - 1 else "",
                )
                if not retryable:
                    return False
            except Exception as e:
                logger.warning("[notifier] Telegram send failed: %s", e,
                               exc_info=True)
            if attempt < SEND_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
        return False

    def _prune_dedupe_state(self, now: float) -> None:
        """Keep the dedupe map from growing without bound in a process
        that runs for 365 days."""
        stale = [k for k, t in self._last_sent.items()
                 if now - t > DEDUPE_WINDOW_SECONDS * 4]
        for k in stale:
            self._last_sent.pop(k, None)
            self._suppressed.pop(k, None)

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
