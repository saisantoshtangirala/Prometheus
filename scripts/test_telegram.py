"""
Send a one-off test Telegram message to confirm your bot is configured
correctly, without waiting for a real trading day to close.

Usage:
  python scripts/test_telegram.py

Requires KRONOS_TELEGRAM_BOT_TOKEN and KRONOS_TELEGRAM_CHAT_ID to already
be set in the environment (e.g. sourced from /etc/kronos.env - see
scripts/hetzner_bootstrap.sh) and notifications.enabled: true in
kronos/config.yaml (or pass --force to bypass the config check for this
one test message).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos import TelegramNotifier, load_config


def main():
    parser = argparse.ArgumentParser(description="Send a test Kronos Telegram message")
    parser.add_argument("--force", action="store_true",
                        help="send even if notifications.enabled is false in config")
    args = parser.parse_args()

    config = load_config()
    notifier = TelegramNotifier(config)

    if not notifier.bot_token or not notifier.chat_id:
        print("FAILED: KRONOS_TELEGRAM_BOT_TOKEN and/or KRONOS_TELEGRAM_CHAT_ID "
              "are not set in the environment.")
        print("Set them in /etc/kronos.env (see hetzner_bootstrap.sh) and "
              "re-run, or export them directly in this shell for a quick test.")
        sys.exit(1)

    if not notifier.enabled and not args.force:
        print("Notifications are configured but notifications.enabled is "
              "false in kronos/config.yaml.")
        print("Set it to true, or re-run this script with --force to send "
              "a one-off test message anyway.")
        sys.exit(1)

    if args.force:
        config.override("notifications.enabled", True)
        notifier = TelegramNotifier(config)

    text = (
        "Kronos test message.\n\n"
        "If you're reading this on Telegram, notifications are configured "
        "correctly. Daily progress reports will arrive after each day's "
        "market close."
    )
    ok = notifier.send(text)
    if ok:
        print("Sent. Check Telegram.")
    else:
        print("Send failed - check the log output above for the reason "
              "(common causes: bot token typo'd, or you haven't sent your "
              "bot a first message yet - Telegram requires that before a "
              "bot can message you back).")
        sys.exit(1)


if __name__ == "__main__":
    main()
