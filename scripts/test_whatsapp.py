"""
Send a one-off test WhatsApp message to confirm CallMeBot is configured
correctly, without waiting for a real trading day to close.

Usage:
  python scripts/test_whatsapp.py

Requires KRONOS_WHATSAPP_PHONE and KRONOS_WHATSAPP_APIKEY to already be
set in the environment (e.g. sourced from /etc/kronos.env - see
scripts/hetzner_bootstrap.sh) and notifications.enabled: true in
kronos/config.yaml (or pass --force to bypass the config check for this
one test message).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos import WhatsAppNotifier, load_config


def main():
    parser = argparse.ArgumentParser(description="Send a test Kronos WhatsApp message")
    parser.add_argument("--force", action="store_true",
                        help="send even if notifications.enabled is false in config")
    args = parser.parse_args()

    config = load_config()
    notifier = WhatsAppNotifier(config)

    if not notifier.phone or not notifier.apikey:
        print("FAILED: KRONOS_WHATSAPP_PHONE and/or KRONOS_WHATSAPP_APIKEY "
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
        notifier = WhatsAppNotifier(config)

    text = (
        "Kronos test message.\n\n"
        "If you're reading this on WhatsApp, notifications are configured "
        "correctly. Daily progress reports will arrive after each day's "
        "market close."
    )
    ok = notifier.send(text)
    if ok:
        print("Sent. Check your WhatsApp.")
    else:
        print("Send failed - check the log output above for the reason "
              "(common causes: API key not activated yet, phone number "
              "format wrong - digits only, with country code, no + or "
              "spaces).")
        sys.exit(1)


if __name__ == "__main__":
    main()
