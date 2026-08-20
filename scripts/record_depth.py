"""
Start the Kite depth recorder. Intended to run on the Hetzner box.

    export KITE_API_KEY=...
    export KITE_ACCESS_TOKEN=...        # expires DAILY
    python scripts/record_depth.py

    # smoke test without waiting for the market to open
    python scripts/record_depth.py --duration 60 --any-hours

This produces nothing useful today. That is expected: market depth
cannot be backfilled from Kite at any price, so the dataset only exists
if it is being written down. Its value is entirely in three months'
time, which is why the right day to start is the earliest one.

RUNNING IT UNATTENDED. The access token expires every morning, so this
is not a fire-and-forget daemon. Two workable patterns:

  systemd with Restart=on-failure, plus a morning job that refreshes
  the token before market open; or

  a cron entry at ~09:10 IST that starts it with --duration 24000
  (~6h40m) so it exits cleanly after the close.

Either way the token refresh is a separate, interactive-ish step. The
recorder deliberately EXITS on an auth error rather than retrying,
because a process that spins on a stale token looks alive while
recording nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from nightevolver.depth_recorder import DEFAULT_DIR, DepthRecorder
from nightevolver.kite import KiteAuthError

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("record_depth")

DEFAULT_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                   "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK"]


def parse_args():
    p = argparse.ArgumentParser(description="Record Kite 5-level depth")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                   help="NSE symbols, space- or comma-separated")
    p.add_argument("--dir", default=str(DEFAULT_DIR),
                   help="output directory for depth_YYYYMMDD.jsonl.gz")
    p.add_argument("--duration", type=float, default=None,
                   help="stop after N seconds (default: run until signalled)")
    p.add_argument("--min-interval-ms", type=int, default=0,
                   help="minimum gap between records per instrument. 0 keeps "
                        "every book change, which is what order-flow research "
                        "wants; raise it only if disk is the binding constraint")
    p.add_argument("--any-hours", action="store_true",
                   help="do not wait for market hours (smoke testing)")
    args = p.parse_args()
    flat: List[str] = []
    for chunk in args.symbols:
        flat.extend(s.strip() for s in str(chunk).split(",") if s.strip())
    args.symbols = flat or DEFAULT_SYMBOLS
    return args


def main() -> int:
    args = parse_args()
    rec = DepthRecorder(args.symbols, directory=Path(args.dir),
                        min_interval_ms=args.min_interval_ms,
                        market_hours_only=not args.any_hours)
    rec.install_signal_handlers()
    logger.info("recording %d symbols -> %s", len(args.symbols), args.dir)
    try:
        stats = asyncio.run(rec.run(max_seconds=args.duration))
    except KiteAuthError as e:
        logger.error("AUTH: %s", e)
        return 2
    except KeyboardInterrupt:
        logger.info("interrupted")
        return 0

    if stats.written == 0:
        logger.warning(
            "recorded ZERO records. If the market was open this is a real "
            "failure, not a quiet day - check the token and the symbol list.")
        return 1
    logger.info("done. %s", stats.line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
