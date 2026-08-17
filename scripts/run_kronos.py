"""
Kronos entry point - starts the 365-day self-evolving paper trading loop.

Usage:
  python scripts/run_kronos.py --mode=paper                 # the real loop
  python scripts/run_kronos.py --mode=paper --days 3        # short run
  python scripts/run_kronos.py --mode=replay --accelerated  # compressed test day

In paper mode Kronos follows the wall clock: it sleeps until each phase
boundary, executes the phase, and ticks the reflex arc every minute during
market hours. In accelerated mode each "day" runs back-to-back with
synthetic market ticks - useful for CI and burn-in testing.
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos import KronosOrchestrator, Phase, load_config

# Must exist before FileHandler below opens it - logging.basicConfig runs at
# import time, well before main()'s own Path("logs").mkdir() would run.
Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/kronos.log"),
    ],
)
logger = logging.getLogger("kronos.main")

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.warning("Signal %s received - finishing current phase then stopping", signum)
    _shutdown = True


def synthetic_ticks(orchestrator, n_ticks: int = 20):
    """Generate plausible market ticks for accelerated/replay mode."""
    rng = np.random.default_rng()
    memory = orchestrator.state.memory
    tickers = memory.tickers if memory else list(orchestrator.cfg.data.tickers)
    base_prices = {
        t: float(memory.prices[t].iloc[-1]) if memory is not None else 100.0
        for t in tickers
    }
    vix = memory.macro.get("vix_last", 20.0) if memory else 20.0
    ticks = []
    for _ in range(n_ticks):
        vix = max(9.0, vix + rng.normal(0, 0.5))
        prices = {
            t: p * (1 + rng.normal(0, 0.002)) for t, p in base_prices.items()
        }
        volumes = {t: float(rng.integers(500_000, 50_000_000)) for t in tickers}
        base_prices = prices
        ticks.append((vix, prices, volumes))
    return ticks


def run_accelerated(orchestrator, n_days: int):
    """Compressed back-to-back days with synthetic intraday ticks."""
    for day in range(1, n_days + 1):
        if _shutdown:
            break
        logger.info("=" * 60)
        logger.info(" KRONOS ACCELERATED DAY %d / %d", day, n_days)
        logger.info("=" * 60)
        orchestrator.state.day = day
        orchestrator.run_digestion()
        ticks = synthetic_ticks(orchestrator)
        orchestrator.run_nightmare()
        orchestrator.run_evolution()
        orchestrator.run_adaptation()
        orchestrator.run_report()
        for vix, prices, volumes in ticks:
            orchestrator.run_reflex_tick(vix, prices, volumes)
            orchestrator.heartbeat()
        orchestrator.run_logging()


PRE_MARKET_PHASES = [Phase.DIGESTION, Phase.NIGHTMARE, Phase.EVOLUTION,
                     Phase.ADAPTATION, Phase.REPORT]


def catch_up(orchestrator, executed_today: set, day: int, now=None) -> None:
    """
    Run any pre-market phase whose window has already opened today but
    hasn't executed yet - covers starting (or restarting) the service
    mid-day, past one or more phase boundaries. Without this, a restart
    during market hours (e.g. every deploy-hetzner.yml redeploy) would
    silently skip that entire day's digestion/nightmare/evolution/
    adaptation/report and leave the reflex arc running with no memory
    until the next midnight UTC boundary.
    """
    now = now or datetime.now(timezone.utc)
    current_phase = orchestrator.phase_for(now)
    try:
        cutoff = PRE_MARKET_PHASES.index(current_phase)
    except ValueError:
        # REFLEX or LOGGING: every pre-market phase's window has passed
        cutoff = len(PRE_MARKET_PHASES) - 1

    for phase in PRE_MARKET_PHASES[: cutoff + 1]:
        if phase.value in executed_today:
            continue
        executed_today.add(phase.value)
        logger.info("[main] catch-up: entering phase: %s (day %d)", phase.value, day)
        if phase == Phase.DIGESTION:
            orchestrator.state.day = day
            orchestrator.run_digestion()
        elif phase == Phase.NIGHTMARE:
            orchestrator.run_nightmare()
        elif phase == Phase.EVOLUTION:
            orchestrator.run_evolution()
        elif phase == Phase.ADAPTATION:
            orchestrator.run_adaptation()
        elif phase == Phase.REPORT:
            orchestrator.run_report()


def run_realtime(orchestrator, n_days: int):
    """Wall-clock loop: execute each phase when its window opens."""
    executed_today = set()
    day = 1
    current_date = datetime.now(timezone.utc).date()
    orchestrator.state.day = day

    logger.info("Kronos realtime loop started (target %d days)", n_days)
    catch_up(orchestrator, executed_today, day)

    while day <= n_days and not _shutdown:
        now = datetime.now(timezone.utc)
        if now.date() != current_date:
            current_date = now.date()
            executed_today.clear()
            day += 1
            orchestrator.state.day = day

        phase = orchestrator.phase_for(now)

        if phase == Phase.REFLEX:
            # tick once per minute during market hours
            memory = orchestrator.state.memory
            if memory is not None:
                vix = memory.macro.get("vix_last", 20.0)
                orchestrator.run_reflex_tick(vix)
                orchestrator.heartbeat(now)
            time.sleep(60)
            continue

        if phase.value not in executed_today:
            executed_today.add(phase.value)
            logger.info("[main] entering phase: %s (day %d)", phase.value, day)
            if phase == Phase.DIGESTION:
                orchestrator.state.day = day
                orchestrator.run_digestion()
            elif phase == Phase.NIGHTMARE:
                orchestrator.run_nightmare()
            elif phase == Phase.EVOLUTION:
                orchestrator.run_evolution()
            elif phase == Phase.ADAPTATION:
                orchestrator.run_adaptation()
            elif phase == Phase.REPORT:
                orchestrator.run_report()
            elif phase == Phase.LOGGING:
                orchestrator.run_logging()

        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description="Kronos 365-day organism")
    parser.add_argument("--mode", choices=["paper", "replay"], default="paper")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--accelerated", action="store_true",
                        help="compress days back-to-back (testing/burn-in)")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    config = load_config(args.config)
    orchestrator = KronosOrchestrator(config)

    logger.info("Kronos initialized: mode=%s days=%d accelerated=%s",
                args.mode, args.days, args.accelerated)

    if args.accelerated or args.mode == "replay":
        run_accelerated(orchestrator, args.days)
    else:
        run_realtime(orchestrator, args.days)

    logger.info("Kronos loop finished. Equity: $%.2f", orchestrator.trader.equity())
    orchestrator.trader.close()


if __name__ == "__main__":
    main()
