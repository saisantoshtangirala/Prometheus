"""
Signal-direction diagnostic CLI.

Answers one narrow question, isolated from position sizing, transaction
costs, and cross-asset netting: does the strategy's predicted direction
correlate with the REALIZED direction of the asset it's predicting, at
all? See kronos/backtest.py's SignalDiagnostic docstring for why this is
a different (and more fundamental) question than the walk-forward
backtest's own Hit Rate column.

With internet (real data, the run that matters):
  python scripts/diagnose_signal_direction.py --tickers RELIANCE.NS TCS.NS ... --start 2015-01-01

Offline / air-gapped:
  python scripts/diagnose_signal_direction.py --csv data/closes.csv --tickers SPY QQQ
  python scripts/diagnose_signal_direction.py --synthetic   # harness validation only
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos.backtest import (
    STRATEGIES,
    WalkForwardBacktester,
    WalkForwardConfig,
    load_history,
    render_signal_diagnostic_report,
    synthetic_history,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kronos.backtest.diagnose")


def main():
    p = argparse.ArgumentParser(description="Kronos signal-direction diagnostic")
    p.add_argument("--tickers", nargs="+",
                   default=["SPY", "QQQ", "IWM", "GLD", "TLT"])
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None, help="offline CSV of close prices")
    p.add_argument("--synthetic", action="store_true",
                   help="regime-switching synthetic data (harness test only)")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=21)
    p.add_argument("--strategies", nargs="+", default=["kronos", "momentum"],
                   choices=list(STRATEGIES))
    args = p.parse_args()

    if args.synthetic:
        closes = synthetic_history(args.tickers)
        data_label = "SYNTHETIC (regime-switching GBM) - says NOTHING about real markets"
    else:
        closes = load_history(args.tickers, args.start, args.end, args.csv)
        data_label = (
            f"{'CSV ' + args.csv if args.csv else 'yfinance'} "
            f"{closes.index[0].date()} .. {closes.index[-1].date()}"
        )

    bt = WalkForwardBacktester(
        closes,
        WalkForwardConfig(train_window=args.train_window, test_window=args.test_window),
    )
    logger.info("data: %s | %d bars | %d walk-forward windows",
                data_label, len(bt.returns), len(bt.windows()))

    print("\n" + "=" * 70)
    for name in args.strategies:
        strategy = STRATEGIES[name]()
        logger.info("[diagnose] running strategy: %s", name)
        diag = bt.diagnose_signal_direction(strategy)
        print(render_signal_diagnostic_report(diag))
        print("=" * 70)


if __name__ == "__main__":
    main()
