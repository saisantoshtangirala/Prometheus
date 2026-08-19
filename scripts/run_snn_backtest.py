"""
Walk-forward backtest CLI for the ACTUAL production model (ReflexArc's
SNN) - see kronos/backtest_snn.py's module docstring for why this is a
different, more fundamental test than scripts/run_backtest.py's.

With internet (real data, the run that matters):
  python scripts/run_snn_backtest.py --tickers RELIANCE.NS TCS.NS ... --start 2015-01-01

Sanity-check a small number of windows before committing to a full run:
  python scripts/run_snn_backtest.py --tickers ... --start 2015-01-01 --max-windows 10
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos.backtest import load_history, render_signal_diagnostic_report, synthetic_history
from kronos.backtest_snn import SNNTrainConfig, SNNWalkForwardBacktester, WalkForwardConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kronos.backtest_snn.cli")


def main():
    p = argparse.ArgumentParser(description="Kronos SNN (production model) walk-forward backtest")
    p.add_argument("--tickers", nargs="+",
                   default=["SPY", "QQQ", "IWM", "GLD", "TLT"])
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None, help="offline CSV of close prices")
    p.add_argument("--synthetic", action="store_true",
                   help="regime-switching synthetic data (harness test only)")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=21)
    p.add_argument("--max-windows", type=int, default=None,
                   help="sanity-check a subset before the full run")
    p.add_argument("--pretrain-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--meta-epochs", type=int, default=10)
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

    bt = SNNWalkForwardBacktester(
        closes, tickers=args.tickers,
        config=WalkForwardConfig(train_window=args.train_window, test_window=args.test_window),
        train_cfg=SNNTrainConfig(
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            meta_epochs=args.meta_epochs,
        ),
    )
    n_total = len(bt.windows())
    n_run = min(args.max_windows, n_total) if args.max_windows else n_total
    logger.info("data: %s | %d bars | %d/%d walk-forward windows",
                data_label, len(bt.returns), n_run, n_total)

    diag = bt.run_signal_diagnostic(max_windows=args.max_windows)

    print("\n" + "=" * 70)
    print(render_signal_diagnostic_report(diag))
    print("=" * 70)


if __name__ == "__main__":
    main()
