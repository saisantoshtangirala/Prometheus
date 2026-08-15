"""
Walk-forward backtest CLI - get a real number instead of a feeling.

With internet (real data, the run that matters):
  python scripts/run_backtest.py --tickers SPY QQQ IWM GLD TLT --start 2015-01-01

Offline / air-gapped:
  python scripts/run_backtest.py --csv data/closes.csv --tickers SPY QQQ
  python scripts/run_backtest.py --synthetic          # harness validation only

Useful knobs:
  --train-window 252   bars of training per fold
  --test-window 21     out-of-sample bars traded per fold
  --cost-bps 10        transaction cost per unit turnover
  --strategies kronos momentum buy_hold
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
    render_report,
    save_history,
    synthetic_history,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kronos.backtest.cli")


def main():
    p = argparse.ArgumentParser(description="Kronos walk-forward backtest")
    p.add_argument("--tickers", nargs="+",
                   default=["SPY", "QQQ", "IWM", "GLD", "TLT"])
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None, help="offline CSV of close prices")
    p.add_argument("--save-csv", default=None,
                   help="save fetched closes here for offline reuse")
    p.add_argument("--synthetic", action="store_true",
                   help="regime-switching synthetic data (harness test only)")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=21)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES),
                   choices=list(STRATEGIES))
    p.add_argument("--out-dir", default="logs/backtests")
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
        if args.save_csv:
            save_history(closes, args.save_csv)
            logger.info("saved closes to %s", args.save_csv)

    bt = WalkForwardBacktester(
        closes,
        WalkForwardConfig(
            train_window=args.train_window,
            test_window=args.test_window,
            cost_bps=args.cost_bps,
            n_trials=len(args.strategies),
        ),
    )
    logger.info("data: %s | %d bars | %d walk-forward windows",
                data_label, len(bt.returns), len(bt.windows()))

    results = bt.compare(args.strategies)
    md_path = render_report(results, data_label, out_dir=args.out_dir)

    print("\n" + "=" * 70)
    for name, res in results.items():
        print(f"  {name:10s} total={res.total_return:+8.1%}  "
              f"sharpe={res.sharpe:6.2f}  deflatedP={res.deflated_sharpe_prob:.2f}  "
              f"maxDD={res.max_drawdown:6.1%}  turnover={res.avg_turnover:.2f}")
    print("=" * 70)
    print(f"\nFull report: {md_path}")


if __name__ == "__main__":
    main()
