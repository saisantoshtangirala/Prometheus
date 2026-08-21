"""
Run the NightEvolver walk-forward. The measurement nothing ever called.

WHY THIS SCRIPT EXISTS. `nightevolver/backtest_evolved.py` implements the
one test that can answer "does this GA produce durable strategies" - a
walk-forward that RE-EVOLVES inside every window and scores strictly
out-of-sample. It has existed, correct and unused, with no caller:
`grep -rl backtest_evolved` found the module, its export, its unit test,
and the README. Not `scripts/train_nightevolver.py`, whose own docstring
says the backtest is "not even imported by this path". Not
`train-runpod.yml`. So every number this project has quoted about the GA
came from a SINGLE train/test split.

WHAT THE SINGLE SPLIT ACTUALLY SAID, in the checkpoint on disk:

    in-sample      3 trades over 459 bars, Sharpe 0.60
    out-of-sample  0 trades
    deflated P     0.0119   (gate is 0.95)
    beats_noise    False

The OOS Sharpe of 0.0 is not "no edge". It is "nothing happened" - the
strategy never fired in the holdout, so there was nothing to measure,
and the 0.60 "overfitting gap" is in-sample minus a number that does not
exist. That is the exact gap this run closes.

WHAT TO READ IN THE OUTPUT, and what to ignore:

  POOLED, trustworthy - these aggregate every window:
    raw hit rate + p-value   directional accuracy over
                             windows x test_bars x assets calls
    net Sharpe               after 22bp NSE round-trip
    deflated P(SR>0)         with n_trials = windows x GA budget, the
                             honest count for a search re-run per window
    mean overfitting gap     mean(in-sample Sharpe - OOS Sharpe)

  PER-WINDOW, NOT trustworthy: at a 21-bar test window, a ~49-day hold
  and a ~0.25 duty cycle, one window yields on the order of ONE trade
  per 10 assets. A per-window Sharpe computed on one trade is noise, and
  reading a run of green windows as a winning streak is exactly the
  mistake `required_validation_bars` was written to prevent. Only the
  pooled series has the sample size to carry a claim.

Offline by default: reads the cached UDiFF bhavcopy under data/cache,
so this reproduces without touching the network or a GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos.backtest import WalkForwardConfig                    # noqa: E402
from nightevolver.backtest_evolved import EvolvedWalkForward     # noqa: E402
from nightevolver.ga_engine import GAConfig                      # noqa: E402
from nightevolver.nse_prices import fetch_nse_prices             # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("walkforward")

DEFAULT_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
]


def resolve_universe(tickers, universe_n, as_of):
    """Explicit ticker list, or the top-N by turnover as of a date.

    BREADTH IS THE CHEAPEST STATISTICAL POWER AVAILABLE HERE. Trade count
    scales linearly with the number of assets, and trade count is the
    binding constraint on validating anything with a multi-week holding
    period - see ga_engine.required_validation_bars, which computes that
    a 90-day hold on 10 assets needs ~720 validation bars at the measured
    duty cycle, more history than the archive holds. At 100 assets the
    same claim needs ~72.

    `top_liquid_symbols` has existed and been correct since it was
    written, with zero callers. It costs no extra network - the bhavcopy
    is one file per session containing every listed security and is
    already downloaded and cached for the ten-name universe.

    POINT-IN-TIME. `as_of` should be at or before the first training
    window's start. Selecting "today's most liquid names" and testing
    them across two years of history is survivorship bias with extra
    steps: every name in that list is one that stayed liquid, which the
    strategy could not have known.
    """
    if not universe_n:
        return list(tickers)
    from nightevolver.nse_prices import top_liquid_symbols
    syms = top_liquid_symbols(as_of, n=universe_n)
    logger.info("universe: top %d by turnover as of %s (point-in-time)",
                len(syms), as_of)
    return syms


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    p.add_argument("--universe", type=int, default=None, metavar="N",
                   help="use the top N NSE equities by turnover instead of "
                        "--tickers. Breadth is the cheapest statistical "
                        "power available: trade count scales linearly with "
                        "the universe.")
    p.add_argument("--universe-as-of", default=None, metavar="DATE",
                   help="date for the point-in-time universe selection. "
                        "Defaults to --start, which is at or before every "
                        "training window and so avoids picking names for "
                        "having survived.")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=21,
                   help="OOS bars per window. See the module docstring on "
                        "why small values make PER-WINDOW stats unreadable "
                        "while leaving the POOLED series valid.")
    p.add_argument("--population", type=int, default=50)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--cost-bps", type=float, default=22.0,
                   help="NSE round-trip. 22bp is STT-dominated and is the "
                        "number the break-even win rate is derived from.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--null", action="store_true",
                   help="NULL CONTROL: block-permute the return series so "
                        "any real structure is destroyed, then run the "
                        "identical search. Answers 'what does this pipeline "
                        "produce on noise at this budget?' - without it, a "
                        "positive Sharpe has nothing to be compared against.")
    p.add_argument("--rebuild", action="store_true",
                   help="put REAL data through the same double-warmup trim "
                        "--null incurs, so the two runs are matched on "
                        "sample length and window count and the only "
                        "difference between them is the permutation.")
    p.add_argument("--block", type=int, default=21,
                   help="permutation block length in bars (default 21 ~ one "
                        "month, preserving short-horizon autocorrelation "
                        "while destroying longer-range predictability)")
    p.add_argument("--out", default="logs/backtests/evolved_walkforward.json")
    return p.parse_args()


def block_permute_prices(close, block: int, seed: int):
    """Destroy predictability, keep the marginal return distribution.

    Returns are cut into contiguous blocks and the BLOCKS are shuffled,
    then re-cumulated into a price path. Blocks rather than individual
    bars so short-horizon autocorrelation and volatility clustering
    survive - a bar-wise shuffle produces unrealistically clean data and
    makes the null too easy to beat, which would flatter the real run.

    Rows are permuted JOINTLY across assets, so the cross-sectional
    correlation structure of any given day is preserved. What is
    destroyed is the relationship between an indicator computed at t and
    the return that follows it - exactly the thing the GA searches for.
    """
    import pandas as pd

    rets = close.pct_change().fillna(0.0)
    n = len(rets)
    starts = list(range(0, n, block))
    rng = np.random.RandomState(seed)
    rng.shuffle(starts)
    order = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
    shuffled = rets.values[order]
    path = close.iloc[0].values * np.cumprod(1.0 + shuffled, axis=0)
    return pd.DataFrame(path, index=close.index[:len(path)],
                        columns=close.columns)


def main() -> int:
    args = parse_args()

    args.tickers = resolve_universe(args.tickers, args.universe,
                                    args.universe_as_of or args.start)
    logger.info("loading %d tickers from %s (cached bhavcopy)",
                len(args.tickers), args.start)
    try:
        # Returns MarketData directly - bhavcopy in, corporate-action
        # adjusted, no yfinance. Not the (close, high, low, volume) tuple
        # build_market_data takes.
        md = fetch_nse_prices(args.tickers, args.start, args.end,
                              use_cache=True, require_actions=True)
    except Exception as e:                                       # noqa: BLE001
        logger.error("could not load prices: %s", e)
        return 2
    if md is None or md.n_bars == 0:
        logger.error("no price data")
        return 2

    logger.info("market data: %d bars x %d tickers", md.n_bars, len(md.tickers))

    if args.rebuild and not args.null:
        # MATCHED CONTROL PATH. --null rebuilds MarketData from md.close,
        # and build_market_data trims WARMUP_BARS again - so the null
        # loses 61 bars the real run keeps (589 -> 528, 16 windows ->
        # 14). Comparing across different sample lengths weakens the one
        # comparison the whole exercise rests on. This flag puts the REAL
        # data through the identical double-build, so the only difference
        # between the two runs is the permutation itself.
        import pandas as pd

        from nightevolver.data_loader import build_market_data
        close = pd.DataFrame(md.close, index=pd.DatetimeIndex(md.dates),
                             columns=list(md.tickers))
        md = build_market_data(close)
        logger.info("rebuilt (matched to --null trimming): %d bars", md.n_bars)

    if args.null:
        # Rebuild the whole MarketData from permuted prices so every
        # indicator and forward return is recomputed consistently.
        # Permuting only the target would leave indicators correlated
        # with the ORIGINAL series and understate the null.
        import pandas as pd

        from nightevolver.data_loader import build_market_data
        close = pd.DataFrame(md.close, index=pd.DatetimeIndex(md.dates),
                             columns=list(md.tickers))
        perm = block_permute_prices(close, args.block, args.seed)
        md = build_market_data(perm)
        logger.warning("NULL CONTROL: prices block-permuted (block=%d, "
                       "seed=%d). Any edge reported below is manufactured "
                       "by the search itself.", args.block, args.seed)
        logger.info("null market data: %d bars x %d tickers",
                    md.n_bars, len(md.tickers))

    wf_cfg = WalkForwardConfig(train_window=args.train_window,
                               test_window=args.test_window,
                               cost_bps=args.cost_bps)
    ga_cfg = GAConfig(population_size=args.population,
                      n_generations=args.generations,
                      cost_bps=args.cost_bps, seed=args.seed)

    wf = EvolvedWalkForward(md, wf_cfg, ga_cfg)
    spans = wf.windows()
    if args.max_windows:
        spans = spans[:args.max_windows]
    budget = args.population * args.generations
    logger.info("%d windows x %d GA evaluations = %d trials",
                len(spans), budget, len(spans) * budget)
    if not spans:
        logger.error("no windows: %d bars is under train_window+1 = %d",
                     md.n_bars, args.train_window + 1)
        return 2

    result = wf.run(max_windows=args.max_windows)

    print("\n" + "=" * 72)
    print(result.summary())
    print("=" * 72)

    # Per-window trade counts, because a pooled Sharpe computed on almost
    # no trades is the failure this project has already hit once.
    trades = [w["oos_trades"] for w in result.windows]
    if trades:
        print(f"\nOOS trades per window: median {np.median(trades):.0f}, "
              f"min {min(trades)}, max {max(trades)}, total {sum(trades)}")
        if np.median(trades) < 2:
            print("  WARNING: a median under ~2 trades/window means the "
                  "PER-WINDOW figures carry no information. Read the pooled "
                  "row only, and widen --test-window to make windows "
                  "individually meaningful.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "tickers": list(args.tickers), "start": args.start, "end": args.end,
            "walk_forward": asdict(wf_cfg), "ga": asdict(ga_cfg),
            "max_windows": args.max_windows,
        },
        "n_bars": int(md.n_bars),
        "result": {
            "hit_rate": float(result.signal.hit_rate),
            "hit_rate_p_value": float(result.signal.hit_rate_p_value),
            "n_calls": int(result.signal.n_calls),
            "pearson_r": float(result.signal.pearson_r),
            "pearson_p": float(result.signal.pearson_p),
            "total_return": float(result.total_return),
            "cagr": float(result.cagr),
            "sharpe": float(result.sharpe),
            "deflated_sharpe_prob": float(result.deflated_sharpe_prob),
            "max_drawdown": float(result.max_drawdown),
            "win_rate": float(result.win_rate),
            "avg_turnover": float(result.avg_turnover),
            "n_windows": int(result.n_windows),
            "total_trials": int(result.total_trials),
            "mean_overfitting_gap": float(result.mean_overfitting_gap),
            "mean_in_sample_sharpe": float(result.mean_in_sample_sharpe),
        },
        "windows": result.windows,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
