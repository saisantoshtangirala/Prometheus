"""
Measure whether the inputs carry information about the targets - before
any optimiser is allowed near them.

    python scripts/run_information_audit.py                  # full audit
    python scripts/run_information_audit.py --no-flows       # indicators only
    python scripts/run_information_audit.py --synthetic      # calibration

This is the step that was missing from every previous cycle in this
project. The GA, the SNN and NEAT all answered "what is the best score a
search can reach on this data", which is a question about the search.
This answers "is there anything here", which is a question about the
data - and it is the one that determines whether searching is worth
doing at all.

Read the null result as informative. If nothing survives FDR correction,
that is a finding: it says the next move is different inputs, not a
different optimiser. It is also cheap to obtain, which is the point -
this runs in seconds, against days of GA and backtest time spent
learning the same thing less clearly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from nightevolver.data_loader import build_market_data
from nightevolver.genome import INDICATOR_NAMES, N_TECHNICAL
from nightevolver.information_audit import (
    DEFAULT_BLOCK_BARS, DEFAULT_FDR_ALPHA, DEFAULT_N_PERMUTATIONS, audit_features,
)
from nightevolver.targets import build_targets

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("audit.run")

DEFAULT_TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                   "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK"]


def parse_args():
    p = argparse.ArgumentParser(description="NightEvolver information audit")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                   help="NSE symbols, with or without .NS")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--no-flows", action="store_true",
                   help="skip FII/DII participant features")
    p.add_argument("--source", choices=["bhav", "yfinance"], default="bhav",
                   help="bhav = official NSE bhavcopy (adjusted via the "
                        "exchange's own PrvsClsgPric); yfinance is a fallback")
    p.add_argument("--permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    p.add_argument("--block-bars", type=int, default=DEFAULT_BLOCK_BARS)
    p.add_argument("--fdr-alpha", type=float, default=DEFAULT_FDR_ALPHA)
    p.add_argument("--synthetic", action="store_true",
                   help="run on a random walk - the calibration check. "
                        "Anything that 'survives' here is a false positive.")
    p.add_argument("--out", default="checkpoints/nightevolver/information_audit.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    flat: List[str] = []
    for chunk in args.tickers:
        flat.extend(t.strip() for t in str(chunk).split(",") if t.strip())
    args.tickers = flat or DEFAULT_TICKERS
    return args


def _synthetic_market(tickers, n_days: int = 900, seed: int = 0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.012, size=(n_days, len(tickers)))
    px = 100 * np.cumprod(1 + rets, axis=0)
    idx = pd.bdate_range("2022-01-01", periods=n_days)
    return build_market_data(pd.DataFrame(px, index=idx, columns=list(tickers)))


def main() -> int:
    args = parse_args()

    if args.synthetic:
        logger.info("SYNTHETIC random-walk mode - this is the calibration run. "
                    "Expect ZERO survivors; any survivor is a false positive "
                    "and means the audit's null is wrong.")
        md = _synthetic_market(args.tickers, seed=args.seed)
    elif args.source == "bhav":
        from nightevolver.nse_prices import fetch_nse_prices
        md = fetch_nse_prices(args.tickers, args.start, args.end,
                              with_flows=not args.no_flows)
    else:
        from nightevolver.data_loader import fetch_nse_data
        tk = [t if t.upper().endswith(".NS") else f"{t}.NS" for t in args.tickers]
        md = fetch_nse_data(tk, args.start, args.end)

    logger.info("data: %d bars x %d assets (%s .. %s)", md.n_bars, md.n_assets,
                md.dates[0].date(), md.dates[-1].date())

    # build_market_data already appends the market-wide flow channels
    # (zeros when unavailable), so the audit sees exactly the channel set
    # the genome votes on - no second, divergent assembly path.
    features = md.indicators                       # [T, A, N_INDICATORS]
    names = list(INDICATOR_NAMES)
    logger.info("features: %d channels (%d technical + %d flow)",
                len(names), N_TECHNICAL, len(names) - N_TECHNICAL)

    targets = build_targets(md.close)
    for t in targets.values():
        logger.info("target %-16s valid rows=%d kind=%s persistence-prone=%s",
                    t.name, t.n_valid, t.kind, t.autocorr_baseline)

    result = audit_features(features, names, targets, md.close,
                            block_bars=args.block_bars,
                            n_permutations=args.permutations,
                            fdr_alpha=args.fdr_alpha, seed=args.seed)
    print("\n" + result.summary())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "n_bars": result.n_bars, "n_assets": result.n_assets,
            "tickers": args.tickers, "start": args.start, "end": args.end,
            "synthetic": args.synthetic, "source": args.source,
            "block_bars": result.block_bars,
            "n_permutations": result.n_permutations,
            "fdr_alpha": result.fdr_alpha,
            "n_pairs": len(result.pairs),
            "n_survivors": len(result.survivors),
            "pairs": [{
                "feature": p.feature, "target": p.target,
                "spearman": p.spearman,
                "spearman_incremental": p.spearman_incremental,
                "p_value": p.p_value,
                "p_value_incremental": p.p_value_incremental,
                "q_value": p.q_value, "significant": p.significant,
                "n_effective": p.n_effective,
            } for p in result.pairs],
        }, f, indent=2)
    logger.info("written: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
