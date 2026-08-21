"""
The null CLOUD: many permutations, not one.

WHY ONE NULL IS NOT ENOUGH. `run_evolved_walkforward.py --null` runs a
single block permutation and reports an overfitting gap of +2.84 against
the real run's +2.75. Two numbers that close look like agreement - but a
single draw has no error bar, so "the same" is an eyeball judgement, not
a measurement. If the null's gap varies by +-1.5 across permutations then
+2.84 and +2.75 are indistinguishable; if it varies by +-0.1 they are
meaningfully different and the real data has slightly LESS overfitting
than noise, which would itself need explaining.

This runs N independent permutations through the identical pipeline and
reports the distribution, so the real result can be placed as a
PERCENTILE within it rather than compared to a point.

WHAT IS HELD FIXED, and why it matters. The GA seed stays at the real
run's value; only the permutation seed varies. The cloud therefore
answers exactly one question - "what does THIS search, at THIS budget,
produce on data with no signal?" - rather than blurring search
randomness and permutation randomness together. Every draw uses the same
window geometry and the same 1,000-evaluation budget as the real run,
because a null run at a smaller budget would overfit less for reasons
that have nothing to do with the data.

MATCHED TRIMMING. Both the real reference and every null draw go through
build_market_data twice (see --rebuild in run_evolved_walkforward.py), so
sample length and window count are identical across all of them and the
permutation is the only difference.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from kronos.backtest import WalkForwardConfig                    # noqa: E402
from nightevolver.backtest_evolved import EvolvedWalkForward     # noqa: E402
from nightevolver.data_loader import build_market_data           # noqa: E402
from nightevolver.ga_engine import GAConfig                      # noqa: E402
from nightevolver.nse_prices import fetch_nse_prices             # noqa: E402
from run_evolved_walkforward import (                            # noqa: E402
    DEFAULT_TICKERS, block_permute_prices, resolve_universe,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nullcloud")


def _one_draw(args):
    """One permutation end to end. Runs in a worker process."""
    values, index, columns, perm_seed, block, wf_kw, ga_kw = args
    # Silence the GA's per-generation logging in workers - 30 draws x 14
    # windows x 20 generations is 8,400 lines of noise.
    logging.disable(logging.INFO)

    # Passed as (values, index, columns) rather than JSON: pandas'
    # read_json no longer accepts a raw JSON string (it reads the
    # argument as a path), and a DataFrame pickles cleanly to a worker
    # anyway, with no epoch-millisecond index round-trip to get wrong.
    close = pd.DataFrame(values, index=index, columns=columns)
    if perm_seed is None:
        md = build_market_data(close)              # the REAL reference
    else:
        md = build_market_data(block_permute_prices(close, block, perm_seed))

    wf = EvolvedWalkForward(md, WalkForwardConfig(**wf_kw), GAConfig(**ga_kw))
    r = wf.run()
    oos = [w["oos_sharpe"] for w in r.windows]
    return {
        "perm_seed": perm_seed,
        "n_bars": int(md.n_bars),
        "n_windows": int(r.n_windows),
        "gap": float(r.mean_overfitting_gap),
        "in_sample_sharpe": float(r.mean_in_sample_sharpe),
        "mean_oos_sharpe": float(np.mean(oos)) if oos else 0.0,
        "pooled_sharpe": float(r.sharpe),
        "total_return": float(r.total_return),
        "hit_rate": float(r.signal.hit_rate),
        "n_calls": int(r.signal.n_calls),
        "oos_trades": int(sum(w["oos_trades"] for w in r.windows)),
        "deflated_sharpe_prob": float(r.deflated_sharpe_prob),
    }


def parse_args():
    p = argparse.ArgumentParser(description="null cloud for the GA walk-forward")
    p.add_argument("--n", type=int, default=30, help="null draws (20-50)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    p.add_argument("--universe", type=int, default=None, metavar="N",
                   help="top N NSE equities by turnover instead of --tickers")
    p.add_argument("--universe-as-of", default=None, metavar="DATE",
                   help="point-in-time date for universe selection (default: --start)")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=21)
    p.add_argument("--population", type=int, default=50)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--cost-bps", type=float, default=22.0)
    p.add_argument("--ga-seed", type=int, default=42)
    p.add_argument("--block", type=int, default=21)
    p.add_argument("--out", default="docs/results/null_cloud.json")
    return p.parse_args()


def pct_of(value: float, cloud: np.ndarray) -> float:
    """Percentile of `value` within `cloud`, 0-100."""
    return float((cloud < value).mean() * 100.0)


def main() -> int:
    args = parse_args()

    args.tickers = resolve_universe(args.tickers, args.universe,
                                    args.universe_as_of or args.start)
    md0 = fetch_nse_prices(args.tickers, args.start, use_cache=True,
                           require_actions=True)
    close = pd.DataFrame(md0.close, index=pd.DatetimeIndex(md0.dates),
                         columns=list(md0.tickers))
    logger.info("source: %d bars x %d tickers", len(close), close.shape[1])

    wf_kw = {"train_window": args.train_window, "test_window": args.test_window,
             "cost_bps": args.cost_bps}
    ga_kw = {"population_size": args.population, "n_generations": args.generations,
             "cost_bps": args.cost_bps, "seed": args.ga_seed}

    # The real reference goes first, through the identical code path.
    cv, ci, cc = close.values, close.index, list(close.columns)
    jobs = [(cv, ci, cc, None, args.block, wf_kw, ga_kw)]
    jobs += [(cv, ci, cc, 1000 + i, args.block, wf_kw, ga_kw)
             for i in range(args.n)]
    logger.info("%d jobs (1 real + %d null) on %d workers; "
                "budget %d evals/window", len(jobs), args.n, args.workers,
                args.population * args.generations)

    real, nulls = None, []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one_draw, j): j[3] for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            done += 1
            if r["perm_seed"] is None:
                real = r
                logger.info("[%d/%d] REAL  gap %+.2f  pooled SR %+.2f",
                            done, len(jobs), r["gap"], r["pooled_sharpe"])
            else:
                nulls.append(r)
                logger.info("[%d/%d] null  gap %+.2f  pooled SR %+.2f",
                            done, len(jobs), r["gap"], r["pooled_sharpe"])

    if real is None or not nulls:
        logger.error("missing real or null results")
        return 2

    gaps = np.array([n["gap"] for n in nulls])
    pooled = np.array([n["pooled_sharpe"] for n in nulls])
    iss = np.array([n["in_sample_sharpe"] for n in nulls])
    hits = np.array([n["hit_rate"] for n in nulls])

    def block(name, cloud, obs):
        lo, hi = np.percentile(cloud, [2.5, 97.5])
        # One-sided empirical p: how often does noise MATCH OR BEAT the
        # observed value? +1 in numerator and denominator is the standard
        # correction that keeps p > 0 with a finite number of draws.
        p = (int((cloud >= obs).sum()) + 1) / (len(cloud) + 1)
        return (f"  {name}\n"
                f"    real   {obs:+.3f}\n"
                f"    null   mean {cloud.mean():+.3f}  sd {cloud.std(ddof=1):.3f}  "
                f"95% [{lo:+.3f}, {hi:+.3f}]  min {cloud.min():+.3f} max {cloud.max():+.3f}\n"
                f"    real sits at the {pct_of(obs, cloud):.0f}th percentile of the null cloud\n"
                f"    P(null >= real) = {p:.3f}")

    print("\n" + "=" * 74)
    print(f"NULL CLOUD - {len(nulls)} block permutations vs the real series")
    print(f"  {real['n_windows']} windows, {real['n_bars']} bars, "
          f"{args.population * args.generations} GA evals/window "
          f"(GA seed fixed at {args.ga_seed}; only the permutation varies)")
    print("=" * 74)
    print(block("OVERFITTING GAP  (mean IS Sharpe - mean OOS Sharpe)",
                gaps, real["gap"]))
    print(block("POOLED OOS SHARPE (after 22bp)", pooled, real["pooled_sharpe"]))
    print(block("MEAN IN-SAMPLE SHARPE", iss, real["in_sample_sharpe"]))
    print(block("RAW HIT RATE", hits, real["hit_rate"]))
    print("=" * 74)
    print("\nHOW TO READ THIS. The gap and the in-sample Sharpe are measures of\n"
          "how hard the SEARCH fits; a real value inside the null's 95% band\n"
          "means the search fits real data no harder than it fits noise, i.e.\n"
          "the in-sample number carries no information about the data. The\n"
          "pooled OOS Sharpe is the payoff question; a real value inside the\n"
          "band means the strategy performs no better out-of-sample than one\n"
          "evolved on a series with the signal removed.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "config": {"n_null": args.n, "block": args.block, "ga_seed": args.ga_seed,
                   "tickers": list(args.tickers), "start": args.start,
                   "walk_forward": asdict(WalkForwardConfig(**wf_kw)),
                   "ga": asdict(GAConfig(**ga_kw))},
        "real": real,
        "null_summary": {
            k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "p2_5": float(np.percentile(v, 2.5)),
                "p97_5": float(np.percentile(v, 97.5)),
                "min": float(v.min()), "max": float(v.max()),
                "real_percentile": pct_of(o, v),
                "p_null_ge_real": (int((v >= o).sum()) + 1) / (len(v) + 1)}
            for k, v, o in [("gap", gaps, real["gap"]),
                            ("pooled_sharpe", pooled, real["pooled_sharpe"]),
                            ("in_sample_sharpe", iss, real["in_sample_sharpe"]),
                            ("hit_rate", hits, real["hit_rate"])]
        },
        "nulls": nulls,
    }, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
