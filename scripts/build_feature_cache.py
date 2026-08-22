"""
Build the derivative/delivery feature panel once, a year at a time.

WHY THIS IS A SEPARATE STEP. Parsing the F&O bhavcopy for 100 names
costs about 430 seconds per year of history - roughly 54 minutes for the
2019-2026 panel - and this environment kills a foreground command at ten
minutes. Run inline, the walk-forward therefore timed out in the SAME
PLACE on every attempt, before a single permutation had been drawn, so
the draw checkpoint never filled and no amount of retrying made progress.

The fix is not to make it faster. It is to make the expensive part
resumable at a granularity that fits the window: one year per invocation,
written to disk, skipped on the next run. Eight invocations get there;
an interrupted one costs at most the year in flight.

The output is a single .npz keyed by channel name, which
run_target_walkforward.py loads directly. Channels are stored as
[T, A] arrays aligned to the price panel's dates and tickers, so the
consumer does no reindexing and cannot silently misalign a calendar - the
derivative archive and the equity archive have different holiday sets,
and a positional join between two calendars that ALMOST agree shifts a
feature by one bar without changing its shape, which is a look-ahead if
it shifts the wrong way.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from nightevolver.delivery import fetch_delivery_features        # noqa: E402
from nightevolver.derivatives import fetch_derivative_features   # noqa: E402
from nightevolver.nse_prices import fetch_nse_prices             # noqa: E402
from run_evolved_walkforward import resolve_universe             # noqa: E402
from run_new_data_audit import align                             # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("featcache")


def year_path(out: Path, label: str) -> Path:
    return out.parent / f"{out.stem}.{label}.npz"


def panel_path(out: Path) -> Path:
    return out.parent / f"{out.stem}.panel.npz"


def build_year(label: str, start: str, end: str, tickers, dates,
               out: Path) -> bool:
    """Returns True if this chunk was built now (False if already cached).

    CHUNKED BY QUARTER, not by year. A year of F&O parsing costs ~430s
    and the panel build ahead of it ~190s, which together overran the
    ten-minute limit and produced nothing on every attempt - the failure
    this whole script exists to avoid, reappearing one level up. A
    quarter is ~110s and leaves comfortable headroom.
    """
    p = year_path(out, label)
    if p.exists():
        logger.info("[%s] already cached", label)
        return False
    t0 = time.time()
    frames = {}
    frames.update(fetch_derivative_features(list(tickers), start, end))
    frames.update(fetch_delivery_features(list(tickers), start, end))

    # Aligned to the FULL panel index, so years concatenate by simple
    # addition of non-NaN cells rather than by index arithmetic.
    idx = pd.DatetimeIndex(dates)
    in_year = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    keep = {}
    for name, f in frames.items():
        arr = align(f, idx, list(tickers))
        arr = np.where(in_year[:, None], arr, np.nan)
        keep[name] = arr.astype(np.float32)

    np.savez_compressed(p, __names__=np.array(sorted(keep)), **keep)
    logger.info("[%s] %d channels in %.0fs -> %s",
                label, len(keep), time.time() - t0, p.name)
    return True


def combine(out: Path, years) -> None:
    parts = [year_path(out, y) for y in years]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        logger.warning("not combining - %d year(s) still missing: %s",
                       len(missing), ", ".join(missing))
        return

    merged, names = {}, None
    for p in parts:
        z = np.load(p, allow_pickle=False)
        ns = [str(n) for n in z["__names__"]]
        names = ns if names is None else names
        for n in ns:
            a = z[n]
            if n not in merged:
                merged[n] = a.copy()
            else:
                # Years are disjoint by construction, so a cell is
                # non-NaN in at most one part; np.where keeps whichever
                # has it rather than summing (which would turn two NaNs
                # into a value).
                merged[n] = np.where(np.isfinite(a), a, merged[n])
    np.savez_compressed(out, __names__=np.array(sorted(merged)), **merged)
    cov = {n: float(np.isfinite(v).mean()) for n, v in merged.items()}
    logger.info("combined %d channels -> %s", len(merged), out)
    for n in sorted(cov):
        logger.info("   %-20s coverage %5.1f%%", n, cov[n] * 100)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--universe", type=int, default=100)
    p.add_argument("--universe-as-of", default="2019-01-01")
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--out", default="logs/wf_ckpt.features.npz")
    p.add_argument("--years", type=int, nargs="*", default=None,
                   help="build only these years (default: all in range)")
    p.add_argument("--max-years", type=int, default=1,
                   help="build at most N years this run, so one "
                        "invocation stays inside the wall-clock limit")
    p.add_argument("--combine-only", action="store_true")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    out = Path(a.out)

    # The panel costs ~190s to rebuild and never changes between chunks,
    # so its dates and tickers are cached too. Without this, a third of
    # every invocation went to recomputing the same thing.
    pp = panel_path(out)
    if pp.exists():
        z = np.load(pp, allow_pickle=False)
        dates = pd.DatetimeIndex([str(d) for d in z["dates"]])
        tickers = [str(t) for t in z["tickers"]]
        logger.info("panel reused: %d bars x %d tickers", len(dates),
                    len(tickers))
    else:
        tickers = resolve_universe(None, a.universe, a.universe_as_of)
        md = fetch_nse_prices(tickers, a.start, a.end, use_cache=True,
                              require_actions=True,
                              min_coverage=a.min_coverage)
        dates, tickers = pd.DatetimeIndex(md.dates), list(md.tickers)
        pp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pp,
                            dates=np.array([str(d) for d in dates]),
                            tickers=np.array(tickers))
        logger.info("panel: %d bars x %d tickers (cached)", len(dates),
                    len(tickers))

    idx = pd.DatetimeIndex(dates)
    chunks = []
    for ts in sorted({(d.year, (d.month - 1) // 3 + 1) for d in idx}):
        y, q = ts
        s_ = pd.Timestamp(year=y, month=3 * (q - 1) + 1, day=1)
        e_ = (s_ + pd.offsets.QuarterEnd(0)).normalize()
        chunks.append((f"{y}Q{q}", f"{s_:%Y-%m-%d}", f"{e_:%Y-%m-%d}"))
    if a.years:
        chunks = [c for c in chunks if int(c[0][:4]) in a.years]
    years = [c[0] for c in chunks]

    if not a.combine_only:
        built = 0
        for label, s_, e_ in chunks:
            if built >= a.max_years:
                logger.info("stopping at %d chunk(s) this run - rerun to "
                            "continue", built)
                break
            if build_year(label, s_, e_, tickers, dates, out):
                built += 1

    combine(out, years)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
