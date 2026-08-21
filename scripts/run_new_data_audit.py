"""
Information audit for the NEW data channels, against the same null.

THE QUESTION THIS ANSWERS, and why it comes before any modelling. The
GA's original 26 channels were shown to carry no directional edge by a
16-window walk-forward against a 30-draw null cloud. Three new data
classes have since been added - derivatives (8 channels), delivery
(5 channels) and news (3) - on the argument that they are genuinely
different INFORMATION rather than different views of the same daily
OHLCV.

That argument is a hypothesis. New channels are also new surface to
overfit, and the measured ceiling from the null cloud is stark: this
pipeline returned a pooled OOS Sharpe of +1.39 on data with the signal
permuted out. Adding thirteen channels and finding "something" proves
nothing without the same control applied to them.

So: score every new channel against every target with the identical
block-permutation null and BH-FDR correction the original audit used
(block=21 bars, 2000 permutations, alpha=0.05), and report survivors.

TWO CONTROLS ARE INJECTED ALONGSIDE THE REAL CHANNELS.

  A NOISE channel - pure gaussian, regenerated per run. It must not
  survive. If it does, the audit is miscalibrated on THIS panel, and
  every other result in the run is void. This has caught a real problem
  before: market-wide features scored |rho| = 0.0126 against noise
  reaching 0.0167, which made a genuine flow feature look significant
  until it was scored at the market level and collapsed to p = 0.60.

  A SHUFFLED copy of one real channel, keeping its marginal
  distribution but destroying its alignment with time. A feature can be
  heavy-tailed, discrete, or mostly-zero in ways that break a rank
  statistic's null; shuffling the real thing tests the audit against
  the actual shape of the data rather than against a gaussian ideal.

INCREMENTAL SCORING is inherited from information_audit.audit_features,
which partials out each target's persistence baseline. This matters more
for the new channels than for price transforms: atm_iv against vol_5d
would otherwise score highly for the trivial reason that implied and
realised volatility are both volatility.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from nightevolver.delivery import fetch_delivery_features          # noqa: E402
from nightevolver.derivatives import fetch_derivative_features     # noqa: E402
from nightevolver.information_audit import audit_features          # noqa: E402
from nightevolver.nse_prices import fetch_bhav_range, fetch_nse_prices  # noqa: E402
from nightevolver.patterns import build_pattern_features           # noqa: E402
from nightevolver.targets import build_targets                     # noqa: E402
from run_evolved_walkforward import DEFAULT_TICKERS, resolve_universe  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("newaudit")


def align(frame: pd.DataFrame, dates, tickers) -> np.ndarray:
    """[date x symbol] frame -> [T, A] aligned to the price panel.

    Reindexed rather than assumed to match: the derivative and delivery
    archives have their own holiday sets and their own missing days, and
    a positional join between two calendars that ALMOST agree is the
    kind of error that shifts a feature by one bar without changing its
    shape - which is a look-ahead if the shift goes the wrong way.
    """
    if frame is None or frame.empty:
        return np.full((len(dates), len(tickers)), np.nan)
    f = frame.copy()
    f.index = pd.DatetimeIndex(f.index)
    f = f.reindex(index=pd.DatetimeIndex(dates))
    for t in tickers:
        if t not in f.columns:
            f[t] = np.nan
    return f[list(tickers)].to_numpy(dtype=float)


def adjusted_ohlc(md, tickers, start, end=None):
    """Raw bhavcopy OHLC rescaled onto the ADJUSTED close.

    md.close is corporate-action back-adjusted; the bhavcopy's open,
    high and low are the actual traded prices. Feeding a mixture into
    the pattern features would repeat the bug measured in bse_prices.py,
    where differencing an adjusted series against an unadjusted one
    produced a stable -16,977 bps "arbitrage" that was purely the two
    conventions.

    Per bar the adjustment is a single multiplicative factor, so
    factor = adjusted_close / raw_close recovers it exactly, and
    applying it to open/high/low puts all four legs on one scale.
    Within-bar ratios (body/range, shadow/range) are invariant to it
    anyway; the cross-bar patterns - engulfing, harami, three soldiers -
    are the ones that would otherwise break across an ex-date.
    """
    raw = fetch_bhav_range(list(tickers), start, end)
    days = sorted(raw)
    if not days:
        return None, None, None

    def col(field):
        return pd.DataFrame(
            {s: {d: raw[d].set_index("TckrSymb")[field].get(s, np.nan)
                 for d in days} for s in tickers})

    o, h, l, c = (col(f) for f in ("OpnPric", "HghPric", "LwPric", "ClsPric"))
    idx = pd.DatetimeIndex(md.dates)
    o, h, l, c = (x.reindex(index=idx, columns=list(tickers)) for x in (o, h, l, c))

    adj = pd.DataFrame(md.close, index=idx, columns=list(tickers))
    factor = adj / c.replace(0.0, np.nan)
    n_adj = int((factor.round(6).nunique() > 1).sum())
    if n_adj:
        logger.info("[audit] %d/%d names carry a non-constant adjustment "
                    "factor - rescaling OHLC onto the adjusted close",
                    n_adj, len(tickers))
    return (o * factor).to_numpy(), (h * factor).to_numpy(), (l * factor).to_numpy()


def parse_args():
    p = argparse.ArgumentParser(description="audit the new data channels")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    p.add_argument("--universe", type=int, default=None, metavar="N")
    p.add_argument("--universe-as-of", default=None)
    p.add_argument("--permutations", type=int, default=2000,
                   help="matching the original audit, so the two are "
                        "directly comparable")
    p.add_argument("--block", type=int, default=21)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="docs/results/new_data_audit.json")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    a.tickers = resolve_universe(a.tickers, a.universe,
                                 a.universe_as_of or a.start)

    md = fetch_nse_prices(a.tickers, a.start, a.end, use_cache=True,
                          require_actions=True)
    dates, tickers = md.dates, list(md.tickers)
    logger.info("price panel: %d bars x %d tickers", md.n_bars, len(tickers))

    logger.info("building classical TA / microstructure channels")
    p_open, p_high, p_low = adjusted_ohlc(md, tickers, a.start, a.end)
    pattern_channels = {}
    if p_open is not None:
        pattern_channels = build_pattern_features(
            md.close, p_high, p_low, md.volume, open_=p_open)
        logger.info("  %d pattern channels", len(pattern_channels))

    logger.info("fetching derivative features (cached after first run)")
    deriv = fetch_derivative_features(tickers, a.start, a.end)
    logger.info("fetching delivery features")
    deliv = fetch_delivery_features(tickers, a.start, a.end)

    channels: Dict[str, np.ndarray] = {}
    for name, arr in pattern_channels.items():
        cov = float(np.isfinite(arr).mean())
        if cov < 0.05:
            logger.warning("[audit] dropping %s - %.1f%% coverage", name, cov * 100)
            continue
        channels[name] = arr

    for name, frame in list(deriv.items()) + list(deliv.items()):
        arr = align(frame, dates, tickers)
        cov = float(np.isfinite(arr).mean())
        if cov < 0.05:
            logger.warning("[audit] dropping %s - only %.1f%% coverage",
                           name, cov * 100)
            continue
        logger.info("  %-20s coverage %5.1f%%", name, cov * 100)
        channels[name] = arr

    if not channels:
        logger.error("no usable channels")
        return 2

    # --- controls, injected alongside the real thing -----------------
    rng = np.random.RandomState(a.seed + 991)
    channels["_CONTROL_noise"] = rng.normal(size=(len(dates), len(tickers)))

    donor = next(iter(sorted(channels)))
    shuffled = channels[donor].copy()
    rng.shuffle(shuffled)                     # permutes rows, keeps marginals
    channels["_CONTROL_shuffled"] = shuffled
    logger.info("controls: gaussian noise, and a row-shuffled copy of %s",
                donor)

    names: List[str] = sorted(channels)
    feats = np.stack([channels[n] for n in names], axis=-1)   # [T, A, F]
    logger.info("auditing %d channels x %d targets, %d permutations",
                len(names), 4, a.permutations)

    targets = build_targets(md.close, horizon=a.horizon)
    res = audit_features(feats, names, targets, md.close,
                         block_bars=a.block, n_permutations=a.permutations,
                         fdr_alpha=a.alpha, seed=a.seed)

    pairs = [p.__dict__ if hasattr(p, "__dict__") else dict(p)
             for p in res.pairs]
    survivors = [p for p in pairs if p.get("significant")]

    print("\n" + "=" * 78)
    print(f"NEW DATA AUDIT - {len(names)} channels, {len(pairs)} pairs, "
          f"{a.permutations} block permutations, BH-FDR alpha={a.alpha}")
    print("=" * 78)

    ctrl = [p for p in survivors if str(p.get("feature", "")).startswith("_CONTROL")]
    if ctrl:
        print("\n  !! CONTROL SURVIVED - THE AUDIT IS MISCALIBRATED ON THIS")
        print("     PANEL AND EVERY RESULT BELOW IS VOID.")
        for p in ctrl:
            print(f"     {p['feature']} -> {p['target']}  q={p.get('q_value'):.4f}")
    else:
        print("\n  controls did NOT survive - audit calibrated on this panel")

    real = [p for p in survivors
            if not str(p.get("feature", "")).startswith("_CONTROL")]
    if not real:
        print("\n  NO new channel survived FDR correction against any target.")
        print("  The new data classes carry no measurable incremental")
        print("  information about these targets on this panel.")
    else:
        print(f"\n  {len(real)} surviving pair(s):\n")
        for p in sorted(real, key=lambda x: x.get("q_value", 1.0)):
            print(f"    {p['feature']:<22} -> {p['target']:<18} "
                  f"rho={p.get('spearman', float('nan')):+.4f} "
                  f"incr={p.get('spearman_incremental', float('nan')):+.4f} "
                  f"q={p.get('q_value', float('nan')):.4f} "
                  f"n_eff={p.get('n_effective', '?')}")
        print("\n  SURVIVING AN AUDIT IS NOT AN EDGE. These pairs cleared a")
        print("  fixed-pair test over the full history. The next question is")
        print("  whether a SEARCH can select them in advance and beat the")
        print("  null cloud out-of-sample - which is what killed the")
        print("  vol_5d/regime result. Run scripts/run_target_walkforward.py.")
    print("=" * 78)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "config": {k: v for k, v in vars(a).items()},
        "n_bars": int(md.n_bars), "tickers": tickers,
        "channels": names,
        "n_pairs": len(pairs),
        "n_survivors": len(real),
        "control_survived": bool(ctrl),
        "pairs": pairs,
    }, indent=2, default=str))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
