"""
The matched null control for the VOLATILITY and REGIME targets.

WHY THIS IS THE INTERESTING ONE. The directional walk-forward had a
foregone conclusion - three independent methods had already found no
edge, and the run confirmed it. Here the information audit DID find
signal: 7 indicator-target pairs survived a 2,000-draw block-permutation
null with BH-FDR control, incremental Spearman of 0.15-0.17 for
atr_pct/bb_width against vol_5d and regime_shift_5d.

But surviving an audit is not the same as surviving a SEARCH. The audit
scored a fixed list of pairs on the full history. A strategy has to pick
a feature without seeing the future and then live with that pick. This
harness reproduces exactly that:

    per window: rank every indicator on the TRAIN slice by incremental
                predictive power, take the best one, then measure that
                one indicator on the untouched TEST slice.

If the audit's signal is durable, the selected feature keeps working out
of sample and the pooled OOS score sits ABOVE the null cloud. If the
audit found a real-but-unusable relationship - or a selection artifact -
the OOS score collapses into the cloud, exactly as the GA's did.

INCREMENTAL, NOT RAW. Volatility is persistent, so a raw correlation
between atr_pct and vol_5d is nearly tautological: ATR *is* recent
volatility, and recent volatility predicts near-future volatility for
free. Every score here is therefore computed on the residual after
projecting out `persistence_baseline` - the "tomorrow looks like today"
forecast that any real model has to beat. A high raw score with zero
incremental score means the model has learned to repeat yesterday.

WHAT THE NULL DESTROYS. Same block permutation as the GA cloud
(tests/test_walkforward_null.py pins its properties), rebuilt end to end
so indicators AND targets AND the persistence baseline all come from the
permuted series. Permuting only the target would leave the baseline
aligned with the original data and understate the null.

A NOTE ON SAMPLE SIZE, because it bounds what this can conclude. The
audit reported n_effective = 28 after block adjustment. Per-window test
slices here are far smaller still, so a single window's score is noise;
only the pooled statistic across windows is read, and it is read against
the cloud rather than against zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from nightevolver.data_loader import build_market_data           # noqa: E402
from nightevolver.genome import INDICATOR_NAMES                  # noqa: E402
from nightevolver.nse_prices import fetch_nse_prices             # noqa: E402
from nightevolver.targets import (                               # noqa: E402
    build_targets, persistence_baseline,
)
from run_evolved_walkforward import (                            # noqa: E402
    DEFAULT_TICKERS, block_permute_prices,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("targetwf")


def _residualise(y: np.ndarray, base: np.ndarray) -> np.ndarray:
    """y with the persistence baseline projected out, rank-space.

    Rank-space because the audit's statistic is Spearman: regressing
    ranks removes the monotone part of the baseline, which is what
    "incremental over persistence" should mean for a rank correlation.
    """
    ry, rb = stats.rankdata(y), stats.rankdata(base)
    rb = rb - rb.mean()
    denom = float(rb @ rb)
    if denom <= 0:
        return ry - ry.mean()
    beta = float(rb @ (ry - ry.mean())) / denom
    return (ry - ry.mean()) - beta * rb


def _score(feature: np.ndarray, target: np.ndarray, base: np.ndarray) -> float:
    """Incremental Spearman of `feature` for `target`, over `base`."""
    ok = np.isfinite(feature) & np.isfinite(target) & np.isfinite(base)
    if ok.sum() < 20:
        return 0.0
    f, t, b = feature[ok], target[ok], base[ok]
    if np.std(f) == 0 or np.std(t) == 0:
        return 0.0
    rf, rt = _residualise(f, b), _residualise(t, b)
    if np.std(rf) == 0 or np.std(rt) == 0:
        return 0.0
    r = float(np.corrcoef(rf, rt)[0, 1])
    return 0.0 if not np.isfinite(r) else r


def _flat(md, target, base, s, e):
    """Flatten a [t0:t1] slice across assets, masked by target validity."""
    ind = md.indicators[s:e]                      # [T, A, F]
    tv, va = target.values[s:e], target.valid[s:e]
    bs = base[s:e]
    m = va & np.isfinite(tv) & np.isfinite(bs)
    if m.sum() < 20:
        return None, None, None
    return (ind.reshape(-1, ind.shape[-1])[m.ravel()],
            tv.ravel()[m.ravel()], bs.ravel()[m.ravel()])


def _one_draw(args):
    """One series (real or permuted) through the whole walk-forward."""
    values, index, columns, perm_seed, block, train_w, test_w, horizon = args
    logging.disable(logging.INFO)

    close_df = pd.DataFrame(values, index=index, columns=columns)
    if perm_seed is not None:
        close_df = block_permute_prices(close_df, block, perm_seed)
    md = build_market_data(close_df)

    targets = build_targets(md.close, horizon=horizon)
    out = {"perm_seed": perm_seed, "n_bars": int(md.n_bars), "targets": {}}

    for tname, target in targets.items():
        base = persistence_baseline(target, md.close, horizon=horizon)
        picks, oos_scores, is_scores = [], [], []

        start = 0
        while start + train_w + 1 < md.n_bars:
            tr_e = start + train_w
            te_e = min(tr_e + test_w, md.n_bars)
            if te_e - tr_e < 1:
                break

            fi, ft, fb = _flat(md, target, base, start, tr_e)
            gi, gt, gb = _flat(md, target, base, tr_e, te_e)
            if fi is None or gi is None:
                start += test_w
                continue

            # SELECT on train only.
            sc = [abs(_score(fi[:, k], ft, fb)) for k in range(fi.shape[1])]
            k = int(np.argmax(sc))
            is_scores.append(float(sc[k]))
            # EVALUATE that one pick on test. Signed, because a feature
            # that flips sign out of sample is not a usable predictor.
            oos_scores.append(abs(_score(gi[:, k], gt, gb)))
            picks.append(INDICATOR_NAMES[k] if k < len(INDICATOR_NAMES) else str(k))
            start += test_w

        if oos_scores:
            out["targets"][tname] = {
                "n_windows": len(oos_scores),
                "mean_is": float(np.mean(is_scores)),
                "mean_oos": float(np.mean(oos_scores)),
                "gap": float(np.mean(is_scores) - np.mean(oos_scores)),
                "picks": picks,
                "n_distinct_picks": len(set(picks)),
            }
    return out


def parse_args():
    p = argparse.ArgumentParser(description="vol/regime targets vs a null cloud")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=63,
                   help="wider than the GA harness's 21: these are "
                        "correlation estimates, which need observations "
                        "rather than trades")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--block", type=int, default=21)
    p.add_argument("--out", default="docs/results/target_null_cloud.json")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    md0 = fetch_nse_prices(a.tickers, a.start, use_cache=True,
                           require_actions=True)
    close = pd.DataFrame(md0.close, index=pd.DatetimeIndex(md0.dates),
                         columns=list(md0.tickers))
    logger.info("source: %d bars x %d tickers", len(close), close.shape[1])

    cv, ci, cc = close.values, close.index, list(close.columns)
    base_job = (cv, ci, cc, None, a.block, a.train_window, a.test_window, a.horizon)
    jobs = [base_job] + [
        (cv, ci, cc, 2000 + i, a.block, a.train_window, a.test_window, a.horizon)
        for i in range(a.n)]
    logger.info("%d jobs (1 real + %d null) on %d workers", len(jobs), a.n, a.workers)

    real, nulls = None, []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one_draw, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            (nulls.append(r) if r["perm_seed"] is not None
             else globals().__setitem__("_r", r))
            if r["perm_seed"] is None:
                real = r
            if i % 5 == 0 or i == len(jobs):
                logger.info("  %d/%d done", i, len(jobs))

    if real is None or not nulls:
        logger.error("missing real or null results")
        return 2

    print("\n" + "=" * 76)
    print(f"TARGET WALK-FORWARD vs NULL CLOUD  ({len(nulls)} permutations)")
    print("  select the best indicator on TRAIN, score that one on TEST")
    print("  scores are INCREMENTAL Spearman over the persistence baseline")
    print("=" * 76)

    summary = {}
    for tname in sorted(real["targets"]):
        r = real["targets"][tname]
        cloud = np.array([n["targets"][tname]["mean_oos"]
                          for n in nulls if tname in n["targets"]])
        if cloud.size == 0:
            continue
        lo, hi = np.percentile(cloud, [2.5, 97.5])
        p = (int((cloud >= r["mean_oos"]).sum()) + 1) / (cloud.size + 1)
        pctile = float((cloud < r["mean_oos"]).mean() * 100)
        verdict = ("ABOVE the null cloud - survives selection"
                   if p < 0.05 else
                   "INSIDE the null cloud - indistinguishable from noise")
        print(f"\n  {tname}   ({r['n_windows']} windows)")
        print(f"    in-sample  (selected) {r['mean_is']:+.4f}")
        print(f"    OOS        (that pick) {r['mean_oos']:+.4f}   real")
        print(f"    null OOS   mean {cloud.mean():+.4f}  sd {cloud.std(ddof=1):.4f}  "
              f"95% [{lo:+.4f}, {hi:+.4f}]")
        print(f"    percentile {pctile:.0f}    P(null >= real) = {p:.3f}")
        print(f"    -> {verdict}")
        print(f"    picks: {r['n_distinct_picks']} distinct across "
              f"{r['n_windows']} windows: {', '.join(r['picks'][:6])}"
              f"{' ...' if len(r['picks']) > 6 else ''}")
        summary[tname] = {
            "n_windows": r["n_windows"], "mean_is": r["mean_is"],
            "mean_oos": r["mean_oos"], "gap": r["gap"],
            "picks": r["picks"], "n_distinct_picks": r["n_distinct_picks"],
            "null_mean": float(cloud.mean()), "null_sd": float(cloud.std(ddof=1)),
            "null_p2_5": float(lo), "null_p97_5": float(hi),
            "real_percentile": pctile, "p_null_ge_real": p,
            "survives": bool(p < 0.05),
        }
    print("\n" + "=" * 76)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "config": vars(a), "real": real, "summary": summary,
        "nulls": nulls,
    }, indent=2, default=str))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
