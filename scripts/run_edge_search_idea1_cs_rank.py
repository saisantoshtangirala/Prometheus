"""
Edge-search idea #1: CROSS-SECTIONAL RANK, not absolute value.

STATE. This is iteration 1 of the rigorous-search loop agreed with the
user in logs/edge_search_state.json, after they explicitly rejected an
uncorrected "search until it works" loop - the framing that would
manufacture a false positive by construction, which is exactly what
happened once already this session (the pcr_volume look-ahead).

THE HYPOTHESIS. Every test run so far - the information audit, the GA
walk-forward, the target walk-forward - correlates a feature's ABSOLUTE
value against a return's ABSOLUTE value, pooled across time and across
names. That construction is dominated by whatever moves the whole market
on a given day: on a day the index falls 2%, every stock's return is
negative regardless of any stock-specific signal, which is noise for a
STOCK-PICKING signal and dilutes a real relative-ordering effect toward
zero in a pooled absolute-value correlation.

This idea removes the common component directly: on each date, convert
every feature and the forward return to a CROSS-SECTIONAL PERCENTILE
RANK among that day's live names, before pooling across time. A
market-wide move ranks every name near its usual position (ranking is
invariant to a common additive/multiplicative shift), so what survives
is purely "did this name do relatively better than its peers, and did
the feature call that in advance" - a genuinely different statistical
object from everything tried before, not a re-run of it.

WHY THE EXISTING NULL MACHINERY STILL APPLIES UNCHANGED. Block
permutation shuffles DATE-BLOCKS, moving each date's entire cross-section
together. Cross-sectional rank is computed per-date BEFORE permutation,
so a permuted date carries its already-computed rank row intact - the
null cloud for rank-transformed data is exactly as valid as it was for
raw data, using the identical permutation code, unmodified.

OUTER CORRECTION. Per the agreed protocol: alpha_effective =
0.05 / n_ideas_tried_so_far. This is idea 1, so alpha_effective = 0.05
this run - but the number written to the state file is what matters for
idea 2 onward, not a hardcoded value here.
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

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from nightevolver.data_loader import WARMUP_BARS, build_market_data   # noqa: E402
from nightevolver.nse_prices import fetch_nse_prices                  # noqa: E402
from nightevolver.targets import build_targets, persistence_baseline  # noqa: E402
from run_evolved_walkforward import (                                 # noqa: E402
    block_permutation_order, block_permute_prices, resolve_universe,
)
from run_target_walkforward import _flat, _score                      # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("idea1_csrank")


def cross_sectional_rank(arr: np.ndarray) -> np.ndarray:
    """[T, A] -> [T, A] percentile rank in (0, 1) among FINITE values on
    each date. NaN stays NaN - a name absent that day contributes no
    rank and does not shift its peers' ranks."""
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for t in range(arr.shape[0]):
        row = arr[t]
        finite = np.isfinite(row)
        n = int(finite.sum())
        if n < 5:                       # too thin a cross-section to rank meaningfully
            continue
        ranks = pd.Series(row[finite]).rank(pct=True).to_numpy()
        out[t, finite] = ranks
    return out


def _one_draw(args):
    (values, index, columns, perm_seed, block, train_w, test_w, horizon,
     extra, extra_names) = args
    logging.disable(logging.INFO)

    close_df = pd.DataFrame(values, index=index, columns=columns)
    if perm_seed is not None:
        close_df = block_permute_prices(close_df, block, perm_seed)
    md = build_market_data(close_df)

    pool, names = md.indicators, list(range(md.indicators.shape[-1]))
    if extra is not None and extra.size:
        ex = extra
        if perm_seed is not None:
            order = block_permutation_order(len(ex), block, perm_seed)
            ex = ex[order]
        lo = WARMUP_BARS
        ex = ex[lo:lo + md.n_bars]
        if ex is not None and len(ex) == md.n_bars:
            pool = np.concatenate([md.indicators, ex], axis=-1)
            names = names + list(extra_names)

    # RANK EVERY CHANNEL AND THE TARGET, CROSS-SECTIONALLY, per date -
    # this is the entire idea. Nothing downstream (window loop, single-
    # feature selection, incremental scoring, null draws) is touched.
    pool_r = np.stack([cross_sectional_rank(pool[:, :, k])
                       for k in range(pool.shape[-1])], axis=-1)

    targets = build_targets(md.close, horizon=horizon)
    out = {"perm_seed": perm_seed, "n_bars": int(md.n_bars), "targets": {}}

    for tname, target in targets.items():
        base = persistence_baseline(target, md.close, horizon=horizon)
        base_r = cross_sectional_rank(base)
        target_r = cross_sectional_rank(np.where(target.valid, target.values, np.nan))

        picks, oos_scores, is_scores = [], [], []
        start = 0
        while start + train_w + 1 < md.n_bars:
            tr_e = start + train_w
            te_e = min(tr_e + test_w, md.n_bars)
            if te_e - tr_e < 1:
                break

            class _T:
                pass
            tobj = _T(); tobj.values = target_r; tobj.valid = np.isfinite(target_r)

            fi, ft, fb = _flat(md, tobj, base_r, start, tr_e, pool_r)
            gi, gt, gb = _flat(md, tobj, base_r, tr_e, te_e, pool_r)
            if fi is None or gi is None:
                start += test_w
                continue

            sc = [abs(_score(fi[:, k], ft, fb)) for k in range(fi.shape[1])]
            k = int(np.argmax(sc))
            is_scores.append(float(sc[k]))
            oos_scores.append(abs(_score(gi[:, k], gt, gb)))
            picks.append(str(names[k]) if k < len(names) else str(k))
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
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20,
                   help="null draws; 20 keeps the p-value floor (1/21=0.048) "
                        "just under the underpowered guard while costing "
                        "less than the 30-draw standard")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--universe", type=int, default=100)
    p.add_argument("--universe-as-of", default="2019-01-01")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=63)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--block", type=int, default=21)
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--checkpoint", default="logs/edge_idea1_ckpt.json")
    p.add_argument("--out", default="docs/results/edge_idea1_cs_rank.json")
    return p.parse_args()


def draw_key(seed):
    return "real" if seed is None else f"null_{seed}"


def load_ckpt(path):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_ckpt(path, done):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(done, default=str))
    tmp.replace(p)


def main() -> int:
    a = parse_args()
    tickers = resolve_universe(None, a.universe, a.universe_as_of)
    md0 = fetch_nse_prices(tickers, a.start, use_cache=True, require_actions=True,
                           min_coverage=a.min_coverage)
    close = pd.DataFrame(md0.close, index=pd.DatetimeIndex(md0.dates),
                         columns=list(md0.tickers))
    logger.info("source: %d bars x %d tickers", len(close), close.shape[1])

    extra, extra_names = None, []
    fc = Path("logs/wf_ckpt.features.npz")
    if fc.exists():
        z = np.load(fc, allow_pickle=False)
        names = sorted(str(n) for n in z["__names__"])
        want = (len(close.index), close.shape[1])
        bad = [n for n in names if z[n].shape != want]
        if not bad:
            extra_names = names
            extra = np.stack([z[n].astype(float) for n in names], axis=-1)
            logger.info("reused feature cache: %d channels", len(names))
        else:
            logger.warning("feature cache shape mismatch - running price-only")

    cv, ci, cc = close.values, close.index, list(close.columns)
    jobs = [(cv, ci, cc, None, a.block, a.train_window, a.test_window,
             a.horizon, extra, extra_names)] + [
        (cv, ci, cc, 3000 + i, a.block, a.train_window, a.test_window,
         a.horizon, extra, extra_names)
        for i in range(a.n)]

    done = load_ckpt(a.checkpoint)
    todo = [j for j in jobs if draw_key(j[3]) not in done]
    logger.info("%d jobs (1 real + %d null), %d already done, %d to run",
                len(jobs), a.n, len(done), len(todo))

    if todo:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(_one_draw, j) for j in todo]
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                done[draw_key(r["perm_seed"])] = r
                save_ckpt(a.checkpoint, done)
                logger.info("  %d/%d this run (%d/%d total)",
                            i, len(todo), len(done), len(jobs))

    if len(done) < len(jobs):
        logger.info("not all draws complete yet - rerun to continue")
        return 1

    real = done[draw_key(None)]
    nulls = [v for k, v in done.items() if k != "real"]

    # OUTER, CROSS-IDEA CORRECTION - read from the shared search state, not
    # hardcoded, so idea 2 onward automatically tightens.
    state_path = Path("logs/edge_search_state.json")
    n_ideas = 1
    if state_path.exists():
        try:
            n_ideas = max(1, json.loads(state_path.read_text())["n_ideas_tried"] + 1)
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    alpha_eff = 0.05 / n_ideas

    print("\n" + "=" * 78)
    print(f"IDEA 1: CROSS-SECTIONAL RANK  ({len(nulls)} null draws, "
          f"alpha_effective={alpha_eff:.4f} for idea #{n_ideas})")
    print("=" * 78)

    summary = {}
    for tname in sorted(real["targets"]):
        r = real["targets"][tname]
        cloud = np.array([n["targets"][tname]["mean_oos"]
                          for n in nulls if tname in n["targets"]])
        if cloud.size == 0:
            continue
        p = (int((cloud >= r["mean_oos"]).sum()) + 1) / (cloud.size + 1)
        underpowered = (1.0 / (cloud.size + 1)) > alpha_eff
        survives = (not underpowered) and p < alpha_eff
        pctile = float((cloud < r["mean_oos"]).mean() * 100)
        verdict = ("UNDERPOWERED" if underpowered
                   else "SURVIVES the outer-corrected bar - CANDIDATE, not a finding"
                   if survives else "does not survive")
        print(f"\n  {tname:<18} ({r['n_windows']} windows, "
              f"{r['n_distinct_picks']} distinct picks)")
        print(f"    OOS (real) {r['mean_oos']:+.4f}   null mean "
              f"{float(cloud.mean()):+.4f} sd {float(cloud.std()):.4f}")
        print(f"    percentile {pctile:.0f}   p={p:.4f}   -> {verdict}")
        summary[tname] = {"p": p, "survives": bool(survives),
                          "underpowered": bool(underpowered),
                          "oos": r["mean_oos"], "null_mean": float(cloud.mean()),
                          "null_sd": float(cloud.std()), "percentile": pctile,
                          "n_windows": r["n_windows"],
                          "n_distinct_picks": r["n_distinct_picks"]}
    print("=" * 78)

    any_survive = any(v["survives"] for v in summary.values())
    out = {
        "idea": "cross_sectional_rank", "idea_id": 1,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_ideas_tried_including_this": n_ideas, "alpha_effective": alpha_eff,
        "any_survive": any_survive, "targets": summary,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
