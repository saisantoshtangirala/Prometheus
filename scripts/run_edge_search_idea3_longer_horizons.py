"""
Edge-search idea #3: LONGER TARGET HORIZONS on the tradeable targets.

STATE. Iteration 2 of the rigorous-search protocol in
logs/edge_search_state.json (idea 1, cross-sectional rank, found nothing
new - see docs/results/edge_idea1_cs_rank.json).

THE GAP THIS FILLS. Every direction/rel-strength test this project has
ever run used a 1-DAY horizon, because direction_1d and rel_strength_1d
in nightevolver/targets.py hardcode it - vol_5d and regime_shift_5d take
a `horizon` parameter, the two TRADEABLE targets never did. That is a
real, unexamined cell: genome.py's own break-even table shows the
correlation bar a signal must clear falls sharply with hold length -

    hold   E|ret|   break-even WR   rho needed
      1d    0.99%          61.1%        0.342
      5d    2.32%          54.7%        0.148
     20d    4.57%          52.4%        0.076

- so if there is ANY real signal in this data, a longer hold is the most
forgiving place to detect it, economically. This idea builds the N-day
analogs directly (not touching targets.py, since this is exploratory,
not yet a vetted permanent addition) and tests N=10 and N=20 alongside
the existing 1-day control.

BASELINE, BUILT CAREFULLY. persistence_baseline() in targets.py branches
by target NAME; anything other than "vol_5d"/"regime_shift_5d" falls
through to a 1-day-lag baseline, which would be too weak a comparison
for a 10/20-day target and could make a feature look like it adds
incremental power it does not - the project's own documented mistake
with the FIRST regime_shift_5d baseline (a too-weak baseline produced
rho=-0.3864, q=0.04 "significant" on a pure random walk). The correct
persistence baseline for an N-day forward return is the TRAILING N-day
compounded return - built explicitly here, matching that lesson.
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
from run_evolved_walkforward import (                                 # noqa: E402
    block_permutation_order, block_permute_prices, resolve_universe,
)
from run_target_walkforward import _flat, _score                      # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("idea3_horizons")

HORIZONS = (1, 10, 20)   # 1d kept as the in-run control


def direction_nd(close: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compounded n-day forward return, close[t] -> close[t+n]."""
    T, A = close.shape
    fwd = np.full((T, A), np.nan)
    if n < T:
        fwd[:T - n] = close[n:] / close[:-n] - 1.0
    return np.nan_to_num(fwd), np.isfinite(fwd)


def rel_strength_nd(fwd: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """fwd minus its cross-sectional mean, count-aware (mirrors
    rel_strength_1d's masked-mean construction in targets.py)."""
    if fwd.shape[1] < 2:
        return np.zeros_like(fwd)
    masked = np.where(valid, fwd, np.nan)
    counts = valid.sum(axis=1, keepdims=True)
    sums = np.nansum(masked, axis=1, keepdims=True)
    mean = np.divide(sums, counts, out=np.zeros_like(sums),
                     where=counts > 0)
    return np.where(valid, fwd - mean, 0.0)


def trailing_nd_baseline(close: np.ndarray, n: int) -> np.ndarray:
    """TRAILING n-day compounded return at t: close[t]/close[t-n] - 1.
    The correct 'yesterday looks like today' comparison for an n-day
    forward target - see module docstring for why a 1-day-lag baseline
    would be too weak here."""
    T, A = close.shape
    out = np.full((T, A), np.nan)
    if n < T:
        out[n:] = close[n:] / close[:-n] - 1.0
    return out


def _one_draw(args):
    (values, index, columns, perm_seed, block, train_w, test_w,
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

    out = {"perm_seed": perm_seed, "n_bars": int(md.n_bars), "targets": {}}

    for n in HORIZONS:
        fwd, valid = direction_nd(md.close, n)
        base = trailing_nd_baseline(md.close, n)

        for kind, target_arr in (("direction", fwd),
                                 ("rel_strength", rel_strength_nd(fwd, valid))):
            tname = f"{kind}_{n}d"

            class _T:
                pass
            tobj = _T(); tobj.values = target_arr; tobj.valid = valid

            picks, oos_scores, is_scores = [], [], []
            start = 0
            while start + train_w + 1 < md.n_bars:
                tr_e = start + train_w
                te_e = min(tr_e + test_w, md.n_bars)
                if te_e - tr_e < 1:
                    break

                fi, ft, fb = _flat(md, tobj, base, start, tr_e, pool)
                gi, gt, gb = _flat(md, tobj, base, tr_e, te_e, pool)
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
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--workers", type=int, default=1,
                   help="default 1: idea 1 found ProcessPoolExecutor stalls "
                        "unreliably on this workload past ~30%% progress; "
                        "sequential is slower but completes reliably")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--universe", type=int, default=100)
    p.add_argument("--universe-as-of", default="2019-01-01")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=63)
    p.add_argument("--block", type=int, default=21)
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--checkpoint", default="logs/edge_idea3_ckpt.json")
    p.add_argument("--out", default="docs/results/edge_idea3_longer_horizons.json")
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

    cv, ci, cc = close.values, close.index, list(close.columns)
    jobs = [(cv, ci, cc, None, a.block, a.train_window, a.test_window,
             extra, extra_names)] + [
        (cv, ci, cc, 4000 + i, a.block, a.train_window, a.test_window,
         extra, extra_names)
        for i in range(a.n)]

    done = load_ckpt(a.checkpoint)
    todo = [j for j in jobs if draw_key(j[3]) not in done]
    logger.info("%d jobs (1 real + %d null), %d already done, %d to run",
                len(jobs), a.n, len(done), len(todo))

    if todo:
        if a.workers == 1:
            for i, j in enumerate(todo, 1):
                r = _one_draw(j)
                done[draw_key(r["perm_seed"])] = r
                save_ckpt(a.checkpoint, done)
                logger.info("  %d/%d this run (%d/%d total)",
                            i, len(todo), len(done), len(jobs))
        else:
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

    state_path = Path("logs/edge_search_state.json")
    n_ideas = 2
    if state_path.exists():
        try:
            n_ideas = max(1, json.loads(state_path.read_text())["n_ideas_tried"] + 1)
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    alpha_eff = 0.05 / n_ideas

    print("\n" + "=" * 78)
    print(f"IDEA 3: LONGER TARGET HORIZONS  ({len(nulls)} null draws, "
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
        print(f"\n  {tname:<16} ({r['n_windows']} windows, "
              f"{r['n_distinct_picks']} distinct picks: {r['picks'][:3]}...)")
        print(f"    OOS (real) {r['mean_oos']:+.4f}   null mean "
              f"{float(cloud.mean()):+.4f} sd {float(cloud.std()):.4f}")
        print(f"    percentile {pctile:.0f}   p={p:.4f}   -> {verdict}")
        summary[tname] = {"p": p, "survives": bool(survives),
                          "underpowered": bool(underpowered),
                          "oos": r["mean_oos"], "null_mean": float(cloud.mean()),
                          "null_sd": float(cloud.std()), "percentile": pctile,
                          "n_windows": r["n_windows"],
                          "n_distinct_picks": r["n_distinct_picks"],
                          "picks": r["picks"]}
    print("=" * 78)

    any_survive = any(v["survives"] for v in summary.values())
    out = {
        "idea": "longer_target_horizons", "idea_id": 3,
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
