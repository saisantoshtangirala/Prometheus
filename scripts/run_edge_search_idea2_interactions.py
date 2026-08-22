"""
Edge-search idea #2: PAIRWISE FEATURE INTERACTIONS.

STATE. Iteration 5 of the rigorous-search protocol in
logs/edge_search_state.json. Ideas 1, 3, 4: no directional edge. Idea 5
(sector pairs): a candidate that survived pooled but not either
independent half of a confirmatory split - set aside by user decision,
not pursued further for now.

THE HYPOTHESIS. No SINGLE feature predicts direction_1d or
rel_strength_1d (established by every prior test this session, and
originally by the 42-channel audit). A specific PAIR might: momentum and
positioning signals are frequently conditional in known microstructure
literature - e.g. open-interest change might only carry information when
skew is also extreme, a joint condition no single-feature test can see.

WHY NO BESPOKE COMBINATORIAL CORRECTION WAS BUILT. The 13 derivative/
delivery channels produce C(13,2)=78 pairwise products. Testing each
separately would need its own outer correction across 78 tests -
expensive machinery to build correctly. Instead: the 78 product terms
are added DIRECTLY into the existing feature pool (indicators + extras),
and the SAME select-best-on-train/evaluate-on-test/compare-to-null-cloud
machinery from ideas 1, 3 and 4 is reused UNCHANGED. That discipline
already handles pool-size inflation correctly by construction - a larger
pool of pure noise columns produces a correspondingly larger apparent
train-pick score AND a correspondingly larger null-cloud ceiling (the
null cloud is built by running the identical select-from-pool procedure
on permuted data with the SAME pool size), so the comparison stays fair
without a separate correction step. This is exactly why the null cloud
sits around the SELECTION step rather than around a single feature in
every walk-forward this project runs.

INTERACTION TERM CONSTRUCTION. Each pair (X, Y) of the 13 derivative/
delivery channels contributes rank(X)*rank(Y) rather than the raw
product X*Y - the channels are on wildly different scales (atm_iv is a
0-1-ish volatility, oi_change_norm is roughly unit-scale, avg_trade_size
is a log-dollar figure), and a raw product would be dominated by
whichever channel has the largest scale that day rather than capturing
genuine interaction. Cross-sectional-rank product is scale-free and
answers "when both X and Y are unusually high (or low) together,
relative to their peers that day, is direction more predictable" -
the standard construction for interaction effects when inputs are not
comparably scaled.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from nightevolver.data_loader import WARMUP_BARS, build_market_data   # noqa: E402
from nightevolver.nse_prices import fetch_nse_prices                  # noqa: E402
from nightevolver.targets import build_targets, persistence_baseline  # noqa: E402
from run_edge_search_idea1_cs_rank import cross_sectional_rank        # noqa: E402
from run_evolved_walkforward import (                                 # noqa: E402
    block_permutation_order, block_permute_prices, resolve_universe,
)
from run_target_walkforward import _flat, _score                      # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("idea2_interactions")


def build_interaction_pool(extra: np.ndarray, extra_names) -> tuple:
    """[T, A, P] product-of-ranks pool from every pair of extra channels,
    plus the interaction names, e.g. 'atm_iv*pcr_volume'."""
    P = len(extra_names)
    ranked = np.stack([cross_sectional_rank(extra[:, :, k]) for k in range(P)],
                      axis=-1)
    cols, names = [], []
    for i, j in itertools.combinations(range(P), 2):
        cols.append(ranked[:, :, i] * ranked[:, :, j])
        names.append(f"{extra_names[i]}*{extra_names[j]}")
    return np.stack(cols, axis=-1), names


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

            inter, inter_names = build_interaction_pool(ex, extra_names)
            pool = np.concatenate([pool, inter], axis=-1)
            names = names + inter_names

    targets = build_targets(md.close, horizon=horizon)
    out = {"perm_seed": perm_seed, "n_bars": int(md.n_bars),
          "n_pool_features": pool.shape[-1], "targets": {}}

    for tname in ("direction_1d", "rel_strength_1d"):
        target = targets[tname]
        base = persistence_baseline(target, md.close, horizon=horizon)
        picks, oos_scores, is_scores = [], [], []
        start = 0
        while start + train_w + 1 < md.n_bars:
            tr_e = start + train_w
            te_e = min(tr_e + test_w, md.n_bars)
            if te_e - tr_e < 1:
                break

            fi, ft, fb = _flat(md, target, base, start, tr_e, pool)
            gi, gt, gb = _flat(md, target, base, tr_e, te_e, pool)
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
    p.add_argument("--n", type=int, default=120,
                   help="idea 2 is the 6th test tried; alpha_effective="
                        "0.05/6=0.0083, needs floor 1/(n+1)<0.0083 i.e. "
                        "n>=120 to be properly powered from the start")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--universe", type=int, default=100)
    p.add_argument("--universe-as-of", default="2019-01-01")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=63)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--block", type=int, default=21)
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--checkpoint", default="logs/edge_idea2_ckpt.json")
    p.add_argument("--out", default="docs/results/edge_idea2_interactions.json")
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
            n_pairs = len(names) * (len(names) - 1) // 2
            logger.info("reused feature cache: %d channels -> %d interaction terms",
                        len(names), n_pairs)

    cv, ci, cc = close.values, close.index, list(close.columns)
    jobs = [(cv, ci, cc, None, a.block, a.train_window, a.test_window,
             a.horizon, extra, extra_names)] + [
        (cv, ci, cc, 7000 + i, a.block, a.train_window, a.test_window,
         a.horizon, extra, extra_names)
        for i in range(a.n)]

    done = load_ckpt(a.checkpoint)
    todo = [j for j in jobs if draw_key(j[3]) not in done]
    logger.info("%d jobs (1 real + %d null), %d already done, %d to run",
                len(jobs), a.n, len(done), len(todo))

    for i, j in enumerate(todo, 1):
        r = _one_draw(j)
        done[draw_key(r["perm_seed"])] = r
        save_ckpt(a.checkpoint, done)
        logger.info("  %d/%d this run (%d/%d total, pool=%d features)",
                    i, len(todo), len(done), len(jobs), r["n_pool_features"])

    if len(done) < len(jobs):
        logger.info("not all draws complete yet - rerun to continue")
        return 1

    real = done[draw_key(None)]
    nulls = [v for k, v in done.items() if k != "real"]

    state_path = Path("logs/edge_search_state.json")
    n_ideas = 6
    if state_path.exists():
        try:
            n_ideas = max(1, json.loads(state_path.read_text())["n_ideas_tried"] + 1)
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    alpha_eff = 0.05 / n_ideas

    print("\n" + "=" * 78)
    print(f"IDEA 2: PAIRWISE FEATURE INTERACTIONS  (pool={real['n_pool_features']} "
          f"features, {len(nulls)} null draws, alpha_effective={alpha_eff:.4f} "
          f"for idea #{n_ideas})")
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
              f"{r['n_distinct_picks']} distinct: {r['picks'][:3]}...)")
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
        "idea": "two_feature_interaction", "idea_id": 2,
        "n_ideas_tried_including_this": n_ideas, "alpha_effective": alpha_eff,
        "n_pool_features": real["n_pool_features"],
        "any_survive": any_survive, "targets": summary,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
