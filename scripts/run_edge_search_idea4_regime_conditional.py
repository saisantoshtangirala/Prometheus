"""
Edge-search idea #4: REGIME-CONDITIONAL predictability of direction_1d.

STATE. Iteration 3 of the rigorous-search protocol in
logs/edge_search_state.json (idea 1 cross-sectional rank, idea 3 longer
horizons: both no directional edge).

THE HYPOTHESIS. Every prior test pools all dates together. direction_1d
may be unpredictable ON AVERAGE while still being predictable during a
specific market regime - e.g. only when volatility is actively
expanding, when microstructure signals like PCR or OI often carry more
information (a real, known effect in market microstructure literature:
positioning signals are frequently regime-dependent). This is a genuinely
different hypothesis from every pooled test run so far, not a re-test.

WHY THE CONDITIONING VARIABLE MUST BE BUILT CAREFULLY. The obvious
conditioning variable, regime_shift_5d, is itself a FORWARD-looking
target (it needs 5 future days of returns to compute) - conditioning a
same-day trading decision on it would be a look-ahead: you cannot know
today whether volatility is about to expand.

The correct conditioning variable is CAUSAL: it must be computable from
information available at or before the decision bar. This uses

    causal_regime[t] = log(trailing_5d_vol[t] / trailing_20d_vol[t])

- purely backward-looking (recent realised vol relative to a longer
baseline), available at t, no future information.

WHY THE THRESHOLD ITSELF MUST BE TRAIN-ONLY. Splitting into "high" vs
"low" regime by a tercile of the FULL series would leak future
distribution shape into the threshold used at the start of the sample.
Instead, each walk-forward window computes its regime tercile cutoff
from ONLY that window's TRAIN slice - exactly the discipline the project
already uses for feature selection (select on train, evaluate on test),
extended to the conditioning variable itself.

NULL VALIDITY. causal_regime is recomputed fresh from each permuted
close series, per draw - not reused from the real data - so the
conditioning variable is subject to the same permutation as everything
else and the null cloud stays a fair comparison.
"""

from __future__ import annotations

import argparse
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
from run_evolved_walkforward import (                                 # noqa: E402
    block_permutation_order, block_permute_prices, resolve_universe,
)
from run_target_walkforward import _score                             # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("idea4_regime")


def causal_regime(close: np.ndarray, short: int = 5, long: int = 20) -> np.ndarray:
    """log(trailing short-window vol / trailing long-window vol).
    Backward-looking only - uses returns up to and including bar t."""
    T, A = close.shape
    rets = np.full((T, A), np.nan)
    rets[1:] = close[1:] / close[:-1] - 1.0

    def trailing_std(win):
        out = np.full((T, A), np.nan)
        for t in range(win, T):
            w = rets[t - win + 1:t + 1]
            ok = np.isfinite(w).all(axis=0)
            out[t, ok] = w[:, ok].std(axis=0, ddof=1)
        return out

    sv, lv = trailing_std(short), trailing_std(long)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(sv / lv)
    return np.where(np.isfinite(out), out, np.nan)


def _flat_masked(md, target, base, extra_mask, s, e, pool):
    ind = pool[s:e]
    tv, va = target.values[s:e], target.valid[s:e]
    bs = base[s:e]
    em = extra_mask[s:e]
    m = va & np.isfinite(tv) & np.isfinite(bs) & em
    if m.sum() < 20:
        return None, None, None
    return (ind.reshape(-1, ind.shape[-1])[m.ravel()],
            tv.ravel()[m.ravel()], bs.ravel()[m.ravel()])


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

    # Regime conditioning variable built from the SAME (possibly
    # permuted) close series that produced everything else this draw -
    # not the real data reused across draws.
    regime_full = causal_regime(close_df.to_numpy())
    trim = len(close_df.index) - md.n_bars
    regime = regime_full[trim:trim + md.n_bars] if trim >= 0 else regime_full

    targets = build_targets(md.close, horizon=horizon)
    out = {"perm_seed": perm_seed, "n_bars": int(md.n_bars), "targets": {}}

    for tname in ("direction_1d", "rel_strength_1d"):
        target = targets[tname]
        base = persistence_baseline(target, md.close, horizon=horizon)

        for regime_label in ("high", "low"):
            key = f"{tname}__{regime_label}_regime"
            picks, oos_scores, is_scores = [], [], []
            start = 0
            while start + train_w + 1 < md.n_bars:
                tr_e = start + train_w
                te_e = min(tr_e + test_w, md.n_bars)
                if te_e - tr_e < 1:
                    break

                # TRAIN-ONLY tercile threshold - the discipline this idea
                # depends on for validity.
                train_regime = regime[start:tr_e]
                finite = train_regime[np.isfinite(train_regime)]
                if finite.size < 50:
                    start += test_w
                    continue
                hi_cut = np.percentile(finite, 200.0 / 3.0)
                lo_cut = np.percentile(finite, 100.0 / 3.0)
                if regime_label == "high":
                    mask = regime > hi_cut
                else:
                    mask = regime < lo_cut

                fi, ft, fb = _flat_masked(md, target, base, mask, start, tr_e, pool)
                gi, gt, gb = _flat_masked(md, target, base, mask, tr_e, te_e, pool)
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
                out["targets"][key] = {
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
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--universe", type=int, default=100)
    p.add_argument("--universe-as-of", default="2019-01-01")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=63)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--block", type=int, default=21)
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--checkpoint", default="logs/edge_idea4_ckpt.json")
    p.add_argument("--out", default="docs/results/edge_idea4_regime_conditional.json")
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
             a.horizon, extra, extra_names)] + [
        (cv, ci, cc, 5000 + i, a.block, a.train_window, a.test_window,
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
        logger.info("  %d/%d this run (%d/%d total)", i, len(todo), len(done), len(jobs))

    if len(done) < len(jobs):
        logger.info("not all draws complete yet - rerun to continue")
        return 1

    real = done[draw_key(None)]
    nulls = [v for k, v in done.items() if k != "real"]

    state_path = Path("logs/edge_search_state.json")
    n_ideas = 3
    if state_path.exists():
        try:
            n_ideas = max(1, json.loads(state_path.read_text())["n_ideas_tried"] + 1)
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    alpha_eff = 0.05 / n_ideas

    print("\n" + "=" * 78)
    print(f"IDEA 4: REGIME-CONDITIONAL DIRECTION  ({len(nulls)} null draws, "
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
        print(f"\n  {tname:<28} ({r['n_windows']} windows, "
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
        "idea": "regime_conditional_predictability", "idea_id": 4,
        "n_ideas_tried_including_this": n_ideas, "alpha_effective": alpha_eff,
        "any_survive": any_survive, "targets": summary,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
