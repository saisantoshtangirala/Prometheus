"""
Edge-search idea #5: SECTOR-RELATIVE PAIRS mean reversion.

STATE. Iteration 4 of the rigorous-search protocol in
logs/edge_search_state.json. Ideas 1, 3, 4: no directional edge.

THE HYPOTHESIS. Structurally different from every prior idea: not
"does a feature predict a stock's own future return" but "does the
SPREAD between two same-sector stocks mean-revert" - a pairs/statistical-
arbitrage signal, using nightevolver/sectors.py's hand-curated grouping
(13 groups, >=3 members each, built and tested this iteration - see that
module's docstring for its provenance and limits).

WHY POOLED RATHER THAN PER-PAIR. The 13 sector groups produce ~240
distinct pairs (all-pairs within each group). Testing each pair
separately would need a combinatorial outer correction across 240 tests
- expensive to run properly and exactly the design idea 2 was deferred
over. Sidestepped by construction: the hypothesis is "does sector-pairs
reversion exist as a general phenomenon", which does not require
per-pair identification. Every pair's (z-score, forward-reversion)
observation is pooled into ONE synthetic [T, n_pairs] panel and scored
with a SINGLE incremental correlation against ONE null cloud - the same
protocol cost as every prior idea, no per-pair correction needed.

CONSTRUCTION.
    ratio[t]  = log(close_A[t]) - log(close_B[t])           (log spread)
    z[t]      = causal rolling z-score of ratio, 60-bar trailing window
    fwd[t]    = ratio[t+H] - ratio[t]                        (H-day forward spread change)
    base[t]   = ratio[t] - ratio[t-H]                        (H-day trailing spread change - the
                                                                "spread has been drifting this way" baseline)

If pairs mean-revert, a large positive z (A rich vs B) should predict a
negative fwd (spread narrows) - a NEGATIVE correlation between z and fwd,
beyond what trailing momentum in the spread already explains.

NULL VALIDITY. The pair panel is rebuilt from md.close AFTER block
permutation each draw (same permuted, trimmed close series everything
else in this search uses), so the null is a fair comparison under the
same permutation discipline as every other idea.
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

from nightevolver.data_loader import build_market_data                # noqa: E402
from nightevolver.nse_prices import fetch_nse_prices                  # noqa: E402
from nightevolver.sectors import sector_groups                        # noqa: E402
from run_evolved_walkforward import block_permute_prices, resolve_universe  # noqa: E402
from run_target_walkforward import _score                             # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("idea5_pairs")

Z_WINDOW = 60


def build_pairs(tickers) -> list:
    groups = sector_groups(list(tickers), min_size=3)
    pairs = []
    for _, members in sorted(groups.items()):
        pairs.extend(itertools.combinations(sorted(members), 2))
    return pairs


def causal_zscore(ratio: np.ndarray, window: int = Z_WINDOW) -> np.ndarray:
    T, P = ratio.shape
    out = np.full((T, P), np.nan)
    for t in range(window, T):
        w = ratio[t - window + 1:t + 1]
        ok = np.isfinite(w).all(axis=0)
        mean = w[:, ok].mean(axis=0)
        std = w[:, ok].std(axis=0, ddof=1)
        std = np.where(std > 1e-10, std, np.nan)
        out[t, ok] = (ratio[t, ok] - mean) / std
    return out


def _one_draw(args):
    (values, index, columns, perm_seed, block, train_w, test_w, horizon,
     pairs) = args
    logging.disable(logging.INFO)

    close_df = pd.DataFrame(values, index=index, columns=columns)
    if perm_seed is not None:
        close_df = block_permute_prices(close_df, block, perm_seed)
    md = build_market_data(close_df)

    idx = {t: i for i, t in enumerate(md.tickers)}
    T = md.n_bars
    log_close = np.log(np.where(md.close > 0, md.close, np.nan))
    valid_pairs = [(a, b) for a, b in pairs if a in idx and b in idx]
    ratio = np.full((T, len(valid_pairs)), np.nan)
    for j, (a, b) in enumerate(valid_pairs):
        ratio[:, j] = log_close[:, idx[a]] - log_close[:, idx[b]]

    z = causal_zscore(ratio)
    fwd = np.full_like(ratio, np.nan)
    if horizon < T:
        # fwd[t] = ratio[t+horizon] - ratio[t], stored at position t.
        fwd[:T - horizon] = ratio[horizon:] - ratio[:-horizon]
    base = np.full_like(ratio, np.nan)
    if horizon < T:
        # base[t] = ratio[t] - ratio[t-horizon], stored at position t -
        # causal (uses only bars <= t), unlike fwd which reaches to t+horizon.
        base[horizon:] = ratio[horizon:] - ratio[:-horizon]

    picks, oos_scores, is_scores = [], [], []
    start = 0
    while start + train_w + 1 < T:
        tr_e = start + train_w
        te_e = min(tr_e + test_w, T)
        if te_e - tr_e < 1:
            break

        def flat(s, e):
            zz, ff, bb = z[s:e], fwd[s:e], base[s:e]
            m = np.isfinite(zz) & np.isfinite(ff) & np.isfinite(bb)
            if m.sum() < 20:
                return None, None, None
            return zz[m], ff[m], bb[m]

        fz, ff, fb = flat(start, tr_e)
        gz, gf, gb = flat(tr_e, te_e)
        if fz is None or gz is None:
            start += test_w
            continue

        is_scores.append(abs(_score(fz, ff, fb)))
        oos_scores.append(abs(_score(gz, gf, gb)))
        picks.append("z_score")
        start += test_w

    out = {"perm_seed": perm_seed, "n_bars": T, "n_pairs": len(valid_pairs),
          "targets": {}}
    if oos_scores:
        out["targets"]["pair_reversion"] = {
            "n_windows": len(oos_scores),
            "mean_is": float(np.mean(is_scores)),
            "mean_oos": float(np.mean(oos_scores)),
            "gap": float(np.mean(is_scores) - np.mean(oos_scores)),
            "picks": picks, "n_distinct_picks": 1,
        }
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None,
                   help="restrict the panel end date - used for the "
                        "confirmatory split-sample re-test")
    p.add_argument("--universe", type=int, default=100)
    p.add_argument("--universe-as-of", default="2019-01-01")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=63)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--block", type=int, default=21)
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--checkpoint", default="logs/edge_idea5_ckpt.json")
    p.add_argument("--out", default="docs/results/edge_idea5_sector_pairs.json")
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
    md0 = fetch_nse_prices(tickers, a.start, a.end, use_cache=True,
                           require_actions=True, min_coverage=a.min_coverage)
    close = pd.DataFrame(md0.close, index=pd.DatetimeIndex(md0.dates),
                         columns=list(md0.tickers))
    pairs = build_pairs(close.columns)
    logger.info("source: %d bars x %d tickers -> %d same-sector pairs",
                len(close), close.shape[1], len(pairs))
    if not pairs:
        logger.error("no sector pairs available - check sectors.py coverage")
        return 2

    cv, ci, cc = close.values, close.index, list(close.columns)
    jobs = [(cv, ci, cc, None, a.block, a.train_window, a.test_window,
             a.horizon, pairs)] + [
        (cv, ci, cc, 6000 + i, a.block, a.train_window, a.test_window,
         a.horizon, pairs)
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
    n_ideas = 4
    if state_path.exists():
        try:
            n_ideas = max(1, json.loads(state_path.read_text())["n_ideas_tried"] + 1)
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    alpha_eff = 0.05 / n_ideas

    print("\n" + "=" * 78)
    print(f"IDEA 5: SECTOR-RELATIVE PAIRS  ({real['n_pairs']} pairs, "
          f"{len(nulls)} null draws, alpha_effective={alpha_eff:.4f} "
          f"for idea #{n_ideas})")
    print("=" * 78)

    r = real["targets"].get("pair_reversion")
    if r is None:
        print("no windows produced - insufficient data")
        return 2
    cloud = np.array([n["targets"]["pair_reversion"]["mean_oos"]
                      for n in nulls if "pair_reversion" in n["targets"]])
    p = (int((cloud >= r["mean_oos"]).sum()) + 1) / (cloud.size + 1)
    underpowered = (1.0 / (cloud.size + 1)) > alpha_eff
    survives = (not underpowered) and p < alpha_eff
    pctile = float((cloud < r["mean_oos"]).mean() * 100)
    verdict = ("UNDERPOWERED" if underpowered
               else "SURVIVES the outer-corrected bar - CANDIDATE, not a finding"
               if survives else "does not survive")
    print(f"\n  pair_reversion  ({r['n_windows']} windows)")
    print(f"    OOS (real) {r['mean_oos']:+.4f}   null mean "
          f"{float(cloud.mean()):+.4f} sd {float(cloud.std()):.4f}")
    print(f"    percentile {pctile:.0f}   p={p:.4f}   -> {verdict}")
    print("=" * 78)

    out = {
        "idea": "sector_relative_pairs", "idea_id": 5, "n_pairs": real["n_pairs"],
        "n_ideas_tried_including_this": n_ideas, "alpha_effective": alpha_eff,
        "any_survive": bool(survives),
        "targets": {"pair_reversion": {
            "p": p, "survives": bool(survives), "underpowered": bool(underpowered),
            "oos": r["mean_oos"], "null_mean": float(cloud.mean()),
            "null_sd": float(cloud.std()), "percentile": pctile,
            "n_windows": r["n_windows"],
        }},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
