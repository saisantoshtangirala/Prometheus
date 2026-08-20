"""
Does this feature carry information about this target? Measure it BEFORE
searching over it.

WHY THIS MODULE EXISTS
----------------------
Every result this project has produced so far came from the same loop:
point an optimiser at some features, let it search, read off the best
score. That loop cannot answer the question that matters, because a
search over 1,050 candidates returns a strong in-sample number whether
or not the data contains anything - measured here in the last RunPod
run, where the GA's winner (+1.29 in-sample) scored *below* the +1.62 a
best-of-1050 search over pure noise would be expected to reach.

An information audit inverts the order. It asks, per (feature, target)
pair and with no optimisation at all, whether there is a dependence that
survives (a) autocorrelation, (b) cross-sectional correlation, (c)
multiple testing, and (d) a trivial persistence baseline. If nothing
survives, no optimiser will find an edge, and the correct action is to
change the inputs rather than the search.

THE FOUR THINGS THIS GETS RIGHT, each of which is a way the naive
version lies:

1. BLOCK permutation, not iid shuffling. Features and targets are both
   strongly autocorrelated. An iid permutation destroys that structure
   and produces a null far narrower than reality, so ordinary p-values
   come out spuriously tiny. Contiguous blocks (default 21 bars ~ one
   month) preserve serial dependence under the null.

2. Time blocks are permuted JOINTLY ACROSS ASSETS. Ten NSE large-caps on
   the same day are not ten independent observations - they share a
   market factor and move together. Permuting each asset separately
   would treat T x A as the sample size, when the effective sample size
   is closer to T. Permuting whole rows keeps the cross-section intact.

3. Benjamini-Hochberg FDR across the WHOLE grid. Testing 26 features
   against 4 targets is 104 simultaneous tests; at alpha=0.05 roughly 5
   will look significant by chance alone. Reporting the best one without
   correction is the multiple-testing error this project already made
   once, in a different costume.

4. Every target is scored against its PERSISTENCE BASELINE. Volatility
   is autocorrelated, so a vol feature predicts a vol target almost by
   definition. The number that matters is the *incremental* dependence
   after the trivial "tomorrow looks like today" forecast is partialled
   out.

Spearman rank correlation is the dependence measure: it needs no
linearity assumption, it is robust to the fat tails that make Pearson
unstable on returns, and it is cheap enough to permute thousands of
times.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from nightevolver.targets import Target, persistence_baseline

logger = logging.getLogger("nightevolver.audit")

DEFAULT_BLOCK_BARS = 21          # ~1 trading month
DEFAULT_N_PERMUTATIONS = 2000
DEFAULT_FDR_ALPHA = 0.05


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks along axis 0 of a 1-D array."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average ties
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def _standardise(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-norm. Returns zeros for a degenerate column."""
    x = x - x.mean()
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else np.zeros_like(x)


def _block_permutation_indices(n_rows: int, block: int,
                               rng: np.random.Generator) -> np.ndarray:
    """A row permutation built from contiguous blocks.

    The series is cut into ceil(n/block) blocks, the blocks are shuffled,
    and the result is truncated back to n rows. This preserves
    within-block serial dependence, which is the point.
    """
    if block <= 1:
        return rng.permutation(n_rows)
    starts = np.arange(0, n_rows, block)
    order = rng.permutation(len(starts))
    idx = np.concatenate([np.arange(s, min(s + block, n_rows)) for s in starts[order]])
    return idx[:n_rows]


def _partial_out(y: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Residual of y after least-squares removal of `control`.

    Both are expected pre-standardised. Used to strip the persistence
    baseline out of a target so the remaining dependence is incremental.
    """
    c = _standardise(control)
    if not np.any(c):
        return y
    return y - float(y @ c) * c


@dataclass
class PairResult:
    feature: str
    target: str
    spearman: float                  # raw pooled rank correlation
    spearman_incremental: float      # after partialling out persistence
    p_value: float                   # block-permutation, two-sided
    p_value_incremental: float
    n_effective: int                 # independent blocks, not T*A
    q_value: float = float("nan")    # BH-FDR adjusted p (incremental)
    significant: bool = False
    market_level: bool = False       # asset-invariant: scored at market level

    def __str__(self) -> str:
        star = "  <-- SURVIVES FDR" if self.significant else ""
        mw = " [mkt]" if self.market_level else ""
        return (f"{self.feature:22s} -> {self.target:16s} "
                f"rho={self.spearman:+.4f}  incr={self.spearman_incremental:+.4f}  "
                f"p={self.p_value_incremental:.4f}  q={self.q_value:.4f}{mw}{star}")


@dataclass
class AuditResult:
    pairs: List[PairResult] = field(default_factory=list)
    n_bars: int = 0
    n_assets: int = 0
    block_bars: int = DEFAULT_BLOCK_BARS
    n_permutations: int = DEFAULT_N_PERMUTATIONS
    fdr_alpha: float = DEFAULT_FDR_ALPHA

    @property
    def survivors(self) -> List[PairResult]:
        return [p for p in self.pairs if p.significant]

    def best_per_target(self) -> Dict[str, PairResult]:
        out: Dict[str, PairResult] = {}
        for p in self.pairs:
            cur = out.get(p.target)
            if cur is None or abs(p.spearman_incremental) > abs(cur.spearman_incremental):
                out[p.target] = p
        return out

    def summary(self) -> str:
        lines = [
            "=" * 78,
            "INFORMATION AUDIT (no optimisation - direct dependence measurement)",
            "=" * 78,
            f"  data           {self.n_bars} bars x {self.n_assets} assets",
            f"  null           block permutation, {self.block_bars}-bar blocks, "
            f"{self.n_permutations} draws, rows permuted jointly across assets",
            f"  correction     Benjamini-Hochberg FDR at alpha={self.fdr_alpha} "
            f"over all {len(self.pairs)} pairs",
            "",
            "  'incr' = Spearman AFTER partialling out the persistence baseline",
            "  (the trivial 'tomorrow looks like today' forecast). That is the",
            "  column that matters; raw rho on a vol target is nearly free.",
            "",
        ]
        for target in sorted({p.target for p in self.pairs}):
            lines.append(f"  --- {target} " + "-" * (68 - len(target)))
            rows = sorted((p for p in self.pairs if p.target == target),
                          key=lambda p: -abs(p.spearman_incremental))
            for p in rows[:8]:
                lines.append("    " + str(p))
            lines.append("")

        surv = self.survivors
        lines.append("=" * 78)
        if surv:
            lines.append(f"  {len(surv)} of {len(self.pairs)} pairs survive FDR correction:")
            for p in sorted(surv, key=lambda p: -abs(p.spearman_incremental)):
                lines.append(f"    {p.feature} -> {p.target}  "
                             f"incr rho={p.spearman_incremental:+.4f}  q={p.q_value:.4f}")
            lines.append("")
            lines.append("  A surviving pair is a licence to SEARCH, not an edge. It says")
            lines.append("  the dependence is unlikely to be an artefact of noise; it says")
            lines.append("  nothing about whether it survives 22bp of round-trip cost.")
        else:
            lines.append("  NOTHING survives FDR correction.")
            lines.append("")
            lines.append("  No optimiser can extract an edge from features that carry no")
            lines.append("  measurable information about the target. Running a bigger GA")
            lines.append("  over these inputs will produce a better in-sample number and")
            lines.append("  the same out-of-sample collapse. The lever is new inputs.")
        lines.append("=" * 78)
        return "\n".join(lines)


def benjamini_hochberg(p_values: Sequence[float],
                       alpha: float = DEFAULT_FDR_ALPHA) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (q_values, rejected). Standard BH step-up procedure."""
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * n / np.arange(1, n + 1)
    # enforce monotonicity from the largest p downwards
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_ranked, 0.0, 1.0)
    return q, q <= alpha


def audit_features(features: np.ndarray,
                   feature_names: Sequence[str],
                   targets: Dict[str, Target],
                   close: np.ndarray,
                   block_bars: int = DEFAULT_BLOCK_BARS,
                   n_permutations: int = DEFAULT_N_PERMUTATIONS,
                   fdr_alpha: float = DEFAULT_FDR_ALPHA,
                   seed: int = 0) -> AuditResult:
    """Measure dependence of every feature on every target.

    features: [T, A, F] causal features (indicators and/or flows)
    targets:  name -> Target, each [T, A] forward-looking
    close:    [T, A], needed to build the persistence baselines

    Returns an AuditResult with BH-FDR-corrected significance.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 3:
        raise ValueError(f"features must be [T, A, F]; got {features.shape}")
    T, A, F = features.shape
    if len(feature_names) != F:
        raise ValueError(f"{len(feature_names)} names for {F} feature channels")

    rng = np.random.default_rng(seed)
    result = AuditResult(n_bars=T, n_assets=A, block_bars=block_bars,
                         n_permutations=n_permutations, fdr_alpha=fdr_alpha)

    # Permutations are drawn ONCE and reused across every (feature,
    # target) pair. That is deliberate: it makes the tests share a null
    # realisation, which is what BH assumes about dependent tests, and
    # it makes the whole audit cheap enough to run 2000 draws.
    perms = [_block_permutation_indices(T, block_bars, rng)
             for _ in range(n_permutations)]

    for tname, target in targets.items():
        valid_rows = target.valid.all(axis=1)          # keep whole rows only
        if valid_rows.sum() < 3 * block_bars:
            logger.warning("[audit] target %s has only %d fully-valid rows - skipping",
                           tname, int(valid_rows.sum()))
            continue

        rows = np.flatnonzero(valid_rows)
        n_rows = len(rows)
        base = persistence_baseline(target, close)[rows]        # [n_rows, A]
        y = target.values[rows]                                 # [n_rows, A]

        # Rank-transform per asset (rank correlation), then flatten.
        y_rank = np.column_stack([_rankdata(y[:, a]) for a in range(A)])
        b_rank = np.column_stack([_rankdata(base[:, a]) for a in range(A)])

        y_flat = _standardise(y_rank.reshape(-1))
        b_flat = _standardise(b_rank.reshape(-1))
        y_incr = _standardise(_partial_out(y_flat, b_flat))

        # MARKET-WIDE FEATURES ARE NOT T*A OBSERVATIONS.
        #
        # The flow features (FII/DII positioning) are one number per
        # session, broadcast identically across every asset. Pooling
        # them as T*A rows claims 5,870 observations from ~588
        # independent days, and the rank transform then breaks the exact
        # cancellation that a cross-sectionally demeaned target should
        # give a market-wide predictor.
        #
        # Measured: six PURE-NOISE market-wide features run through the
        # pooled path reached |incremental rho| up to 0.0126 on
        # rel_strength_1d and produced 2-3 spurious FDR survivors per
        # seed. The real fii_stk_fut_net scored 0.0167 - the same order
        # of magnitude as the artefact, i.e. not established.
        #
        # The fix is to evaluate an asset-invariant feature against the
        # cross-sectional MEAN of the target, which is what it can
        # actually speak to, at its true sample size of n_rows. For a
        # demeaned target that mean is ~0, so such a feature correctly
        # scores nothing rather than borrowing significance from
        # duplication.
        market_wide = np.zeros(F, dtype=bool)
        for f in range(F):
            chan = features[rows, :, f]
            market_wide[f] = bool(np.allclose(chan, chan[:, :1], atol=1e-12))
        if market_wide.any():
            logger.info("[audit] %d/%d feature channels are asset-invariant; "
                        "scoring them at market level (n=%d) rather than "
                        "pooled (n=%d)", int(market_wide.sum()), F, n_rows,
                        n_rows * A)

        X = np.empty((n_rows * A, F))
        for f in range(F):
            cols = [_rankdata(features[rows, a, f]) for a in range(A)]
            X[:, f] = _standardise(np.column_stack(cols).reshape(-1))

        # Market-level design for the asset-invariant channels.
        mw_idx = np.flatnonzero(market_wide)
        if mw_idx.size:
            Xm = np.empty((n_rows, mw_idx.size))
            for j, f in enumerate(mw_idx):
                Xm[:, j] = _standardise(_rankdata(features[rows, 0, f]))
            ym_raw = _rankdata(y.mean(axis=1))
            bm_raw = _rankdata(base.mean(axis=1))
            ym = _standardise(ym_raw)
            bm = _standardise(bm_raw)
            ym_incr = _standardise(_partial_out(ym, bm))
            obs_raw_m = Xm.T @ ym
            obs_incr_m = Xm.T @ ym_incr
            cnt_raw_m = np.zeros(mw_idx.size)
            cnt_incr_m = np.zeros(mw_idx.size)

        obs_raw = X.T @ y_flat                    # [F]
        obs_incr = X.T @ y_incr

        # Null: permute the TARGET's time blocks, jointly across assets.
        cnt_raw = np.zeros(F)
        cnt_incr = np.zeros(F)
        for idx in perms:
            keep = idx[idx < n_rows] if n_rows != T else idx
            if len(keep) != n_rows:
                # rows were subset (target had invalid tail); rebuild a
                # block permutation on the reduced length instead of
                # silently comparing mismatched samples
                keep = _block_permutation_indices(n_rows, block_bars, rng)
            yp = _standardise(y_rank[keep].reshape(-1))
            yp_incr = _standardise(_partial_out(yp, b_flat))
            cnt_raw += (np.abs(X.T @ yp) >= np.abs(obs_raw))
            cnt_incr += (np.abs(X.T @ yp_incr) >= np.abs(obs_incr))
            if mw_idx.size:
                ypm = _standardise(ym_raw[keep])
                ypm_incr = _standardise(_partial_out(ypm, bm))
                cnt_raw_m += (np.abs(Xm.T @ ypm) >= np.abs(obs_raw_m))
                cnt_incr_m += (np.abs(Xm.T @ ypm_incr) >= np.abs(obs_incr_m))

        # +1 in numerator and denominator: a permutation p-value is an
        # estimate and can never legitimately be 0.
        p_raw = (cnt_raw + 1.0) / (n_permutations + 1.0)
        p_incr = (cnt_incr + 1.0) / (n_permutations + 1.0)

        # Overwrite the pooled statistics for asset-invariant channels
        # with the market-level ones computed at the honest sample size.
        for j, f in enumerate(mw_idx):
            obs_raw[f] = obs_raw_m[j]
            obs_incr[f] = obs_incr_m[j]
            p_raw[f] = (cnt_raw_m[j] + 1.0) / (n_permutations + 1.0)
            p_incr[f] = (cnt_incr_m[j] + 1.0) / (n_permutations + 1.0)

        for f, fname in enumerate(feature_names):
            result.pairs.append(PairResult(
                feature=fname, target=tname,
                spearman=float(obs_raw[f]),
                spearman_incremental=float(obs_incr[f]),
                p_value=float(p_raw[f]),
                p_value_incremental=float(p_incr[f]),
                n_effective=int(np.ceil(n_rows / block_bars)),
                market_level=bool(market_wide[f]),
            ))

    q, rejected = benjamini_hochberg([p.p_value_incremental for p in result.pairs],
                                     alpha=fdr_alpha)
    for p, qi, ri in zip(result.pairs, q, rejected):
        p.q_value = float(qi)
        p.significant = bool(ri)
    return result
