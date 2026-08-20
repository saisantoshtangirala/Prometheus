"""
Dependence tests that do NOT assume monotonicity.

WHY THIS EXISTS
---------------
`information_audit.py` uses Spearman rank correlation. Spearman is
robust, cheap enough to permute thousands of times, and needs no
linearity assumption - but it detects MONOTONIC dependence only. Two
real signal shapes score approximately zero under it:

  U-SHAPED / non-monotonic. "Extreme RSI in either direction predicts
  reversal" has a strong V in the scatter and a Spearman rho near 0,
  because the two halves cancel.

  CONDITIONAL / interaction. "Momentum works only when volatility is
  low" is invisible marginally: pooled across regimes the two
  conditional effects average out.

So a null result from the Spearman audit means "no monotonic
dependence", which is strictly weaker than "no information". This
module closes that gap with two tests that make no shape assumption at
all.

1. DISTANCE CORRELATION (Szekely, Rizzo & Bakirov 2007).
   dCor(X, Y) = 0 **if and only if** X and Y are independent. That
   biconditional is the whole point: unlike Pearson or Spearman, a zero
   is a genuine certificate of independence rather than an absence of
   one particular shape. It detects U-shapes, rings, and any other
   structure.

   Cost is O(n^2) memory, so this runs PER ASSET (n ~ 588) and averages,
   rather than pooling to n = T*A. That is also the statistically
   honest choice here: pooling would inflate the sample the same way the
   market-wide flow features did.

2. QUANTILE-BIN TEST (Kruskal-Wallis).
   Bin the feature into q quantiles and ask whether the target's
   distribution differs across bins. A V-shape produces clearly
   different bin medians at the extremes while its Spearman rho is ~0,
   so this catches exactly the case dCor is expensive to run on.

Both use the SAME block-permutation null and the SAME Benjamini-Hochberg
correction as the monotonic audit, so results are directly comparable
and the multiple-testing budget is shared honestly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from nightevolver.information_audit import (
    DEFAULT_BLOCK_BARS, DEFAULT_FDR_ALPHA, _block_permutation_indices,
    benjamini_hochberg,
)
from nightevolver.targets import Target

logger = logging.getLogger("nightevolver.nonlinear")

DEFAULT_N_PERMUTATIONS = 500      # O(n^2) per draw, so fewer than the linear audit
DEFAULT_QUANTILE_BINS = 5
MAX_N_FOR_DCOR = 1200             # memory guard: n^2 float64


def _double_center(d: np.ndarray) -> np.ndarray:
    """A_ij = d_ij - row_mean_i - col_mean_j + grand_mean."""
    row = d.mean(axis=1, keepdims=True)
    col = d.mean(axis=0, keepdims=True)
    return d - row - col + d.mean()


def _pairwise_abs(x: np.ndarray) -> np.ndarray:
    return np.abs(x[:, None] - x[None, :])


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """dCor in [0, 1]. Zero IFF x and y are independent.

    For 1-D x and y the distance matrix is just |x_i - x_j|, which keeps
    this to two n x n allocations.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n != y.size:
        raise ValueError(f"length mismatch: {n} vs {y.size}")
    if n < 4:
        return 0.0

    A = _double_center(_pairwise_abs(x))
    B = _double_center(_pairwise_abs(y))
    dcov2 = float((A * B).mean())
    dvarx = float((A * A).mean())
    dvary = float((B * B).mean())
    denom = np.sqrt(dvarx * dvary)
    if denom <= 1e-15 or dcov2 <= 0:
        return 0.0
    return float(np.sqrt(dcov2 / denom))


def kruskal_bin_statistic(x: np.ndarray, y: np.ndarray,
                          n_bins: int = DEFAULT_QUANTILE_BINS) -> float:
    """Kruskal-Wallis H for `y` grouped by quantile bins of `x`.

    Catches non-monotonic structure that rank correlation cancels out:
    a V-shape gives high H with Spearman rho near zero.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n < n_bins * 4:
        return 0.0

    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    groups = np.searchsorted(edges, x, side="right")

    # Ranks of y, average-tie handled by argsort twice (adequate here -
    # the statistic is used only against its own permutation null).
    order = np.argsort(y, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    h = 0.0
    for g in range(n_bins):
        m = groups == g
        n_g = int(m.sum())
        if n_g == 0:
            return 0.0
        h += n_g * (ranks[m].mean() - (n + 1) / 2.0) ** 2
    return float(12.0 / (n * (n + 1)) * h)


@dataclass
class NonlinearPair:
    feature: str
    target: str
    dcor: float
    dcor_p: float
    kruskal: float
    kruskal_p: float
    spearman_ref: float          # what the monotonic audit saw
    n_per_asset: int
    q_value: float = float("nan")
    significant: bool = False

    @property
    def hidden_by_monotonic(self) -> bool:
        """Significant here but invisible to Spearman - the case this
        module was built to find."""
        return self.significant and abs(self.spearman_ref) < 0.05

    def __str__(self) -> str:
        flag = "  <-- SURVIVES FDR" if self.significant else ""
        hid = "  [MISSED BY SPEARMAN]" if self.hidden_by_monotonic else ""
        return (f"{self.feature:22s} -> {self.target:16s} "
                f"dCor={self.dcor:.4f} p={self.dcor_p:.4f}  "
                f"KW={self.kruskal:6.2f} p={self.kruskal_p:.4f}  "
                f"(rho={self.spearman_ref:+.3f})  q={self.q_value:.4f}{flag}{hid}")


@dataclass
class NonlinearResult:
    pairs: List[NonlinearPair] = field(default_factory=list)
    n_bars: int = 0
    n_assets: int = 0
    n_permutations: int = 0
    fdr_alpha: float = DEFAULT_FDR_ALPHA

    @property
    def survivors(self) -> List[NonlinearPair]:
        return [p for p in self.pairs if p.significant]

    @property
    def hidden(self) -> List[NonlinearPair]:
        return [p for p in self.pairs if p.hidden_by_monotonic]

    def summary(self) -> str:
        lines = [
            "=" * 78,
            "NON-MONOTONIC DEPENDENCE AUDIT",
            "=" * 78,
            f"  data        {self.n_bars} bars x {self.n_assets} assets, per-asset tests",
            f"  statistics  distance correlation (zero IFF independent) + "
            f"Kruskal-Wallis over {DEFAULT_QUANTILE_BINS} quantile bins",
            f"  null        block permutation, {self.n_permutations} draws",
            f"  correction  Benjamini-Hochberg at alpha={self.fdr_alpha} "
            f"over {len(self.pairs)} pairs",
            "",
            "  Spearman detects MONOTONIC dependence only. These tests assume",
            "  no shape at all, so they can see U-shapes and interactions that",
            "  rank correlation cancels to zero.",
            "",
        ]
        for target in sorted({p.target for p in self.pairs}):
            rows = sorted((p for p in self.pairs if p.target == target),
                          key=lambda p: p.dcor_p)
            lines.append(f"  --- {target} " + "-" * (68 - len(target)))
            for p in rows[:6]:
                lines.append("    " + str(p))
            lines.append("")

        lines.append("=" * 78)
        surv, hid = self.survivors, self.hidden
        if not surv:
            lines.append("  NOTHING survives FDR correction.")
            lines.append("")
            lines.append("  This is the stronger null. Distance correlation is zero if and")
            lines.append("  ONLY if the variables are independent, so a null here rules out")
            lines.append("  non-monotonic and interaction structure too - not just the")
            lines.append("  monotonic dependence Spearman was testing for.")
        else:
            lines.append(f"  {len(surv)} of {len(self.pairs)} pairs survive FDR:")
            for p in sorted(surv, key=lambda p: p.dcor_p):
                lines.append(f"    {p.feature} -> {p.target}  dCor={p.dcor:.4f}  q={p.q_value:.4f}")
            if hid:
                lines.append("")
                lines.append(f"  {len(hid)} of these were INVISIBLE to the Spearman audit")
                lines.append("  (|rho| < 0.05). That is exactly the blind spot this module")
                lines.append("  exists to cover, and it means the monotonic null was too weak.")
        lines.append("=" * 78)
        return "\n".join(lines)


def audit_nonlinear(features: np.ndarray,
                    feature_names: Sequence[str],
                    targets: Dict[str, Target],
                    spearman_ref: Optional[Dict[Tuple[str, str], float]] = None,
                    block_bars: int = DEFAULT_BLOCK_BARS,
                    n_permutations: int = DEFAULT_N_PERMUTATIONS,
                    fdr_alpha: float = DEFAULT_FDR_ALPHA,
                    max_n: int = MAX_N_FOR_DCOR,
                    seed: int = 0) -> NonlinearResult:
    """Distance correlation + quantile-bin test for every (feature, target).

    Runs PER ASSET and averages the statistic across assets, rather than
    pooling to T*A. Pooling would claim an independence certificate on a
    sample whose effective size is far smaller - the same inflation that
    made market-wide flow features look significant in the linear audit.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 3:
        raise ValueError(f"features must be [T, A, F]; got {features.shape}")
    T, A, F = features.shape
    if len(feature_names) != F:
        raise ValueError(f"{len(feature_names)} names for {F} channels")

    # PERMUTATION FLOOR. A permutation p-value cannot go below
    # 1/(n_perm+1), and Benjamini-Hochberg multiplies the smallest p by
    # n_pairs. So the best achievable q is
    #
    #     q_min = n_pairs / (n_perm + 1)
    #
    # and if that exceeds alpha, NO pair can be rejected no matter how
    # strong the dependence is. The audit would return "nothing
    # survives" as a property of its own configuration rather than of
    # the data - a null result that means nothing at all.
    #
    # Measured while building this: 104 pairs at 200 permutations gives
    # q_min = 0.517, so the first real run was structurally incapable of
    # finding anything. Requirement:  n_perm + 1 >= n_pairs / alpha.
    n_pairs_expected = len(feature_names) * len(targets)
    q_floor = n_pairs_expected / (n_permutations + 1.0)
    if q_floor > fdr_alpha:
        raise ValueError(
            f"{n_permutations} permutations cannot support {n_pairs_expected} "
            f"pairs at alpha={fdr_alpha}: the smallest achievable q is "
            f"{q_floor:.3f}. Every pair would be reported as 'not significant' "
            f"regardless of the data. Use n_permutations >= "
            f"{int(np.ceil(n_pairs_expected / fdr_alpha)) - 1}, or test fewer "
            f"pairs."
        )

    rng = np.random.default_rng(seed)
    result = NonlinearResult(n_bars=T, n_assets=A,
                             n_permutations=n_permutations, fdr_alpha=fdr_alpha)

    for tname, target in targets.items():
        rows = np.flatnonzero(target.valid.all(axis=1))
        if rows.size < 5 * block_bars:
            logger.warning("[nonlinear] target %s has %d valid rows - skipping",
                           tname, rows.size)
            continue
        # Subsample contiguously if the asset series is long enough to
        # make the n^2 distance matrix expensive.
        if rows.size > max_n:
            start = (rows.size - max_n) // 2
            rows = rows[start:start + max_n]
        n = rows.size
        y_all = target.values[rows]                      # [n, A]

        perms = [_block_permutation_indices(n, block_bars, rng)
                 for _ in range(n_permutations)]

        for f, fname in enumerate(feature_names):
            x_all = features[rows, :, f]                 # [n, A]
            if np.allclose(x_all.std(axis=0), 0.0):
                continue                                 # inert channel

            # PRECOMPUTE THE DISTANCE MATRICES ONCE PER ASSET.
            #
            # The naive loop recomputes both n x n matrices for every
            # permutation, which is ~26 minutes for a 104-pair grid.
            # Two facts make that unnecessary:
            #
            #   1. X never changes under a permutation of Y, so A is
            #      constant.
            #   2. Permuting Y then double-centering equals double-
            #      centering then permuting: a permutation preserves the
            #      set of row and column means, so
            #      B_centered(y[idx]) == B_centered(y)[idx][:, idx].
            #      dVar(Y) is therefore invariant too.
            #
            # Each permutation collapses to one fancy-index and one
            # elementwise product.
            mats, obs_d, obs_k = [], [], []
            for a in range(A):
                x, y = x_all[:, a], y_all[:, a]
                if x.std() < 1e-12 or y.std() < 1e-12:
                    continue
                Ac = _double_center(_pairwise_abs(x))
                Bc = _double_center(_pairwise_abs(y))
                dvarx = float((Ac * Ac).mean())
                dvary = float((Bc * Bc).mean())
                denom = np.sqrt(dvarx * dvary)
                if denom <= 1e-15:
                    continue
                dcov2 = float((Ac * Bc).mean())
                mats.append((Ac, Bc, denom, x, y))
                obs_d.append(np.sqrt(dcov2 / denom) if dcov2 > 0 else 0.0)
                obs_k.append(kruskal_bin_statistic(x, y))
            if not obs_d:
                continue
            dcor_obs = float(np.mean(obs_d))
            kw_obs = float(np.mean(obs_k))

            # Null: permute the target's blocks, jointly across assets so
            # the cross-section stays intact.
            cnt_d, cnt_k = 0, 0
            for idx in perms:
                d_perm, k_perm = [], []
                for Ac, Bc, denom, x, y in mats:
                    Bp = Bc[np.ix_(idx, idx)]
                    dcov2 = float((Ac * Bp).mean())
                    d_perm.append(np.sqrt(dcov2 / denom) if dcov2 > 0 else 0.0)
                    k_perm.append(kruskal_bin_statistic(x, y[idx]))
                cnt_d += float(np.mean(d_perm)) >= dcor_obs
                cnt_k += float(np.mean(k_perm)) >= kw_obs

            ref = 0.0
            if spearman_ref is not None:
                ref = float(spearman_ref.get((fname, tname), 0.0))

            result.pairs.append(NonlinearPair(
                feature=fname, target=tname,
                dcor=dcor_obs,
                dcor_p=(cnt_d + 1.0) / (n_permutations + 1.0),
                kruskal=kw_obs,
                kruskal_p=(cnt_k + 1.0) / (n_permutations + 1.0),
                spearman_ref=ref, n_per_asset=n,
            ))

    q, rejected = benjamini_hochberg([p.dcor_p for p in result.pairs],
                                     alpha=fdr_alpha)
    for p, qi, ri in zip(result.pairs, q, rejected):
        p.q_value = float(qi)
        p.significant = bool(ri)
    return result
