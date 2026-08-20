"""
Component 1/6 - Financial Connectome: dynamic market network connectivity.

The premise (and the reason this is a feature source rather than a
visualisation): a market's *connectivity structure* carries information
that its marginal returns do not. Correlation alone is the wrong tool -
two assets can be highly correlated purely because both track the index.
What matters is CONDITIONAL dependence: the partial correlation network,
which is read off the precision matrix (inverse covariance), and which
strips out the common factor before asking whether i and j are still
linked.

Concretely, for each rolling window this computes:

  covariance -> Ledoit-Wolf shrinkage -> precision -> partial correlations
             -> weighted graph -> per-node and global network statistics

Per-node (each becomes a per-asset feature):
  strength            weighted degree - how wired into the market this
                      asset currently is
  eigen_centrality    influence accounting for neighbours' influence
  clustering          Onnela weighted clustering - is this asset in a
                      tight cluster or a bridge between clusters
  participation       how evenly its connections spread across the book

Global (each becomes a market-regime feature, shared across assets):
  fiedler             algebraic connectivity: 2nd-smallest Laplacian
                      eigenvalue. Low = market fragmenting into blocs,
                      high = one tightly-coupled mass (the crisis
                      signature)
  spectral_radius     largest adjacency eigenvalue - overall coupling
  mean_abs_pcorr      average conditional dependence
  vn_entropy          von Neumann graph entropy of the normalised
                      Laplacian - structural disorder of the network
  density             fraction of edges above threshold

Everything is numpy and closed-form; no networkx, because this runs
inside a 125-window walk-forward loop and the graph is dense and small
(N=10), so direct linear algebra is both faster and easier to verify.

NO LOOK-AHEAD: every statistic is a function of one window of returns.
The caller is responsible for only ever passing it data it is allowed to
see; `rolling_features` enforces this by construction (bar t's features
use bars [t-window, t), strictly excluding t).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Partial correlations below this are treated as no edge. 0.05 is not
# tuned - it is a deliberately low bar, chosen so the threshold removes
# numerical dust rather than doing implicit feature selection (that is
# component 2's job, and it should get an unbiased feature bank).
EDGE_THRESHOLD = 0.05

NODE_FEATURE_NAMES = ("strength", "eigen_centrality", "clustering", "participation")
GLOBAL_FEATURE_NAMES = (
    "fiedler", "spectral_radius", "mean_abs_pcorr", "vn_entropy", "density",
)


@dataclass(frozen=True)
class ConnectomeFeatures:
    """One window's network snapshot.

    node:   [n_assets, len(NODE_FEATURE_NAMES)]
    glob:   [len(GLOBAL_FEATURE_NAMES)]
    pcorr:  [n_assets, n_assets] signed partial correlations (diag 0)
    """

    node: np.ndarray
    glob: np.ndarray
    pcorr: np.ndarray

    @property
    def adjacency(self) -> np.ndarray:
        """Thresholded |partial correlation| weights, zero diagonal."""
        a = np.abs(self.pcorr).copy()
        a[a < EDGE_THRESHOLD] = 0.0
        np.fill_diagonal(a, 0.0)
        return a


def ledoit_wolf_shrinkage(returns: np.ndarray) -> Tuple[np.ndarray, float]:
    """Covariance with Ledoit-Wolf shrinkage toward a scaled identity.

    Necessary, not optional: with T=252 and N=10 the sample covariance is
    usable, but the walk-forward harness also runs short windows, and an
    ill-conditioned covariance makes its INVERSE (which is the whole
    point here - the precision matrix IS the network) explode. Shrinkage
    keeps the precision matrix finite and the partial correlations
    meaningful.

    Returns (sigma, shrinkage_intensity).
    """
    T, N = returns.shape
    X = returns - returns.mean(axis=0, keepdims=True)
    S = (X.T @ X) / max(T - 1, 1)

    mu = float(np.trace(S)) / N
    target = mu * np.eye(N)

    # LW optimal intensity: b2/d2, where d2 is distance to target and b2
    # is the average per-observation deviation from the sample covariance.
    d2 = float(((S - target) ** 2).sum())
    if d2 <= 1e-18:
        return S, 0.0
    b2_sum = 0.0
    for t in range(T):
        xt = X[t : t + 1]
        b2_sum += float((((xt.T @ xt) - S) ** 2).sum())
    b2 = min(b2_sum / (T ** 2), d2)
    intensity = float(np.clip(b2 / d2, 0.0, 1.0))
    return (1.0 - intensity) * S + intensity * target, intensity


def partial_correlations(sigma: np.ndarray) -> np.ndarray:
    """Signed partial correlation matrix from a covariance matrix.

    pcorr_ij = -P_ij / sqrt(P_ii * P_jj), where P = sigma^-1. This is the
    correlation between i and j with ALL other assets partialled out -
    the conditional-dependence edge weight the connectome is built on.
    """
    N = sigma.shape[0]
    # Ridge before inversion: shrinkage already conditions sigma, but a
    # degenerate window (e.g. a holiday run of identical prices) can
    # still produce a singular matrix, and this must never raise inside
    # a 125-window loop.
    ridge = 1e-10 * float(np.trace(sigma)) / max(N, 1)
    precision = np.linalg.pinv(sigma + ridge * np.eye(N))

    d = np.sqrt(np.clip(np.diag(precision), 1e-18, None))
    pc = -precision / np.outer(d, d)
    np.fill_diagonal(pc, 0.0)
    return np.clip(pc, -1.0, 1.0)


def _eigen_centrality(adj: np.ndarray, iters: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Power iteration for the leading eigenvector (non-negative adj)."""
    n = adj.shape[0]
    if n == 0 or not np.any(adj):
        return np.zeros(n)
    v = np.ones(n) / np.sqrt(n)
    for _ in range(iters):
        w = adj @ v
        norm = np.linalg.norm(w)
        if norm < 1e-18:
            return np.zeros(n)
        w = w / norm
        if np.linalg.norm(w - v) < tol:
            v = w
            break
        v = w
    return np.abs(v)


def _onnela_clustering(adj: np.ndarray) -> np.ndarray:
    """Onnela weighted clustering coefficient per node.

    C_i = [ (W^(1/3))^3 ]_ii / (k_i (k_i - 1)), with W scaled to max 1.
    Measures whether an asset's neighbours are themselves connected -
    i.e. whether it sits inside a bloc or bridges between blocs.
    """
    n = adj.shape[0]
    mx = adj.max()
    if mx <= 0:
        return np.zeros(n)
    w = (adj / mx) ** (1.0 / 3.0)
    triangles = np.diag(w @ w @ w)
    degree = (adj > 0).sum(axis=1).astype(float)
    denom = degree * (degree - 1.0)
    out = np.zeros(n)
    ok = denom > 0
    out[ok] = triangles[ok] / denom[ok]
    return out


def _participation(adj: np.ndarray) -> np.ndarray:
    """1 - HHI of a node's normalised edge weights.

    0 = all connectivity concentrated in one counterpart, ->1 = spread
    evenly across the book. Distinguishes an asset tethered to a single
    name from one moving with the whole market.
    """
    n = adj.shape[0]
    strength = adj.sum(axis=1)
    out = np.zeros(n)
    ok = strength > 1e-18
    if not np.any(ok):
        return out
    shares = adj[ok] / strength[ok, None]
    out[ok] = 1.0 - (shares ** 2).sum(axis=1)
    return out


def _global_stats(adj: np.ndarray, pcorr: np.ndarray) -> np.ndarray:
    """Spectral / structural statistics of the whole network."""
    n = adj.shape[0]
    degree = adj.sum(axis=1)
    laplacian = np.diag(degree) - adj

    lap_eigs = np.linalg.eigvalsh(laplacian)
    lap_eigs = np.sort(np.real(lap_eigs))
    # Algebraic connectivity: 2nd-smallest Laplacian eigenvalue. The
    # smallest is always ~0 (constant vector); the second says how hard
    # the graph is to cut in two.
    fiedler = float(lap_eigs[1]) if n > 1 else 0.0

    adj_eigs = np.linalg.eigvalsh(adj) if np.any(adj) else np.zeros(n)
    spectral_radius = float(np.max(np.abs(adj_eigs))) if n else 0.0

    off = ~np.eye(n, dtype=bool)
    mean_abs_pcorr = float(np.abs(pcorr[off]).mean()) if n > 1 else 0.0
    density = float((adj[off] > 0).mean()) if n > 1 else 0.0

    # von Neumann graph entropy: spectrum of the trace-normalised
    # Laplacian read as a probability distribution.
    tr = float(np.trace(laplacian))
    if tr > 1e-18:
        p = np.clip(lap_eigs / tr, 0.0, None)
        p = p[p > 1e-12]
        vn_entropy = float(-(p * np.log(p)).sum()) if p.size else 0.0
    else:
        vn_entropy = 0.0

    return np.array(
        [fiedler, spectral_radius, mean_abs_pcorr, vn_entropy, density],
        dtype=np.float64,
    )


class FinancialConnectome:
    """Builds network features from a rolling window of returns.

    window: lookback used for each snapshot. 60 trading days by default -
    long enough that a 10x10 covariance is estimable, short enough that
    the network can actually move (a 252-day window produces a nearly
    static graph, which defeats the purpose of calling it *dynamic*
    connectivity).
    """

    def __init__(self, window: int = 60):
        if window < 4:
            raise ValueError("connectome window must be >= 4 bars")
        self.window = window

    def snapshot(self, returns: np.ndarray) -> ConnectomeFeatures:
        """Network features for ONE window of returns [T, n_assets]."""
        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim != 2:
            raise ValueError(f"expected [T, n_assets], got {returns.shape}")
        T, N = returns.shape
        if T < 2:
            raise ValueError(f"need >= 2 bars for a covariance, got {T}")

        sigma, _ = ledoit_wolf_shrinkage(returns)
        pcorr = partial_correlations(sigma)

        adj = np.abs(pcorr).copy()
        adj[adj < EDGE_THRESHOLD] = 0.0
        np.fill_diagonal(adj, 0.0)

        node = np.column_stack([
            adj.sum(axis=1),
            _eigen_centrality(adj),
            _onnela_clustering(adj),
            _participation(adj),
        ])
        return ConnectomeFeatures(node=node, glob=_global_stats(adj, pcorr), pcorr=pcorr)

    def rolling_features(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Per-bar connectome features over a whole return series.

        Returns (node_feats, global_feats) with shapes
          node_feats:   [T, n_assets, 4]
          global_feats: [T, 5]

        NO LOOK-AHEAD BY CONSTRUCTION: row t is computed from
        returns[t-window : t] - strictly excluding bar t itself. Rows
        before the first full window are zero-filled and should be
        dropped by the caller (`warmup` tells you how many).

        This is the single most important property in the module, so it
        is asserted by tests rather than left to a comment: shifting a
        future bar must not change any earlier row.
        """
        returns = np.asarray(returns, dtype=np.float64)
        T, N = returns.shape
        node_out = np.zeros((T, N, len(NODE_FEATURE_NAMES)))
        glob_out = np.zeros((T, len(GLOBAL_FEATURE_NAMES)))

        for t in range(self.window, T):
            snap = self.snapshot(returns[t - self.window : t])
            node_out[t] = snap.node
            glob_out[t] = snap.glob
        return node_out, glob_out

    @property
    def warmup(self) -> int:
        """Rows at the start of rolling_features() output that are unusable."""
        return self.window
