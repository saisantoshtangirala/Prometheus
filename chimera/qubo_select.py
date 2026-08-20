"""
Component 2/6 - Quantum-inspired feature selection via QUBO + Simulated
Bifurcation.

Why this is not just a rebranded greedy filter: feature selection under a
relevance/redundancy trade-off is a genuine combinatorial problem -
choosing k of M features has C(M,k) solutions, and greedy mRMR is a
heuristic that commits to early picks it cannot revisit. Written as a
QUBO:

    minimise  x^T Q x,   x in {0,1}^M

    Q_ii = -alpha * relevance_i          (reward informative features)
    Q_ij = +beta  * redundancy_ij        (punish saying the same thing twice)
    + lambda * (sum_i x_i - k)^2         (cardinality, folded into Q)

...it becomes exactly the problem class that quantum and quantum-inspired
annealers target. This module solves it with **ballistic Simulated
Bifurcation** (Goto et al., Sci. Adv. 2019 / 2021), a real
quantum-inspired algorithm derived from the adiabatic evolution of a
network of Kerr-nonlinear parametric oscillators - not a metaphor. It is
also fully vectorised and runs many independent replicas at once, which
is what makes it practical inside a 125-window loop.

Cardinality expansion (why Q picks up those extra terms), using
x_i^2 = x_i for binary x:

    (sum x - k)^2 = (sum x)^2 - 2k sum x + k^2
                  = sum_i x_i + sum_{i!=j} x_i x_j - 2k sum_i x_i + k^2
    => Q_ii += lambda (1 - 2k)
       Q_ij += lambda            (i != j)

Relevance and redundancy are both **distance correlation**, not Pearson.
That matters: Pearson would score a feature with a clean U-shaped
relationship to the target as irrelevant. Distance correlation is zero
if and only if the variables are independent, so it catches nonlinear
dependence - which is the entire reason for using a nonlinear model
downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


def distance_correlation(x: np.ndarray, y: np.ndarray, max_n: int = 400,
                         seed: int = 0) -> float:
    """Distance correlation between two 1-D samples, in [0, 1].

    dCor = 0 iff x and y are independent - unlike Pearson, which is 0 for
    plenty of strong nonlinear relationships. This is what makes the
    relevance term worth computing.

    O(n^2) in memory, so long series are subsampled to `max_n` points
    (deterministically, via `seed`) - dCor is a population quantity and a
    few hundred draws estimate it well enough to rank features.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    if n < 4:
        return 0.0
    if n > max_n:
        idx = np.random.default_rng(seed).choice(n, size=max_n, replace=False)
        idx.sort()
        x, y = x[idx], y[idx]
        n = max_n

    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    # Double centring is what turns raw distance matrices into something
    # whose covariance vanishes exactly under independence.
    A = a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()
    B = b - b.mean(0, keepdims=True) - b.mean(1, keepdims=True) + b.mean()

    dcov2 = float((A * B).mean())
    dvarx = float((A * A).mean())
    dvary = float((B * B).mean())
    denom = np.sqrt(dvarx * dvary)
    if denom <= 1e-18 or dcov2 <= 0:
        return 0.0
    return float(np.clip(np.sqrt(dcov2 / denom), 0.0, 1.0))


@dataclass
class SBConfig:
    """Ballistic Simulated Bifurcation hyperparameters.

    Defaults follow the bSB paper's stable regime. `n_replicas` is the
    real lever: SB is cheap per replica and the replicas are independent,
    so running 32 in parallel and keeping the best is nearly free and
    substantially improves the solution.
    """

    n_steps: int = 400
    dt: float = 0.5
    a0: float = 1.0          # final pump amplitude
    c0: Optional[float] = None   # coupling scale; auto-set from ||Q|| if None
    n_replicas: int = 32
    seed: int = 0


class SimulatedBifurcation:
    """Ballistic SB solver for QUBO: minimise x^T Q x over x in {0,1}^M.

    The oscillator picture: each variable is a particle with position
    x_i and momentum y_i in a potential whose shape is ramped over time
    (the "pump" a(t): 0 -> a0). Early on the potential is a single well
    and everything sits near 0; as the pump rises the well bifurcates
    into a double well and each particle is driven toward +1 or -1, with
    the coupling term biasing which. That bifurcation is the adiabatic
    evolution being emulated. "Ballistic" = perfectly inelastic walls at
    |x|=1, which prevents the runaway that plain SB suffers.
    """

    def __init__(self, config: Optional[SBConfig] = None):
        self.cfg = config or SBConfig()

    def solve(self, Q: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (best_x_binary [M], best_energy)."""
        Q = np.asarray(Q, dtype=np.float64)
        M = Q.shape[0]
        if Q.shape != (M, M):
            raise ValueError(f"Q must be square, got {Q.shape}")
        Qs = 0.5 * (Q + Q.T)   # SB assumes a symmetric coupling matrix

        cfg = self.cfg
        dev = torch.device("cpu")
        Qt = torch.tensor(Qs, dtype=torch.float32, device=dev)

        # QUBO -> Ising. With x = (s+1)/2:
        #   x^T Q x = (1/4) s^T Q s + (1/2) (Q1)^T s + const
        # so the energy gradient wrt the continuous spin proxy is
        #   dE/ds = (1/2) Q s + (1/2) Q1
        h0 = Qt.sum(dim=1)

        # Coupling scale: bSB is only stable when c0 is O(1/sqrt(M)/||Q||).
        # Auto-scaling by the spectral norm makes the solver robust to the
        # absolute magnitude of relevance/redundancy, which otherwise
        # silently determines whether it converges at all.
        if cfg.c0 is not None:
            c0 = float(cfg.c0)
        else:
            spectral = float(torch.linalg.matrix_norm(Qt, ord=2).item())
            c0 = 0.5 / (np.sqrt(max(M, 1)) * max(spectral, 1e-6))

        g = torch.Generator(device=dev).manual_seed(cfg.seed)
        R = cfg.n_replicas
        # Small random initial conditions: every replica starts near the
        # unbifurcated origin and is pushed apart by the pump.
        x = 0.1 * (torch.rand((R, M), generator=g, device=dev) * 2 - 1)
        y = 0.1 * (torch.rand((R, M), generator=g, device=dev) * 2 - 1)

        for step in range(cfg.n_steps):
            a_t = cfg.a0 * (step / max(cfg.n_steps - 1, 1))   # pump ramp
            grad = 0.5 * (x @ Qt + h0.unsqueeze(0))
            y = y + cfg.dt * (-(cfg.a0 - a_t) * x - c0 * grad)
            x = x + cfg.dt * cfg.a0 * y
            # Inelastic walls: clamp position to [-1,1] and kill the
            # momentum of anything that hit a wall. This is the
            # "ballistic" correction and it is load-bearing - without it
            # x diverges and the run is meaningless.
            hit = x.abs() > 1.0
            if bool(hit.any()):
                x = torch.clamp(x, -1.0, 1.0)
                y = torch.where(hit, torch.zeros_like(y), y)

        spins = torch.sign(x)
        spins[spins == 0] = 1.0
        xb = ((spins + 1) / 2)                       # [R, M] in {0,1}
        energies = torch.einsum("rm,mn,rn->r", xb, Qt, xb)
        best = int(torch.argmin(energies).item())
        return xb[best].numpy().astype(np.int64), float(energies[best].item())


class QUBOFeatureSelector:
    """Select k features by minimising a relevance/redundancy QUBO.

    alpha: weight on relevance (feature vs target dependence)
    beta:  weight on redundancy (feature vs feature dependence)
    lam:   cardinality penalty weight. Auto-scaled to the magnitude of
           the relevance/redundancy terms if left None, because a fixed
           lambda either swamps the objective or fails to bind depending
           on how strong the dCor values happen to be in a given window.
    """

    def __init__(
        self,
        k: int,
        alpha: float = 1.0,
        beta: float = 0.5,
        lam: Optional[float] = None,
        sb_config: Optional[SBConfig] = None,
        dcor_max_n: int = 400,
    ):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.sb = SimulatedBifurcation(sb_config)
        self.dcor_max_n = dcor_max_n

        self.relevance_: Optional[np.ndarray] = None
        self.redundancy_: Optional[np.ndarray] = None
        self.selected_: Optional[np.ndarray] = None
        self.energy_: Optional[float] = None

    def build_qubo(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Assemble Q from distance correlations. X: [T, M], y: [T]."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        T, M = X.shape
        if y.size != T:
            raise ValueError(f"X has {T} rows but y has {y.size}")
        k = min(self.k, M)

        relevance = np.array(
            [distance_correlation(X[:, i], y, self.dcor_max_n, seed=i) for i in range(M)]
        )
        redundancy = np.zeros((M, M))
        for i in range(M):
            for j in range(i + 1, M):
                d = distance_correlation(X[:, i], X[:, j], self.dcor_max_n, seed=1000 + i * M + j)
                redundancy[i, j] = redundancy[j, i] = d

        self.relevance_ = relevance
        self.redundancy_ = redundancy

        Q = np.zeros((M, M))
        np.fill_diagonal(Q, -self.alpha * relevance)
        Q += self.beta * redundancy
        np.fill_diagonal(Q, np.diag(Q) - self.beta * np.diag(redundancy))  # keep diag clean

        # Auto-scale the cardinality penalty to the objective's own scale,
        # so the constraint binds without dominating.
        if self.lam is None:
            scale = max(self.alpha * float(np.abs(relevance).mean()),
                        self.beta * float(np.abs(redundancy).mean()), 1e-6)
            lam = 4.0 * scale
        else:
            lam = float(self.lam)

        Q[np.diag_indices(M)] += lam * (1.0 - 2.0 * k)
        off = ~np.eye(M, dtype=bool)
        Q[off] += lam
        return Q

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QUBOFeatureSelector":
        """Solve for the best k-subset. Sets .selected_ (sorted indices)."""
        X = np.asarray(X, dtype=np.float64)
        M = X.shape[1]
        k = min(self.k, M)

        Q = self.build_qubo(X, y)
        bits, energy = self.sb.solve(Q)
        self.energy_ = energy

        chosen = np.flatnonzero(bits > 0)
        # Repair to EXACTLY k. The cardinality penalty is a soft
        # constraint, so SB can land at k+-1; downstream shapes are fixed,
        # so this must be exact. Repair by relevance, breaking ties
        # toward lower redundancy against what is already in.
        if chosen.size > k:
            order = np.argsort(-self.relevance_[chosen])
            chosen = chosen[order[:k]]
        elif chosen.size < k:
            missing = k - chosen.size
            rest = np.setdiff1d(np.arange(M), chosen)
            if rest.size:
                if chosen.size:
                    penalty = self.redundancy_[np.ix_(rest, chosen)].mean(axis=1)
                else:
                    penalty = np.zeros(rest.size)
                score = self.relevance_[rest] - self.beta * penalty
                chosen = np.concatenate([chosen, rest[np.argsort(-score)[:missing]]])

        self.selected_ = np.sort(chosen.astype(np.int64))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_ is None:
            raise RuntimeError("call fit() before transform()")
        return np.asarray(X)[:, self.selected_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
