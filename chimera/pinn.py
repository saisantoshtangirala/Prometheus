"""
Component 5/6 - Physics-informed constraints: no-arbitrage as a
differentiable penalty.

Two constraints, because they bind on different things.

(A) CROSS-SECTIONAL NO-ARBITRAGE - the one that actually matters here.

    A portfolio with (near) zero variance must have (near) zero expected
    return, or it is a money pump. Formally: if w^T Sigma w ~ 0 then
    w^T mu ~ 0. Any predicted return vector mu-hat that violates this is
    asserting a riskless profit, which is the cleanest possible statement
    that the model has fit noise.

    Implementation: take the eigenvectors of the train-window covariance
    with the SMALLEST eigenvalues - these span the near-null space, the
    directions in which the market barely moves - and penalise the
    squared predicted return along each, weighted by how riskless it is:

        L = sum_j  (1 / (lambda_j + eps)) * (v_j . mu_hat)^2

    The 1/lambda weighting is what makes this a no-arbitrage penalty
    rather than a generic shrinkage: it is nearly free to predict return
    along a high-variance direction (that is just a risk premium, which
    is allowed), and very expensive along a zero-variance one.

    This is a genuine economic constraint that a purely statistical
    regulariser (L2, dropout) does not express.

(B) BLACK-SCHOLES PDE RESIDUAL - the textbook PINN term.

        dV/dt + (1/2) sigma^2 S^2 d2V/dS2 + r S dV/dS - r V = 0

    Computed by real autograd differentiation of an auxiliary value head
    V(S, t) w.r.t. its inputs (second derivative included), not a finite
    difference. Its role is auxiliary-task regularisation: forcing a
    shared encoder to also support a no-arbitrage-consistent value
    surface constrains what representations it is allowed to learn.

    Stated plainly: (B) is the weaker of the two for a directional
    cross-sectional predictor - it constrains an auxiliary head rather
    than the trading signal itself. It is included because it is what
    "physics-informed, no-arbitrage PDE residual" actually specifies, and
    it is implemented properly rather than gestured at.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def black_scholes_residual(
    value_fn: nn.Module,
    S: torch.Tensor,
    t: torch.Tensor,
    r: float = 0.065,
    sigma: float = 0.20,
) -> torch.Tensor:
    """Black-Scholes PDE residual via autograd. Returns [N] residuals.

    value_fn: maps [N, 2] (S, t) -> [N, 1]
    S, t:     [N] collocation points. Do NOT pre-set requires_grad; this
              function handles it.
    r:        risk-free rate. Default 6.5% - the Indian 10y, since this
              system trades NSE. Using a US rate here would be a real
              (if quiet) modelling error.
    sigma:    volatility for the diffusion term.

    A zero residual means the value surface admits no arbitrage under
    the model's own dynamics.
    """
    S = S.reshape(-1).clone().requires_grad_(True)
    t = t.reshape(-1).clone().requires_grad_(True)

    V = value_fn(torch.stack([S, t], dim=1)).squeeze(-1)

    # create_graph=True on the first derivatives: required both to take
    # the SECOND derivative and to let this residual be backpropagated
    # into the network's weights as a loss term.
    dV_dS, dV_dt = torch.autograd.grad(
        V, (S, t), grad_outputs=torch.ones_like(V), create_graph=True,
    )
    d2V_dS2 = torch.autograd.grad(
        dV_dS, S, grad_outputs=torch.ones_like(dV_dS), create_graph=True,
    )[0]

    return dV_dt + 0.5 * (sigma ** 2) * S.pow(2) * d2V_dS2 + r * S * dV_dS - r * V


class ValueSurface(nn.Module):
    """Small MLP V(S, t) used as the PINN auxiliary head.

    tanh activations, deliberately: the PDE needs a well-behaved SECOND
    derivative, and ReLU's is zero almost everywhere (with a delta at the
    kink), which makes the residual meaningless. Any PINN built on ReLU
    is silently broken.
    """

    def __init__(self, hidden: int = 32, n_layers: int = 3, context_dim: int = 0):
        super().__init__()
        in_dim = 2 + context_dim
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.context_dim = context_dim
        self._context: Optional[torch.Tensor] = None

    def set_context(self, context: Optional[torch.Tensor]) -> None:
        """Condition the surface on an encoder latent (optional)."""
        self._context = context

    def forward(self, st: torch.Tensor) -> torch.Tensor:
        if self.context_dim and self._context is not None:
            ctx = self._context
            if ctx.dim() == 1:
                ctx = ctx.unsqueeze(0).expand(st.shape[0], -1)
            st = torch.cat([st, ctx], dim=1)
        return self.net(st)


class NoArbitragePenalty(nn.Module):
    """Cross-sectional no-arbitrage penalty on predicted returns.

    Fit the covariance ONCE per walk-forward window on train data only
    (`fit`), then penalise predictions (`forward`). Keeping fit and apply
    separate is what makes this look-ahead-free: the null-space basis
    comes strictly from the train window.
    """

    def __init__(self, n_null_directions: int = 3, eps: float = 1e-6,
                 max_weight: float = 1e4):
        super().__init__()
        self.n_null = n_null_directions
        self.eps = eps
        self.max_weight = max_weight
        self.register_buffer("null_basis", torch.zeros(0), persistent=False)
        self.register_buffer("null_weights", torch.zeros(0), persistent=False)
        self._fitted = False

    def fit(self, train_returns: torch.Tensor) -> "NoArbitragePenalty":
        """train_returns: [T, n_assets] - TRAIN WINDOW ONLY."""
        if not torch.is_tensor(train_returns):
            train_returns = torch.tensor(train_returns, dtype=torch.float32)
        X = train_returns.to(torch.float32)
        X = X - X.mean(dim=0, keepdim=True)
        T, N = X.shape
        cov = (X.T @ X) / max(T - 1, 1)

        evals, evecs = torch.linalg.eigh(cov)          # ascending
        k = min(self.n_null, N)
        lam = evals[:k].clamp_min(0.0)
        basis = evecs[:, :k]                            # [N, k] column vectors

        # Weight each direction by how riskless it is. Capped so a
        # numerically-zero eigenvalue cannot produce an infinite penalty
        # that swamps every other term in the loss.
        weights = (1.0 / (lam + self.eps)).clamp(max=self.max_weight)
        # Normalise so the penalty's scale does not depend on how
        # degenerate this particular window happened to be.
        weights = weights / weights.sum().clamp_min(1e-12)

        self.null_basis = basis
        self.null_weights = weights
        self._fitted = True
        return self

    def forward(self, predicted_returns: torch.Tensor) -> torch.Tensor:
        """predicted_returns: [B, n_assets] or [n_assets] -> scalar penalty."""
        if not self._fitted:
            raise RuntimeError("call fit() on the train window before forward()")
        mu = predicted_returns
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
        # Projection of each prediction onto each near-riskless direction.
        proj = mu @ self.null_basis                     # [B, k]
        return (self.null_weights.unsqueeze(0) * proj.pow(2)).sum(dim=1).mean()

    def arbitrage_score(self, predicted_returns: torch.Tensor) -> float:
        """Diagnostic: unweighted RMS return along the null space.

        Reported in units of return, so it is directly interpretable -
        'this model claims X% return on a portfolio that cannot move'.
        """
        with torch.no_grad():
            mu = predicted_returns
            if mu.dim() == 1:
                mu = mu.unsqueeze(0)
            proj = mu @ self.null_basis
            return float(proj.pow(2).mean().sqrt().item())
