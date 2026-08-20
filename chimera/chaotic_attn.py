"""
Component 3/6 - Chaotic Oscillator-Enhanced Attention.

A Lorenz oscillator, SEEDED FROM THE MARKET STATE, drives the attention
temperature of every head.

The argument for why this is more than decoration, stated so it can be
attacked: a chaotic system has sensitive dependence on initial
conditions. Seed the oscillator from a learned projection of the input
window and two nearby market states produce trajectories that diverge
exponentially - i.e. the oscillator is a *nonlinear basis expansion*
that separates market states a linear encoder would blur together. This
is the reservoir-computing argument (chaotic reservoirs sit near the
edge of chaos precisely because that maximises separability), applied
inside attention rather than as a separate readout layer.

Lorenz system:
    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z
Chaotic at the classical sigma=10, rho=28, beta=8/3.

Two engineering problems have to be solved for this to be trainable at
all, and both are handled explicitly below:

1. EXPLODING GRADIENTS. Backpropagating through a chaotic rollout is the
   textbook way to get infinite gradients. Controlled by bounding the
   integration horizon: the largest Lyapunov exponent of Lorenz is ~0.9,
   so with dt=0.01 over T=64 steps total integration time is 0.64 and
   worst-case gradient growth is e^{0.9*0.64} ~ 1.8x. That is a
   completely ordinary amount of gradient scaling. Chaos is only
   dangerous here if you integrate for a long time, so we do not.

2. STAYING IN THE CHAOTIC REGIME. sigma/rho/beta are learnable (the
   network can tune its own chaos), but unconstrained they will drift to
   a fixed point and the whole component silently degrades into a
   constant. They are therefore parameterised through softplus with
   offsets that keep them near the chaotic regime by construction.

The readout modulates attention TEMPERATURE rather than adding a bias:
temperature controls how sharply each head attends, so the oscillator is
gating "how decisively should this head commit right now" - which is a
meaningful thing for a market model to modulate, unlike an arbitrary
additive logit.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical chaotic parameters, used as the centre of the learnable range.
LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0

# Integration step. At dt=0.02 a T=64 rollout spans 1.28 time units, so
# worst-case gradient growth is e^{0.9*1.28} ~ 3.2x - still an entirely
# ordinary amount of scaling, while giving the trajectory room to
# actually traverse the attractor.
LORENZ_DT = 0.02

# Burn-in steps discarded before the recorded trajectory begins.
#
# This is load-bearing and was found empirically, not assumed: without
# it, two nearby seeds CONVERGE rather than diverge. Lorenz has one
# positive Lyapunov exponent (~0.9) but strongly negative transverse
# ones (~-14.6), so an initial condition off the attractor collapses
# onto it much faster than it separates along it. Over a short window
# that contraction dominates and the "chaotic" reservoir silently
# behaves like a fixed encoding.
#
# Burn-in lands both trajectories ON the attractor first, where the
# positive exponent governs. It is kept short (1 time unit) on purpose:
# burn in too long and nearby seeds decorrelate completely, which
# destroys the data-dependence that makes this a market-state
# expansion rather than noise.
LORENZ_BURN_IN = 50

# Bounds for the learnable chaos parameters. Ranges are centred on the
# classical values and kept inside the region where the Lorenz system is
# genuinely chaotic - a plain softplus+floor can drift to sigma~4, where
# the attractor is not reliably strange and the component degenerates.
SIGMA_RANGE = (8.0, 13.0)
RHO_RANGE = (24.0, 32.0)
BETA_RANGE = (2.0, 3.4)

# tanh-bounded temperature range. Attention temperature outside roughly
# [0.5, 2.0] either collapses to argmax or flattens to uniform; both
# destroy the signal, so the modulation is clamped into a useful band.
TEMP_MIN, TEMP_MAX = 0.5, 2.0


class LorenzReservoir(nn.Module):
    """Integrates a Lorenz system seeded from a data-dependent state.

    seed_dim -> (x0, y0, z0) -> T-step trajectory -> [B, T, 3] normalised.

    The seed projection is what makes this data-dependent rather than a
    fixed sinusoid-substitute: different market windows land on
    different parts of the attractor, and sensitive dependence pulls them
    apart as the rollout proceeds.
    """

    def __init__(self, seed_dim: int, dt: float = LORENZ_DT,
                 learnable_params: bool = True, burn_in: int = LORENZ_BURN_IN):
        super().__init__()
        self.dt = dt
        self.burn_in = burn_in
        self.seed_proj = nn.Linear(seed_dim, 3)

        # sigmoid-bounded into an explicitly chaotic range. Raw params are
        # initialised so sigmoid(raw) lands on the classical values, and
        # NO value of raw can leave the range - so "learnable chaos"
        # cannot quietly become "learned fixed point".
        def inv_sigmoid_for(value: float, lo: float, hi: float) -> float:
            frac = min(max((value - lo) / (hi - lo), 1e-4), 1 - 1e-4)
            return math.log(frac / (1 - frac))

        self._sigma_raw = nn.Parameter(
            torch.tensor(inv_sigmoid_for(LORENZ_SIGMA, *SIGMA_RANGE)),
            requires_grad=learnable_params)
        self._rho_raw = nn.Parameter(
            torch.tensor(inv_sigmoid_for(LORENZ_RHO, *RHO_RANGE)),
            requires_grad=learnable_params)
        self._beta_raw = nn.Parameter(
            torch.tensor(inv_sigmoid_for(LORENZ_BETA, *BETA_RANGE)),
            requires_grad=learnable_params)

    @staticmethod
    def _bounded(raw: torch.Tensor, rng: Tuple[float, float]) -> torch.Tensor:
        lo, hi = rng
        return lo + (hi - lo) * torch.sigmoid(raw)

    @property
    def sigma(self) -> torch.Tensor:
        return self._bounded(self._sigma_raw, SIGMA_RANGE)

    @property
    def rho(self) -> torch.Tensor:
        return self._bounded(self._rho_raw, RHO_RANGE)

    @property
    def beta(self) -> torch.Tensor:
        return self._bounded(self._beta_raw, BETA_RANGE)

    def _derivs(self, state: torch.Tensor) -> torch.Tensor:
        x, y, z = state[..., 0], state[..., 1], state[..., 2]
        return torch.stack([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        ], dim=-1)

    def raw_trajectory(self, seed_input: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Un-standardised trajectory [B, n_steps, 3], in attractor units.

        Exposed because per-sample standardisation (what `forward`
        returns) is an affine map that removes overall scale - and
        therefore MASKS the very separation that makes this reservoir
        useful. Any honest test of sensitive dependence has to measure
        the raw state, so the raw state has to be reachable.
        """
        return self._integrate(seed_input, n_steps)

    def lyapunov_estimate(self, seed_input: torch.Tensor, n_steps: int,
                          epsilon: float = 1e-4) -> float:
        """Finite-time largest Lyapunov exponent, by trajectory separation.

        Perturbs the seed by `epsilon`, integrates both, and fits the
        slope of log||delta|| against time. Positive => chaotic. This is
        the direct empirical check that the component is doing what its
        name claims, rather than a proxy.
        """
        with torch.no_grad():
            base = self._integrate(seed_input, n_steps)
            pert = self._integrate(seed_input + epsilon, n_steps)
            sep = (base - pert).norm(dim=-1).mean(dim=0)         # [n_steps]
            sep = torch.clamp(sep, min=1e-12)
            log_sep = torch.log(sep)
            t = torch.arange(n_steps, dtype=torch.float32) * self.dt
            tc, lc = t - t.mean(), log_sep - log_sep.mean()
            denom = float((tc * tc).sum())
            return float((tc * lc).sum() / denom) if denom > 0 else 0.0

    def _integrate(self, seed_input: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Burn-in then record. Returns the raw [B, n_steps, 3] state."""
        return self.forward(seed_input, n_steps, _raw=True)

    def forward(self, seed_input: torch.Tensor, n_steps: int,
                _raw: bool = False) -> torch.Tensor:
        """[B, seed_dim] -> [B, n_steps, 3], standardised per sample.

        RK4 rather than Euler: Euler at this dt accumulates enough error
        to drift off the attractor over 64 steps, which would make the
        "chaotic" trajectory an artefact of integration error rather
        than of the dynamics.
        """
        # tanh-bound the seed into the attractor's actual scale (~|x|<20).
        # An unbounded projection can start the state far outside the
        # basin, where the dynamics diverge instead of orbiting.
        state = torch.tanh(self.seed_proj(seed_input)) * 15.0

        dt = self.dt

        def rk4(s: torch.Tensor) -> torch.Tensor:
            k1 = self._derivs(s)
            k2 = self._derivs(s + 0.5 * dt * k1)
            k3 = self._derivs(s + 0.5 * dt * k2)
            k4 = self._derivs(s + dt * k3)
            s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            # Hard safety clamp. Should never bind for in-basin seeds; it
            # exists so a pathological input can never produce NaNs that
            # poison a 125-window backtest hours in.
            return torch.clamp(s, -60.0, 60.0)

        # Burn-in: settle onto the attractor before recording. Without
        # this the strongly-negative transverse Lyapunov exponents make
        # nearby seeds CONVERGE, and the reservoir stops being chaotic in
        # any useful sense (see LORENZ_BURN_IN).
        for _ in range(self.burn_in):
            state = rk4(state)

        traj = []
        for _ in range(n_steps):
            state = rk4(state)
            traj.append(state)

        out = torch.stack(traj, dim=1)                       # [B, T, 3]
        if _raw:
            return out
        # Standardise per sample: the three Lorenz coordinates live on
        # very different scales (z is ~0..50, x/y are ~+-20) and the
        # readout should see shape, not units.
        mean = out.mean(dim=1, keepdim=True)
        std = out.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (out - mean) / std


class ChaoticMultiHeadAttention(nn.Module):
    """Multi-head self-attention whose per-head temperature is driven by
    the Lorenz reservoir, with a causal mask.

    scores = (Q K^T / sqrt(d_head)) * temperature[b, h, t_query]

    Temperature is indexed by QUERY position, so the oscillator sets how
    sharply each position attends to its own history - a per-timestep,
    per-head decisiveness gate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Reservoir -> per-head log-temperature.
        self.temp_head = nn.Linear(3, n_heads)

    def forward(self, x: torch.Tensor, chaos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, d_model], chaos: [B, T, 3] -> (out, attn_weights)."""
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)                       # each [B, T, H, dh]
        q = q.transpose(1, 2)                             # [B, H, T, dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # [B,H,T,T]

        # Chaotic temperature, bounded into [TEMP_MIN, TEMP_MAX].
        raw = self.temp_head(chaos)                       # [B, T, H]
        span = TEMP_MAX - TEMP_MIN
        temp = TEMP_MIN + 0.5 * span * (1.0 + torch.tanh(raw))
        temp = temp.permute(0, 2, 1).unsqueeze(-1)        # [B, H, T, 1]
        scores = scores * temp

        # Causal mask - a market model may never attend forward in time.
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out), attn


class ChaoticTransformerLayer(nn.Module):
    """Pre-norm transformer block using chaotic attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = ChaoticMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, chaos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.attn(self.norm1(x), chaos)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, attn_w


class ChaoticAttentionEncoder(nn.Module):
    """Sequence encoder: [B, T, n_features] -> [B, d_model] pooled latent.

    The reservoir is seeded from a mean-pooled summary of the INPUT
    window, so the chaotic trajectory is a function of the market state
    being encoded - which is the entire point.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        learnable_chaos: bool = True,
    ):
        super().__init__()
        d_ff = d_ff or 2 * d_model
        self.d_model = d_model
        self.input_proj = nn.Linear(n_features, d_model)
        self.reservoir = LorenzReservoir(seed_dim=n_features, learnable_params=learnable_chaos)
        self.layers = nn.ModuleList([
            ChaoticTransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_chaos: bool = False):
        """x: [B, T, n_features] -> [B, d_model] (plus chaos if requested)."""
        B, T, _ = x.shape
        chaos = self.reservoir(x.mean(dim=1), n_steps=T)      # [B, T, 3]
        h = self.input_proj(x)
        for layer in self.layers:
            h, _ = layer(h, chaos)
        # Last-position pooling, not mean: with a causal mask only the
        # final position has seen the whole window, and this is a
        # forecasting model - the most recent state is what matters.
        pooled = self.norm(h[:, -1, :])
        return (pooled, chaos) if return_chaos else pooled
