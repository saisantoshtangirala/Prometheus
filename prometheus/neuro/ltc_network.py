"""
Liquid Time-Constant (LTC) Network for financial time-series.

LTC networks have time-constants that vary dynamically with input, making them
ideal for financial data where volatility regimes shift abruptly.  Unlike fixed
RNNs, LTCs solve an ODE at each step, allowing continuous-time reasoning about
market state.

Reference: Hasani et al. (2021) "Liquid Time-constant Networks"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCCell(nn.Module):
    """
    A single Liquid Time-Constant cell.

    State update: dx/dt = -x/τ(x,u) + f(x, u)
    where τ(x, u) is an input-dependent time constant.

    Discretized via explicit Euler with adaptive step size.
    """

    def __init__(self, input_size: int, hidden_size: int, n_steps: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_steps = n_steps  # ODE integration steps per timestep

        # Synaptic weights
        self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)

        # Time-constant network: τ is a learned function of (x, u)
        self.tau_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),  # τ ∈ (0, 1)
        )

        # Modulation gates (inspired by NCP wiring)
        self.A = nn.Parameter(torch.ones(hidden_size))    # synaptic amplitude
        self.erev = nn.Parameter(torch.zeros(hidden_size))  # reversal potentials

    def forward(
        self,
        x: torch.Tensor,        # [batch, input_size]
        h: torch.Tensor,        # [batch, hidden_size]
        delta_t: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """Run one LTC step, integrating the ODE."""
        meta = {}
        dt = delta_t / self.n_steps

        for _ in range(self.n_steps):
            # Time constants — larger τ = slower dynamics (regime persistence)
            tau_input = torch.cat([x, h], dim=-1)
            tau = self.tau_net(tau_input) + 0.01  # avoid division by zero

            # Synaptic input
            f_xu = torch.tanh(self.W_in(x) + self.W_rec(h))

            # ODE step: dh/dt = (-h + A * f_xu + erev) / tau
            dh = (-h + self.A * f_xu + self.erev) / tau
            h = h + dt * dh
            h = torch.clamp(h, -10.0, 10.0)

        meta["tau_mean"] = tau.mean().item()  # track volatility regime
        return h, meta


class LiquidTimeConstantNetwork(nn.Module):
    """
    Multi-layer LTC network for sequential financial data.

    Key advantage over LSTM: the time constant τ naturally encodes market
    regime — low-τ (fast dynamics) during crises, high-τ during trending
    conditions. No explicit regime-switching required.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes

        sizes = [input_size] + hidden_sizes
        self.cells = nn.ModuleList([
            LTCCell(sizes[i], sizes[i + 1])
            for i in range(len(hidden_sizes))
        ])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in hidden_sizes])
        self.output_proj = nn.Linear(hidden_sizes[-1], output_size)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(h) for h in hidden_sizes])

    def forward(
        self,
        x: torch.Tensor,              # [batch, seq_len, input_size]
        hidden: Optional[List[torch.Tensor]] = None,
        delta_t: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
          outputs:   [batch, seq_len, output_size]
          last_hidden: list of [batch, hidden_size] per layer
          meta: dict with tau traces and regime indicators
        """
        B, T, _ = x.shape
        if hidden is None:
            hidden = [torch.zeros(B, h, device=x.device) for h in self.hidden_sizes]

        all_outputs = []
        tau_traces: List[List[float]] = [[] for _ in self.cells]

        for t in range(T):
            inp = x[:, t, :]
            for layer_idx, (cell, dropout, norm) in enumerate(
                zip(self.cells, self.dropouts, self.layer_norms)
            ):
                h, meta = cell(inp, hidden[layer_idx], delta_t)
                h = norm(h)
                hidden[layer_idx] = dropout(h)
                inp = hidden[layer_idx]
                tau_traces[layer_idx].append(meta["tau_mean"])

            out = self.output_proj(hidden[-1])
            all_outputs.append(out)

        outputs = torch.stack(all_outputs, dim=1)  # [B, T, output_size]

        # Compute regime indicator from tau of last layer
        tau_arr = torch.tensor(tau_traces[-1])
        regime_label = "HIGH_VOLATILITY" if tau_arr.mean() < 0.3 else (
            "TRENDING" if tau_arr.mean() > 0.7 else "MEAN_REVERTING"
        )

        meta_out = {
            "tau_traces": tau_traces,
            "regime": regime_label,
            "mean_tau": float(tau_arr.mean()),
        }
        return outputs, hidden, meta_out

    def get_regime(self, x: torch.Tensor) -> str:
        """Quick regime detection from input sequence."""
        with torch.no_grad():
            _, _, meta = self.forward(x)
        return meta["regime"]
