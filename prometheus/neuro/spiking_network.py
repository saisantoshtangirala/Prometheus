"""
Spiking Neural Network (SNN) encoder for market microstructure.

SNNs communicate via discrete spike events rather than continuous activations,
making them uniquely suited for encoding order-book events, trade prints, and
other event-driven market data.  We use Leaky Integrate-and-Fire (LIF) neurons
with surrogate gradients for backpropagation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateSpike(torch.autograd.Function):
    """
    Heaviside spike with surrogate gradient (fast sigmoid) for backprop.
    Forward: H(v - v_thresh)
    Backward: σ'(v - v_thresh) · 4  (arctan surrogate)
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input,) = ctx.saved_tensors
        threshold = ctx.threshold
        surrogate = 1.0 / (1.0 + (torch.pi * (input - threshold)).pow(2))
        return grad_output * surrogate, None


spike_fn = SurrogateSpike.apply


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer.

    Membrane dynamics: τ * dV/dt = -V + I(t)
    Spike: z(t) = H(V(t) - V_thresh)
    Reset: V(t) ← V(t) - V_thresh * z(t)  [soft reset]
    """

    def __init__(
        self,
        n_neurons: int,
        tau_mem: float = 20.0,   # membrane time constant (ms)
        v_thresh: float = 1.0,
        v_rest: float = 0.0,
        dt: float = 1.0,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.v_thresh = v_thresh
        self.v_rest = v_rest
        # Leak factor: α = exp(-dt / τ)
        self.alpha = nn.Parameter(
            torch.tensor(torch.exp(torch.tensor(-dt / tau_mem)).item()),
            requires_grad=True,
        )

    def forward(
        self,
        current: torch.Tensor,   # [batch, n_neurons] synaptic current
        v_mem: torch.Tensor,     # [batch, n_neurons] membrane voltage
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One LIF timestep. Returns (spikes, new_v_mem)."""
        alpha = torch.sigmoid(self.alpha)  # keep in (0, 1)
        v_new = alpha * v_mem + (1 - alpha) * current
        spikes = spike_fn(v_new, self.v_thresh)
        v_reset = v_new - self.v_thresh * spikes  # soft reset
        return spikes, v_reset


class SpikingLayer(nn.Module):
    """Fully connected + LIF neuron layer."""

    def __init__(self, in_size: int, out_size: int, tau_mem: float = 20.0):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.lif = LIFNeuron(out_size, tau_mem)

    def forward(
        self,
        x: torch.Tensor,
        v_mem: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        current = self.fc(x)
        spikes, v_new = self.lif(current, v_mem)
        return spikes, v_new


class SpikingMarketEncoder(nn.Module):
    """
    Multi-layer SNN for encoding market microstructure events.

    Input: tick-by-tick features (bid, ask, size, direction, time_delta, ...)
    Output: spike-rate encoded market state vector + population activity metrics.

    The population activity (firing rate, synchrony, burst index) provides
    implicit signals about market stress — high synchrony = herding behavior.
    """

    def __init__(
        self,
        input_size: int,
        layer_sizes: List[int],
        output_size: int,
        tau_mem: float = 20.0,
        n_timesteps: int = 100,
    ):
        super().__init__()
        self.n_timesteps = n_timesteps

        sizes = [input_size] + layer_sizes
        self.layers = nn.ModuleList([
            SpikingLayer(sizes[i], sizes[i + 1], tau_mem)
            for i in range(len(layer_sizes))
        ])
        self.readout = nn.Linear(layer_sizes[-1], output_size)

    def forward(
        self,
        x: torch.Tensor,   # [batch, n_timesteps, input_size]
    ) -> Tuple[torch.Tensor, Dict]:
        B = x.shape[0]
        T = min(x.shape[1], self.n_timesteps)

        # Initialize membrane voltages
        v_mems = [
            torch.zeros(B, layer.lif.n_neurons, device=x.device)
            for layer in self.layers
        ]

        all_spikes: List[List[torch.Tensor]] = [[] for _ in self.layers]

        for t in range(T):
            inp = x[:, t, :]
            for l_idx, layer in enumerate(self.layers):
                spikes, v_mems[l_idx] = layer(inp, v_mems[l_idx])
                inp = spikes
                all_spikes[l_idx].append(spikes)

        # Spike rate (average over time) for each layer
        spike_rates = [
            torch.stack(all_spikes[i], dim=1).mean(dim=1)
            for i in range(len(self.layers))
        ]

        # Readout from final layer spike rate
        output = self.readout(spike_rates[-1])

        # Population metrics for market stress detection
        final_spikes = torch.stack(all_spikes[-1], dim=1)  # [B, T, N]
        firing_rate = final_spikes.mean().item()
        synchrony = final_spikes.float().var(dim=-1).mean().item()  # low = synchronized
        burst_idx = (final_spikes.sum(dim=-1) > final_spikes.shape[-1] * 0.8).float().mean().item()

        meta = {
            "firing_rate": firing_rate,
            "synchrony": synchrony,
            "burst_index": burst_idx,
            "stress_signal": float(burst_idx * (1 - synchrony)),
            "herding_detected": synchrony < 0.1 and firing_rate > 0.6,
        }
        return output, meta
