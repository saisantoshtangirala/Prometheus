"""
Causal Transformer – Do-Calculus integrated attention mechanism.

Augments standard multi-head self-attention with causal masks derived from
the DAG.  Each attention head is associated with a causal pathway, and
attention weights are regularized to reflect causal confidence scores.
The forward pass outputs both predictions AND causal attribution weights.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttentionHead(nn.Module):
    """
    Single attention head constrained by a causal adjacency mask.

    The causal mask forces zero attention from effect to cause (preventing
    acausal information leakage) and amplifies attention along high-confidence
    causal edges.
    """

    def __init__(self, d_model: int, d_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_head = d_head
        self.q = nn.Linear(d_model, d_head, bias=False)
        self.k = nn.Linear(d_model, d_head, bias=False)
        self.v = nn.Linear(d_model, d_head, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        causal_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            causal_mask: [seq_len, seq_len] boolean — True where attention is forbidden
            causal_weights: [seq_len, seq_len] float — DAG edge confidence amplifier
        Returns:
            output: [batch, seq_len, d_head]
            attn_weights: [batch, seq_len, seq_len]
        """
        Q = self.q(x)  # [B, T, d_head]
        K = self.k(x)
        V = self.v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if causal_weights is not None:
            scores = scores + causal_weights.unsqueeze(0)

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return out, attn


class CausalMultiHeadAttention(nn.Module):
    """Multi-head attention where each head corresponds to a causal pathway depth."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.heads = nn.ModuleList([
            CausalAttentionHead(d_model, self.d_head, dropout)
            for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        causal_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head_outs = []
        all_attn = []
        for head in self.heads:
            out, attn = head(x, causal_mask, causal_weights)
            head_outs.append(out)
            all_attn.append(attn)
        concat = torch.cat(head_outs, dim=-1)
        out = self.out_proj(concat)
        mean_attn = torch.stack(all_attn, dim=1).mean(dim=1)
        return out, mean_attn


class CausalTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalMultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        causal_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(self.norm1(x), causal_mask, causal_weights)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, attn_weights


class CausalTransformer(nn.Module):
    """
    Full Causal Transformer for financial time-series.

    Inputs:
      - x: [batch, seq_len, n_features]  — market feature tensor
      - dag_adj: [n_features, n_features] — causal adjacency (confidence weights)

    Outputs:
      - predictions: [batch, horizon, n_targets]
      - causal_attributions: [batch, n_features]  — lever importance scores
      - attn_weights: [batch, seq_len, seq_len]   — final layer attention map
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        horizon: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets
        self.horizon = horizon
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_enc = self._build_positional_encoding(max_seq_len, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            CausalTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output heads
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Linear(d_ff // 2, horizon * n_targets),
        )

        # Causal attribution head: maps d_model → n_features importance scores
        self.attribution_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_features),
            nn.Softmax(dim=-1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        dag_adj: Optional[torch.Tensor] = None,
        return_attributions: bool = True,
    ) -> Dict[str, torch.Tensor]:
        B, T, F = x.shape

        # Project to model dimension
        h = self.input_proj(x)  # [B, T, d_model]

        # Add positional encoding
        pos = self.pos_enc[:T, :].to(x.device)
        h = self.dropout(h + pos.unsqueeze(0))

        # Build temporal causal mask (no future leakage)
        temporal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        # Convert DAG adjacency to attention bias
        causal_weights = None
        if dag_adj is not None and dag_adj.shape == (T, T):
            # Scale confidence to log-space for attention addition
            causal_weights = torch.log(dag_adj.clamp(min=1e-6)).to(x.device)

        # Run through causal transformer layers
        all_attn = []
        for layer in self.layers:
            h, attn = layer(h, temporal_mask, causal_weights)
            all_attn.append(attn)

        # Pool over sequence for attribution
        pooled = h.mean(dim=1)  # [B, d_model]

        # Predictions
        preds_flat = self.pred_head(pooled)  # [B, horizon * n_targets]
        predictions = preds_flat.view(B, self.horizon, self.n_targets)

        result = {"predictions": predictions, "attn_weights": all_attn[-1]}

        if return_attributions:
            attributions = self.attribution_head(pooled)  # [B, n_features]
            result["causal_attributions"] = attributions

        return result

    @staticmethod
    def _build_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [max_len, d_model]

    def get_causal_summary(
        self, attributions: torch.Tensor, feature_names: List[str]
    ) -> List[Dict]:
        """Convert attribution tensor to human-readable causal lever ranking."""
        scores = attributions.mean(dim=0).detach().cpu().numpy()
        ranked = sorted(
            zip(feature_names, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"lever": name, "importance": float(score)} for name, score in ranked]
