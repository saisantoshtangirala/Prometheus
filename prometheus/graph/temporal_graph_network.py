"""
Temporal Graph Network (TGN) – The "Hive Mind" of Market Participants.

Models the market as a dynamic graph where:
  - Nodes = individual stocks, ETFs, commodities, currencies
  - Edges = hidden institutional order-flow spillover (not visible in prices)
  - Edge weights = time-varying "Systemic Influence Score"

Uses GATv2 (dynamic attention) to identify the 3 stocks that act as the
"heart" of the market. Trading is heavily restricted when the heart is unstable.

Reference:
  Rossi et al. (2020) "Temporal Graph Networks for Deep Learning on Dynamic Graphs"
  Brody et al. (2021) "How Attentive are Graph Attention Networks?"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """
    GATv2 graph attention layer with dynamic attention.

    GATv2 fixes the static attention problem of GATv1 by computing attention
    as a_ij = LeakyReLU(W[h_i || h_j]) instead of GATv1's additive approach.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features * n_heads if concat else out_features)

    def forward(
        self,
        x: torch.Tensor,        # [n_nodes, in_features]
        adj: torch.Tensor,      # [n_nodes, n_nodes] adjacency (0/1 or weighted)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (node_embeddings, attention_matrix)."""
        n = x.shape[0]
        Wh = self.W(x).view(n, self.n_heads, self.out_features)  # [N, H, F]

        # Broadcast to all pairs
        Wh_i = Wh.unsqueeze(1).expand(-1, n, -1, -1)  # [N, N, H, F]
        Wh_j = Wh.unsqueeze(0).expand(n, -1, -1, -1)  # [N, N, H, F]
        e = torch.cat([Wh_i, Wh_j], dim=-1)            # [N, N, H, 2F]
        e = self.leaky(self.a(e)).squeeze(-1)           # [N, N, H]

        # Mask non-edges
        mask = (adj == 0).unsqueeze(-1).expand_as(e)
        e = e.masked_fill(mask, float("-inf"))
        attn = F.softmax(e, dim=1)                     # [N, N, H]
        attn = self.dropout(attn)

        # Aggregate
        out = torch.einsum("ijh,jhf->ihf", attn, Wh)  # [N, H, F]
        attn_mean = attn.mean(dim=-1)                  # [N, N]

        if self.concat:
            out = out.reshape(n, -1)
        else:
            out = out.mean(dim=1)

        return self.norm(out), attn_mean


class NodeMemoryModule(nn.Module):
    """
    TGN memory module: each node maintains a persistent memory vector
    that accumulates interaction history via a GRU update.
    """

    def __init__(self, n_nodes: int, memory_dim: int, message_dim: int):
        super().__init__()
        self.memory = nn.Parameter(
            torch.zeros(n_nodes, memory_dim), requires_grad=False
        )
        self.memory_updater = nn.GRUCell(message_dim, memory_dim)
        self.message_agg = nn.Linear(memory_dim * 2, message_dim)

    def update(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> None:
        """Update memory for nodes involved in recent interactions."""
        # Build messages: src_memory + edge_feature → dst message
        src_mem = self.memory[src_nodes]
        dst_mem = self.memory[dst_nodes]
        msg_input = torch.cat([src_mem, dst_mem], dim=-1)
        messages = F.relu(self.message_agg(msg_input))

        # Aggregate messages per destination node
        new_mem = self.memory.clone()
        for i, dst in enumerate(dst_nodes):
            new_mem[dst] = self.memory_updater(messages[i].unsqueeze(0), new_mem[dst].unsqueeze(0)).squeeze(0)

        self.memory.data.copy_(new_mem)

    def get(self, node_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if node_indices is None:
            return self.memory
        return self.memory[node_indices]


class TemporalGraphNetwork(nn.Module):
    """
    Full TGN for market graph representation learning.

    Processes a sequence of market interactions (trades, order fills, news events)
    and outputs:
      - Node embeddings (per-asset latent representation)
      - Systemic Influence Scores (centrality in the causal flow graph)
      - Heart of Market identification (top-3 most systemically important nodes)
      - Market stability score
    """

    def __init__(
        self,
        n_nodes: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        memory_dim: int = 128,
        embed_dim: int = 128,
        n_gat_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.memory_dim = memory_dim

        # Node memory
        self.memory = NodeMemoryModule(n_nodes, memory_dim, memory_dim)

        # Node feature + memory projection
        self.node_proj = nn.Linear(node_feat_dim + memory_dim, embed_dim)

        # GATv2 layers
        dims = [embed_dim] + [embed_dim] * n_gat_layers
        self.gat_layers = nn.ModuleList([
            GATv2Layer(dims[i], dims[i + 1] // n_heads, n_heads, dropout)
            for i in range(n_gat_layers)
        ])

        # Systemic influence scorer
        self.influence_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Market stability classifier
        self.stability_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_features: torch.Tensor,     # [n_nodes, node_feat_dim]
        adj_matrix: torch.Tensor,        # [n_nodes, n_nodes] weighted adjacency
        src_nodes: Optional[torch.Tensor] = None,
        dst_nodes: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
          node_embeddings: [n_nodes, embed_dim]
          influence_scores: [n_nodes] — systemic importance
          attention_maps: list of [n_nodes, n_nodes] per GAT layer
          market_stability: float in [0, 1]
          heart_nodes: top-3 node indices
        """
        n = node_features.shape[0]

        # Update memory from recent interactions
        if src_nodes is not None and edge_features is not None:
            self.memory.update(src_nodes, dst_nodes, edge_features)

        # Concatenate node features with memory
        mem = self.memory.get()
        h = self.node_proj(torch.cat([node_features, mem], dim=-1))

        # GAT forward
        attn_maps = []
        for gat in self.gat_layers:
            h, attn = gat(h, adj_matrix)
            attn_maps.append(attn)
            h = F.elu(h)

        # Compute systemic influence (global pooling as market context)
        market_context = h.mean(dim=0, keepdim=True).expand(n, -1)
        influence_input = h + market_context
        influence_scores = self.influence_head(influence_input).squeeze(-1)

        # Market stability from global representation
        market_stability = self.stability_head(h.mean(dim=0)).squeeze()

        # Heart of market: top-3 by influence score
        heart_nodes = torch.topk(influence_scores, k=min(3, n)).indices

        return {
            "node_embeddings": h,
            "influence_scores": influence_scores,
            "attention_maps": attn_maps,
            "market_stability": market_stability,
            "heart_nodes": heart_nodes,
            "heart_stable": bool(market_stability.item() > 0.6),
        }

    def get_systemic_importance_ranking(
        self,
        node_names: List[str],
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> List[Dict]:
        """Return ranked list of nodes by systemic importance."""
        with torch.no_grad():
            result = self.forward(node_features, adj_matrix)
        scores = result["influence_scores"].cpu().numpy()
        return sorted(
            [{"asset": name, "influence_score": float(scores[i])}
             for i, name in enumerate(node_names)],
            key=lambda x: x["influence_score"],
            reverse=True,
        )
