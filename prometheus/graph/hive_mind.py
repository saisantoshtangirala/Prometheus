"""
Hive Mind Graph Engine – orchestrates TGN for real-time market surveillance.

Maintains a live graph of assets, computes the Heart of Market,
enforces trading restrictions when systemic stability is low, and
detects institutional order-flow spillover between seemingly unrelated assets.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .temporal_graph_network import TemporalGraphNetwork

logger = logging.getLogger(__name__)


class HiveMindGraphEngine:
    """
    Real-time market graph intelligence engine.

    Key responsibilities:
      1. Build and update the asset adjacency matrix from return correlations
         + order-flow imbalance signals.
      2. Run TGN to compute systemic influence scores.
      3. Identify the "Heart of Market" (3 nodes with highest influence).
      4. Issue TRADING_ALLOWED / TRADING_RESTRICTED signals based on heart stability.
      5. Detect cross-asset spillover: which 2 nodes are about to move together.
    """

    def __init__(
        self,
        asset_names: List[str],
        node_feat_dim: int = 32,
        edge_feat_dim: int = 8,
        memory_dim: int = 64,
        device: str = "cpu",
        stability_threshold: float = 0.60,
        correlation_window: int = 60,
    ):
        # Duplicate ticker detection — graph nodes must be unique
        seen, dupes = set(), []
        for a in asset_names:
            (dupes if a in seen else seen).append(a) if a in seen else seen.add(a)
        if dupes:
            raise ValueError(f"Duplicate tickers detected in graph: {dupes}")

        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.stability_threshold = stability_threshold
        self.correlation_window = correlation_window
        self.device = device

        self.tgn = TemporalGraphNetwork(
            n_nodes=self.n_assets,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            memory_dim=memory_dim,
        ).to(device)

        self._return_history: List[np.ndarray] = []  # rolling window
        self._last_result: Optional[Dict] = None
        self._heart_history: List[List[int]] = []

    # ------------------------------------------------------------------
    # Main update cycle
    # ------------------------------------------------------------------

    def update(
        self,
        current_returns: np.ndarray,          # [n_assets] latest bar returns
        order_flow_imbalance: Optional[np.ndarray] = None,  # [n_assets]
        news_sentiment: Optional[np.ndarray] = None,        # [n_assets]
    ) -> Dict:
        """
        Process latest bar data. Returns current market intelligence state.
        """
        self._return_history.append(current_returns)
        if len(self._return_history) > self.correlation_window:
            self._return_history.pop(0)

        # Build node feature vector
        node_features = self._build_node_features(
            current_returns, order_flow_imbalance, news_sentiment
        )

        # Build adjacency matrix
        adj = self._build_adjacency_matrix()

        # Run TGN
        node_feat_tensor = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        adj_tensor = torch.tensor(adj, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            result = self.tgn(node_feat_tensor, adj_tensor)

        # Extract heart of market
        heart_indices = result["heart_nodes"].cpu().numpy().tolist()
        self._heart_history.append(heart_indices)
        heart_names = [self.asset_names[i] for i in heart_indices if i < self.n_assets]

        # Check stability
        stability = float(result["market_stability"].item())
        heart_stable = stability >= self.stability_threshold

        # Detect spillover
        spillover = self._detect_spillover(result["attention_maps"])

        self._last_result = {
            "timestamp": self._get_timestamp(),
            "heart_of_market": heart_names,
            "heart_indices": heart_indices,
            "market_stability": stability,
            "heart_stable": heart_stable,
            "trading_signal": "ALLOWED" if heart_stable else "RESTRICTED",
            "influence_scores": {
                name: float(result["influence_scores"][i].item())
                for i, name in enumerate(self.asset_names)
                if i < result["influence_scores"].shape[0]
            },
            "spillover_pairs": spillover,
            "systemic_risk_level": self._classify_risk(stability),
        }
        return self._last_result

    def get_trading_permission(self, asset: str) -> Tuple[bool, str]:
        """
        Should we trade this asset right now?
        Returns (allowed, reason).
        """
        if self._last_result is None:
            return True, "No graph data yet — trading cautiously allowed"

        if not self._last_result["heart_stable"]:
            stability = self._last_result["market_stability"]
            return False, (
                f"Heart of market unstable (stability={stability:.2f} < {self.stability_threshold}). "
                f"Heart assets: {self._last_result['heart_of_market']}. "
                "Reducing all positions until stability recovers."
            )

        # Check if asset is in a high-influence spillover pair
        for pair in self._last_result["spillover_pairs"]:
            if asset in pair["assets"] and pair["spillover_strength"] > 0.8:
                return True, f"CAUTION: {asset} in active spillover pair — size down 50%"

        return True, "Market heart stable — normal trading permitted"

    def get_heart_stability_trend(self) -> Dict:
        """Trend analysis of heart stability over recent history."""
        if len(self._heart_history) < 5:
            return {"stable": True, "trend": "INSUFFICIENT_DATA"}

        # Heart consistency: same 3 nodes over time = stable
        recent = self._heart_history[-10:]
        unique_hearts = len(set(tuple(sorted(h)) for h in recent))
        consistency = 1.0 - (unique_hearts - 1) / max(len(recent), 1)

        return {
            "consistency": float(consistency),
            "stable": consistency > 0.7,
            "trend": "STABLE" if consistency > 0.7 else (
                "ROTATING" if consistency > 0.4 else "CHAOTIC"
            ),
            "recent_hearts": recent[-3:],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_node_features(
        self,
        returns: np.ndarray,
        ofi: Optional[np.ndarray],
        sentiment: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build [n_assets, node_feat_dim] feature matrix."""
        n = self.n_assets
        # Volatility from history
        if len(self._return_history) >= 5:
            hist = np.stack(self._return_history[-20:])
            vol = hist.std(axis=0)
            momentum = hist[-5:].mean(axis=0) - hist[:5].mean(axis=0)
        else:
            vol = np.ones(n) * 0.01
            momentum = np.zeros(n)

        features = np.stack([
            returns[:n],
            vol[:n],
            momentum[:n],
            ofi[:n] if ofi is not None else np.zeros(n),
            sentiment[:n] if sentiment is not None else np.zeros(n),
        ], axis=1)  # [n, 5]

        # Pad to node_feat_dim=32 with zeros
        pad_width = 32 - features.shape[1]
        if pad_width > 0:
            features = np.hstack([features, np.zeros((n, pad_width))])
        return features.astype(np.float32)

    def _build_adjacency_matrix(self) -> np.ndarray:
        """Compute correlation-based adjacency matrix."""
        n = self.n_assets
        if len(self._return_history) < 10:
            return np.eye(n, dtype=np.float32)
        hist = np.stack(self._return_history)
        corr = np.corrcoef(hist.T)
        # Only keep strong correlations as edges (|r| > 0.3)
        adj = np.where(np.abs(corr) > 0.3, np.abs(corr), 0.0)
        np.fill_diagonal(adj, 1.0)
        return adj.astype(np.float32)

    def _detect_spillover(self, attention_maps: List[torch.Tensor]) -> List[Dict]:
        """
        Identify pairs of assets with unusually strong attention (order-flow spillover).
        """
        if not attention_maps:
            return []
        final_attn = attention_maps[-1].cpu().numpy()  # [N, N]
        n = final_attn.shape[0]
        threshold = final_attn.mean() + 2 * final_attn.std()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                strength = (final_attn[i, j] + final_attn[j, i]) / 2
                if strength > threshold:
                    pairs.append({
                        "assets": [self.asset_names[i], self.asset_names[j]],
                        "spillover_strength": float(strength),
                        "direction": "bidirectional",
                    })
        return sorted(pairs, key=lambda x: x["spillover_strength"], reverse=True)[:10]

    def _classify_risk(self, stability: float) -> str:
        if stability > 0.8:
            return "LOW"
        if stability > 0.6:
            return "MODERATE"
        if stability > 0.4:
            return "ELEVATED"
        return "CRITICAL"

    @staticmethod
    def _get_timestamp() -> str:
        from datetime import datetime
        return datetime.utcnow().isoformat()
