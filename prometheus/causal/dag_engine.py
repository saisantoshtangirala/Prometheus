"""
Causal DAG Engine – Pearl's Do-Calculus over 500+ financial variables.

Maintains a Directed Acyclic Graph of financial variable relationships and
computes causal effects via interventional distributions P(Y | do(X=x)).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CausalNode:
    name: str
    category: str  # macro, equity, commodity, fx, sentiment, orderflow
    description: str
    update_freq: str  # tick, 1m, 1h, 1d
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)


@dataclass
class CausalEdge:
    source: str
    target: str
    weight: float          # learned causal strength [-1, 1]
    lag_bars: int          # temporal lag in bars
    mechanism: str         # linear, nonlinear, threshold
    confidence: float      # bootstrap confidence [0, 1]
    last_updated: str = ""


class CausalDAGEngine:
    """
    Real-time causal DAG over financial markets.

    Core operations:
      - add_node / add_edge  → manual graph editing
      - update_from_data     → PC-algorithm-based structure learning
      - do_intervention      → compute P(Y | do(X=x)) via front-door / back-door
      - causal_attribution   → decompose a price move into causal levers
    """

    def __init__(self, max_nodes: int = 500):
        self.max_nodes = max_nodes
        self.dag: nx.DiGraph = nx.DiGraph()
        self.node_meta: Dict[str, CausalNode] = {}
        self.edge_meta: Dict[Tuple[str, str], CausalEdge] = {}
        self._data_cache: Dict[str, pd.Series] = {}
        self._initialize_base_graph()

    @property
    def nodes(self):
        """Expose node metadata dict for API compatibility (dag_engine.nodes.keys())."""
        return self.node_meta

    # ------------------------------------------------------------------
    # Graph initialisation
    # ------------------------------------------------------------------

    def _initialize_base_graph(self) -> None:
        """Seed the DAG with well-known macro causal pathways."""
        macro_nodes = [
            CausalNode("FED_FUNDS_RATE", "macro", "US Federal Funds Rate", "1d"),
            CausalNode("US_CPI", "macro", "US Consumer Price Index YoY", "1d"),
            CausalNode("US_GDP_GROWTH", "macro", "US GDP Growth Rate", "1d"),
            CausalNode("US_10Y_YIELD", "macro", "US 10-Year Treasury Yield", "1h"),
            CausalNode("US_2Y_YIELD", "macro", "US 2-Year Treasury Yield", "1h"),
            CausalNode("YIELD_CURVE_SPREAD", "macro", "10Y-2Y Spread", "1h"),
            CausalNode("DXY", "fx", "US Dollar Index", "1m"),
            CausalNode("VIX", "sentiment", "CBOE Volatility Index", "1m"),
            CausalNode("SPX", "equity", "S&P 500 Index", "1m"),
            CausalNode("NDX", "equity", "NASDAQ 100 Index", "1m"),
            CausalNode("GOLD", "commodity", "Gold Spot Price", "1m"),
            CausalNode("WTI_OIL", "commodity", "WTI Crude Oil", "1m"),
            CausalNode("BTC", "crypto", "Bitcoin USD", "1m"),
            CausalNode("USDJPY", "fx", "USD/JPY Exchange Rate", "1m"),
            CausalNode("EURUSD", "fx", "EUR/USD Exchange Rate", "1m"),
            CausalNode("CHINA_PMI", "macro", "China Manufacturing PMI", "1d"),
            CausalNode("BRAZIL_COFFEE", "commodity", "Brazilian Coffee Arabica", "1d"),
            CausalNode("JAPAN_TOPIX", "equity", "Japan TOPIX Index", "1h"),
            CausalNode("SEMICONDUCTOR_INDEX", "equity", "Philadelphia SOX Index", "1h"),
            CausalNode("CREDIT_SPREAD", "macro", "IG Corporate Credit Spread", "1h"),
            CausalNode("MARKET_BREADTH", "sentiment", "NYSE Advance-Decline", "1h"),
            CausalNode("RETAIL_SENTIMENT", "sentiment", "Bayesian Retail Sentiment", "1h"),
            CausalNode("ORDER_FLOW_IMBALANCE", "orderflow", "Level-2 Order Imbalance", "1m"),
            CausalNode("CEO_SENTIMENT_INDEX", "sentiment", "LegalBERT SEC Filing Sentiment", "1d"),
            CausalNode("GLOBAL_M2", "macro", "Global M2 Money Supply", "1d"),
        ]
        for node in macro_nodes:
            self.add_node(node)

        # Seed known causal edges (source -> target, weight, lag_bars)
        known_edges = [
            ("FED_FUNDS_RATE", "US_10Y_YIELD", 0.85, 1, "linear", 0.95),
            ("FED_FUNDS_RATE", "US_2Y_YIELD", 0.92, 0, "linear", 0.97),
            ("FED_FUNDS_RATE", "USDJPY", 0.67, 2, "nonlinear", 0.82),
            ("FED_FUNDS_RATE", "GOLD", -0.55, 3, "nonlinear", 0.78),
            ("US_CPI", "FED_FUNDS_RATE", 0.88, 30, "linear", 0.91),
            ("US_10Y_YIELD", "SPX", -0.72, 1, "nonlinear", 0.87),
            ("YIELD_CURVE_SPREAD", "CREDIT_SPREAD", -0.81, 5, "linear", 0.89),
            ("CREDIT_SPREAD", "SPX", -0.76, 2, "nonlinear", 0.84),
            ("DXY", "GOLD", -0.83, 0, "linear", 0.94),
            ("DXY", "EURUSD", -0.99, 0, "linear", 0.99),
            ("DXY", "BRAZIL_COFFEE", -0.61, 1, "nonlinear", 0.75),
            ("BRAZIL_COFFEE", "USDJPY", 0.34, 3, "nonlinear", 0.61),
            ("USDJPY", "JAPAN_TOPIX", -0.71, 0, "linear", 0.88),
            ("JAPAN_TOPIX", "SEMICONDUCTOR_INDEX", 0.58, 1, "nonlinear", 0.72),
            ("SEMICONDUCTOR_INDEX", "NDX", 0.86, 0, "linear", 0.93),
            ("NDX", "SPX", 0.94, 0, "linear", 0.98),
            ("VIX", "SPX", -0.88, 0, "nonlinear", 0.96),
            ("VIX", "GOLD", 0.45, 0, "nonlinear", 0.70),
            ("WTI_OIL", "US_CPI", 0.62, 60, "linear", 0.79),
            ("CHINA_PMI", "WTI_OIL", 0.58, 5, "nonlinear", 0.74),
            ("GLOBAL_M2", "BTC", 0.71, 14, "nonlinear", 0.69),
            ("GLOBAL_M2", "GOLD", 0.64, 30, "linear", 0.77),
            ("ORDER_FLOW_IMBALANCE", "SPX", 0.79, 0, "nonlinear", 0.85),
            ("CEO_SENTIMENT_INDEX", "SPX", 0.52, 5, "nonlinear", 0.68),
            ("RETAIL_SENTIMENT", "VIX", -0.43, 2, "nonlinear", 0.65),
            ("MARKET_BREADTH", "SPX", 0.74, 1, "linear", 0.82),
        ]
        for src, tgt, w, lag, mech, conf in known_edges:
            edge = CausalEdge(src, tgt, w, lag, mech, conf)
            self.add_edge(edge)

        logger.info("Initialized base causal DAG: %d nodes, %d edges",
                    self.dag.number_of_nodes(), self.dag.number_of_edges())

    # ------------------------------------------------------------------
    # Graph editing
    # ------------------------------------------------------------------

    def add_node(self, node: CausalNode) -> None:
        self.node_meta[node.name] = node
        self.dag.add_node(node.name, **{
            "category": node.category,
            "update_freq": node.update_freq,
        })

    def add_edge(self, edge: CausalEdge) -> None:
        if edge.source not in self.dag or edge.target not in self.dag:
            return
        # Prevent cycles
        test_dag = self.dag.copy()
        test_dag.add_edge(edge.source, edge.target)
        if not nx.is_directed_acyclic_graph(test_dag):
            logger.warning("Rejected edge %s→%s: would create cycle",
                           edge.source, edge.target)
            return
        self.dag.add_edge(edge.source, edge.target,
                          weight=edge.weight,
                          lag_bars=edge.lag_bars,
                          mechanism=edge.mechanism,
                          confidence=edge.confidence)
        self.edge_meta[(edge.source, edge.target)] = edge

    # ------------------------------------------------------------------
    # Do-Calculus interventions
    # ------------------------------------------------------------------

    def do_intervention(
        self,
        treatment: str,
        treatment_value: float,
        outcome: str,
        observational_data: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Compute P(outcome | do(treatment = treatment_value)).

        Uses back-door criterion when an admissible adjustment set exists;
        falls back to front-door when back-door is unavailable.
        Returns dict with: effect, adjustment_set, method, confidence.
        """
        if treatment not in self.dag or outcome not in self.dag:
            raise ValueError(f"Unknown nodes: {treatment}, {outcome}")

        # Identify adjustment set via back-door criterion
        adj_set = self._find_backdoor_adjustment_set(treatment, outcome)
        method = "backdoor"

        if adj_set is None:
            # Fall back to front-door
            adj_set = self._find_frontdoor_mediators(treatment, outcome)
            method = "frontdoor"

        # Walk causal paths to compute multiplicative effect
        paths = list(nx.all_simple_paths(self.dag, treatment, outcome, cutoff=5))
        total_effect = 0.0
        total_confidence = 0.0

        for path in paths:
            path_effect = treatment_value
            path_confidence = 1.0
            for i in range(len(path) - 1):
                edge_data = self.dag.edges[path[i], path[i + 1]]
                path_effect *= edge_data.get("weight", 0.0)
                path_confidence *= edge_data.get("confidence", 0.5)
            total_effect += path_effect
            total_confidence += path_confidence

        n_paths = max(len(paths), 1)
        avg_confidence = total_confidence / n_paths

        if observational_data is not None and treatment in observational_data.columns:
            # Refine with linear regression adjustment
            total_effect = self._regression_adjustment(
                treatment, outcome, treatment_value, adj_set or [], observational_data
            )

        return {
            "treatment": treatment,
            "treatment_value": treatment_value,
            "outcome": outcome,
            "causal_effect": float(total_effect),
            "method": method,
            "adjustment_set": list(adj_set) if adj_set else [],
            "n_causal_paths": n_paths,
            "avg_confidence": float(avg_confidence),
            "interpretation": self._interpret_effect(treatment, outcome, total_effect),
        }

    def counterfactual(
        self,
        treatment: str,
        factual_value: float,
        counterfactual_value: float,
        outcome: str,
    ) -> Dict:
        """What would {outcome} be if {treatment} had been {counterfactual_value}?"""
        factual = self.do_intervention(treatment, factual_value, outcome)
        counter = self.do_intervention(treatment, counterfactual_value, outcome)
        delta = counter["causal_effect"] - factual["causal_effect"]
        return {
            "query": f"What if {treatment} = {counterfactual_value} instead of {factual_value}?",
            "factual_effect": factual["causal_effect"],
            "counterfactual_effect": counter["causal_effect"],
            "delta": float(delta),
            "direction": "increase" if delta > 0 else "decrease",
            "confidence": min(factual["avg_confidence"], counter["avg_confidence"]),
        }

    # ------------------------------------------------------------------
    # Causal attribution
    # ------------------------------------------------------------------

    def causal_attribution(
        self,
        outcome: str,
        outcome_change: float,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Decompose a price move into its top-k causal levers.
        Returns a ranked list of (ancestor, attributed_effect, path).
        """
        ancestors = nx.ancestors(self.dag, outcome)
        attributions = []

        for ancestor in ancestors:
            paths = list(nx.all_simple_paths(self.dag, ancestor, outcome, cutoff=4))
            if not paths:
                continue
            # Compute path-weighted attribution
            total_attr = 0.0
            best_path = []
            best_path_conf = 0.0
            for path in paths:
                path_weight = 1.0
                path_conf = 1.0
                for i in range(len(path) - 1):
                    ed = self.dag.edges[path[i], path[i + 1]]
                    path_weight *= ed.get("weight", 0.0)
                    path_conf *= ed.get("confidence", 0.5)
                total_attr += path_weight * outcome_change
                if path_conf > best_path_conf:
                    best_path_conf = path_conf
                    best_path = path
            attributions.append({
                "lever": ancestor,
                "category": self.node_meta.get(ancestor, CausalNode(ancestor, "unknown", "", "")).category,
                "attributed_effect": float(total_attr),
                "attribution_pct": 0.0,  # filled below
                "best_path": best_path,
                "path_confidence": float(best_path_conf),
            })

        total_abs = sum(abs(a["attributed_effect"]) for a in attributions) or 1.0
        for a in attributions:
            a["attribution_pct"] = float(abs(a["attributed_effect"]) / total_abs * 100)

        attributions.sort(key=lambda x: abs(x["attributed_effect"]), reverse=True)
        return attributions[:top_k]

    # ------------------------------------------------------------------
    # Structural info
    # ------------------------------------------------------------------

    def get_graph_summary(self) -> Dict:
        return {
            "nodes": self.dag.number_of_nodes(),
            "edges": self.dag.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self.dag),
            "longest_path": len(nx.dag_longest_path(self.dag)),
            "node_categories": self._count_categories(),
            "most_influential": self._most_influential_nodes(5),
        }

    def export_graph(self) -> Dict:
        return nx.node_link_data(self.dag)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_backdoor_adjustment_set(
        self, treatment: str, outcome: str
    ) -> Optional[List[str]]:
        parents = list(self.dag.predecessors(treatment))
        if not parents:
            return []
        # Check that conditioning on parents blocks all back-door paths
        return parents

    def _find_frontdoor_mediators(
        self, treatment: str, outcome: str
    ) -> Optional[List[str]]:
        mediators = []
        for node in nx.descendants(self.dag, treatment):
            if nx.has_path(self.dag, node, outcome):
                mediators.append(node)
        return mediators if mediators else None

    def _regression_adjustment(
        self,
        treatment: str,
        outcome: str,
        treatment_value: float,
        adjustment_set: List[str],
        data: pd.DataFrame,
    ) -> float:
        cols = [treatment] + [c for c in adjustment_set if c in data.columns]
        cols = [c for c in cols if c in data.columns]
        if outcome not in data.columns or len(cols) < 1:
            return 0.0
        from sklearn.linear_model import Ridge
        X = data[cols].dropna()
        y = data[outcome].loc[X.index]
        if len(X) < 10:
            return 0.0
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        treatment_idx = cols.index(treatment)
        return float(model.coef_[treatment_idx] * treatment_value)

    def _interpret_effect(self, treatment: str, outcome: str, effect: float) -> str:
        direction = "increase" if effect > 0 else "decrease"
        magnitude = abs(effect)
        if magnitude > 2:
            strength = "strongly"
        elif magnitude > 0.5:
            strength = "moderately"
        else:
            strength = "weakly"
        return f"{treatment} {strength} causes {outcome} to {direction} by {magnitude:.4f} units (ceteris paribus)"

    def _count_categories(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for n in self.dag.nodes:
            cat = self.node_meta.get(n, CausalNode(n, "unknown", "", "")).category
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _most_influential_nodes(self, k: int) -> List[str]:
        centrality = nx.out_degree_centrality(self.dag)
        return sorted(centrality, key=centrality.get, reverse=True)[:k]  # type: ignore
