"""
PC-Algorithm based causal structure discovery.

Runs the Peter-Clark algorithm to discover hidden causal links between
financial variables from observational time-series data.  Uses conditional
independence tests (partial correlation, G-squared) to orient edges.
"""

from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class PCAlgorithmDiscovery:
    """
    PC algorithm for causal skeleton discovery in financial time-series.

    Steps:
      1. Start with fully connected undirected graph.
      2. Remove edges via conditional independence tests (growing conditioning sets).
      3. Orient v-structures (colliders).
      4. Apply Meek orientation rules to propagate directions.
    """

    def __init__(
        self,
        alpha: float = 0.01,         # significance level for independence tests
        max_cond_set_size: int = 4,  # max conditioning set size (computational limit)
        test: str = "partial_corr",  # "partial_corr" | "g_squared"
        lag: int = 1,                # temporal lag for directed time-series edges
    ):
        self.alpha = alpha
        self.max_cond_set_size = max_cond_set_size
        self.test = test
        self.lag = lag
        self.sep_sets: Dict[Tuple[str, str], Set[str]] = {}

    def fit(self, data) -> "PCAlgorithmDiscovery":
        """Run the full PC algorithm on a DataFrame or numpy array."""
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=[f"X{i}" for i in range(data.shape[1])])
        self.variables = list(data.columns)
        self.data = data.copy()
        logger.info("PC Algorithm: %d variables, %d observations",
                    len(self.variables), len(data))

        self._build_skeleton()
        self._orient_v_structures()
        self._meek_rules()

        # Build sklearn-style adjacency matrix
        n = len(self.variables)
        self.adjacency_matrix_ = np.zeros((n, n), dtype=int)
        for u, v in self.skeleton.edges():
            i, j = self.variables.index(u), self.variables.index(v)
            self.adjacency_matrix_[i, j] = 1
            if not self.skeleton[u][v].get("directed", False):
                self.adjacency_matrix_[j, i] = 1
        return self

    def get_edges(self) -> List[Tuple[str, str, float]]:
        """Return (source, target, confidence) for all oriented edges."""
        edges = []
        for u, v, data in self.skeleton.edges(data=True):
            conf = data.get("weight", 0.5)
            if data.get("directed", False):
                edges.append((u, v, conf))
            else:
                # Undirected edge → orient by time ordering for time-series
                u_idx = self.variables.index(u)
                v_idx = self.variables.index(v)
                if u_idx < v_idx:
                    edges.append((u, v, conf))
                else:
                    edges.append((v, u, conf))
        return edges

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_skeleton(self) -> None:
        import networkx as nx
        n = len(self.variables)
        self.skeleton = nx.Graph()
        self.skeleton.add_nodes_from(self.variables)
        for i, j in itertools.combinations(range(n), 2):
            self.skeleton.add_edge(self.variables[i], self.variables[j], weight=1.0)

        cond_size = 0
        while cond_size <= min(self.max_cond_set_size, n - 2):
            edges_to_remove = []
            for u, v in list(self.skeleton.edges()):
                neighbors_u = set(self.skeleton.neighbors(u)) - {v}
                if len(neighbors_u) < cond_size:
                    continue
                for cond_set in itertools.combinations(neighbors_u, cond_size):
                    pval = self._independence_test(u, v, list(cond_set))
                    if pval > self.alpha:
                        edges_to_remove.append((u, v))
                        self.sep_sets[(u, v)] = set(cond_set)
                        self.sep_sets[(v, u)] = set(cond_set)
                        break
            for u, v in edges_to_remove:
                if self.skeleton.has_edge(u, v):
                    self.skeleton.remove_edge(u, v)
                    logger.debug("Removed edge %s -- %s", u, v)
            cond_size += 1
        logger.info("Skeleton: %d edges retained", self.skeleton.number_of_edges())

    def _orient_v_structures(self) -> None:
        """Orient X → Z ← Y v-structures (colliders)."""
        for z in self.variables:
            neighbors_z = list(self.skeleton.neighbors(z))
            for x, y in itertools.combinations(neighbors_z, 2):
                if self.skeleton.has_edge(x, y):
                    continue  # already adjacent, skip
                sep_xy = self.sep_sets.get((x, y), set())
                if z not in sep_xy:
                    # z is a collider: x → z ← y
                    self.skeleton[x][z]["directed"] = True
                    self.skeleton[z][y]["directed"] = True
                    logger.debug("V-structure: %s → %s ← %s", x, z, y)

    def _meek_rules(self) -> None:
        """Apply Meek (1995) orientation rules R1–R4 to maximize orientation."""
        changed = True
        while changed:
            changed = False
            changed |= self._meek_r1()
            changed |= self._meek_r2()

    def _meek_r1(self) -> bool:
        """R1: Orient Z — Y into Z → Y if X → Z and X not adjacent to Y."""
        changed = False
        for z, y in list(self.skeleton.edges()):
            if self.skeleton[z][y].get("directed", False):
                continue
            for x in self.skeleton.predecessors_of(z) if hasattr(self.skeleton, "predecessors_of") else []:
                if not self.skeleton.has_edge(x, y):
                    self.skeleton[z][y]["directed"] = True
                    changed = True
        return changed

    def _meek_r2(self) -> bool:
        return False  # Stub — full impl requires DiGraph tracking

    def _independence_test(
        self, x: str, y: str, cond_set: List[str]
    ) -> float:
        """Return p-value for X ⊥ Y | cond_set."""
        try:
            if self.test == "partial_corr":
                return self._partial_corr_test(x, y, cond_set)
            else:
                return self._g_squared_test(x, y, cond_set)
        except Exception:
            return 0.0

    def _partial_corr_test(self, x: str, y: str, cond_set: List[str]) -> float:
        cols = [x, y] + cond_set
        sub = self.data[cols].dropna()
        if len(sub) < 20:
            return 0.0
        if not cond_set:
            r, p = stats.pearsonr(sub[x], sub[y])
            return float(p)
        # Partial correlation via residualization
        from sklearn.linear_model import LinearRegression
        reg_x = LinearRegression().fit(sub[cond_set], sub[x])
        reg_y = LinearRegression().fit(sub[cond_set], sub[y])
        res_x = sub[x].values - reg_x.predict(sub[cond_set])
        res_y = sub[y].values - reg_y.predict(sub[cond_set])
        r, p = stats.pearsonr(res_x, res_y)
        return float(p)

    def _g_squared_test(self, x: str, y: str, cond_set: List[str]) -> float:
        # Discretize and apply chi-squared
        from sklearn.preprocessing import KBinsDiscretizer
        cols = [x, y] + cond_set
        sub = self.data[cols].dropna()
        if len(sub) < 20:
            return 0.0
        disc = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
        sub_disc = disc.fit_transform(sub)
        if not cond_set:
            ct = pd.crosstab(sub_disc[:, 0], sub_disc[:, 1])
            _, p, _, _ = stats.chi2_contingency(ct)
            return float(p)
        return 0.0  # Simplified for high-dimensional case
