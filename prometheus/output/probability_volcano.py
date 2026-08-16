"""
Probability Volcano – 4D visualization of future return distributions.

Outputs a 3D surface plot over time showing the entire probability density
function of future returns, color-coded by causal confidence.

The "volcano" metaphor: the rim represents high-probability (consensus)
paths; the crater represents the tail-risk zone; steam vents mark
causal high-confidence signals.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class ProbabilityVolcano:
    """
    Computes and renders the 4D Probability Volcano visualization.

    Inputs:
      - predictions: [n_samples, horizon, n_assets] — Monte Carlo paths
      - causal_confidence: [horizon, n_assets] — DAG confidence per step
      - asset_names: list of asset names

    Outputs:
      - Plotly figure with interactive 3D surface
      - Statistical summary (percentiles, VaR, CVaR)
    """

    def __init__(
        self,
        n_percentile_bands: int = 11,    # 0,10,20,...,100 percentile bands
        horizon_labels: Optional[List[str]] = None,
        n_assets: Optional[int] = None,  # unused; kept for API compatibility
        horizon: Optional[int] = None,   # unused; kept for API compatibility
    ):
        self.n_bands = n_percentile_bands
        self.horizon_labels = horizon_labels

    def compute_distribution(
        self,
        paths: np.ndarray,        # [n_samples, horizon, n_assets]
        causal_confidence: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute the probability distribution surface.
        Returns percentile bands and confidence-weighted density.
        """
        n_samples, horizon, n_assets = paths.shape
        percentiles = np.linspace(0, 100, self.n_bands)

        # Compute percentile bands for each (timestep, asset)
        pct_surface = np.zeros((self.n_bands, horizon, n_assets))
        for t in range(horizon):
            for a in range(n_assets):
                pct_surface[:, t, a] = np.percentile(paths[:, t, a], percentiles)

        # Compute cumulative return (compounded)
        cum_paths = np.cumprod(1 + paths.clip(-0.5, 2.0), axis=1) - 1

        # VaR and CVaR — computed per asset
        final_returns = cum_paths[:, -1, :]  # [n_samples, n_assets]
        var_95 = np.percentile(final_returns, 5, axis=0)  # [n_assets]
        cvar_95 = np.array([
            float(final_returns[:, a][final_returns[:, a] < var_95[a]].mean())
            if (final_returns[:, a] < var_95[a]).any() else float(var_95[a])
            for a in range(n_assets)
        ])

        # Probability mass in positive territory
        p_positive = (paths > 0).mean(axis=0)  # [horizon, n_assets]

        # Confidence-weighted density peak
        if causal_confidence is not None:
            conf_weighted_peak = pct_surface[self.n_bands // 2] * causal_confidence
        else:
            conf_weighted_peak = pct_surface[self.n_bands // 2]

        return {
            "percentile_surface": pct_surface,
            "percentiles": percentiles.tolist(),
            "cumulative_paths": cum_paths,
            "var_95": var_95.tolist(),
            "cvar_95": cvar_95.tolist(),
            "p_positive": p_positive.tolist(),
            "confidence_weighted_peak": conf_weighted_peak.tolist(),
            "n_samples": n_samples,
            "horizon": horizon,
            "n_assets": n_assets,
        }

    def render_plotly(
        self,
        paths: np.ndarray,
        asset_idx: int = 0,
        asset_name: str = "Asset",
        causal_confidence: Optional[np.ndarray] = None,
        show: bool = False,
    ):
        """
        Render interactive 3D probability volcano for one asset.
        Returns a plotly Figure object.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise RuntimeError("plotly is required for volcano visualization")

        dist = self.compute_distribution(paths, causal_confidence)
        pct_surface = dist["percentile_surface"][:, :, asset_idx]
        horizon = paths.shape[1]
        n_bands = self.n_bands
        percentiles = np.linspace(0, 100, n_bands)

        # Build meshgrid
        T = np.arange(horizon)
        P = percentiles

        T_mesh, P_mesh = np.meshgrid(T, P)
        Z_mesh = pct_surface  # [n_bands, horizon]

        # Color by causal confidence
        if causal_confidence is not None:
            cc = np.asarray(causal_confidence)
            conf = cc[:, asset_idx] if cc.ndim == 2 else cc
            conf_expanded = np.tile(conf, (n_bands, 1))
            colorscale = "Viridis"
            surf_color = conf_expanded
        else:
            surf_color = Z_mesh
            colorscale = "RdBu"

        fig = go.Figure()

        # Main probability surface
        fig.add_trace(go.Surface(
            x=T_mesh,
            y=P_mesh,
            z=Z_mesh,
            surfacecolor=surf_color,
            colorscale=colorscale,
            name="Probability Distribution",
            colorbar=dict(title="Causal Confidence" if causal_confidence is not None else "Return"),
            opacity=0.85,
        ))

        # Median path (the "crater rim")
        median_path = np.percentile(paths[:, :, asset_idx], 50, axis=0)
        fig.add_trace(go.Scatter3d(
            x=T,
            y=[50.0] * horizon,
            z=median_path,
            mode="lines",
            line=dict(color="gold", width=5),
            name="Median (50th percentile)",
        ))

        # 5th percentile path (downside risk floor)
        var_path = np.percentile(paths[:, :, asset_idx], 5, axis=0)
        fig.add_trace(go.Scatter3d(
            x=T,
            y=[5.0] * horizon,
            z=var_path,
            mode="lines",
            line=dict(color="red", width=4, dash="dash"),
            name="VaR 95% (5th pct)",
        ))

        fig.update_layout(
            title=dict(
                text=f"PROMETHEUS – Probability Volcano: {asset_name}",
                font=dict(size=16, color="white"),
            ),
            scene=dict(
                xaxis=dict(title="Forecast Horizon (bars)", gridcolor="gray"),
                yaxis=dict(title="Percentile", gridcolor="gray"),
                zaxis=dict(title="Expected Return", gridcolor="gray"),
                bgcolor="black",
            ),
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
        )

        if show:
            fig.show()
        return fig

    def render_html(
        self,
        paths: np.ndarray,
        asset_idx: int,
        asset_name: str,
        output_path: str = "volcano.html",
        causal_confidence: Optional[np.ndarray] = None,
    ) -> str:
        """Save interactive volcano to HTML file. Returns file path."""
        fig = self.render_plotly(paths, asset_idx, asset_name, causal_confidence)
        fig.write_html(output_path, include_plotlyjs=True)
        return output_path

    def summary_statistics(
        self,
        paths: np.ndarray,
        asset_names: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Statistical summary table for all assets."""
        dist = self.compute_distribution(paths)
        n_assets = paths.shape[2]
        if asset_names is None:
            asset_names = [f"Asset{i}" for i in range(n_assets)]
        results = []
        for i, name in enumerate(asset_names):
            if i >= paths.shape[2]:
                break
            asset_paths = paths[:, :, i]
            final_returns = asset_paths[:, -1]
            results.append({
                "asset": name,
                "expected_return": float(np.mean(final_returns)),
                "median_return": float(np.median(final_returns)),
                "std": float(np.std(final_returns)),
                "skewness": float(
                    ((final_returns - final_returns.mean()) ** 3).mean()
                    / (final_returns.std() ** 3 + 1e-8)
                ),
                "p_positive": float((final_returns > 0).mean()),
                "var_95": float(dist["var_95"][i]),
                "cvar_95": float(dist["cvar_95"][i]),
                "best_case_99": float(np.percentile(final_returns, 99)),
                "worst_case_1": float(np.percentile(final_returns, 1)),
            })
        return results
