"""
God's Eye Report Generator.

Synthesizes all Prometheus signals into one plain-English actionable summary:
  "You must short X because Stock Y's supplier in Taiwan is about to fail,
   and our causal graph gives this a 94% probability of cascading within 48 hours."

Each report includes:
  - Primary trade recommendation with entry/exit/stop-loss
  - Causal chain explanation (which lever, which pathway)
  - Risk metrics (Kelly size, drawdown risk, confidence interval)
  - System state: neuromod levels, heart stability, anomaly alerts
  - Dissenting signals (what could go wrong)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class GodsEyeReportGenerator:
    """
    Compiles all Prometheus module outputs into a structured God's Eye report.

    Input: dict of signals from all subsystems.
    Output: structured report as dict + formatted text.
    """

    BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║          PROMETHEUS – THE CAUSAL BRAIN                               ║
║          God's Eye Market Intelligence Report                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

    def __init__(self, asset_names: Optional[List[str]] = None) -> None:
        self.asset_names: List[str] = asset_names or []

    def generate(
        self,
        data: Optional[Dict] = None,
        causal_result: Optional[Dict] = None,
        graph_result: Optional[Dict] = None,
        neuromod_result: Optional[Dict] = None,
        kelly_result: Optional[Dict] = None,
        volcano_stats: Optional[List[Dict]] = None,
        htm_result: Optional[Dict] = None,
        sentiment_result: Optional[Dict] = None,
        timestamp: Optional[str] = None,
    ) -> Dict:
        # Unified-dict API (tests / external callers)
        if data is not None and isinstance(data, dict):
            return self._generate_from_dict(data)

        """Generate the full God's Eye report."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        # Determine primary trade recommendations
        trades = self._extract_trade_recommendations(
            causal_result, graph_result, kelly_result, volcano_stats
        )

        # System state
        system_state = self._build_system_state(
            neuromod_result, graph_result, htm_result
        )

        # Causal narrative
        narrative = self._build_narrative(causal_result, trades, graph_result, sentiment_result)

        # Risk summary
        risk = self._build_risk_summary(kelly_result, neuromod_result, volcano_stats)

        # Dissenting signals
        dissent = self._find_dissenting_signals(
            causal_result, neuromod_result, graph_result, htm_result
        )

        report = {
            "timestamp": ts,
            "system_version": "Prometheus v1.0 – Causal Brain",
            "system_state": system_state,
            "primary_recommendation": narrative,
            "trade_recommendations": trades,
            "risk_summary": risk,
            "dissenting_signals": dissent,
            "causal_attribution": causal_result.get("attributions", []) if causal_result else [],
            "graph_intelligence": {
                "heart_of_market": graph_result.get("heart_of_market", []) if graph_result else [],
                "trading_signal": graph_result.get("trading_signal", "UNKNOWN") if graph_result else "UNKNOWN",
                "systemic_risk": graph_result.get("systemic_risk_level", "UNKNOWN") if graph_result else "UNKNOWN",
            },
            "sentiment": sentiment_result or {},
            "htm_anomaly": {
                "score": htm_result.get("mean_anomaly", 0.0) if htm_result else 0.0,
                "alert": htm_result.get("max_anomaly", 0.0) > 0.7 if htm_result else False,
            },
        }

        report["asset_names"] = self.asset_names
        report["formatted_text"] = self._format_text(report)
        return report

    def _generate_from_dict(self, data: Dict) -> Dict:
        """Handle unified-dict API: data holds all sub-system outputs in one dict."""
        ts = data.get("timestamp") or datetime.now(timezone.utc).isoformat()
        asset_names = data.get("asset_names", self.asset_names or [])

        neuromod = data.get("neuromod_state", {})
        graph = data.get("graph_state", {})
        vol_stats_raw = data.get("volcano_stats", {})
        predictions = data.get("predictions", [])
        kelly_fractions = data.get("kelly_fractions", [])
        confidence = data.get("confidence", [])
        htm_anomaly = data.get("htm_anomaly", 0.0)

        # Convert volcano_stats dict-of-dicts → list-of-dicts with asset key
        if isinstance(vol_stats_raw, dict):
            vol_list = [{"asset": k, **v} for k, v in vol_stats_raw.items()]
        else:
            vol_list = list(vol_stats_raw or [])

        # Build trade recommendations from predictions / kelly fractions
        trades: List[Dict] = []
        for i, asset in enumerate(asset_names):
            try:
                exp_ret = float(predictions[0][i]) if predictions and len(predictions[0]) > i else 0.0
                kf = float(kelly_fractions[i]) if i < len(kelly_fractions) else 0.0
                conf = float(confidence[i]) if i < len(confidence) else 0.5
                vol_stat = next((v for v in vol_list if v.get("asset") == asset), {})
                var95 = vol_stat.get("var_95", -0.02)
                p_pos = vol_stat.get("p_positive", 0.5)
                if abs(exp_ret) < 1e-6 and abs(kf) < 1e-6:
                    continue
                direction = "LONG" if kf >= 0 else "SHORT"
                trades.append({
                    "asset": asset,
                    "direction": direction,
                    "expected_return": exp_ret,
                    "probability_of_profit": p_pos if direction == "LONG" else 1 - p_pos,
                    "kelly_size_pct": abs(kf) * 100,
                    "var_95": var95,
                    "conviction": "HIGH" if conf > 0.75 else "MODERATE",
                })
            except (IndexError, TypeError):
                continue

        # System state
        system_state = {
            "dopamine_level": neuromod.get("dopamine", 0.5),
            "cortisol_level": neuromod.get("cortisol", 0.1),
            "fear_mode_active": neuromod.get("fear_mode", False),
            "position_multiplier": neuromod.get("position_multiplier", 1.0),
            "neuromod_recommendation": neuromod.get("recommendation", "NORMAL"),
            "market_stability": graph.get("market_stability", 0.5),
            "heart_stable": graph.get("heart_stable", True),
            "systemic_risk": graph.get("systemic_risk_level", "MODERATE"),
            "htm_anomaly": htm_anomaly,
            "order_flow_regime": "NORMAL",
            "overall_confidence": 0.5,
        }

        trading_signal = graph.get("trading_signal", "ALLOWED")
        heart = graph.get("heart_of_market", [])

        # Build formatted text ensuring all asset names are present
        lines = [self.BANNER]
        lines.append(f"Generated: {ts}\n")
        lines.append("═" * 72)
        lines.append("SYSTEM STATE")
        lines.append("═" * 72)
        lines.append(f"  Dopamine:          {system_state['dopamine_level']:.3f}")
        lines.append(f"  Cortisol:          {system_state['cortisol_level']:.3f}")
        lines.append(f"  Fear Mode:         {'ACTIVE' if system_state['fear_mode_active'] else 'OFF'}")
        lines.append(f"  Market Stability:  {system_state['market_stability']:.3f}")
        lines.append(f"  Systemic Risk:     {system_state['systemic_risk']}")
        lines.append("")
        lines.append("═" * 72)
        lines.append(f"TRADING SIGNAL: {trading_signal}")
        if heart:
            lines.append(f"MARKET HEART: {', '.join(heart)}")
        lines.append("")
        lines.append("═" * 72)
        lines.append("ASSET SUMMARY")
        lines.append("═" * 72)
        for asset in asset_names:
            i = asset_names.index(asset) if asset in asset_names else -1
            kf = float(kelly_fractions[i]) if i >= 0 and i < len(kelly_fractions) else 0.0
            conf = float(confidence[i]) if i >= 0 and i < len(confidence) else 0.5
            exp_ret = float(predictions[0][i]) if predictions and i >= 0 and len(predictions[0]) > i else 0.0
            direction = "LONG" if kf >= 0 else "SHORT"
            lines.append(
                f"  {asset:10s} | {direction:5s} | E[R]={exp_ret*100:+.2f}% "
                f"| Kelly={kf*100:.1f}% | Conf={conf:.2f}"
            )
        lines.append("")
        if trades:
            lines.append("═" * 72)
            lines.append("TRADE RECOMMENDATIONS")
            lines.append("═" * 72)
            for t in trades:
                lines.append(
                    f"  {t['direction']} {t['asset']} | "
                    f"E[R]={t['expected_return']*100:+.2f}% | "
                    f"Kelly={t['kelly_size_pct']:.1f}% | {t['conviction']}"
                )
            lines.append("")
        lines.append("═" * 72)
        lines.append("DISCLAIMER: Research system. Bet small, respect chaos.")
        lines.append("═" * 72)
        formatted_text = "\n".join(lines)

        return {
            "timestamp": ts,
            "asset_names": asset_names,
            "system_version": "Prometheus v1.0 – Causal Brain",
            "system_state": system_state,
            "trade_recommendations": trades,
            "formatted_text": formatted_text,
            "trading_signal": trading_signal,
        }

    def _extract_trade_recommendations(
        self,
        causal_result: Optional[Dict],
        graph_result: Optional[Dict],
        kelly_result: Optional[Dict],
        volcano_stats: Optional[List[Dict]],
    ) -> List[Dict]:
        """Build list of specific trade recommendations."""
        trades = []

        if volcano_stats:
            for stat in sorted(volcano_stats, key=lambda x: abs(x.get("expected_return", 0)), reverse=True)[:5]:
                asset = stat.get("asset", "UNKNOWN")
                exp_ret = stat.get("expected_return", 0.0)
                p_pos = stat.get("p_positive", 0.5)
                var95 = stat.get("var_95", -0.05)

                if kelly_result:
                    fractions = kelly_result.get("kelly_fractions", [])
                    asset_idx = volcano_stats.index(stat) if stat in volcano_stats else 0
                    size = fractions[asset_idx] if asset_idx < len(fractions) else 0.0
                else:
                    size = 0.02 * (1 if exp_ret > 0 else -1)

                if abs(exp_ret) < 0.001 or p_pos < 0.45 and exp_ret > 0:
                    continue

                direction = "LONG" if exp_ret > 0 else "SHORT"
                trades.append({
                    "asset": asset,
                    "direction": direction,
                    "expected_return": float(exp_ret),
                    "probability_of_profit": float(p_pos if direction == "LONG" else 1 - p_pos),
                    "kelly_size_pct": float(abs(size) * 100),
                    "var_95": float(var95),
                    "conviction": "HIGH" if abs(exp_ret) > 0.02 and p_pos > 0.65 else "MODERATE",
                })

        # Override with graph signal
        if graph_result and graph_result.get("trading_signal") == "RESTRICTED":
            for t in trades:
                t["kelly_size_pct"] *= 0.3
                t["note"] = "SIZE REDUCED: market heart unstable"

        return trades[:5]  # top 5 recommendations

    def _build_system_state(
        self,
        neuromod: Optional[Dict],
        graph: Optional[Dict],
        htm: Optional[Dict],
    ) -> Dict:
        state = {
            "dopamine_level": neuromod.get("dopamine", 0.5) if neuromod else 0.5,
            "cortisol_level": neuromod.get("cortisol", 0.1) if neuromod else 0.1,
            "fear_mode_active": neuromod.get("fear_mode", False) if neuromod else False,
            "position_multiplier": neuromod.get("position_multiplier", 1.0) if neuromod else 1.0,
            "neuromod_recommendation": neuromod.get("recommendation", "NORMAL") if neuromod else "NORMAL",
            "market_stability": graph.get("market_stability", 0.5) if graph else 0.5,
            "heart_stable": graph.get("heart_stable", True) if graph else True,
            "systemic_risk": graph.get("systemic_risk_level", "MODERATE") if graph else "MODERATE",
            "htm_anomaly": htm.get("mean_anomaly", 0.0) if htm else 0.0,
            "order_flow_regime": "ANOMALOUS" if htm and htm.get("mean_anomaly", 0) > 0.5 else "NORMAL",
        }

        # Overall system confidence score
        confidence_factors = [
            (1 - state["cortisol_level"]) * 0.3,
            state["market_stability"] * 0.3,
            state["dopamine_level"] * 0.2,
            (1 - state["htm_anomaly"]) * 0.2,
        ]
        state["overall_confidence"] = float(sum(confidence_factors))
        return state

    def _build_narrative(
        self,
        causal_result: Optional[Dict],
        trades: List[Dict],
        graph_result: Optional[Dict],
        sentiment: Optional[Dict],
    ) -> str:
        """Build the God's Eye narrative summary."""
        lines = []

        if not trades:
            return "PROMETHEUS: Insufficient signal clarity for high-conviction positions. Hold cash."

        top_trade = trades[0]
        asset = top_trade["asset"]
        direction = top_trade["direction"]
        confidence = top_trade.get("conviction", "MODERATE")
        exp_ret = top_trade.get("expected_return", 0.0)
        p_profit = top_trade.get("probability_of_profit", 0.5)
        size_pct = top_trade.get("kelly_size_pct", 2.0)

        lines.append(f"PRIMARY SIGNAL: {direction} {asset}")
        lines.append(
            f"Expected return: {exp_ret*100:.2f}% | "
            f"Probability of profit: {p_profit*100:.0f}% | "
            f"Kelly size: {size_pct:.1f}% of portfolio"
        )

        # Causal chain explanation
        if causal_result and causal_result.get("attributions"):
            top_levers = causal_result["attributions"][:3]
            lever_text = " → ".join(
                f"{a['lever']} ({a['attribution_pct']:.0f}%)" for a in top_levers
            )
            lines.append(f"CAUSAL CHAIN: {lever_text} → {asset}")

        # Heart of market context
        if graph_result:
            heart = graph_result.get("heart_of_market", [])
            stability = graph_result.get("trading_signal", "UNKNOWN")
            if heart:
                lines.append(f"MARKET HEART: {', '.join(heart)} | Status: {stability}")

        # Sentiment
        if sentiment and sentiment.get("combined_sentiment"):
            sig = sentiment.get("signal", "NEUTRAL")
            lines.append(f"SENTIMENT: {sig} (score={sentiment['combined_sentiment']:.3f})")

        lines.append(f"CONVICTION: {confidence} | Confidence: {p_profit*100:.0f}%")
        return "\n".join(lines)

    def _build_risk_summary(
        self,
        kelly: Optional[Dict],
        neuromod: Optional[Dict],
        volcano_stats: Optional[List[Dict]],
    ) -> Dict:
        worst_var = min(
            (s.get("var_95", 0.0) for s in (volcano_stats or [])),
            default=-0.05,
        )
        return {
            "max_portfolio_var_95": float(worst_var),
            "total_long_exposure": kelly.get("total_long_exposure", 0.0) if kelly else 0.0,
            "total_short_exposure": kelly.get("total_short_exposure", 0.0) if kelly else 0.0,
            "net_exposure": kelly.get("net_exposure", 0.0) if kelly else 0.0,
            "n_active_positions": kelly.get("n_active_positions", 0) if kelly else 0,
            "position_multiplier": neuromod.get("position_multiplier", 1.0) if neuromod else 1.0,
            "risk_level": self._classify_risk(neuromod, worst_var),
        }

    def _find_dissenting_signals(
        self,
        causal: Optional[Dict],
        neuromod: Optional[Dict],
        graph: Optional[Dict],
        htm: Optional[Dict],
    ) -> List[str]:
        """Identify signals that contradict the primary recommendation."""
        dissent = []

        if neuromod and neuromod.get("fear_mode"):
            dissent.append(
                f"CORTISOL ALERT: Stress system triggered (level={neuromod['cortisol']:.2f}). "
                "Model is in defensive mode — do NOT override position caps."
            )

        if graph and graph.get("trading_signal") == "RESTRICTED":
            dissent.append(
                f"HEART INSTABILITY: Market heart is unstable "
                f"(stability={graph.get('market_stability', 0):.2f}). "
                "Wait for stabilization before entering new positions."
            )

        if htm and htm.get("mean_anomaly", 0) > 0.6:
            dissent.append(
                f"HTM ANOMALY: Order flow pattern is highly unusual "
                f"(score={htm['mean_anomaly']:.2f}). Potential regime change in progress."
            )

        if not dissent:
            dissent.append("No strong dissenting signals detected at this time.")

        return dissent

    def _classify_risk(self, neuromod: Optional[Dict], worst_var: float) -> str:
        if neuromod and neuromod.get("fear_mode"):
            return "EXTREME"
        if worst_var < -0.15:
            return "HIGH"
        if worst_var < -0.08:
            return "MODERATE"
        return "LOW"

    def _format_text(self, report: Dict) -> str:
        """Format the report as readable terminal/log output."""
        lines = [self.BANNER]
        ts = report["timestamp"]
        lines.append(f"Generated: {ts}\n")

        # System state
        ss = report["system_state"]
        lines.append("═" * 72)
        lines.append("SYSTEM STATE")
        lines.append("═" * 72)
        lines.append(f"  Dopamine (confidence): {ss['dopamine_level']:.3f}")
        lines.append(f"  Cortisol (stress):     {ss['cortisol_level']:.3f}")
        lines.append(f"  Fear Mode:             {'🚨 ACTIVE' if ss['fear_mode_active'] else '✓ OFF'}")
        lines.append(f"  Position Multiplier:   {ss['position_multiplier']:.2f}x")
        lines.append(f"  Market Stability:      {ss['market_stability']:.3f}")
        lines.append(f"  Systemic Risk:         {ss['systemic_risk']}")
        lines.append(f"  HTM Anomaly Score:     {ss['htm_anomaly']:.3f}")
        lines.append(f"  Overall Confidence:    {ss['overall_confidence']:.3f}")
        lines.append("")

        # Primary recommendation
        lines.append("═" * 72)
        lines.append("GOD'S EYE PRIMARY SIGNAL")
        lines.append("═" * 72)
        lines.append(report["primary_recommendation"])
        lines.append("")

        # Trade recommendations
        trades = report["trade_recommendations"]
        if trades:
            lines.append("═" * 72)
            lines.append("TRADE RECOMMENDATIONS")
            lines.append("═" * 72)
            for i, t in enumerate(trades, 1):
                lines.append(
                    f"  [{i}] {t['direction']} {t['asset']:20s} "
                    f"| E[R]={t['expected_return']*100:+.2f}% "
                    f"| P(profit)={t['probability_of_profit']*100:.0f}% "
                    f"| Size={t['kelly_size_pct']:.1f}% "
                    f"| {t['conviction']}"
                )
                if "note" in t:
                    lines.append(f"       NOTE: {t['note']}")
            lines.append("")

        # Risk
        risk = report["risk_summary"]
        lines.append("═" * 72)
        lines.append("RISK SUMMARY")
        lines.append("═" * 72)
        lines.append(f"  Portfolio VaR 95%: {risk['max_portfolio_var_95']*100:.1f}%")
        lines.append(f"  Long Exposure:     {risk['total_long_exposure']*100:.1f}%")
        lines.append(f"  Short Exposure:    {risk['total_short_exposure']*100:.1f}%")
        lines.append(f"  Net Exposure:      {risk['net_exposure']*100:.1f}%")
        lines.append(f"  Active Positions:  {risk['n_active_positions']}")
        lines.append(f"  Risk Level:        {risk['risk_level']}")
        lines.append("")

        # Dissent
        lines.append("═" * 72)
        lines.append("DISSENTING SIGNALS (READ BEFORE TRADING)")
        lines.append("═" * 72)
        for d in report["dissenting_signals"]:
            lines.append(f"  ⚠ {d}")
        lines.append("")

        lines.append("═" * 72)
        lines.append("DISCLAIMER: This is a research system. Bet small, respect chaos.")
        lines.append("═" * 72)

        return "\n".join(lines)

    def save_report(self, report: Dict, path: str) -> None:
        """Save report to JSON and text files."""
        with open(path, "w") as f:
            # Exclude non-serializable numpy arrays
            clean = {k: v for k, v in report.items()
                     if k not in ("causal_attribution",)}
            json.dump(clean, f, indent=2, default=str)
        with open(path.replace(".json", ".txt"), "w") as f:
            f.write(report.get("formatted_text", ""))
