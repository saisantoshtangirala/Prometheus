"""
Kronos "God's Eye" Reporter - 06:00 daily markdown briefing.

Summarizes portfolio exposure (Kelly cap enforced), expected volatility,
top predicted movers, and a plain-English trading note for the day ahead.
Written to logs/reports/GodsEye_<date>.md and also as GodsEye.md (latest).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class GodsEyeReporter:
    """Daily markdown report generator."""

    def __init__(self, config):
        self.cfg = config
        self.report_dir = config.orchestrator.report_dir
        self.last_report_md: Optional[str] = None   # REP-03 in-memory fallback
        try:
            os.makedirs(self.report_dir, exist_ok=True)
        except OSError as e:
            logger.error("[reporter] cannot create report dir: %s", e)

    @staticmethod
    def sanitize_tickers(text: str, valid_tickers: List[str]) -> str:
        """
        REP-02: strip hallucinated ticker mentions. Any $TICKER token whose
        symbol is not in the portfolio is removed from the summary text.
        Bare uppercase words are left alone (too many false positives);
        the $-prefixed convention is the contract for ticker mentions.
        """
        import re
        valid = set(valid_tickers)

        def _replace(match):
            symbol = match.group(1)
            return f"${symbol}" if symbol in valid else ""

        cleaned = re.sub(r"\$([A-Z]{1,5})\b", _replace, text)
        return re.sub(r"\s{2,}", " ", cleaned).strip()

    # -- computation helpers -------------------------------------------------

    def _expected_volatility(self, memory) -> Dict[str, float]:
        out = {}
        for window in self.cfg.reporting.volatility_windows:
            rets = memory.returns.tail(int(window))
            out[f"{window}d"] = float(rets.std().mean() * np.sqrt(252) * 100)
        return out

    def _top_movers(
        self, memory, signals: Optional[np.ndarray]
    ) -> List[Dict]:
        tickers = memory.tickers
        if signals is None or len(signals) < len(tickers):
            # fall back to momentum ranking
            signals = memory.returns.tail(5).mean().values
        k = int(self.cfg.reporting.top_movers)
        order = np.argsort(-np.abs(signals[: len(tickers)]))[:k]
        return [
            {
                "ticker": tickers[i],
                "direction": "LONG" if signals[i] > 0 else "SHORT",
                "signal": float(signals[i]),
            }
            for i in order
        ]

    def _human_summary(
        self, memory, vol: Dict[str, float], regime: str, position_cap: float
    ) -> str:
        vix = memory.macro.get("vix_last", 20.0)
        vix_mean = memory.macro.get("vix_mean_20d", 20.0)
        lines = []
        if vix > vix_mean * 1.25:
            lines.append(
                f"VIX at {vix:.1f} is {vix / (vix_mean + 1e-9):.1f}x its 20-day "
                "average. Expect elevated volatility around the open; reduce "
                "position sizing by 20% until the first hour settles."
            )
        elif regime == "panic":
            lines.append(
                "Reflex gate is in PANIC - no new long positions until the "
                "lockout expires. Focus on risk reduction only."
            )
        else:
            lines.append(
                f"Volatility regime is normal (VIX {vix:.1f}). Standard Kelly "
                f"sizing applies, capped at "
                f"{self.cfg.trading.max_position_pct:.0%} per position."
            )
        if position_cap < 1.0:
            lines.append(
                f"Active position cap: {position_cap:.0%} of normal sizing."
            )
        drift = memory.macro.get("market_return_1d", 0.0)
        if abs(drift) > 0.01:
            direction = "up" if drift > 0 else "down"
            lines.append(
                f"Yesterday closed {direction} {abs(drift):.1%} across the book - "
                "watch for continuation vs. mean-reversion in the first 30 minutes."
            )
        return " ".join(lines)

    # -- report --------------------------------------------------------------

    def generate(
        self,
        day: int,
        memory,
        trader,
        signals: Optional[np.ndarray] = None,
        regime: str = "calm",
        position_cap: float = 1.0,
        evolution_summary: Optional[Dict] = None,
        warmup_summary: Optional[Dict] = None,
    ) -> str:
        """Build and write the markdown report. Returns the file path."""
        now = datetime.now(timezone.utc)
        vol = self._expected_volatility(memory)
        movers = self._top_movers(memory, signals)
        equity = trader.equity() if trader else self.cfg.trading.initial_capital

        exposure_lines = []
        if trader:
            for ticker in sorted(trader.positions):
                pct = trader.position_pct(ticker)
                if pct > 0.001:
                    exposure_lines.append(
                        f"| {ticker} | {trader.positions[ticker]:+.1f} | {pct:.1%} |"
                    )
        # REP-01: an explicitly neutral statement on zero-activity days -
        # and no Sharpe division anywhere (close_day already guards std=0).
        exposure_table = (
            "| Ticker | Shares | % of Equity |\n|---|---|---|\n"
            + "\n".join(exposure_lines)
            if exposure_lines else
            "_No trading opportunities identified. Position: Neutral._"
        )

        movers_lines = "\n".join(
            f"{i + 1}. **{m['ticker']}** - {m['direction']} "
            f"(signal {m['signal']:+.3f})"
            for i, m in enumerate(movers)
        )

        vol_lines = ", ".join(
            f"{k}: {v:.1f}% annualized" for k, v in vol.items()
        )

        evo_block = ""
        if evolution_summary:
            evo_block = (
                f"\n## Overnight Evolution\n"
                f"- Population: {evolution_summary.get('population_size', '?')}"
                f" ({'degraded' if evolution_summary.get('degraded') else 'full'})\n"
                f"- Top-{len(evolution_summary.get('top_fitness', []))} fitness: "
                f"{['%.3f' % f for f in evolution_summary.get('top_fitness', [])]}\n"
            )
        warm_block = ""
        if warmup_summary:
            warm_block = (
                f"\n## MAML Warm-up\n"
                f"- Regime estimate: **{warmup_summary.get('regime', '?')}**\n"
                f"- Inner losses: "
                f"{['%.5f' % l for l in warmup_summary.get('inner_losses', [])]}\n"
            )

        summary = self.sanitize_tickers(
            self._human_summary(memory, vol, regime, position_cap),
            memory.tickers,
        )

        md = f"""# God's Eye Report - Day {day}

_Generated {now.strftime('%Y-%m-%d %H:%M UTC')} | data source: {memory.source_used}_

## Human Summary

> {summary}

## Portfolio

- **Equity:** ${equity:,.2f}
- **Kelly cap:** {self.cfg.trading.max_position_pct:.0%} per position (enforced)
- **Reflex gate:** {regime} (cap {position_cap:.0%})

{exposure_table}

## Expected Volatility

{vol_lines}

## Top {len(movers)} Predicted Movers

{movers_lines}
{evo_block}{warm_block}
## Data Quality

- Source used: `{memory.source_used}`
- Flags: {memory.quality_flags if memory.quality_flags else 'none'}
"""

        # REP-03: a full/read-only disk must not kill the orchestration -
        # keep the report in memory and carry on.
        self.last_report_md = md
        date_str = now.strftime("%Y%m%d")
        path = os.path.join(self.report_dir, f"GodsEye_{date_str}_day{day}.md")
        try:
            with open(path, "w") as f:
                f.write(md)
            latest = os.path.join(self.report_dir, "GodsEye.md")
            with open(latest, "w") as f:
                f.write(md)
        except OSError as e:
            logger.error(
                "[reporter] cannot write report to disk (%s) - "
                "report retained in memory buffer", e,
            )
            return "<in-memory>"
        logger.info("[reporter] God's Eye written: %s", path)
        return path
