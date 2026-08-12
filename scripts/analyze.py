"""
Prometheus Analysis Script – Generate God's Eye Report for a symbol or portfolio.

Usage:
  python scripts/analyze.py --symbols SPY QQQ GLD TLT --checkpoint checkpoints/finetune
  python scripts/analyze.py --symbols SPY --volcano --save-html output/volcano_SPY.html
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from prometheus.engine import PrometheusEngine, PrometheusConfig
from prometheus.data import MarketDataFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prometheus.analyze")


def parse_args():
    p = argparse.ArgumentParser(description="Prometheus God's Eye Analysis")
    p.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--volcano", action="store_true")
    p.add_argument("--save-html", default=None)
    p.add_argument("--save-report", default="output/gods_eye_report.json")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    n_assets = len(args.symbols)
    os.makedirs("output", exist_ok=True)

    cfg = PrometheusConfig(
        n_assets=n_assets,
        seq_len=args.seq_len,
        horizon=args.horizon,
        d_model=128,
        n_heads=4,
        n_layers=4,
        device=args.device,
    )
    engine = PrometheusEngine(cfg)

    if args.checkpoint:
        logger.info("Loading checkpoint: %s", args.checkpoint)
        engine.load(args.checkpoint)

    # Fetch data
    logger.info("Fetching market data for: %s", args.symbols)
    fetcher = MarketDataFetcher()
    raw = fetcher.fetch_all(tickers=args.symbols)

    if raw.empty:
        logger.warning("No real data — using synthetic data")
        returns = np.random.normal(0, 0.01, (args.seq_len + 10, n_assets))
    else:
        ret_df = fetcher.get_returns(raw).fillna(0)
        # Select Close columns for symbols
        if hasattr(ret_df.columns, "levels"):
            try:
                ret_df = ret_df.xs("Close", axis=1, level=0)
            except Exception:
                pass
        returns = ret_df.values[-args.seq_len:, :n_assets]
        if returns.shape[0] < args.seq_len:
            pad = np.zeros((args.seq_len - returns.shape[0], n_assets))
            returns = np.vstack([pad, returns])

    # Update graph engine with correct asset names
    engine.hive_mind.asset_names = args.symbols

    # Run full analysis
    logger.info("Running Prometheus analysis...")
    report = engine.analyze(
        market_data=returns,
        asset_names=args.symbols,
    )

    # Print God's Eye report
    print(report["formatted_text"])

    # Save report
    engine.report_gen.save_report(report, args.save_report)
    logger.info("Report saved to: %s", args.save_report)

    # Probability volcano visualization
    if args.volcano and report.get("mc_paths") is not None:
        mc_paths = report["mc_paths"]
        logger.info("Rendering Probability Volcano...")
        try:
            html_path = args.save_html or f"output/volcano_{args.symbols[0]}.html"
            engine.volcano.render_html(mc_paths, 0, args.symbols[0], html_path)
            logger.info("Volcano saved to: %s", html_path)
        except Exception as e:
            logger.warning("Could not render volcano: %s", e)


if __name__ == "__main__":
    main()
