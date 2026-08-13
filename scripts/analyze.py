"""
Prometheus Analysis Script – Generate God's Eye Report for a symbol or portfolio.

Usage:
  python scripts/analyze.py --symbols SPY QQQ GLD TLT --checkpoint checkpoints/finetune
  python scripts/analyze.py --symbols SPY --volcano --save-html output/volcano_SPY.html
  python scripts/analyze.py --config configs/mac_mini.yaml --symbols SPY QQQ GLD
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


def _load_yaml_config(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")


def parse_args():
    p = argparse.ArgumentParser(description="Prometheus God's Eye Analysis")
    p.add_argument("--config", default=None, help="YAML config file (e.g. configs/mac_mini.yaml)")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--volcano", action="store_true")
    p.add_argument("--save-html", default=None)
    p.add_argument("--save-report", default="output/gods_eye_report.json")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Defaults (overridden by --config, then by explicit CLI flags)
    symbols = ["SPY", "QQQ", "GLD", "TLT"]
    seq_len = 64
    horizon = 5
    device = "cpu"
    d_model, n_heads, n_layers = 128, 4, 4

    if args.config:
        cfg_yaml = _load_yaml_config(args.config)
        sys_cfg = cfg_yaml.get("system", {})
        model_cfg = cfg_yaml.get("model", {})
        data_cfg = cfg_yaml.get("data", {})
        device = sys_cfg.get("device", device)
        symbols = data_cfg.get("symbols", symbols)
        seq_len = model_cfg.get("seq_len", seq_len)
        horizon = model_cfg.get("horizon", horizon)
        d_model = model_cfg.get("d_model", d_model)
        n_heads = model_cfg.get("n_heads", n_heads)
        n_layers = model_cfg.get("n_layers", n_layers)

    # CLI flags take precedence over config file
    if args.symbols:
        symbols = args.symbols
    if args.seq_len:
        seq_len = args.seq_len
    if args.horizon:
        horizon = args.horizon
    if args.device:
        device = args.device

    # Auto-detect MPS if device not explicitly set and no config provided
    if not args.config and not args.device:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"

    n_assets = len(symbols)
    os.makedirs("output", exist_ok=True)

    logger.info("Device: %s | Assets: %d | seq_len: %d", device, n_assets, seq_len)

    cfg = PrometheusConfig(
        n_assets=n_assets,
        seq_len=seq_len,
        horizon=horizon,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        device=device,
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
