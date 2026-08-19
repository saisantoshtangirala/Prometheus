"""
Prometheus Training Script.

Usage:
  python scripts/train.py --mode pretrain    # black-swan pre-training (1 week budget)
  python scripts/train.py --mode finetune    # fine-tune on real market data
  python scripts/train.py --mode meta        # MAML meta-training across regimes
  python scripts/train.py --mode evolve      # nightly NEAT architecture evolution
  python scripts/train.py --mode full        # full pipeline (pretrain → finetune → meta)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prometheus.train")


def parse_args():
    p = argparse.ArgumentParser(description="Prometheus Training")
    p.add_argument("--mode", choices=["pretrain", "finetune", "meta", "evolve", "full"],
                   default="pretrain")
    p.add_argument("--n-assets", type=int, default=20)
    p.add_argument("--tickers", default=None,
                   help="comma-separated ticker list to fetch for the real-data "
                        "phases (run_finetune), e.g. 'RELIANCE.NS,TCS.NS,...'. "
                        "Overrides MarketDataFetcher's default combined US+India "
                        "universe. Must match the live trading ticker list, in "
                        "the same order, for a produced checkpoint to load "
                        "correctly into ReflexArc.snn (kronos/orchestrator.py).")
    p.add_argument("--snn-layer-sizes", default="32,16",
                   help="comma-separated SNN hidden layer sizes, e.g. '32,16'. "
                        "Defaults to kronos/reflex.py ReflexArc's hardcoded "
                        "shape (not PrometheusEngine's own general-purpose "
                        "default of [128, 64]) since this script's checkpoint "
                        "output is loaded straight into ReflexArc.snn and "
                        "must match its architecture exactly to be adopted.")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--pretrain-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--meta-epochs", type=int, default=10)
    p.add_argument("--n-black-swans", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def run_pretrain(engine: PrometheusEngine, args) -> None:
    """Phase 1: Pre-train on synthetic black-swan chaos."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Black-Swan Pre-Training")
    logger.info("  Generating %d doomsday scenarios...", args.n_black_swans)
    logger.info("=" * 60)

    def on_epoch(epoch, stats):
        logger.info("  [Pretrain] Epoch %d | Loss: %.4f", epoch, stats["mean_loss"])

    history = engine.train_on_black_swans(
        n_scenarios=args.n_black_swans,
        n_epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        on_epoch_end=on_epoch,
    )
    engine.save(f"{args.checkpoint_dir}/pretrain")
    logger.info("Pre-training complete. Checkpoint saved.")
    return history


def run_finetune(engine: PrometheusEngine, args) -> None:
    """Phase 2: Fine-tune on real market data."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Real Market Data Fine-Tuning")
    logger.info("=" * 60)

    fetcher = MarketDataFetcher()
    explicit_tickers = args.tickers.split(",") if args.tickers else None
    raw_data = fetcher.fetch_all(tickers=explicit_tickers)
    if raw_data.empty:
        logger.warning("Could not fetch real market data — using synthetic fallback")
        raw_data = MarketDataFetcher._synthetic_data(
            [f"ASSET_{i}" for i in range(args.n_assets)]
        )
        returns = fetcher.get_returns(raw_data)
    else:
        returns = fetcher.get_returns(raw_data)

    # Prepare training tensors
    ret_array = returns.fillna(0).values
    n_bars, n_cols = ret_array.shape
    n_assets = min(args.n_assets, n_cols)
    ret_array = ret_array[:, :n_assets]

    engine.config.n_assets = n_assets
    engine.causal_transformer.eval()
    engine.ltc.eval()

    history = []
    for epoch in range(args.finetune_epochs):
        epoch_losses = []
        snn_epoch_losses = []
        # Rolling window mini-batches
        for start in range(0, n_bars - args.seq_len - args.horizon, args.seq_len):
            end = start + args.seq_len
            x_np = ret_array[start:end]
            y_np = ret_array[end:end + args.horizon]

            if x_np.shape[0] < 4 or y_np.shape[0] < 1:
                continue

            x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(args.device)
            y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(0).to(args.device)

            # Reshape y to match prediction shape [B, horizon, n_assets]
            if y.dim() == 2:
                y = y.unsqueeze(0)

            step_result = engine.train_step(x, y)
            epoch_losses.append(step_result["loss"])

            # self.snn (ReflexArc's live model, kronos/reflex.py) is a
            # separate network from causal_transformer/ltc above and needs
            # its own training step - see PrometheusEngine.train_snn_step.
            # ReflexArc.infer() feeds it a returns window and reads a
            # single next-tick signal back out, so its target is one bar
            # ahead (y's first horizon step), not the full horizon.
            snn_result = engine.train_snn_step(x, y[:, 0, :])
            snn_epoch_losses.append(snn_result["loss"])

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        dir_acc = step_result.get("directional_accuracy", 0.0) if epoch_losses else 0.0
        snn_mean_loss = float(np.mean(snn_epoch_losses)) if snn_epoch_losses else 0.0
        snn_dir_acc = snn_result.get("directional_accuracy", 0.0) if snn_epoch_losses else 0.0
        history.append({
            "epoch": epoch, "loss": mean_loss, "dir_acc": dir_acc,
            "snn_loss": snn_mean_loss, "snn_dir_acc": snn_dir_acc,
        })
        logger.info(
            "  [Finetune] Epoch %d | Loss: %.4f | DirAcc: %.2f%% | "
            "SNN Loss: %.4f | SNN DirAcc: %.2f%%",
            epoch, mean_loss, dir_acc * 100, snn_mean_loss, snn_dir_acc * 100,
        )

    engine.save(f"{args.checkpoint_dir}/finetune")
    return history


def run_meta_training(engine: PrometheusEngine, args) -> None:
    """Phase 3: MAML meta-training across synthetic regimes."""
    logger.info("=" * 60)
    logger.info("PHASE 3: MAML Meta-Learning")
    logger.info("  Learning to adapt in 3 gradient steps...")
    logger.info("=" * 60)

    # Generate tasks from different synthetic regimes
    n = args.n_assets
    seq = args.seq_len
    h = args.horizon
    tasks = []

    for regime_type in ["trending", "mean_reverting", "volatile", "crash", "recovery"]:
        for _ in range(args.meta_epochs):
            if regime_type == "trending":
                drift = np.random.uniform(0.001, 0.003)
                data = np.cumsum(np.random.normal(drift, 0.01, (seq + h, n)), axis=0)
            elif regime_type == "mean_reverting":
                data = np.random.normal(0, 0.01, (seq + h, n))
                for t in range(1, seq + h):
                    data[t] = 0.9 * data[t - 1] + np.random.normal(0, 0.005, n)
            elif regime_type == "volatile":
                data = np.random.normal(0, 0.05, (seq + h, n))
            elif regime_type == "crash":
                data = np.random.normal(-0.02, 0.03, (seq + h, n))
            else:
                data = np.random.normal(0.01, 0.02, (seq + h, n))

            x = torch.tensor(data[:seq], dtype=torch.float32).unsqueeze(0).to(args.device)
            y = torch.tensor(data[seq:], dtype=torch.float32).unsqueeze(0).to(args.device)
            tasks.append(((x, y), (x, y)))  # support, query (simplified)

    meta_losses = []
    for i in range(0, len(tasks), 5):
        batch_tasks = tasks[i:i + 5]
        meta_loss = engine.maml.meta_train_step(
            tasks=batch_tasks,
            loss_fn=lambda p, t: engine.loss_fn(p, t),
        )
        meta_losses.append(meta_loss)
        if i % 10 == 0:
            logger.info("  [MAML] Batch %d | Meta-loss: %.4f", i // 5, meta_loss)

    engine.save(f"{args.checkpoint_dir}/meta")
    logger.info("Meta-training complete. Mean meta-loss: %.4f", np.mean(meta_losses))


def run_evolution(engine: PrometheusEngine, args) -> None:
    """Phase 4: Nightly NEAT architecture evolution."""
    logger.info("=" * 60)
    logger.info("PHASE 4: NEAT Architecture Evolution")
    logger.info("  Evolving model topology (population=30, generations=10)...")
    logger.info("=" * 60)

    n = args.n_assets
    seq = args.seq_len
    h = args.horizon

    # Validation dataset (synthetic for now)
    np.random.seed(99)
    X_val = torch.randn(50, seq * n).to(args.device)
    y_val = torch.randn(50, h * n).to(args.device)

    engine.neat.population_size = 20
    engine.neat.n_generations = 5

    best_model, best_genome = engine.neat.build_best_model(
        val_data=(X_val, y_val),
        loss_fn=lambda p, t: torch.nn.functional.mse_loss(p, t),
    )

    logger.info("Best genome: %s", best_genome.to_dict())
    engine.neat.save_evolution_history(f"{args.output_dir}/evolution_history.json")
    logger.info("Evolution complete.")


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = PrometheusConfig(
        n_assets=args.n_assets,
        seq_len=args.seq_len,
        horizon=args.horizon,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        device=args.device,
        output_dir=args.output_dir,
        # Must match kronos/reflex.py's ReflexArc.snn exactly (layer_sizes
        # and output_size == n_assets, not PrometheusEngine's own default
        # of n_assets // 2) - this checkpoint's only consumer is Kronos's
        # maybe_adopt_runpod_checkpoint(), and a shape mismatch there fails
        # closed (logs a warning, keeps training from scratch) rather than
        # loudly, so a silent drift here would go unnoticed indefinitely.
        snn_layer_sizes=[int(x) for x in args.snn_layer_sizes.split(",")],
        snn_output_size=args.n_assets,
    )

    logger.info("Initializing Prometheus Engine...")
    engine = PrometheusEngine(cfg)

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        engine.load(args.resume)

    if args.mode == "pretrain":
        run_pretrain(engine, args)
    elif args.mode == "finetune":
        run_finetune(engine, args)
    elif args.mode == "meta":
        run_meta_training(engine, args)
    elif args.mode == "evolve":
        run_evolution(engine, args)
    elif args.mode == "full":
        run_pretrain(engine, args)
        run_finetune(engine, args)
        run_meta_training(engine, args)
        run_evolution(engine, args)
        logger.info("Full training pipeline complete!")

    logger.info("Done. Checkpoints saved to: %s", args.checkpoint_dir)


if __name__ == "__main__":
    main()
