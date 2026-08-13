"""
One-command AWS training launcher.

Run from your LOCAL machine:
  python scripts/aws_train.py launch --bucket my-bucket --repo https://github.com/you/Prometheus
  python scripts/aws_train.py wait   --bucket my-bucket --job job-20260812T190000Z
  python scripts/aws_train.py pull   --bucket my-bucket --tag latest
  python scripts/aws_train.py cost   --instance g4dn.xlarge --hours 8
  python scripts/aws_train.py ls     --bucket my-bucket
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("prometheus.aws")


def cmd_launch(args):
    from prometheus.cloud import CheckpointStore, RemoteTrainer

    # Push config to S3 first so the remote job uses it
    store = CheckpointStore(bucket=args.bucket, prefix=args.prefix, region=args.region)
    if Path("configs/default.yaml").exists():
        logger.info("Syncing config to S3...")
        store.sync_config("configs/default.yaml")

    trainer = RemoteTrainer(
        bucket=args.bucket,
        repo_url=args.repo,
        branch=args.branch,
        prefix=args.prefix,
        region=args.region,
        key_pair=args.key_pair,
        security_group=args.security_group,
        iam_instance_profile=args.iam_profile,
        subnet_id=args.subnet,
    )

    job = trainer.launch(
        instance_type=args.instance,
        train_mode=args.mode,
        n_assets=args.n_assets,
        seq_len=args.seq_len,
        horizon=args.horizon,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        n_black_swans=args.n_black_swans,
        spot_price_limit=args.spot_max_price,
        volume_gb=args.volume_gb,
    )

    print("\n" + "=" * 60)
    print("JOB LAUNCHED")
    print("=" * 60)
    for k, v in job.items():
        print(f"  {k:25s} {v}")
    print("=" * 60)
    print(f"\nTo stream logs:")
    print(f"  python scripts/aws_train.py wait --bucket {args.bucket} --job {job['job_id']}")
    print(f"\nTo pull checkpoint when done:")
    print(f"  python scripts/aws_train.py pull --bucket {args.bucket} --tag {job['job_id']}")

    # Save job info locally
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{job['job_id']}.json", "w") as f:
        json.dump(job, f, indent=2)
    print(f"\nJob info saved: logs/{job['job_id']}.json")


def cmd_wait(args):
    from prometheus.cloud import RemoteTrainer
    trainer = RemoteTrainer(
        bucket=args.bucket, repo_url="", prefix=args.prefix, region=args.region
    )
    ok = trainer.wait(
        job_id=args.job,
        stream_logs=not args.no_stream,
        poll_interval=args.poll,
        timeout_hours=args.timeout,
    )
    if ok:
        print(f"\nJob {args.job} completed. Run:")
        print(f"  python scripts/aws_train.py pull --bucket {args.bucket} --tag {args.job}")
    else:
        print(f"\nJob {args.job} may have failed or timed out. Check logs:")
        print(f"  aws s3 cp s3://{args.bucket}/{args.prefix}/logs/{args.job}/train.log -")
    sys.exit(0 if ok else 1)


def cmd_pull(args):
    from prometheus.cloud import CheckpointStore
    store = CheckpointStore(bucket=args.bucket, prefix=args.prefix, region=args.region)

    local_dir = args.local_dir or "checkpoints"
    logger.info("Pulling checkpoint tag='%s' → %s/", args.tag, local_dir)
    key = store.pull(local_dir, tag=args.tag)
    print(f"\nCheckpoint downloaded: {local_dir}/")
    print(f"S3 key: {key}")
    print(f"\nTo analyze with local machine:")
    print(f"  python scripts/analyze.py --checkpoint {local_dir} --symbols SPY QQQ GLD")


def cmd_push(args):
    from prometheus.cloud import CheckpointStore
    store = CheckpointStore(bucket=args.bucket, prefix=args.prefix, region=args.region)
    local_dir = args.local_dir or "checkpoints"
    uri = store.push(local_dir, tag=args.tag)
    print(f"Pushed: {uri}")


def cmd_ls(args):
    from prometheus.cloud import CheckpointStore
    store = CheckpointStore(bucket=args.bucket, prefix=args.prefix, region=args.region)
    checkpoints = store.list_checkpoints(tag=args.tag)
    if not checkpoints:
        print("No checkpoints found.")
        return
    print(f"\n{'TAG':20s} {'TIMESTAMP':20s} {'SIZE':10s} {'S3 URI'}")
    print("-" * 100)
    for c in checkpoints:
        print(f"{c['tag']:20s} {c['timestamp']:20s} {c['size_mb']:8.1f} MB  {c['s3_uri']}")


def cmd_cost(args):
    from prometheus.cloud import RemoteTrainer
    trainer = RemoteTrainer(bucket="", repo_url="", region=args.region)
    est = trainer.estimate_cost(instance_type=args.instance, duration_hours=args.hours)
    print("\nCost Estimate")
    print("=" * 50)
    for k, v in est.items():
        print(f"  {k:25s} {v}")


def main():
    p = argparse.ArgumentParser(description="Prometheus AWS Training CLI")
    p.add_argument("--bucket", help="S3 bucket name")
    p.add_argument("--prefix", default="prometheus", help="S3 key prefix")
    p.add_argument("--region", default="us-east-1")

    sub = p.add_subparsers(dest="command", required=True)

    # launch
    sl = sub.add_parser("launch", help="Launch training on EC2 Spot")
    sl.add_argument("--repo", required=True, help="GitHub HTTPS URL")
    sl.add_argument("--branch", default="claude/prometheus-causal-market-ol2pau")
    sl.add_argument("--instance", default="g4dn.xlarge")
    sl.add_argument("--mode", default="full", choices=["pretrain", "finetune", "meta", "evolve", "full"])
    sl.add_argument("--n-assets", type=int, default=20)
    sl.add_argument("--seq-len", type=int, default=64)
    sl.add_argument("--horizon", type=int, default=5)
    sl.add_argument("--d-model", type=int, default=256)
    sl.add_argument("--n-heads", type=int, default=8)
    sl.add_argument("--n-layers", type=int, default=6)
    sl.add_argument("--pretrain-epochs", type=int, default=10)
    sl.add_argument("--finetune-epochs", type=int, default=50)
    sl.add_argument("--n-black-swans", type=int, default=2000)
    sl.add_argument("--spot-max-price", default=None)
    sl.add_argument("--volume-gb", type=int, default=100)
    sl.add_argument("--key-pair", default=None)
    sl.add_argument("--security-group", default=None)
    sl.add_argument("--iam-profile", default="PrometheusEC2Role")
    sl.add_argument("--subnet", default=None)

    # wait
    sw = sub.add_parser("wait", help="Wait for a job and stream logs")
    sw.add_argument("--job", required=True)
    sw.add_argument("--no-stream", action="store_true")
    sw.add_argument("--poll", type=int, default=30, help="Poll interval (seconds)")
    sw.add_argument("--timeout", type=float, default=12.0, help="Timeout (hours)")

    # pull
    spl = sub.add_parser("pull", help="Download checkpoint from S3")
    spl.add_argument("--tag", default="latest")
    spl.add_argument("--local-dir", default=None)

    # push
    sph = sub.add_parser("push", help="Upload local checkpoint to S3")
    sph.add_argument("--tag", default="latest")
    sph.add_argument("--local-dir", default=None)

    # ls
    sls = sub.add_parser("ls", help="List S3 checkpoints")
    sls.add_argument("--tag", default=None)

    # cost
    sc = sub.add_parser("cost", help="Estimate training cost")
    sc.add_argument("--instance", default="g4dn.xlarge")
    sc.add_argument("--hours", type=float, default=8.0)

    args = p.parse_args()
    dispatch = {
        "launch": cmd_launch,
        "wait": cmd_wait,
        "pull": cmd_pull,
        "push": cmd_push,
        "ls": cmd_ls,
        "cost": cmd_cost,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
