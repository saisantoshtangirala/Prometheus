"""
Remote trainer – launches and monitors training jobs on AWS EC2 Spot instances.

Flow:
  local machine calls RemoteTrainer.launch()
    → starts an EC2 Spot instance with a startup script
    → script pulls the repo, runs scripts/train.py, pushes checkpoints to S3
    → RemoteTrainer.wait() polls CloudWatch logs and streams them to stdout
    → when complete, call CheckpointStore.pull() to grab the weights

Supported instance types (choose based on budget):
  g4dn.xlarge  – T4 GPU  16 GB VRAM,  4 vCPU,  ~$0.16/hr spot  ← default
  g4dn.2xlarge – T4 GPU  16 GB VRAM,  8 vCPU,  ~$0.23/hr spot
  p3.2xlarge   – V100    16 GB VRAM,  8 vCPU,  ~$0.90/hr spot
  p3.8xlarge   – 4×V100  64 GB VRAM, 32 vCPU,  ~$3.60/hr spot (full training)
  g5.2xlarge   – A10G    24 GB VRAM,  8 vCPU,  ~$0.34/hr spot
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Deep Learning AMI (GPU) — Amazon Linux 2, us-east-1
# Check https://aws.amazon.com/machine-learning/amis/ for your region
DEFAULT_AMI = {
    "us-east-1": "ami-0a2928e1c02fe0b30",   # DLAMI AL2 PyTorch 2.1
    "us-west-2": "ami-0c5f63e5a4e9c7b8e",
    "eu-west-1": "ami-0e3a6b2b93f4a1c7d",
}

STARTUP_SCRIPT_TEMPLATE = """\
#!/bin/bash
set -e
exec > /var/log/prometheus-train.log 2>&1

echo "[$(date)] Starting Prometheus training job: {job_id}"

# Activate PyTorch conda environment (DLAMI default)
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

# Clone or update the repository
if [ -d /opt/prometheus ]; then
    cd /opt/prometheus && git pull origin {branch}
else
    git clone {repo_url} /opt/prometheus
    cd /opt/prometheus && git checkout {branch}
fi

cd /opt/prometheus

# Install dependencies
pip install -e . -q
pip install boto3 awscli -q

# Pull scenario library from S3 if it exists (saves regeneration time)
aws s3 sync s3://{bucket}/{prefix}/scenarios/ data/scenarios/ --quiet || true

# Run training
python scripts/train.py {train_args} 2>&1 | tee /var/log/prometheus-job.log

echo "[$(date)] Training complete — pushing checkpoint to S3"

# Push checkpoint
python -c "
from prometheus.cloud import CheckpointStore
store = CheckpointStore(bucket='{bucket}', prefix='{prefix}')
store.push('checkpoints', tag='{job_id}')
store.push('checkpoints', tag='latest')
store.push_scenarios()
print('Checkpoint pushed to S3.')
"

# Signal completion
aws s3 cp /var/log/prometheus-job.log s3://{bucket}/{prefix}/logs/{job_id}/train.log
echo "[$(date)] Job {job_id} complete."

# Self-terminate to stop billing
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region {region}
"""


class RemoteTrainer:
    """
    Launches a Spot EC2 training job, streams logs, auto-terminates on completion.

    Example:
        trainer = RemoteTrainer(
            bucket="my-prometheus-bucket",
            repo_url="https://github.com/you/Prometheus",
            key_pair="my-key",
            security_group="sg-xxxxxx",
        )
        job = trainer.launch(
            instance_type="g4dn.xlarge",
            train_mode="full",
            n_assets=20,
            pretrain_epochs=10,
        )
        trainer.wait(job["job_id"], stream_logs=True)
    """

    def __init__(
        self,
        bucket: str,
        repo_url: str,
        branch: str = "claude/prometheus-causal-market-ol2pau",
        prefix: str = "prometheus",
        region: str = "us-east-1",
        key_pair: Optional[str] = None,
        security_group: Optional[str] = None,
        iam_instance_profile: Optional[str] = None,
        subnet_id: Optional[str] = None,
    ):
        self.bucket = bucket
        self.repo_url = repo_url
        self.branch = branch
        self.prefix = prefix
        self.region = region
        self.key_pair = key_pair
        self.security_group = security_group
        self.iam_profile = iam_instance_profile
        self.subnet_id = subnet_id
        self._ec2 = None
        self._logs = None

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(
        self,
        instance_type: str = "g4dn.xlarge",
        train_mode: str = "full",
        n_assets: int = 20,
        seq_len: int = 64,
        horizon: int = 5,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        pretrain_epochs: int = 10,
        finetune_epochs: int = 50,
        n_black_swans: int = 2000,
        spot_price_limit: Optional[str] = None,  # e.g. "0.25" for max $0.25/hr
        ami_id: Optional[str] = None,
        volume_gb: int = 100,
    ) -> Dict:
        """
        Launch a Spot EC2 instance, run training, auto-terminate when done.

        Returns:
            dict with job_id, instance_id, spot_request_id
        """
        ec2 = self._get_ec2()
        job_id = datetime.now(timezone.utc).strftime("job-%Y%m%dT%H%M%S")

        train_args = (
            f"--mode {train_mode} "
            f"--n-assets {n_assets} "
            f"--seq-len {seq_len} "
            f"--horizon {horizon} "
            f"--d-model {d_model} "
            f"--n-heads {n_heads} "
            f"--n-layers {n_layers} "
            f"--pretrain-epochs {pretrain_epochs} "
            f"--finetune-epochs {finetune_epochs} "
            f"--n-black-swans {n_black_swans} "
            f"--device cuda"
        )

        startup = STARTUP_SCRIPT_TEMPLATE.format(
            job_id=job_id,
            branch=self.branch,
            repo_url=self.repo_url,
            bucket=self.bucket,
            prefix=self.prefix,
            region=self.region,
            train_args=train_args,
        )
        user_data = base64.b64encode(startup.encode()).decode()

        ami = ami_id or DEFAULT_AMI.get(self.region, DEFAULT_AMI["us-east-1"])

        # Build launch spec
        launch_spec: Dict = {
            "ImageId": ami,
            "InstanceType": instance_type,
            "UserData": user_data,
            "BlockDeviceMappings": [{
                "DeviceName": "/dev/xvda",
                "Ebs": {"VolumeSize": volume_gb, "VolumeType": "gp3", "DeleteOnTermination": True},
            }],
        }
        if self.key_pair:
            launch_spec["KeyName"] = self.key_pair
        if self.security_group:
            launch_spec["SecurityGroupIds"] = [self.security_group]
        if self.iam_profile:
            launch_spec["IamInstanceProfile"] = {"Name": self.iam_profile}
        if self.subnet_id:
            launch_spec["SubnetId"] = self.subnet_id

        # Spot request
        spot_kwargs: Dict = {
            "LaunchSpecification": launch_spec,
            "InstanceCount": 1,
            "Type": "one-time",
            "TagSpecifications": [{
                "ResourceType": "spot-instances-request",
                "Tags": [
                    {"Key": "Name", "Value": f"prometheus-{job_id}"},
                    {"Key": "Project", "Value": "Prometheus"},
                    {"Key": "JobId", "Value": job_id},
                ],
            }],
        }
        if spot_price_limit:
            spot_kwargs["SpotPrice"] = spot_price_limit

        logger.info("Requesting Spot instance: %s for job %s", instance_type, job_id)
        response = ec2.request_spot_instances(**spot_kwargs)
        spot_request_id = response["SpotInstanceRequests"][0]["SpotInstanceRequestId"]

        # Wait for fulfillment
        instance_id = self._wait_for_spot_fulfillment(ec2, spot_request_id)

        logger.info("Instance launched: %s (job=%s)", instance_id, job_id)
        logger.info("Training started. Monitor with:")
        logger.info("  trainer.wait('%s', stream_logs=True)", job_id)
        logger.info("  or: aws logs tail /prometheus/%s", job_id)

        return {
            "job_id": job_id,
            "instance_id": instance_id,
            "spot_request_id": spot_request_id,
            "instance_type": instance_type,
            "s3_checkpoint": f"s3://{self.bucket}/{self.prefix}/checkpoints/{job_id}/",
            "log_uri": f"s3://{self.bucket}/{self.prefix}/logs/{job_id}/train.log",
        }

    # ------------------------------------------------------------------
    # Monitor
    # ------------------------------------------------------------------

    def wait(
        self,
        job_id: str,
        stream_logs: bool = True,
        poll_interval: int = 30,
        timeout_hours: float = 12.0,
    ) -> bool:
        """
        Wait for a training job to complete by polling for its S3 log file.

        Returns True if completed successfully, False if timed out.
        """
        s3_log_key = f"{self.prefix}/logs/{job_id}/train.log"
        deadline = time.time() + timeout_hours * 3600
        last_size = 0

        logger.info("Waiting for job %s (timeout=%.1fh)...", job_id, timeout_hours)
        import boto3
        s3 = boto3.client("s3", region_name=self.region)

        while time.time() < deadline:
            time.sleep(poll_interval)
            try:
                head = s3.head_object(Bucket=self.bucket, Key=s3_log_key)
                size = head["ContentLength"]

                if stream_logs and size > last_size:
                    obj = s3.get_object(Bucket=self.bucket, Key=s3_log_key,
                                        Range=f"bytes={last_size}-{size - 1}")
                    chunk = obj["Body"].read().decode("utf-8", errors="replace")
                    print(chunk, end="", flush=True)
                    last_size = size

                # Job complete when log contains terminal marker
                if size > 0:
                    tail = s3.get_object(
                        Bucket=self.bucket, Key=s3_log_key,
                        Range=f"bytes={max(0, size - 500)}-{size - 1}"
                    )["Body"].read().decode("utf-8", errors="replace")
                    if "Job " + job_id + " complete" in tail:
                        logger.info("Job %s completed successfully.", job_id)
                        return True
                    if "Error" in tail or "Traceback" in tail:
                        logger.error("Job %s may have failed. Check logs.", job_id)
                        return False

            except s3.exceptions.NoSuchKey if hasattr(s3, "exceptions") else Exception:
                pass  # log not yet created (instance still starting)

        logger.warning("Job %s timed out after %.1fh", job_id, timeout_hours)
        return False

    def estimate_cost(
        self,
        instance_type: str = "g4dn.xlarge",
        duration_hours: float = 8.0,
    ) -> Dict:
        """Rough cost estimate before launching."""
        # Approximate spot prices (subject to change — check AWS console)
        spot_prices = {
            "g4dn.xlarge": 0.16,
            "g4dn.2xlarge": 0.23,
            "g4dn.4xlarge": 0.45,
            "g5.2xlarge": 0.34,
            "p3.2xlarge": 0.90,
            "p3.8xlarge": 3.60,
        }
        hourly = spot_prices.get(instance_type, 0.50)
        storage_cost = 0.023 * 100 / 30 / 24 * duration_hours  # 100 GB gp3 prorated
        s3_cost = 0.023 * 10 * 0.001  # ~10 GB at $0.023/GB
        total = hourly * duration_hours + storage_cost + s3_cost

        return {
            "instance_type": instance_type,
            "spot_price_per_hour": f"${hourly:.2f}",
            "duration_hours": duration_hours,
            "estimated_compute": f"${hourly * duration_hours:.2f}",
            "estimated_storage": f"${storage_cost + s3_cost:.2f}",
            "estimated_total": f"${total:.2f}",
            "note": (
                "Spot pricing fluctuates. Set spot_price_limit to cap spend. "
                "Use g4dn.xlarge for testing, p3.8xlarge for full production training."
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ec2(self):
        if self._ec2 is None:
            try:
                import boto3
                self._ec2 = boto3.client("ec2", region_name=self.region)
            except ImportError:
                raise RuntimeError("boto3 not installed. Run: pip install boto3")
        return self._ec2

    def _wait_for_spot_fulfillment(
        self, ec2, spot_request_id: str, timeout: int = 300
    ) -> str:
        """Wait until the Spot request is fulfilled. Returns instance_id."""
        logger.info("Waiting for Spot fulfillment (request=%s)...", spot_request_id)
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[spot_request_id]
            )
            req = resp["SpotInstanceRequests"][0]
            state = req["State"]
            if state == "active":
                return req["InstanceId"]
            if state in ("cancelled", "failed", "closed"):
                raise RuntimeError(f"Spot request {state}: {req.get('Status', {})}")
            logger.debug("Spot state: %s — waiting...", state)
            time.sleep(10)
        raise TimeoutError(f"Spot request not fulfilled within {timeout}s")
