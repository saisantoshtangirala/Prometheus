"""
S3-backed checkpoint store.

Handles upload/download of model checkpoints and scenario libraries between
the local machine and AWS.  Every upload is versioned with a UTC timestamp so
you can roll back to any training run without touching git.

Usage:
    store = CheckpointStore(bucket="my-prometheus-bucket")
    store.push("checkpoints/finetune")          # local dir → S3
    store.pull("checkpoints/finetune")           # S3 → local dir
    store.push_scenarios("data/scenarios")       # scenario library → S3
    latest = store.list_checkpoints()            # show all remote versions
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import tarfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckpointStore:
    """S3-backed versioned checkpoint manager."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "prometheus",
        region: str = "us-east-1",
        local_base: str = ".",
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self.local_base = Path(local_base)
        self._s3 = None  # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        local_dir: str,
        tag: str = "latest",
        compress: bool = True,
    ) -> str:
        """
        Upload a checkpoint directory to S3.

        Args:
            local_dir:  local path to checkpoint directory (e.g. "checkpoints/finetune")
            tag:        label for this version ("latest", "pretrain", "v2", …)
            compress:   gzip-compress before upload (saves ~60% bandwidth)

        Returns:
            S3 URI of the uploaded checkpoint
        """
        s3 = self._get_s3()
        local_path = self.local_base / local_dir
        if not local_path.exists():
            raise FileNotFoundError(f"Checkpoint dir not found: {local_path}")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        s3_key = f"{self.prefix}/checkpoints/{tag}/{ts}.tar.gz"

        # Pack to tar.gz
        archive = self.local_base / f"_tmp_ckpt_{ts}.tar.gz"
        try:
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(local_path, arcname=local_dir)

            file_size_mb = os.path.getsize(archive) / 1e6
            logger.info("Uploading %.1f MB → s3://%s/%s", file_size_mb, self.bucket, s3_key)
            s3.upload_file(
                str(archive),
                self.bucket,
                s3_key,
                Callback=_ProgressCallback(file_size_mb),
            )
        finally:
            archive.unlink(missing_ok=True)

        # Write/update "latest" pointer
        self._write_pointer(s3, tag, s3_key, ts)

        uri = f"s3://{self.bucket}/{s3_key}"
        logger.info("Push complete: %s", uri)
        return uri

    def pull(
        self,
        local_dir: str,
        tag: str = "latest",
        overwrite: bool = True,
    ) -> str:
        """
        Download a checkpoint from S3 to local_dir.

        Returns:
            S3 key that was downloaded
        """
        s3 = self._get_s3()
        s3_key = self._resolve_tag(s3, tag)
        if s3_key is None:
            raise RuntimeError(f"No checkpoint found for tag '{tag}' in s3://{self.bucket}/{self.prefix}/")

        local_path = self.local_base / local_dir
        if local_path.exists() and overwrite:
            shutil.rmtree(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        archive = self.local_base / "_tmp_pull.tar.gz"
        try:
            size_mb = self._get_object_size_mb(s3, s3_key)
            logger.info("Downloading %.1f MB from s3://%s/%s", size_mb, self.bucket, s3_key)
            s3.download_file(
                self.bucket,
                s3_key,
                str(archive),
                Callback=_ProgressCallback(size_mb),
            )
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(self.local_base)
        finally:
            archive.unlink(missing_ok=True)

        logger.info("Pull complete → %s", local_path)
        return s3_key

    def push_scenarios(
        self,
        local_dir: str = "data/scenarios",
        tag: str = "scenarios-latest",
    ) -> str:
        """Upload the scenario library (can be large: ~2–5 GB)."""
        return self.push(local_dir, tag=tag, compress=True)

    def pull_scenarios(
        self,
        local_dir: str = "data/scenarios",
        tag: str = "scenarios-latest",
    ) -> str:
        return self.pull(local_dir, tag=tag)

    def list_checkpoints(self, tag: Optional[str] = None) -> List[Dict]:
        """List all checkpoint versions in S3, newest first."""
        s3 = self._get_s3()
        prefix = f"{self.prefix}/checkpoints/"
        if tag:
            prefix += f"{tag}/"

        paginator = s3.get_paginator("list_objects_v2")
        results = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".tar.gz"):
                    parts = key.split("/")
                    results.append({
                        "s3_uri": f"s3://{self.bucket}/{key}",
                        "tag": parts[-2] if len(parts) >= 2 else "unknown",
                        "timestamp": parts[-1].replace(".tar.gz", ""),
                        "size_mb": obj["Size"] / 1e6,
                        "last_modified": obj["LastModified"].isoformat(),
                    })
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results

    def sync_config(self, config_path: str = "configs/default.yaml") -> str:
        """Push config file to S3 (for reproducibility)."""
        s3 = self._get_s3()
        key = f"{self.prefix}/configs/{Path(config_path).name}"
        s3.upload_file(config_path, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def download_logs(self, job_id: str, local_path: str = "logs/") -> None:
        """Download training logs from a remote job."""
        s3 = self._get_s3()
        log_prefix = f"{self.prefix}/logs/{job_id}/"
        paginator = s3.get_paginator("list_objects_v2")
        os.makedirs(local_path, exist_ok=True)
        for page in paginator.paginate(Bucket=self.bucket, Prefix=log_prefix):
            for obj in page.get("Contents", []):
                fname = obj["Key"].split("/")[-1]
                s3.download_file(self.bucket, obj["Key"], os.path.join(local_path, fname))
                logger.info("Downloaded log: %s", fname)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_s3(self):
        if self._s3 is None:
            try:
                import boto3
                self._s3 = boto3.client("s3", region_name=self.region)
            except ImportError:
                raise RuntimeError("boto3 not installed. Run: pip install boto3")
        return self._s3

    def _write_pointer(self, s3, tag: str, s3_key: str, ts: str) -> None:
        """Write a JSON pointer file so 'latest' can be resolved."""
        pointer_key = f"{self.prefix}/checkpoints/{tag}/_pointer.json"
        pointer = json.dumps({"s3_key": s3_key, "timestamp": ts}).encode()
        s3.put_object(Bucket=self.bucket, Key=pointer_key, Body=pointer)

    def _resolve_tag(self, s3, tag: str) -> Optional[str]:
        pointer_key = f"{self.prefix}/checkpoints/{tag}/_pointer.json"
        try:
            obj = s3.get_object(Bucket=self.bucket, Key=pointer_key)
            data = json.loads(obj["Body"].read())
            return data["s3_key"]
        except Exception:
            return None

    def _get_object_size_mb(self, s3, key: str) -> float:
        try:
            head = s3.head_object(Bucket=self.bucket, Key=key)
            return head["ContentLength"] / 1e6
        except Exception:
            return 0.0


class _ProgressCallback:
    """Simple upload/download progress logger."""

    def __init__(self, total_mb: float):
        self._total = max(total_mb, 0.001)
        self._seen = 0.0
        self._lock = threading.Lock()
        self._last_pct = 0

    def __call__(self, bytes_transferred: int) -> None:
        with self._lock:
            self._seen += bytes_transferred / 1e6
            pct = int(self._seen / self._total * 100)
            if pct >= self._last_pct + 10:
                self._last_pct = pct
                logger.info("  Transfer: %d%% (%.1f / %.1f MB)", pct, self._seen, self._total)
