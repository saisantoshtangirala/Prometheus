#!/usr/bin/env bash
#
# Provider-agnostic remote GPU training launcher.
#
# Works with any box you can SSH into: RunPod, Hetzner, Lambda, Vast.ai,
# a friend's gaming PC. No cloud SDK, no quota approvals - just SSH.
#
# Usage:
#   ./scripts/ssh_train.sh -h root@1.2.3.4 [-p 22] [-k ~/.ssh/id_ed25519] \
#       [-b claude/prometheus-causal-market-ol2pau] [-m full] [-d cuda] \
#       [-o checkpoints/] [--shutdown]
#
# What it does:
#   1. SSH to the host, clone/update the repo at the given branch
#   2. Install dependencies into a venv (idempotent - safe to re-run)
#   3. Run scripts/train.py with the requested mode/device
#   4. Tar the checkpoints and scp them back to the local output dir
#   5. Optionally power the box off (--shutdown) so hourly billing stops
#
# RunPod tip: use the "SSH over exposed TCP" host/port from the pod page,
# and ALWAYS pass --shutdown (or stop the pod in the console) when done.

set -euo pipefail

HOST=""
PORT="22"
KEY=""
BRANCH="claude/prometheus-causal-market-ol2pau"
REPO_URL="https://github.com/saisantoshtangirala/prometheus"
MODE="full"
DEVICE="cuda"
OUT_DIR="checkpoints"
SHUTDOWN="no"
EXTRA_ARGS=""

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h) HOST="$2"; shift 2 ;;
    -p) PORT="$2"; shift 2 ;;
    -k) KEY="$2"; shift 2 ;;
    -b) BRANCH="$2"; shift 2 ;;
    -r) REPO_URL="$2"; shift 2 ;;
    -m) MODE="$2"; shift 2 ;;
    -d) DEVICE="$2"; shift 2 ;;
    -o) OUT_DIR="$2"; shift 2 ;;
    --extra) EXTRA_ARGS="$2"; shift 2 ;;
    --shutdown) SHUTDOWN="yes"; shift ;;
    --help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

[[ -z "$HOST" ]] && { echo "ERROR: -h user@host is required"; usage; }

SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
SCP_OPTS=(-P "$PORT" -o StrictHostKeyChecking=accept-new)
[[ -n "$KEY" ]] && { SSH_OPTS+=(-i "$KEY"); SCP_OPTS+=(-i "$KEY"); }

JOB_ID="job-$(date -u +%Y%m%dT%H%M%SZ)"
echo "=== Remote training: $JOB_ID on $HOST (mode=$MODE device=$DEVICE) ==="

# ---------------------------------------------------------------------------
# 1-3. Setup + train on the remote box
# ---------------------------------------------------------------------------
ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- <<REMOTE
set -euo pipefail
echo "[remote] \$(uname -a)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[remote] no GPU visible"

command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git) || true

if [ -d /opt/prometheus/.git ]; then
    cd /opt/prometheus
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" /opt/prometheus
    cd /opt/prometheus
fi

if [ ! -d /opt/prometheus/.venv ]; then
    python3 -m venv /opt/prometheus/.venv
fi
source /opt/prometheus/.venv/bin/activate
pip install -q --upgrade pip
pip install -q -e . || pip install -q torch numpy scipy scikit-learn networkx pandas
pip install -q pyyaml

echo "[remote] starting training ($MODE / $DEVICE)..."
python scripts/train.py --mode "$MODE" --device "$DEVICE" $EXTRA_ARGS \
    2>&1 | tee /tmp/train_$JOB_ID.log

echo "[remote] packaging checkpoints..."
tar czf /tmp/checkpoints_$JOB_ID.tar.gz -C /opt/prometheus checkpoints
echo "[remote] done: /tmp/checkpoints_$JOB_ID.tar.gz"
REMOTE

# ---------------------------------------------------------------------------
# 4. Pull the checkpoints + log back
# ---------------------------------------------------------------------------
mkdir -p "$OUT_DIR"
scp "${SCP_OPTS[@]}" "$HOST:/tmp/checkpoints_$JOB_ID.tar.gz" "$OUT_DIR/"
scp "${SCP_OPTS[@]}" "$HOST:/tmp/train_$JOB_ID.log" "$OUT_DIR/" || true
tar xzf "$OUT_DIR/checkpoints_$JOB_ID.tar.gz" -C "$OUT_DIR" --strip-components=1
echo "=== Checkpoints extracted to $OUT_DIR/ ==="

# ---------------------------------------------------------------------------
# 5. Optional shutdown (stops hourly billing on RunPod/Hetzner cloud)
# ---------------------------------------------------------------------------
if [[ "$SHUTDOWN" == "yes" ]]; then
    echo "=== Powering off remote host ==="
    ssh "${SSH_OPTS[@]}" "$HOST" "shutdown -h now || poweroff" || true
fi

echo "=== $JOB_ID complete ==="
