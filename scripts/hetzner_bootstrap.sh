#!/usr/bin/env bash
#
# One-time Hetzner server setup for the Kronos 365-day orchestrator.
# Run this ONCE, as root, on a fresh Hetzner Cloud CX32 (Ubuntu 22.04/24.04).
#
#   ssh root@<hetzner-ip>
#   curl -fsSL https://raw.githubusercontent.com/<you>/prometheus/<branch>/scripts/hetzner_bootstrap.sh | bash
#
# or copy the repo up first and run it locally:
#   scp scripts/hetzner_bootstrap.sh root@<hetzner-ip>:/root/
#   ssh root@<hetzner-ip> bash /root/hetzner_bootstrap.sh
#
# After this script finishes:
#   - /opt/prometheus is a working clone on the target branch
#   - kronos.service is enabled and running the 365-day paper loop
#   - GitHub Actions can redeploy via `systemctl restart kronos.service`
#     once you've added the deploy key this script prints at the end.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/saisantoshtangirala/prometheus}"
BRANCH="${BRANCH:-claude/prometheus-causal-market-ol2pau}"
INSTALL_DIR="/opt/prometheus"

echo "=== Installing system dependencies ==="
apt-get update -qq
apt-get install -y -qq git python3-venv python3-pip

echo "=== Cloning $REPO_URL @ $BRANCH ==="
if [ -d "$INSTALL_DIR/.git" ]; then
    cd "$INSTALL_DIR" && git fetch origin "$BRANCH" && git checkout "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi
cd "$INSTALL_DIR"

echo "=== Setting up Python environment ==="
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e . || pip install -q torch numpy scipy scikit-learn networkx pandas
pip install -q pyyaml yfinance

echo "=== Installing systemd service ==="
cat > /etc/systemd/system/kronos.service <<UNIT
[Unit]
Description=Kronos 365-day paper trading orchestrator
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/.venv/bin/python scripts/run_kronos.py --mode=paper
Restart=always
RestartSec=30
StandardOutput=append:/var/log/kronos-service.log
StandardError=append:/var/log/kronos-service.log

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable kronos.service
systemctl restart kronos.service
sleep 3
systemctl status kronos.service --no-pager || true

echo ""
echo "=== Generating a dedicated GitHub Actions deploy key ==="
DEPLOY_KEY_PATH="/root/.ssh/github_actions_deploy"
if [ ! -f "$DEPLOY_KEY_PATH" ]; then
    mkdir -p /root/.ssh
    ssh-keygen -t ed25519 -N "" -f "$DEPLOY_KEY_PATH" -C "github-actions-deploy"
    cat "$DEPLOY_KEY_PATH.pub" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

echo ""
echo "============================================================"
echo " Bootstrap complete."
echo ""
echo " kronos.service is running: tail -f /var/log/kronos-service.log"
echo ""
echo " Copy this PRIVATE key into the GitHub repo secret HETZNER_SSH_KEY:"
echo "------------------------------------------------------------"
cat "$DEPLOY_KEY_PATH"
echo "------------------------------------------------------------"
echo ""
echo " Also set these repo secrets:"
echo "   HETZNER_HOST = $(curl -s ifconfig.me 2>/dev/null || echo '<this server IP>')"
echo "   HETZNER_USER = root"
echo "============================================================"
