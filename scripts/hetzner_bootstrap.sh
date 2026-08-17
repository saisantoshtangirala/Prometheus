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
#
# PRIVATE REPO: this script generates and uses its own SSH "git deploy
# key" for outbound clone/fetch (a DIFFERENT keypair from the inbound
# "GitHub Actions -> Hetzner" one below - two keys, two directions of
# trust). If REPO_URL is left as the SSH form (default), the very first
# clone in this script will fail until you've added that key's public
# half as a read-only GitHub Deploy Key - see the printed instructions
# at the end and re-run this script once it's added.

set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:saisantoshtangirala/prometheus.git}"
BRANCH="${BRANCH:-claude/prometheus-causal-market-ol2pau}"
INSTALL_DIR="/opt/prometheus"
GIT_DEPLOY_KEY="/root/.ssh/git_deploy_key"

echo "=== Installing system dependencies ==="
apt-get update -qq
apt-get install -y -qq git python3-venv python3-pip

echo "=== Setting up outbound git deploy key (Hetzner -> GitHub) ==="
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [ ! -f "$GIT_DEPLOY_KEY" ]; then
    ssh-keygen -t ed25519 -N "" -f "$GIT_DEPLOY_KEY" -C "hetzner-git-deploy"
fi
ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null
cat > /root/.ssh/config <<SSHCFG
Host github.com
    IdentityFile $GIT_DEPLOY_KEY
    IdentitiesOnly yes
SSHCFG
chmod 600 /root/.ssh/config

if [[ "$REPO_URL" == git@github.com:* ]] && [ ! -d "$INSTALL_DIR/.git" ]; then
    echo ""
    echo "============================================================"
    echo " If this is a PRIVATE repo, add this key as a read-only"
    echo " GitHub Deploy Key BEFORE continuing: repo -> Settings ->"
    echo " Deploy keys -> Add deploy key (leave 'Allow write access' off)."
    echo "------------------------------------------------------------"
    cat "$GIT_DEPLOY_KEY.pub"
    echo "------------------------------------------------------------"
    echo " Public repos: ignore this, the clone below will just work."
    echo "============================================================"
    echo ""
fi

echo "=== Cloning $REPO_URL @ $BRANCH ==="
if [ -d "$INSTALL_DIR/.git" ]; then
    cd "$INSTALL_DIR" && git fetch origin "$BRANCH" && git checkout "$BRANCH"
else
    if ! git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"; then
        echo ""
        echo "!!! Clone failed. If this is a private repo, add the deploy"
        echo "!!! key printed above under repo Settings -> Deploy keys,"
        echo "!!! then re-run this script (it's safe to run again - the"
        echo "!!! key won't be regenerated and nothing else has changed)."
        exit 1
    fi
fi
cd "$INSTALL_DIR"

echo "=== Setting up Python environment ==="
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e . || pip install -q torch numpy scipy scikit-learn networkx pandas
pip install -q pyyaml yfinance

echo "=== Setting up secrets env file (Telegram, RunPod nightly training) ==="
# Kept OUTSIDE the git repo on purpose - never commit bot tokens or API
# keys. The leading '-' on EnvironmentFile below means systemd starts
# fine even if this file is empty or missing entirely.
if [ ! -f /etc/kronos.env ]; then
    cat > /etc/kronos.env <<'ENVFILE'
# Kronos secrets - not tracked in git. Uncomment and fill in to enable
# daily Telegram progress reports (see kronos/notifier.py for setup):
#KRONOS_TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
#KRONOS_TELEGRAM_CHAT_ID=987654321

# Uncomment to enable nightly RunPod GPU training (see
# kronos/runpod_trigger.py). Both are required - if RUNPOD_API_KEY is
# unset, Kronos never touches RunPod at all and behaves exactly as
# before this feature existed.
#RUNPOD_API_KEY=rp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Private half of an SSH keypair whose PUBLIC half is registered under
# RunPod Settings -> SSH Public Keys (RunPod bakes it into every new
# pod's authorized_keys automatically). Generate with:
#   ssh-keygen -t ed25519 -N "" -f /root/.ssh/runpod_key
#RUNPOD_SSH_KEY_PATH=/root/.ssh/runpod_key
ENVFILE
    chmod 600 /etc/kronos.env
fi

echo "=== Installing systemd service ==="
cat > /etc/systemd/system/kronos.service <<UNIT
[Unit]
Description=Kronos 365-day paper trading orchestrator
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=-/etc/kronos.env
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
echo "=== Generating a dedicated GitHub Actions -> Hetzner deploy key ==="
echo "    (this is the OTHER direction from the git_deploy_key above:"
echo "     that one lets Hetzner pull FROM GitHub, this one lets"
echo "     GitHub Actions SSH INTO Hetzner to redeploy)"
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
echo " (If your repo is private, scroll up for the git_deploy_key"
echo " public half - that one goes on GitHub as a Deploy Key, not"
echo " a repo secret. This next one is different.)"
echo ""
echo " Copy this PRIVATE key into the GitHub repo secret HETZNER_SSH_KEY:"
echo "------------------------------------------------------------"
cat "$DEPLOY_KEY_PATH"
echo "------------------------------------------------------------"
echo ""
echo " Also set these repo secrets:"
echo "   HETZNER_HOST = $(curl -s ifconfig.me 2>/dev/null || echo '<this server IP>')"
echo "   HETZNER_USER = root"
echo ""
echo " Want daily Telegram progress reports? Edit /etc/kronos.env with"
echo " your bot token/chat id, set notifications.enabled: true in"
echo " kronos/config.yaml, then: systemctl restart kronos"
echo " Test it anytime with: python scripts/test_telegram.py"
echo "============================================================"
