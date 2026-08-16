# Project Kronos

**The self-evolving daily lifecycle layer on top of Prometheus.**

Kronos runs the Prometheus trading engine through a 365-day continuous
paper-trading loop. Every night it dreams up adversarial market futures,
evolves its own architecture against them, warms up to the current regime,
and briefs you at 06:00. During market hours it runs a low-latency reflex
path with a hard volatility kill-switch. Everything is logged, forever.

```
00:00 ─ DIGESTION   multi-source fetch, cross-validation, Kalman repair
02:00 ─ NIGHTMARE   10,000 adversarial futures (conditional diffusion)
04:00 ─ EVOLUTION   20 NEAT variants vs. the nightmares -> top-5 ensemble
05:00 ─ ADAPTATION  exactly 3 MAML gradient steps on the last 3 real days
06:00 ─ REPORT      God's Eye markdown briefing
09:30 ─ REFLEX      SNN inference only, VIX 2-sigma kill gate, NO training
16:00 ─ LOGGING     close the books, persist PnL/Sharpe/accuracy to SQLite
```

## Quick Start

```bash
pip install -e .
pip install apscheduler pyyaml

# Full 365-day wall-clock loop (paper trading)
python scripts/run_kronos.py --mode=paper

# 3 compressed test days (no waiting for the clock)
python scripts/run_kronos.py --mode=replay --accelerated --days 3

# Run the test suite
python -m pytest tests/test_kronos.py -v
```

All tunables live in `kronos/config.yaml` - population sizes, mutation
rates, slippage tiers, VIX thresholds, schedule times. No magic numbers in
code.

## What Kronos Reuses From Prometheus

| Kronos module | Prometheus component |
|---|---|
| `nightmare_generator.py` | `prometheus.generative.diffusion_simulator.MarketDiffusionSimulator` |
| `evolver.py` | `prometheus.meta.neat_evolver.NEATArchitectureEvolver` + `GenomeDecoder` |
| `warmer.py` | `prometheus.meta.maml_engine.MAMLMetaLearner` |
| `reflex.py` | `prometheus.neuro.spiking_network.SpikingMarketEncoder` |
| `data_pipeline.py` | `prometheus.data.data_validator.KalmanFilter1D`, `prometheus.data.sentiment_analyzer.SentimentAnalyzer` |

Nothing under `prometheus/` was modified.

## Design Decisions Worth Knowing

**The "weighted average" of top-5 variants is prediction-space, not
parameter-space.** Distinct NEAT genomes decode to distinct topologies, so
averaging their weights is mathematically undefined. The Master Model is a
`WeightedEnsemble` whose output is the fitness-weighted mean of the top-5
variants' outputs. Same intent, actually computable.

**The reflex gate runs before the SNN.** A VIX panic print zeroes the
position cap even if inference were to hang - safety is not queued behind
the model.

**Vetoes take 24 hours.** Drop a `veto.txt` containing `FLATTEN` or `HALT`
in the repo root. Kronos logs it immediately, applies it one day later.
This is a feature: it converts panic into a decision you must still agree
with tomorrow.

**Failure never stops the year.** Every phase retries exactly once, then
logs a fatal audit row and the day continues with yesterday's model. A
failed evolution automatically retries in degraded mode (population 6
instead of 20).

## Data Sources & API Keys

Priority order (config: `data.sources`): `yfinance` -> `polygon` ->
`alphavantage`. yfinance needs no key. The others activate automatically
when their keys are present:

```bash
export POLYGON_API_KEY=...
export ALPHAVANTAGE_API_KEY=...
```

Sources are fetched **in parallel**; the first configured source that
succeeded becomes primary, and close prices are cross-validated against
every other source that responded. Disagreement beyond
`cross_validation_tolerance_pct` flags the day in `DailyMemory.quality_flags`.

## Recommended Deployment: Hetzner + RunPod (~$30-40/month)

The cost-optimal split for the nightly-cycle workload, with no cloud
quota approvals anywhere:

- **Hetzner Cloud CX32 (~EUR 7/mo)** - the always-on brain. Runs the
  Kronos orchestrator 24/7: digestion, reflex arc, paper trading,
  reporting. All CPU work.
- **RunPod hourly (~$0.40/hr, per-second billing)** - the nightly muscle.
  ~3h/night of GPU for the heavy phases via `scripts/ssh_train.sh`.

```bash
# One-shot remote training on ANY SSH-able GPU box (RunPod, Hetzner, ...):
./scripts/ssh_train.sh -h root@<pod-ip> -p <ssh-port> \
    -m full -d cuda --shutdown
# clones the repo, installs deps, trains, pulls checkpoints/ back,
# powers the box off so billing stops.
```

Why not a dedicated Hetzner GPU (GEX44, ~EUR 184/mo)? Break-even vs
hourly rental is ~500 GPU-hours/month; the nightly cycle uses ~90.
Why not AWS Spot ($0.16/hr, cheapest on paper)? New accounts routinely
get GPU quota requests denied. The CloudFormation stack in
`cloudformation/` is ready if a quota is ever granted.

## GitHub CI/CD - Automated Deploy to Hetzner + RunPod

Three workflows live in `.github/workflows/`:

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | every push/PR, any branch | Runs the full 266-test suite |
| `deploy-hetzner.yml` | push to the working branch | Tests must pass, then SSHes into Hetzner, pulls latest code, restarts `kronos.service` |
| `train-runpod.yml` | manual button (Actions tab) | Starts your RunPod pod, trains, downloads the checkpoint as a workflow artifact, **always** stops the pod again |

The Hetzner deploy is safe to fire on every push: Kronos's crash-recovery
logic (`kronos/orchestrator.py` - `save_checkpoint`/`load_checkpoint`,
tested by ORC-07) means a mid-cycle `systemctl restart` resumes exactly
where it left off instead of losing the day.

### Step 1 - Bootstrap the Hetzner server (one time)

1. Create a Hetzner Cloud CX32 (Ubuntu 22.04 or 24.04) in the
   [Hetzner Console](https://console.hetzner.cloud).
2. SSH in as root and run the bootstrap script committed in this repo:
   ```bash
   ssh root@<hetzner-ip>
   curl -fsSL https://raw.githubusercontent.com/<you>/prometheus/claude/prometheus-causal-market-ol2pau/scripts/hetzner_bootstrap.sh | bash
   ```
   This clones the repo, sets up a venv, installs `kronos.service` under
   systemd, starts the 365-day paper loop, and **prints a fresh SSH
   private key** at the end - generated specifically for GitHub Actions,
   never your personal key.
3. Copy the private key block it prints, and note the server's IP.

### Step 2 - Bootstrap a RunPod pod (one time)

1. In the [RunPod Console](https://runpod.io/console/pods), deploy a pod:
   template **PyTorch 2.8.0**, GPU **RTX A5000** (or your pick from the
   cost comparison above), enable **SSH over exposed TCP** in the pod's
   network settings.
2. Once running, open the pod's **Connect** panel and copy the
   "SSH over exposed TCP" details - a real IP and port (not the
   `xxx@ssh.runpod.io` proxy string, which changes and isn't suited to
   automation). Note the pod's ID too (shown on the pod card).
3. Register a dedicated CI keypair with RunPod (Settings -> SSH Public
   Keys) - don't reuse your personal key:
   ```bash
   ssh-keygen -t ed25519 -N "" -f ~/.ssh/runpod_ci -C "github-actions-runpod"
   cat ~/.ssh/runpod_ci.pub    # paste this into RunPod's SSH Public Keys
   ```
4. Leave the pod **running** for now - Step 3 wires it up, and after
   that the workflow starts/stops it for you on every training run.

### Step 3 - Add GitHub repository secrets

Repo -> **Settings -> Secrets and variables -> Actions -> New repository secret**:

| Secret | Value |
|---|---|
| `HETZNER_HOST` | Hetzner server IP |
| `HETZNER_USER` | `root` |
| `HETZNER_SSH_KEY` | the private key `hetzner_bootstrap.sh` printed |
| `RUNPOD_API_KEY` | RunPod Console -> Settings -> API Keys |
| `RUNPOD_SSH_PRIVATE_KEY` | contents of `~/.ssh/runpod_ci` (the *private* half) |
| `RUNPOD_POD_ID` | the pod ID from the RunPod console |
| `RUNPOD_HOST` | the IP from "SSH over exposed TCP" |
| `RUNPOD_PORT` | the port from "SSH over exposed TCP" |
| `RUNPOD_USER` | `root` (RunPod pods default to root) |

### Step 4 - Verify

- **Push a commit** to the working branch -> Actions tab -> `ci.yml`
  should go green, then `deploy-hetzner.yml` deploys automatically.
  Check it landed: `ssh root@<hetzner-ip> "systemctl status kronos"`.
- **Actions tab -> "Train on RunPod GPU" -> Run workflow** (manual
  button) to fire off a GPU training run on demand. Watch the log the
  first time - the RunPod steps use fixed connection secrets from Step 3
  rather than parsing a freshly-created pod's address, which is the
  reliable part; if `runpodctl start/stop` ever changes its CLI verbs,
  that's the one spot to check against `runpodctl --help`.
- Trained checkpoints download from the workflow run's **Artifacts**
  section (top of the run page), retained 30 days.

### Why train-runpod.yml starts/stops one pod instead of creating a fresh one per run

An earlier design created and destroyed a pod on every run, which is
tidier in principle but means parsing a brand-new SSH address out of
`runpodctl` CLI output every single time - a genuinely fragile
dependency, since exact output formatting isn't something to guess at
for a tool that changes across versions. Starting/stopping one
persistent pod keeps its address stable across runs, so the workflow
only ever calls `runpodctl start pod` / `stop pod` on a known ID -
verbs stable enough to trust. The tradeoff: a stopped RunPod pod still
bills a small amount for its container disk while idle (a few cents/day
for 20GB) - trivial next to GPU-hour costs, and cheaper than dedicated
hosting.

## Alternative Deployment: Local Machine + AWS Hybrid

The intended production split:

- **Local machine (e.g. Mac Mini)** - always on, runs the orchestrator loop:
  digestion, reflex arc (inference only), paper trading, reporting. Light,
  CPU-friendly work.
- **AWS Spot instance (g4dn.xlarge)** - fired nightly for the heavy phases:
  nightmare generation and NEAT evolution on GPU.

### Local machine setup

```bash
git clone <repo> && cd Prometheus
python -m venv .venv && source .venv/bin/activate
pip install -e . && pip install apscheduler pyyaml

# Run under launchd/systemd so it survives reboots. Example launchd plist:
# ~/Library/LaunchAgents/com.kronos.daemon.plist -> runs:
python scripts/run_kronos.py --mode=paper
```

### AWS nightly heavy compute

The repo already ships the full AWS stack (see `cloudformation/prometheus-training.yml`
and `scripts/aws_train.py`). The nightly pattern:

1. At 02:00 the local machine launches a Spot instance
   (`scripts/aws_train.py launch --mode pretrain`), which generates
   scenarios/futures on GPU and pushes them to S3.
2. The local machine polls S3 (`aws_train.py wait`), pulls the artifacts,
   and feeds them into the evolution phase.
3. **Graceful degradation (tested):** if the Spot instance is preempted or
   the quota is unavailable, `KronosEvolver` automatically runs locally
   with `evolution.fallback_population_size` (6) and
   `fallback_generations` (1). The year never stops.

### Surviving 365 days

- **Spot preemption** -> degraded local evolution, audit-logged.
- **API rate limits** -> multi-source fallback chain; the day is flagged,
  not lost.
- **Silent data corruption** -> Kalman repair + cross-source validation +
  flash-crash detection; tickers exceeding `max_missing_pct` are dropped
  for the day and flagged.
- **Process crash** -> every trade, model checkpoint, and heartbeat is
  already on disk (`logs/trades.db`, `logs/models/`); restart resumes with
  yesterday's checkpoint.
- **Human panic** -> `veto.txt`, 24-hour delay, fully audit-logged.

## The Audit Trail

Everything lands in `logs/`:

```
logs/
├── trades.db          # SQLite: trades, daily_performance, audit_log
├── kronos.log         # process log
├── models/            # daily master-model checkpoints (master_dayNNN.pt)
└── reports/           # GodsEye_<date>_dayN.md + GodsEye.md (latest)
```

The `audit_log` table records every phase outcome, every veto event, every
heartbeat (hourly portfolio snapshot), and every fatal error - the complete
flight recorder for the year.
