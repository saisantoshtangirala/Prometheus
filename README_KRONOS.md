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
| `ci.yml` | every push/PR, any branch | Runs the full test suite |
| `deploy-hetzner.yml` | push to the working branch | Tests must pass, then SSHes into Hetzner, pulls latest code, restarts `kronos.service` |
| `train-runpod.yml` | manual button (Actions tab) | Creates a fresh RunPod pod via the REST API, trains, downloads the checkpoint as a workflow artifact, **always terminates the pod** - even on failure |

The Hetzner deploy is safe to fire on every push: Kronos's crash-recovery
logic (`kronos/orchestrator.py` - `save_checkpoint`/`load_checkpoint`,
tested by ORC-07) means a mid-cycle `systemctl restart` resumes exactly
where it left off instead of losing the day.

`train-runpod.yml` is built directly on RunPod's documented REST API
(`https://rest.runpod.io/v1`, see [docs.runpod.io](https://docs.runpod.io/api-reference/pods/POST/pods)) -
`POST /pods` to create, `GET /pods/{id}` to poll for a running state and
read its real IP/port, `DELETE /pods/{id}` to terminate. No CLI, no
output-parsing, no persistent pod to keep configured between runs: every
run gets a brand-new pod and it's gone by the time the workflow ends,
successful or not.

### Private repository? Read this first

Both workflows need to clone this repo onto machines that aren't you,
which needs its own authentication if the repo isn't public. RunPod and
Hetzner get **different** solutions on purpose, because they're
different kinds of machine:

- **RunPod pods are ephemeral** - a fresh one is created and destroyed
  every run, so there's no persistent place to store a long-lived
  credential even if you wanted to. `train-runpod.yml` instead uses the
  `secrets.GITHUB_TOKEN` GitHub automatically hands every Actions run -
  no setup, expires when the job ends, scoped to only this repo. Nothing
  to configure for this - it already works.
- **The Hetzner box is permanent** - it needs its own long-term
  credential that survives independently of any GitHub Actions run,
  including if you SSH in and `git pull` by hand months from now. For
  that, `hetzner_bootstrap.sh` generates an SSH **deploy key** (a
  GitHub feature for exactly this - one keypair, read-only, scoped to a
  single repo) and clones over SSH instead of HTTPS. This needs one
  manual step below.

If your repo is public, ignore all of this - both clones work with zero
extra setup either way.

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
3. **Private repo only:** the script prints a *different* key near the
   start - `git_deploy_key.pub` - and the clone step right after it will
   fail on a first run. That's expected: copy that public key to
   **repo -> Settings -> Deploy keys -> Add deploy key** (leave "Allow
   write access" unchecked), then re-run the exact same bootstrap
   command - it's idempotent, so it picks up cleanly from where it
   stopped.
4. Copy the private key block it prints at the very end (this one goes
   in a GitHub *secret*, not a deploy key), and note the server's IP.

### Step 2 - Get a RunPod API key and register a CI SSH key (one time, no pod needed)

Unlike Hetzner, there is nothing to pre-create for RunPod - the workflow
creates and destroys the pod itself on every run.

1. RunPod Console -> **Settings -> API Keys** -> create a key with
   **All** permissions (needed to create/delete pods). Copy it.
2. Register a dedicated CI keypair under **Settings -> SSH Public Keys**
   - don't reuse your personal key. RunPod bakes every account-level
   registered key into `authorized_keys` automatically when a new pod
   boots, which is exactly what the workflow needs:
   ```bash
   ssh-keygen -t ed25519 -N "" -f ~/.ssh/runpod_ci -C "github-actions-runpod"
   cat ~/.ssh/runpod_ci.pub    # paste this into RunPod's SSH Public Keys
   ```

### Step 3 - Add GitHub repository secrets

Repo -> **Settings -> Secrets and variables -> Actions -> New repository secret**:

| Secret | Value |
|---|---|
| `HETZNER_HOST` | Hetzner server IP |
| `HETZNER_USER` | `root` |
| `HETZNER_SSH_KEY` | the private key `hetzner_bootstrap.sh` printed |
| `RUNPOD_API_KEY` | the key from RunPod Console -> Settings -> API Keys |
| `RUNPOD_SSH_PRIVATE_KEY` | contents of `~/.ssh/runpod_ci` (the *private* half) |

Just five secrets total - no pod ID, host, or port to track, since none
of that exists until the workflow creates it.

### Step 4 - Verify

- **Push a commit** to the working branch -> Actions tab -> `ci.yml`
  should go green, then `deploy-hetzner.yml` deploys automatically.
  Check it landed: `ssh root@<hetzner-ip> "systemctl status kronos"`.
- **Actions tab -> "Train on RunPod GPU" -> Run workflow** (manual
  button, pick the GPU type and mode) to fire off a fresh GPU training
  run on demand. Watch the "Create pod" and "Poll until running" steps
  the first time - they print the full JSON response from RunPod at
  each stage, so if anything about the account (spend limits, GPU
  availability) blocks pod creation, the error is visible immediately
  rather than hidden behind CLI output.
- Trained checkpoints download from the workflow run's **Artifacts**
  section (top of the run page), retained 30 days.
- Confirm no pod is left running: RunPod Console -> Pods should be
  empty after the workflow finishes, success or failure - the
  terminate step runs unconditionally (`if: always()`).

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

## Daily WhatsApp Progress Reports (optional)

Kronos can text you a short progress digest to your own phone once a
day, right after market close: current day and % through the 365-day
run, today's equity/PnL/Sharpe/trade count, market regime, NEAT's best
fitness, and any warnings from that day.

Sent via [CallMeBot](https://www.callmebot.com/blog/free-api-whatsapp-messages/) -
a free, unofficial service built specifically for "let my script message
my own WhatsApp." Not affiliated with WhatsApp/Meta, no business account
needed, but also no uptime guarantee - fine for a personal digest, not
something to depend on for anything time-critical. (Twilio's WhatsApp
Business API is the paid, official alternative if you ever need one.)

### Setup

1. **On your phone**, save this contact: `+34 644 84 71 64`
2. WhatsApp it exactly: `I allow callmebot to send me messages`
3. CallMeBot replies with your personal API key.
4. **On the Hetzner box**, edit the secrets file `hetzner_bootstrap.sh`
   already created for you:
   ```bash
   nano /etc/kronos.env
   ```
   Uncomment and fill in:
   ```
   KRONOS_WHATSAPP_PHONE=15551234567    # your number, digits only, with country code
   KRONOS_WHATSAPP_APIKEY=123456        # the key CallMeBot sent you
   ```
5. Turn it on in `kronos/config.yaml`:
   ```yaml
   notifications:
     enabled: true
   ```
6. Test it immediately, without waiting for a real day to close:
   ```bash
   python scripts/test_whatsapp.py
   ```
7. Restart the service so it picks up the new environment file:
   ```bash
   systemctl restart kronos
   ```

Notifications are fully optional and fail silently - a broken API key or
a CallMeBot outage logs a warning and moves on, it never affects trading
or the daily cycle.

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
