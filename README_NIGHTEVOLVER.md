# NightEvolver-Hybrid

Genetic-algorithm strategy evolution on RunPod, execution on Hetzner —
reusing the existing Prometheus/Kronos infrastructure, with **zero LLMs
and zero neural architecture search**.

---

## Corrections to the specification, up front

Three premises in the brief are factually wrong about this codebase, and
one of them motivated a headline benefit. All are verifiable:

**1. There are no LLMs in Prometheus/Kronos.** A grep across
`kronos/`, `prometheus/` and `scripts/` for `openai|anthropic|deepseek|
gpt-|llm|langchain|chat.completions` returns **zero** hits. The system
used NEAT, MAML, an SNN and a diffusion model — all local torch. So
"eliminating all LLM costs" eliminates a cost of **₹0**, and "no
hallucination risk" removes a risk that was never present. Everything
else in the redesign stands on its own; this particular benefit does
not exist.

**2. `kronos/checkpoint_loader.py` does not exist.** The brief says to
reuse it. The only checkpoint code is
`kronos/runpod_trigger.load_runpod_checkpoint()`, which is
SNN-specific (it takes a `torch.nn.Module`). `nightevolver/saver.py`
is new, not reused.

**3. `vectorbt` has no CUDA backend**, and neither `vectorbt` nor
`TA-Lib` is installed. vectorbt is numba/numpy. Both are replaced with
direct numpy here — fewer dependencies that can fail unattended.

**Unverifiable references.** I could not verify that *TradeSight*
(Sharpe 2.53), *StratEvo* (484-factor GA, 55.5% win rate) or *RLER*
exist as described, so I have not treated their numbers as achievable
targets. That matters more than it sounds — see the next section for
why a "Sharpe 2.53" from a GA is the number you would *expect* from a
large search over noise.

**Also worth knowing:** last night's scheduled RunPod run failed with
`Pod never reported RUNNING with a public IP within 10 minutes`. GPU
pod provisioning is an observed, recurring failure mode.

---

## The one thing this design gets right that the spec missed

A GA with population 50 over 20 generations evaluates **~1,000
strategies** on one training window and returns the best. That is a
best-of-1,000 search, and the expected maximum Sharpe of *N worthless*
strategies grows like `sqrt(2 ln N)`:

| Search budget | Expected best Sharpe from pure noise (250 bars) |
|---|---|
| 10 | +1.58 |
| 100 | +2.54 |
| 1,000 | +3.27 |
| 10,000 | +3.88 |

**Searching 1,000 random strategies over one year of data is expected to
produce an in-sample Sharpe above 3 even if every strategy is
worthless.** Reporting that as the strategy's Sharpe is not mild
optimism — it is the entire result.

This is measured here, not argued. Running the GA on a **pure random
walk** (unpredictable by construction):

```
IN-SAMPLE   Sharpe +1.94   win rate 69.2%   maxDD 1.1%
OUT-SAMPLE  Sharpe +1.17   win rate 50.0%
OVERFITTING GAP  +0.77 Sharpe
NOISE BENCHMARK  best-of-390 noise scores ~+2.32 -> winner DOES NOT BEAT noise
deflated P(SR>0) = 0.031   (gate is 0.95)
```

A naive report would announce "**Sharpe 1.94, 69% win rate**" on data
that cannot be predicted. A longer run produced an in-sample Sharpe of
+1.18 against an out-of-sample **−3.12** — a **+4.29 Sharpe overfitting
gap**.

Three controls make this visible instead of shippable:

1. **Deflated Sharpe uses the real search budget.** `n_trials =
   population × generations`, and in the walk-forward, `windows × GA
   budget`. A 125-window backtest re-running a 1,000-evaluation search
   is **125,000 trials**, not 3.
2. **In-sample and out-of-sample Sharpe are always reported together.**
   The gap is the overfitting measure and it is printed first.
3. **The executor refuses under-powered checkpoints.**
   `load_checkpoint(require_gate=True)` raises if deflated P(SR>0) <
   0.95. Verified: the Hetzner side refuses the checkpoint RunPod just
   produced, with the reason stated.

---

## Honest comparison table

The spec's table, corrected where it is wrong and sharpened where it is
right.

| Kronos problem | Real status | How NightEvolver addresses it |
|---|---|---|
| MAML was a no-op | **True** — confirmed and fixed in the audit (`_forward_with_params` ignored adapted params) | No MAML. GA re-runs nightly on fresh data. |
| Backtest tested the wrong model | **True** — the strongest point in the spec | Genome encode/decode, indicators and position rules are **one shared implementation**; the walk-forward re-runs the *same GA* per window. |
| Training objective was MSE vs synthetic futures | **True** | Fitness is `Sharpe × (1−MaxDD) × (WinRate/0.5)` with penalty gates — the live objective directly. |
| Position sizing was near-zero `tanh(pred)` | **True** — `size_scale` collapsed to 0.005–0.007 | Kelly-derived, capped at 10%, with an explicit conviction floor. |
| NEAT overfitted noise | **Partly** — NEAT was *reporting-only*; the 49.7% came from the SNN | GA searches a **65-gene constrained space** of interpretable rules, not architectures. Smaller space ≠ immune: hence the controls above. |
| LLMs added cost and hallucination risk | **False** — no LLMs exist | No change; there was nothing to remove. |

---

## Architecture

Same split as Kronos. Nothing new is introduced.

```
RunPod (ephemeral)                    Hetzner (persistent)
─────────────────────                 ────────────────────────
scripts/train_nightevolver.py         nightevolver/strategy_decoder.py
  data_loader  20 indicators            load_checkpoint  (verify + gate)
  ga_engine    tournament GA            signal()         < 100ms/tick
  rl_trainer   optional Q-learning      kronos/paper_trader.py  (unchanged)
  saver        JSON checkpoint          kronos/notifier.py      (unchanged)
        │                                        ▲
        └────── checkpoints/nightevolver/ ───────┘
```

**One package, not two.** The brief asks for `nightevolver_runpod/`
plus a separate `kronos/` copy. Genome encode/decode **must** be
byte-identical on both machines — two copies is exactly the "backtest
tested the wrong model" failure being fixed. Training and execution
import the same modules; there is nothing to drift.

**The checkpoint is JSON, not pickle.** It is fetched over the network
from an ephemeral pod onto the box holding the trading account. A
pickle there is arbitrary code execution on load.

**Verification on load.** `genome_version` and the full
`indicator_names` ordering are checked. If channel order changed, gene 7
would attach to a different indicator and the strategy would be
silently scrambled — so that raises rather than trades.

---

## On the GPU

This workload does not need one. The GA's inner loop is a vectorised
backtest over ~250 bars × 10 assets (~25,000 floats), repeated ~1,000
times: **13 seconds for a 10-generation run on CPU**, no torch import on
the path. There is no dense linear algebra to accelerate.

Combined with the observed pod-provisioning failure, running this on the
Hetzner box directly would be cheaper and more reliable. `--device` is
accepted for CLI parity but unused. Keeping RunPod is fine — depending
on it for a CPU-bound job is a liability worth knowing you have.

---

## Usage

```bash
# RunPod (or anywhere)
python scripts/train_nightevolver.py --mode ga --generations 20 --population 50
python scripts/train_nightevolver.py --mode rl --episodes 1000
python scripts/train_nightevolver.py --mode ga --synthetic     # offline

# Walk-forward validation (GA re-run per window)
python -c "
from nightevolver import fetch_nse_data
from nightevolver.backtest_evolved import EvolvedWalkForward
md = fetch_nse_data(['RELIANCE.NS','TCS.NS'], '2015-01-01')
print(EvolvedWalkForward(md).run().summary())"
```

Deployment uses the existing `HETZNER_HOST` / `HETZNER_USER` GitHub
secrets. The brief's literal `ssh -p 22141 root@69.30.85.174` is
deliberately **not** committed anywhere — infrastructure addresses
belong in secrets, not in a repo file.

---

## Acceptance criteria — realistic assessment

| # | Criterion | Assessment |
|---|---|---|
| 1 | Hit rate > 55%, Sharpe > 0.5, MaxDD < 10% | **Measurable, not guaranteed.** The spec calls these "now achievable"; changing model class does not create signal. The prior model measured 49.7% at n=26,210 *after* six real bug fixes. |
| 2 | Deflated Sharpe > 0.95 vs 10 baselines | **Implemented, and harder than stated.** n_trials is the real search budget (≥1,000), not 10. This is the correct bar and it is a high one. |
| 3 | Paper trade 3 months, positive returns | Executor + gate ready; not yet run. |
| 4 | RunPod runtime < 15 min | **Met with room to spare** — 13s for 10 generations; a full 20×50 run is well under a minute. |
| 5 | Signal latency < 100ms/tick | **Met and asserted by test.** |

**The honest prediction:** criterion 4 and 5 are comfortably met, 2 is
correctly enforced, and 1 is the open question. A constrained GA over
interpretable rules genuinely has a smaller hypothesis space than
architecture search — a real anti-overfitting argument. But it runs on
the **same daily OHLCV over the same ten NSE names** that produced
49.7%. If the information is not in that data, no search finds it, and
the deflated-Sharpe gate will correctly refuse to trade.

That refusal is the system working, not failing. The next move at that
point is **new inputs** — intraday microstructure, order flow,
cross-asset, fundamentals — not a different optimiser over the same
inputs.

---

## Tests

34 tests in `tests/test_nightevolver.py`, each able to fail if its
component is wrong. Headline:
`test_ga_overfits_pure_noise_and_the_controls_catch_it` asserts that a
GA on a random walk **fails** the deflated-Sharpe gate. If that ever
passes, the statistics are broken and every downstream number is
worthless.

Also covered: no-look-ahead probes on indicators and on `simulate()`
(mutate future bars, assert earlier rows unchanged), forward-return
alignment, checkpoint version/ordering/gate refusal, and that live
`signal()` output matches the training-time scorer exactly.
