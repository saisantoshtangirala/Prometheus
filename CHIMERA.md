# CHIMERA

A six-component hybrid trading system: dynamic market-network features,
quantum-inspired feature selection, chaotic-oscillator attention,
critic-free RL, physics-informed no-arbitrage constraints, and
quality-diversity evolution — composed as one pipeline and evaluated by
the same walk-forward machinery as everything else in this repo.

## Standing caveat, stated once and up front

This repository has a measured result, at n=26,210 observations across a
full 125-window walk-forward, that the previous production model has
**no detectable directional edge** (hit rate 49.7%, p=0.31; Pearson
p=0.86; Spearman p=0.57). That result was obtained *after* fixing six
genuine bugs, so it is not an artefact of broken code.

None of CHIMERA's six components addresses the most likely reason for
that. They add model capacity, better feature *selection*, better
*optimisation*, and better *deployment* — but the binding constraint is
most likely the **information content of the inputs and labels**: daily
OHLCV on ten NSE large-caps, predicting next-day direction. If the
signal is not in the data, no architecture recovers it.

So CHIMERA is built and documented as **an experimental platform for
testing that hypothesis**, not as a system expected to reach a 55% win
rate. The honest use of it is: run it, and if the raw-signal diagnostic
still reads ~50% with a non-significant p-value, that is strong evidence
the problem is the data, and the next move is new *inputs* (intraday
microstructure, order flow, cross-asset, fundamentals) rather than a
seventh component.

## Why the ₹10,000 target needs restating

At ₹10,000 (~$120) across ten NSE large-caps, the account cannot express
the portfolio. This is measured, not asserted — from
`tests/test_chimera.py` and the harness's own diagnostics:

| Capital | Untradeable names | Quantisation error (L1/bar) |
|---|---|---|
| ₹10,000 | 3.37 / 5 | 0.264 |
| ₹50,000 | 0 / 5 | 0.061 |
| ₹200,000 | 0.24 / 5 | 0.018 |
| ₹1,000,000 | 0 / 5 | 0.003 |

A 20% target weight in a ₹3,000 stock is ₹2,000 — which buys **zero
shares**, since NSE's cash segment has no fractional trading. In one
backtest run the identical model produced turnover of 0.007/bar at
₹10,000 versus 0.101/bar at ₹200,000: same signal, an order of magnitude
difference in whether it can be deployed at all.

Second constraint: NSE delivery round-trip cost is **~22bp**, dominated
by STT. At 100% daily turnover that is ~56%/year in costs. A 55% win rate
with symmetric win/loss magnitudes implies an expected edge of ~10% of
average move size; NSE large-caps move ~1.2%/day, giving ~12bp/bar —
which does **not** clear 22bp. This is why turnover control (component
6's deadband and smoothing genes) is load-bearing rather than cosmetic,
and why the reward function charges costs *inside* training.

Neither point is a reason not to build the system. Both are reasons the
evaluation reports raw signal and net portfolio **separately** — a model
can have real edge and still be undeployable at ₹10,000, and that is a
completely different problem from having no edge.

## The six components

| # | Component | What it actually does | File |
|---|---|---|---|
| 1 | Financial Connectome | Ledoit-Wolf shrinkage → precision matrix → partial-correlation graph → centrality/clustering + Fiedler value, von Neumann entropy | `connectome.py` |
| 2 | Quantum-inspired selection | mRMR as QUBO (distance correlation), solved by **ballistic Simulated Bifurcation** | `qubo_select.py` |
| 3 | Chaotic attention | Lorenz oscillator seeded *from the market state*, driving per-head attention temperature | `chaotic_attn.py` |
| 4 | DAPO/GRPO | Group-relative advantage (no critic) + clip-higher, dynamic sampling, asset-level loss | `grpo.py` |
| 5 | Physics-informed | Cross-sectional no-arbitrage penalty + Black-Scholes PDE residual via autograd | `pinn.py` |
| 6 | Quality-Diversity | MAP-Elites over (turnover, net exposure, concentration) | `qd_archive.py` |

Composition:

```
OHLCV → connectome ──→ feature bank → QUBO subset
      → chaotic-attention encoder ─┬→ PINN value head   (constraint)
                                   ├→ GRPO policy head  (action)
                                   └→ return head       (signal)
      → MAP-Elites ensemble → NSE integer-share sizing
```

### Design decisions worth defending

**Why partial correlation, not correlation (1).** Two assets that both
track the index are highly correlated but conditionally independent.
Correlation cannot tell that apart from a genuine direct link; the
precision matrix can. Verified by test: a planted common factor drops
the partial correlation to under half the marginal.

**Why distance correlation, not Pearson (2).** dCor is zero *iff*
variables are independent, so it detects nonlinear dependence. Measured:
for `y = x²`, dCor = 0.554 vs Pearson 0.201. Using Pearson to select
features for a nonlinear model discards exactly the features that model
exists to exploit.

**Why the Lorenz burn-in is load-bearing (3).** Without it, nearby seeds
**converge** rather than diverge — Lorenz has one positive Lyapunov
exponent (~0.9) but strongly negative transverse ones (~−14.6), so an
off-attractor seed collapses onto the attractor faster than it separates
along it. This was found empirically when the first sensitive-dependence
test failed. With a 1-time-unit burn-in the measured largest Lyapunov
exponent is **+0.38 to +0.82**, converging toward the classical value.
Chaos parameters are sigmoid-bounded into a genuinely chaotic range so
"learnable chaos" can never silently become "learned fixed point".

**Why critic-free RL specifically suits this problem (4).** A value
critic must learn E[return | state] — which is precisely the quantity
this project has failed to predict. Using it as a variance-reduction
baseline is circular. GRPO's group-relative baseline only asks "which of
these G portfolios did better *this bar*", which is well-posed even when
the absolute expected return is unlearnable.

**Why the PINN test is the strongest in the suite (5).** The residual is
checked against the **closed-form analytic Black-Scholes solution**:
max|residual| = 5.1e-06 on the exact solution, and 2.2e+01 with a wrong
volatility — seven orders of magnitude apart. A wrong autograd second
derivative cannot pass both. (`ValueSurface` uses tanh, not ReLU, whose
second derivative is zero almost everywhere and would make any PINN
built on it silently meaningless.)

**Why MAP-Elites rather than plain evolution (6).** Plain evolution
converges to N near-copies of one optimum; an "ensemble" of correlated
strategies has the risk profile of a single strategy. MAP-Elites keeps a
*worse* genome if it behaves differently, so the ensemble is genuinely
diversified. Verified by test: a fitness-0.1 genome in an empty
behaviour cell is retained over a fitness-5.0 incumbent elsewhere.

## A real bug this build surfaced

Integration testing caught a scale-mismatch bug that every unit test
passed through, and it is worth recording because it is the *same class*
of bug as the SNN `size_scale` collapse found in this repo's audit:

The return head predicts on the scale of daily returns (~0.1), but the
MAP-Elites genome's `signal_threshold` gene is sampled in [0, 0.9] and
compared against `tanh(gain × signal)`. With a raw signal of 0.17 and
gain 0.70, the squashed value is ~0.12 — under almost every sampled
threshold. **Every genome therefore produced an all-zero portfolio**, and
MAP-Elites then *selected for* that outcome, because never trading
scores exactly 0.0 while any real position loses money to 22bp costs.
The archive's best fitness was exactly `0.0000`.

That is a rational conclusion reached for entirely the wrong reason, and
it made the whole search space unreachable. Fixed by normalising the
signal by its train-set standard deviation before the genome sees it, so
the gain and threshold genes address the distribution they were designed
for. Regression test:
`test_signal_scale_prevents_the_all_zero_deadband_collapse`.

## Running it

```python
from kronos.backtest import WalkForwardConfig, load_history
from chimera.backtest_chimera import ChimeraWalkForward
from chimera.strategy import ChimeraStrategyConfig

tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
           "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS"]
closes = load_history(tickers, "2015-01-01")

result = ChimeraWalkForward(
    closes, tickers,
    config=WalkForwardConfig(train_window=252, test_window=21),
    strategy_config=ChimeraStrategyConfig(),      # .fast() for smoke runs
    capital=10_000.0,
).run()                                            # all ~125 windows

print(result.summary())
```

`ChimeraStrategyConfig.fast()` collapses the expensive stages for tests.
Measured cost in fast mode: ~19s/window, so ~40 minutes for 125 windows.
Full config is meaningfully slower; time it before committing to a run.

## Test coverage

53 tests in `tests/test_chimera.py`, written so each *can fail* if its
component is wrong. Wherever ground truth exists it is used rather than
the code's own output — the analytic Black-Scholes solution, a planted
conditional dependency, a QUBO with a known optimum, a known-optimal
portfolio direction, an exactly-riskless spread. Look-ahead is probed by
mutating future bars and asserting past features are unchanged, at both
component and end-to-end level.

This standard is deliberate: this repo's audit found a MAML
implementation that was a **mathematical no-op** while every shape test
passed.

## What is NOT built

- **LLM-enhanced RL.** The brief mentions it; no LLM is in this loop.
  `GroupRelativePolicy.collect_and_step` takes the reward function as a
  callback, so an LLM-derived reward can be supplied without touching
  the optimiser — but claiming an LLM component that does not exist
  would be worse than leaving the seam.
- **Live paper-trading integration.** CHIMERA deliberately does not
  touch `kronos.service`. The 3-month paper-trade gate should run only
  after the walk-forward raw-signal diagnostic clears — deploying an
  unvalidated model into the live orchestrator is how attribution gets
  lost.
- **Research citations.** These components are implemented from their
  underlying mathematics. Where a method has a canonical source
  (Simulated Bifurcation, GRPO/DAPO, MAP-Elites, Ledoit-Wolf) it is
  named in the module docstring, but no claim is made to have surveyed a
  2025–26 literature this build did not read.
