# NightEvolver walk-forward: the measurement nothing ever ran

**2026-08-21.** Raw results in `docs/results/` (tracked deliberately —
`logs/` is gitignored, and 16,800 GA evaluations of evidence should not
live only in a container that gets reclaimed).

Reproduce offline, no network or GPU, against the cached UDiFF bhavcopy:

```
python scripts/run_evolved_walkforward.py                    # 16 windows
python scripts/run_evolved_walkforward.py --rebuild          # matched to null
python scripts/run_evolved_walkforward.py --null             # control
```

---

## What was missing

`nightevolver/backtest_evolved.py` implements the one test that can
answer whether this GA produces durable strategies: a walk-forward that
**re-evolves inside every window** and scores strictly out-of-sample. It
existed, correct, with **no caller**. `grep -rl backtest_evolved` found
the module, its export, its unit test and the README — not
`scripts/train_nightevolver.py`, whose own docstring says the backtest is
*"not even imported by this path"*, and not `train-runpod.yml`.

Every number this project had quoted about the GA came from a **single
train/test split**. That split, as stored in
`checkpoints/nightevolver/nightevolver_best.json`:

| | |
|---|---|
| in-sample | 3 trades over 459 bars, Sharpe 0.60 |
| out-of-sample | **0 trades** |
| deflated P(SR>0) | 0.0119 (gate 0.95) |
| beats_noise | False |

The OOS Sharpe of 0.0 is not "no edge" — it is *nothing happened*. The
strategy never fired in the holdout, so there was nothing to measure, and
the reported 0.60 "overfitting gap" is in-sample minus a number that does
not exist. Both figures looked like measurements.

---

## Results

| | real (full) | real (matched) | **null** |
|---|---|---|---|
| bars / windows | 589 / 16 | 528 / 14 | 528 / 14 |
| hit rate | 49.8% (p=0.836) | 50.0% (p=1.000) | 49.3% (p=0.505) |
| n directional calls | 3,359 | 2,748 | 2,758 |
| Pearson r | −0.0118 (p=0.49) | +0.0083 (p=0.66) | −0.0090 (p=0.64) |
| net Sharpe | −1.07 | −1.12 | −1.39 |
| total return | −6.15% | −4.48% | −8.65% |
| mean in-sample Sharpe | +2.02 | +1.90 | +1.71 |
| **overfitting gap** | **+2.75** | **+3.25** | **+2.84** |
| OOS trades | 103 | 59 | 114 |
| deflated P(SR>0) | 0.000 | 0.000 | 0.000 |

`n_trials` for the deflated Sharpe is windows × GA budget — 16,800 and
14,700 — the honest count when a 1,000-evaluation search is re-run every
window.

**Which column to quote.** The full real run is the primary result: most
data, most windows, 103 out-of-sample trades. The matched pair exists
only for the real-vs-null comparison, because `--null` rebuilds
MarketData and incurs a second `WARMUP_BARS` trim (589 − 61 = 528); the
matched run puts real data through that identical path so permutation is
the only difference. Its 59 pooled trades make it thinner evidence on its
own.

---

## Definitions

**Overfitting gap.** Per window, the GA evolves on the training slice and
the winning genome is scored on the untouched test slice. The gap is the
mean across windows of that per-window difference:

```
gap = (1/W) * SUM_w [ Sharpe_in-sample(w) - Sharpe_out-of-sample(w) ]
```

`nightevolver/backtest_evolved.py:141`. Both terms are annualised Sharpe
from the same `simulate()` call — same 22bp cost model, same long-only
constraint, same vol targeting — so the difference is attributable to the
data slice and not to a change of measurement.

Three things this definition is easy to misread:

- **The OOS term is the mean of per-window Sharpes, not the pooled
  Sharpe.** For the full real run those differ: mean per-window OOS
  Sharpe is **−0.73**, while the Sharpe of the concatenated daily-return
  series is **−1.07**. The gap of +2.75 uses the former
  (2.02 − (−0.73) = 2.75). The pooled figure is the better estimate of
  what the strategy would have *earned*; the per-window mean is the right
  input to a gap, because it pairs each in-sample number with the
  out-of-sample number from the same evolution.
- Mean-of-differences equals difference-of-means here, so quoting it
  either way gives the same number — but only because every window
  contributes one of each.
- **It is a difference of Sharpes, so it is in Sharpe units and inherits
  their noise.** At 2–10 trades per window a single window's OOS Sharpe
  is unstable, which is why the gap is only quoted as a mean over windows
  and compared against a null cloud rather than read as a point estimate.

**In-sample Sharpe** is the winning genome's Sharpe on the training
window it was evolved on — `res.in_sample.sharpe`, the GA's own fitness
view. It is not a held-out number and is reported only to form the gap.

**n_trials** in the deflated Sharpe is `windows × GA search budget`
(`backtest_evolved.py:176`), i.e. every genome evaluation across the
whole walk-forward. This is the honest count when a search is re-run per
window; using the per-window budget alone would understate selection by
the number of windows.

---

## What it means

**1. The GA overfits, by a factor that is now measured.** In-sample
Sharpe is positive in **16 of 16 windows** (mean +2.02). Out-of-sample it
delivers −1.07. A gap of +2.75, reproduced across sixteen independent
re-evolutions.

**2. The in-sample number is a search artifact, not signal we lost.**
This is what the null control settles, and it is the finding that changes
what to do next. Block-permuted data — signal destroyed by construction —
produces in-sample +1.71 and a gap of **+2.84**, statistically the same
as real data's +2.75, and on the matched comparison real data's gap
(+3.25) is *larger* than the null's. A +2.75 gap alone invites the
response *"we have signal, we just need better regularisation"* — smaller
population, stronger penalties, fewer generations. The null shows that is
wrong. Tuning the search cannot recover an edge that is not there.

**3. The search never finds the same thing twice.** Across 16 windows
there are **15 distinct indicators in the #1 slot**; the most any one
repeats is 2, and the top-3 has touched 23 of 26 available indicators.
Durable structure would show the same indicators winning repeatedly.
This is what fitting noise looks like from the inside — and permuted data
behaves the same way.

**4. The per-window win rate actively misleads.** 10 of 16 windows were
OOS-profitable (62.5%, binomial p=0.45), which reads as encouraging until
the sizes are compared:

```
wins:    +3.45  +3.05  +2.73  +2.30  +1.41  +1.33  +1.11  +1.02  +0.37  +0.02
losses:  −7.20  −6.88  −5.14  −4.18  −3.17  −1.97
```

Win often, lose big; mean −0.73. Reporting "62.5% of windows profitable"
would state the exact opposite of the truth.

**5. There is no edge hiding behind a sign error.** Hit rates are 49.8%,
50.0%, 49.3% — one is *exactly* a coin flip at p=1.000 on 2,748 calls.
At n=3,359 the 95% interval is roughly 48.1–51.5%: not "no edge
detected", but "an edge large enough to matter is excluded".

---

## The null cloud (30 permutations)

`docs/results/null_cloud.json` — `python scripts/run_null_cloud.py --n 30`.
Matched trimming, 14 windows each, GA seed fixed at 42 so only the
permutation varies.

| statistic | real | null mean ± sd | null 95% | pct | P(null ≥ real) |
|---|---|---|---|---|---|
| overfitting gap | +3.254 | +2.523 ± **1.559** | [−0.510, +5.738] | 80th | 0.226 |
| pooled OOS Sharpe | −1.125 | −0.472 ± 1.008 | [−2.428, +1.183] | 20th | 0.806 |
| mean in-sample Sharpe | +1.897 | +1.808 ± 0.213 | [+1.496, +2.275] | 77th | 0.258 |
| raw hit rate | 0.500 | 0.491 ± 0.009 | [0.476, 0.506] | 80th | 0.226 |

**Every real statistic falls inside the null's 95% band.** Real market
data is not distinguishable from data with its signal permuted out.

**A single null draw was not enough, and an earlier version of this
document over-claimed on one.** It read "the gaps are essentially
identical: +2.75 real vs +2.84 null" and treated that as agreement. The
cloud shows the null gap has **sd 1.559** and spans −1.94 to +5.84; two
draws landing 0.09 apart is coincidence, not agreement. The conclusion
survives, but it rests on the **in-sample Sharpe** — real +1.897 against
null +1.808 ± 0.213, 77th percentile — not on the gap.

**Power, stated honestly.** With sd 1.559 at 14 windows, the gap could
not detect a real difference of ±1.5 Sharpe; it is close to uninformative
here. The **hit rate** is the tight statistic (sd 0.009) and carries most
of the evidence. Note also that the null's hit rate sits systematically
below 0.5 (mean 0.491, max 0.506) — an asymmetry from long-only plus
abstention, not a defect — so "real at the 80th percentile" describes a
0.9-percentage-point difference, not an edge.

---

## The volatility and regime targets, same control

`docs/results/target_null_cloud.json` —
`python scripts/run_target_walkforward.py --n 30`. Per window: rank all
26 indicators on TRAIN by incremental Spearman over the persistence
baseline, take the best, score that one pick on TEST. 5 windows
(train 252, test 63), 30 permutations.

This is the case where the information audit **did** find signal — 7
pairs surviving a 2,000-draw block-permutation null under BH-FDR,
incremental ρ of 0.15–0.17 for `atr_pct` and `bb_width`.

| target | IS (selected) | OOS | null OOS mean ± sd | pct | P(null ≥ real) |
|---|---|---|---|---|---|
| `vol_5d` | +0.2352 | +0.0779 | +0.0852 ± 0.0304 | 37th | 0.645 |
| `regime_shift_5d` | +0.2090 | +0.0725 | +0.0822 ± 0.0220 | 33rd | 0.677 |
| `rel_strength_1d` | +0.0474 | +0.0401 | +0.0394 ± 0.0150 | 57th | 0.452 |
| `direction_1d` | +0.0487 | +0.0377 | +0.0501 ± 0.0141 | 17th | 0.839 |

**All four inside the cloud**, and on the two targets the audit
flagged, the real OOS score is *below* the null mean.

**But the picks are stable, unlike the GA's.** `vol_5d` and
`regime_shift_5d` both select `bb_width, bb_width, bb_width, atr_pct,
atr_pct` — two distinct features across five windows, and exactly the two
the audit named, against the GA's 15-distinct-in-16. The selection is not
thrashing on noise; it reliably locates the relationship the audit found.
That relationship simply does not pay out-of-sample above what permuted
data delivers, because a max-over-26 selection extracts ~+0.085
incremental Spearman from a series with no signal in it.

**These are not the same test as the audit**, and both results stand.
The audit asked *"does a relationship exist in this history?"* and
answered yes at p=0.0005. This asks *"can it be selected in advance and
beat noise?"* and answers no. A real but unusable relationship satisfies
both.

**Limits.** Five windows is thin. What argues against "underpowered
rather than absent" is the direction: the real point estimate is below
the null mean on three of four targets. The OOS score also takes `abs()`,
so a feature that flips sign out-of-sample still counts as a success —
generous to the real data, and it still does not clear the null.

---

## Reading per-window figures

At a 21-bar test window, individual windows carry 2–10 trades. **A
per-window Sharpe on that many trades is noise.** Window 15 shows +3.45
on two trades; window 13 shows −5.14 on twenty-six. Only the pooled
series has the sample size to carry a claim. `run_evolved_walkforward.py`
prints a warning when the median falls below ~2 trades/window, and
`required_validation_bars()` in `nightevolver/ga_engine.py` exists
because this project already made this mistake once.

---

## Consequences

The `nightevolver.enabled: false` default in `kronos/config.yaml` and the
0.95 deflated-Sharpe gate in `nightevolver/saver.py` are now backed by a
direct measurement rather than inferred from the information audit. The
gate rejects the current checkpoint at P(SR>0) = 0.000.

Worth naming: had the bridge been wired and switched on without this run,
the gate would have refused the checkpoint anyway — but nobody would have
known *why*, and the obvious next move would have been to loosen the
gate. That is precisely the wrong move, and only the null control makes
it visibly wrong.

Three independent lines now agree — the information audit (no directional
edge by three methods), the cost arithmetic (break-even win rate above
what measured correlations support), and this walk-forward with its
control. The direction they point is away from tuning the search and
toward the data itself.

---

# The 2019–2026 extension: `atm_iv → vol_5d` does not survive

## What this run was for

One result in this project was never resolved. The information audit
found `atm_iv` (implied volatility from the F&O bhavcopy) predicting
`vol_5d` (realised forward volatility) at **p = 0.065**, 97th percentile
of its null cloud, with a *negative* overfitting gap and 5/5 unanimous
window picks. Everything about it looked like an effect limited by
statistical power rather than by size: ~16 windows over ~2.5 years of
UDiFF-era data, against a floor of `1/(n+1)` on the empirical p-value.

The stated fix was more history. That required backfilling the legacy
(pre-2024) NSE archive — a different file format for both equity and
F&O — to reach ~7.6 years and roughly 30 windows.

## The panel

| | |
|---|---|
| Equity sessions cached | 1,886 (2019-01-01 → 2026-08-21) |
| F&O sessions cached | 1,884 |
| Corrupt files | 0 |
| Universe | top 100 by turnover **as of 2019-01-01** |
| Panel | 1,825 bars × 100 tickers, 93.9% of cells finite |
| Windows | 24 (up from ~16) |
| Null draws | 30 block permutations, 21-bar blocks |

The universe is point-in-time and the panel is *ragged*: names that
delisted are absent after their last trade rather than dropped from the
study or carried forward as flat lines. JETAIRWAYS ends after 60 bars,
DHFL after 542, HDFC after 1,059 — which lands on July 2023, its actual
merger date.

## The result

| target | selected | OOS (real) | null mean | null 95% | pct | p |
|---|---|---|---|---|---|---|
| `vol_5d` | `atm_iv` 24/24 | +0.3467 | +0.3587 | [0.3380, 0.3798] | 17 | 0.839 |
| `regime_shift_5d` | `atm_iv` 24/24 | +0.3238 | +0.3353 | [0.3169, 0.3558] | 10 | 0.903 |
| `direction_1d` | 10 distinct | +0.0412 | +0.0367 | [0.0291, 0.0454] | 77 | 0.258 |
| `rel_strength_1d` | 7 distinct | +0.0366 | +0.0288 | [0.0213, 0.0362] | 97 | 0.065 |

**`atm_iv → vol_5d` is inside the null cloud, and below the null mean.**
More history did not rescue p = 0.065; it removed it. The honest reading
is that the earlier result was a small-sample artifact of ~16 windows,
not an effect waiting for power.

The selection is still unanimous — `atm_iv` is chosen in 24/24 windows
for both volatility targets — which is worth separating from the
conclusion. `atm_iv` genuinely *is* the best available predictor of
realised volatility. It is simply not better than what the same
machinery extracts from a permuted series, because implied and realised
volatility share a contemporaneous level that block permutation
preserves. Being reliably the best of a set is not the same as carrying
information, and this is what distinguishes the two.

`rel_strength_1d` at p = 0.065 is inside the cloud and should not be
read as a near-miss: its picks are unstable (7 distinct features across
24 windows), which is the signature of selection noise rather than a
signal being tracked.

## Why the first run of this had to be thrown away

The first execution reported `pcr_volume → direction_1d` **above** the
null cloud at p = 0.032, 24/24 windows — a directional edge.

It was a one-bar look-ahead. `build_market_data` keeps
`slice(WARMUP_BARS, len-1)` — 60 bars off the front and one off the back
— and the extra channels were trimmed by length arithmetic
(`trim = len(ex) - md.n_bars` = 61), putting the whole trim on the front.
Every derivative and delivery channel sat one bar early, so `feature[t]`
held day `t+1`'s data.

    pcr_volume vs same-bar return (t-1 → t)   rho = -0.2826
    pcr_volume vs FORWARD return (t → t+1)    rho = +0.0239

The entire effect was contemporaneous: high put volume accompanies down
days, and dating it one bar early turns that everyday fact into a
forecast.

**Two tells were visible in the output before the bug was found**, and
both are now tests:

* A feature five times more correlated with the future than with the
  present is not a predictor, it is a calendar error. Genuine predictors
  are *weaker* on the future.
* The directional null cloud sat at **+0.283** rather than near zero. A
  null that far from zero means the statistic is measuring something
  mechanical. After the fix the same clouds sit at +0.0367 and +0.0288 —
  which is what a directional null is supposed to look like, and is the
  clearest single confirmation that the fix is correct.

The null cloud could not have caught this on its own. Extra channels ride
the same permutation as prices by design, so a same-row relationship
survives every draw intact. The control is structurally blind to
misalignment — which is why alignment needs its own test rather than
trust in the null.

## Where this leaves the project

Four independent lines now agree that there is no directional edge in
this data at this horizon: the information audit, the cost arithmetic
(break-even needs ρ ≈ 0.31 against ρ ≈ 0.046 measured), the GA
walk-forward against its null cloud, and now the target walk-forward on
7.6 years with a point-in-time universe and an honestly ragged panel.

The volatility result, which was the last open question, closes with
them. Nothing here says volatility is unpredictable — `atm_iv` tracks it
well. It says this pipeline extracts no *incremental* forecast beyond
what the same procedure finds in permuted data.
