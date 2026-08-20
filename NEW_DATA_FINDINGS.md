# New data sources and new targets — what was obtainable, and what it showed

Two things were asked for: add new data sources and re-run the GA, and
change the prediction target. Both are done. This is what the data
actually says.

The headline: **the first statistically defensible positive result in
this project — and it is not on direction.** Seven feature/target pairs
survive multiple-testing correction, all of them on volatility and
regime targets, none on direction, and none from the new flow data.

---

## 1. Feasibility, measured rather than assumed

Every source in the brief was probed directly before any code was
written against it.

| Source | Verdict | Evidence |
|---|---|---|
| **Tick / order-flow (Option A)** | **Cannot obtain** | Needs a paid vendor subscription and broker credentials. Historical tick data is sold separately and is not free. Nothing was built against it. |
| **Order book depth L2/L3** | **Cannot obtain** | Same. NSE Data & Analytics is a commercial licence. |
| **FII/DII cash daily** (`api/fiidiiTradeReact`) | **Not backtestable** | The endpoint **ignores its date parameters.** `?date=01-Aug-2026` and `?from=…&to=…` both returned byte-identical 216-byte payloads containing only the latest session. There is no history behind it. |
| **FII/DII derivatives positioning** (NSCCL archive) | **✅ Obtained** | `fao_participant_vol_DDMMYYYY.csv` *is* date-addressable. Verified back to 2019-04-01. 893 sessions cached. |
| **Corporate actions** | **✅ Obtained** | `api/corporates-corporateActions` works. Returned `Bonus 1:1, exDate 28-Oct-2024` for RELIANCE. |
| **Prices** (bhavcopy) | **✅ Obtained** | Official UDiFF bhavcopy, replacing yfinance — which is blocked outright from this sandbox. |
| **Change of target (Option C)** | **✅ Done** | Four targets: direction, relative strength, forward volatility, regime shift. |

A note on the recommendation that gets made most often: the free
`fiidiiTradeReact` endpoint is the one everyone points at for FII/DII,
and it is useless for research. It looks fine until you try to backtest
it. The NSCCL archive is the one that works.

**Corporate actions turned out not to be an alpha source at all — they
are a correctness prerequisite.** See §3.

---

## 2. The information audit: measure before searching

Every previous cycle in this project asked an optimiser for its best
score. That is a question about the search, and a 1,050-candidate search
returns a strong in-sample number whether or not the data contains
anything — measured last run, where the GA's winner (+1.29) scored
*below* the +1.62 that a best-of-1050 search over pure noise reaches.

`nightevolver/information_audit.py` inverts the order. No optimisation:
just the dependence between each feature and each target, with

1. **block permutation** (21-bar blocks) — features and targets are both
   autocorrelated, and iid shuffling gives a null far too narrow, so
   ordinary p-values come out spuriously tiny;
2. **rows permuted jointly across assets** — ten NSE large-caps on one
   day are not ten independent observations;
3. **Benjamini–Hochberg FDR across all 104 pairs** — at α=0.05 about 5
   of 104 look significant by chance alone;
4. **a persistence baseline partialled out of every target** — vol is
   autocorrelated, so "vol predicts vol" is nearly free, and only the
   incremental part counts.

### Result — 588 bars × 10 assets, 2024-04 to 2026-08, 2000 permutations

```
7 of 104 pairs survive FDR correction:
  atr_pct         -> vol_5d           incr rho=+0.1682  q=0.0208
  bb_width        -> vol_5d           incr rho=+0.1504  q=0.0208
  atr_pct         -> regime_shift_5d  incr rho=+0.1485  q=0.0173
  bb_width        -> regime_shift_5d  incr rho=+0.1384  q=0.0173
  ema_21_50_cross -> regime_shift_5d  incr rho=-0.0571  q=0.0173
  price_vs_ema50  -> regime_shift_5d  incr rho=-0.0514  q=0.0297
  rsi_28          -> regime_shift_5d  incr rho=-0.0492  q=0.0260
```

**Nothing survives on `direction_1d`.** Best was `fii_idx_fut_net_chg`
at ρ=+0.0412, q=0.47. **Nothing survives on `rel_strength_1d`.**
**No flow channel survives on anything.**

So the user's hypothesis — that volatility is more predictable than
direction — is **confirmed, on this data, with correction applied.** The
volatility findings are not enormous (ρ≈0.15–0.17) but they are real and
they are incremental to the trivial "tomorrow looks like today"
forecast.

### The caveat that has to travel with that number

**Predicting volatility is not the same as making money.** This system
trades cash equity direction. A vol forecast produces P&L only through
an instrument whose price is a function of vol (options — not traded
here) or through position sizing, whose edge has to come from somewhere
else. A ρ=0.17 vol forecast is a genuine finding and not yet a strategy.

---

## 3. Four bugs the controls caught

Each would have produced a confident, wrong, publishable-looking number.

**1. `PrvsClsgPric` is not corporate-action adjusted.** The first
version of `nse_prices.py` assumed it was and said so in its docstring.
It is the raw prior close:

```
RELIANCE, 1:1 bonus, ex-date 2024-10-28
  PrvsClsgPric = 2655.70   ClsPric = 1334.35   ->  -49.76%
```

That fake crash slipped under a ±50% "data error" clamp — a sanity
filter tuned to a round number misses the most common corporate action
there is. It gave RELIANCE 49.9% annualised vol (peers ~20%) and a
−53.8% total return. After real adjustment: **21.5% and −6.6%.**

**2. The archive answers HTTP 403 under load, not 404 — intermittently.**
The same URL returned 403 five times then `OK 955` on the sixth attempt.
Collapsing every failure to `None` turned throttling into phantom
holidays and silently lost ~20% of sessions (190/year against NSE's
~245), which forward-fill then papered over by presenting stale
positioning as current. After retry-with-backoff and 404/403
separation: **245 / 245 / 248 sessions per year, unresolved=0.**

**3. A mechanical artefact in my own regime target.**
`regime_shift_5d = log(fwd_vol / trail_vol)` carries trailing vol in its
denominator, and `atr_pct` *is* trailing vol — so they correlate with no
information involved. On a **pure random walk** the audit reported
`atr_pct → regime_shift_5d` at **incremental ρ = −0.3864, q = 0.04**. A
significant finding on structureless data.

It escaped the first noise test because that test used *random*
features. The artefact only appears when features are computed from the
same price series as the target — the real configuration. The
persistence baseline now controls for `log(trail_vol)` itself, and
`tests/test_new_data_and_targets.py` has a regression test that builds
**real indicators from a random walk** and asserts zero survivors.

Note what fixing it did to the headline: the sign **flipped**, from
−0.2182 to **+0.1485**. The artefact was masking a real positive
residual.

**4. Market-wide features inflating their own sample size.** The flow
features are one number per session broadcast across 10 assets. Pooled
as T×A they claim 5,870 observations from ~588 independent days, and the
rank transform breaks the cancellation a demeaned target should give a
market-wide predictor. Control: six **pure-noise** market-wide features
reached |incr ρ| up to **0.0126** on `rel_strength_1d` and produced 2–3
spurious FDR survivors per seed. The real `fii_stk_fut_net` scored
**0.0167** — the same order of magnitude.

Asset-invariant channels are now scored at market level, at their true
sample size. `fii_stk_fut_net → rel_strength_1d` went from "survives
FDR, q=0.039" to **p=0.60**. It was an artefact.

---

## 4. Re-running the GA with the new data

Flow channels are wired into the genome (`GENOME_VERSION` 2, length
65 → 83) so the GA can vote on them. Same data, same budget, flows off
vs on. Neither arm passes the gate; neither beats the noise benchmark of
+2.26.

The single-seed run is instructive precisely because of how good it
looks:

```
WITH flow channels
  IN-SAMPLE   Sharpe +1.71  win 51.1%  trades 31
  OUT-SAMPLE  Sharpe +1.18  win 66.7%  trades 3     <-- three trades
  deflated P(SR>0) = 0.004   (gate is 0.95)
```

A naive report writes "adding FII flow data lifted out-of-sample Sharpe
from −1.46 to +1.18 and win rate to 66.7%". That is **two winning trades
out of three.** The multi-seed spread in §5 settles it: across 8 seeds
the +1.18 never reappears, every run is ≤ 0.00, and the two arms are
statistically indistinguishable (p=0.27).

---

## 5. Multi-seed stability — the single-seed result was noise

8 seeds × 2 arms, same data, same budget:

```
seed | no-flow OOS  trd |  flow OOS  trd
   0 |       -2.32    3 |     -0.67    2
   1 |       -1.04    1 |      0.00    0
   2 |       -1.45    2 |     -2.53    3
   3 |       -2.13    5 |     -1.03    2
   4 |       -2.48    3 |      0.00    0
   5 |       -2.33    3 |     -0.82    2
   6 |       -1.04    1 |     -2.18    3
   7 |       -3.26    3 |      0.00    0

no flows    OOS Sharpe mean=-2.01 sd=0.72 range=[-3.26,-1.04] | median trades=3
with flows  OOS Sharpe mean=-0.90 sd=0.92 range=[-2.53,+0.00] | median trades=2
```

**The +1.18 does not reappear. Every one of the 16 runs is ≤ 0.00.** The
seed-42 result reported in §4 was a lucky draw on three trades, exactly
as suspected.

**And the apparent improvement is not real.** `with flows` looks better
(−0.90 vs −2.01) for one reason: **3 of its 8 seeds placed ZERO trades**
and were recorded as Sharpe 0.00. A strategy that never trades is not a
strategy scoring zero — it is an abstention, and averaging it in as a
zero drags the mean up. Excluding the non-trading seeds:

```
flows, seeds that actually traded:  mean -1.45  (n=5)
no flows, all seeds traded:         mean -2.01  (n=8)
Welch t-test:  t=-1.20  p=0.265     -> not distinguishable
```

**This is the same failure mode as the CHIMERA deadband collapse found
earlier in this project**: never-trading scores exactly 0.0, which beats
any negative score, so a search that is failing will select for
abstention and then report the abstention as a good number. Adding six
uninformative channels widened the search space 30% and made the GA
trade *less* (median 3 → 2), which is what "no information, more
parameters" looks like from the inside.

So: **adding the flow data did not help.** The audit said so in three
seconds; the GA took sixteen runs to agree.

---

## 6. What this changes about what to do next

- **Direction is done.** Three independent measurements now agree: the
  SNN at 49.7% (n=26,210), the GA's out-of-sample collapse, and an audit
  that finds no feature/target dependence surviving correction. More
  search over these inputs will keep producing better in-sample numbers
  and the same collapse.
- **Volatility and regime carry real, corrected signal.** That is worth
  building on — but through sizing or an instrument that pays for vol,
  not by relabelling it as a directional edge.
- **The new flow data carries nothing measurable** at its honest sample
  size, and the GA confirmed it independently: 8 seeds, every run ≤ 0.00
  out-of-sample, two arms indistinguishable at p=0.27. That is a real
  answer, and the audit reached it in three seconds against sixteen GA
  runs.
- **Watch for abstention being scored as success.** Three of eight
  flow-arm seeds placed zero trades and were recorded as Sharpe 0.00,
  which pulled the arm's mean *above* the arm that actually traded. This
  is the second time this exact pattern has appeared in this project
  (the first was CHIMERA's deadband collapse). A zero-trade run should
  be reported as an abstention, never averaged in as a zero.
- **The genuinely untested lever is still intraday microstructure** —
  the one source that could not be obtained free. Everything daily and
  free has now been measured.

---

## Reproducing

```bash
python scripts/run_information_audit.py --start 2024-01-01 --end 2026-08-19
python scripts/run_information_audit.py --synthetic     # calibration: expect 0 survivors
python -m pytest tests/test_new_data_and_targets.py -q  # 26 tests
```

The audit runs in about three seconds against 2000 permutations. That is
the point of doing it before the search rather than after.
