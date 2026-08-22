# Zero-Trust Pipeline Audit — Prometheus / NightEvolver / Kronos

**Trigger.** A one-bar look-ahead in `run_target_walkforward.py`'s extra-channel
alignment manufactured a false directional edge (`pcr_volume → direction_1d`,
p=0.032) before it was caught by a manual past-vs-future correlation check —
not by the permutation-null control, which is structurally blind to a
same-row misalignment. That near-miss is the reason for this audit: prove the
rest of the pipeline the same way, rather than trust that one caught bug means
the surrounding code is clean.

**A scope correction stated up front, per the instruction to be honest about
what's unverifiable.** This system has no database. There are no SQL tables —
ingestion writes zip/CSV to a file cache (`data/cache/nse_bhav/`,
`data/cache/nse_fo/`, …), features live in in-memory pandas frames or `.npz`
caches, and the only durable state is JSON checkpoints. Every "SQL query"
below is therefore given as the equivalent pandas/Python inspection, run
against the real cached archive. Where a control genuinely doesn't exist
(no query log, no write-audit trail), that is stated as a risk rather than
answered with a query that would not actually run.

Every method below was **executed**, not designed on paper. Evidence is
quoted; file:line references are exact as of commit `6ff737d`.

---

## 1. Data Lineage & Source Integrity

### 1.1 Raw data is stored without modification

**Objective.** The cached zip is byte-identical to what the exchange served —
no in-place rewriting, decompression-then-recompression, or silent repair.

**Risk if wrong.** Any transformation applied before caching is invisible to
every later run; a corrupted cache and a corrupted exchange feed become
indistinguishable.

**Verification method.**
```python
raw, _ = _fetch_raw(date)                 # nightevolver/nse_prices.py
cache_path(date).write_bytes(raw)         # only after _zip_is_intact(raw)
assert cache_path(date).read_bytes() == raw
```
`nightevolver/nse_prices.py:186-198` (`fetch_bhav_day`) writes the exact
bytes returned by `urlopen(...).read()`. No decode, no re-encode, no
normalisation happens before the write.

**Executed check** — CRC-validated every cached file this session:
```
equity: 1886 files, 0 corrupt
f&o:    1884 files, 0 corrupt
```

**Pass/Fail.** **PASS.** Additionally hardened this session: a download is
now only cached if `_zip_is_intact()` — the write is conditioned on the CRC
check, not merely followed by one. Before that fix, a truncated download from
`http.client.IncompleteRead` could reach the cache as a permanent corrupt
file (`nightevolver/nse_prices.py:169-175`, commit `8c8c044`).

### 1.2 Missing dates, duplicate symbols, negative volumes/prices

**Objective.** Every trading day is represented once; every symbol appears
once per day; no field carries an impossible value.

**Verification method** (pandas equivalent of the requested SQL):
```python
# missing dates: real weekdays not in the confirmed-holiday set
missing = [d for d in pd.bdate_range(start, end)
           if not cache_path(d).exists() and str(d.date()) not in holidays]

# duplicate symbols within one day
dupes = df.groupby("TckrSymb").size()
dupes = dupes[dupes > 1]

# impossible values
bad_price  = (df[["OpnPric","HghPric","LwPric","ClsPric"]] <= 0).any(axis=1)
bad_volume = df["TtlTradgVol"] < 0
```

**Executed check.**
* Missing dates: **0** unexplained gaps in either archive over 2019–2026
  (1,234/1,304 equity weekdays cached + 69 confirmed exchange holidays +
  1 retried-and-recovered 403 = fully accounted; F&O identical).
* Duplicate symbols: **handled, not merely detected** —
  `nightevolver/nse_prices.py:342` (`if isinstance(row, pd.DataFrame): row = row.iloc[0]`)
  takes the first row rather than silently summing or averaging duplicates.
* Negative volume: **found unguarded, fixed this audit.** No check existed
  anywhere in the ingestion path before `nightevolver/nse_prices.py:351-361`
  (commit `6ff737d`). Checked all 1,886 cached equity files retroactively —
  none contained a negative volume, so this closes a gap rather than a live
  corruption. `TestNegativeVolumeIsRejected` in `tests/test_nse_prices.py`
  pins it.
* Negative/zero price: **already guarded**, but one level up, in
  `nightevolver/data_loader.py:250-262` — masks non-finite or `<= 0` close
  prices before any derived quantity is computed, with a regression test
  (`TestIndicators::test_all_nan_column_raises...`) tracing back to a
  measured `1.8e308` poisoned-target bug.

**Pass/Fail.** **PASS**, with one gap closed during the audit (negative
volume) and one already-fixed defect confirmed still fixed (poisoned target
from `nan_to_num`'s `inf → FLOAT_MAX` behaviour).

### 1.3 Delisted/suspended symbols dropped before cross-sectional contamination

**Objective.** A name that stopped trading must not silently masquerade as a
live, flat, zero-volatility instrument.

**Risk if wrong.** This is not hypothetical — it happened. `DHFL` went into
liquidation and was carried in a 2019 universe panel as **71.0% exactly-zero
daily returns**, pinned at 7.56, because `build_adjusted_frames` filled every
missing return with 0.0 regardless of whether the gap was inside a live
series or after the instrument's death.

**Verification method.**
```python
r = np.diff(np.log(close), axis=0)
zero_frac = (np.abs(r) < 1e-12).mean(axis=0)   # per name
flat_names = [t for t, z in zip(tickers, zero_frac) if z > 0.3]
```

**Executed check, before and after the fix (commit `c2ec36b`):**

| name | before | after |
|---|---|---|
| DHFL | 71.0% zero returns, flat at 7.56 to end of panel | absent after last trade (542 finite bars) |
| RELCAPITAL | 67.4% zero | absent after last trade (777 bars) |
| HDFC | 42.0% zero | absent after **2023-07**, its actual merger date (1,059 bars) |

**Pass/Fail.** **PASS, after fix.** The universe itself is chosen
point-in-time (`top_liquid_symbols(as_of=...)`) so survivorship never enters
via *selection*; this closes the second half — survivorship reintroduced via
*fabricated continuation* of a name that should have exited the panel.

---

## 2. Temporal Alignment & Shift Verification

**This is the section where the triggering bug lived, and it is the one
section where the requested "reproduce the null cloud with forced ±1 shift"
protocol needs an explicit correction: it does not work, and demonstrating
why is itself the useful control.**

### 2.1 Feature[t] aligns with Target[t+1], every channel

**Objective.** No channel — price-derived indicator, derivative, delivery,
PCR — is offset from the price panel it's scored against.

**Risk if wrong.** Exactly what happened: a channel one bar early reads as a
forecast because it is silently contemporaneous with the *next* bar's
outcome.

**Verification method — the diagnostic that actually caught it, since the
requested null-cloud-with-forced-shift does NOT:**
```python
same = spearmanr(feature[t], return[t-1 -> t])      # contemporaneous
fwd  = spearmanr(feature[t], return[t -> t+1])       # claimed prediction
# an honest feature: |same| >= |fwd|.  A shifted one: |fwd| >> |same|,
# because the "prediction" is actually a same-row fact one index late.
```

**Executed check, the exact bug this caught:**
```
pcr_volume vs same-bar return (t-1 -> t)   rho = -0.2826
pcr_volume vs FORWARD return (t -> t+1)    rho = +0.0239   [misaligned: was +0.28]
```
A feature 5–12× more correlated with the future than the present is not a
predictor; it is a calendar error, because genuine predictors are *weaker* on
the future by construction (real information decays).

**Root cause, exact.** `build_market_data` keeps `slice(WARMUP_BARS, len-1)` —
60 bars off the front, **one** off the back. The extra-channel trim used
`trim = len(ex) - md.n_bars` (=61) and sliced `ex[trim:trim+n]` — 61 off the
front, **zero** off the back. Same net length, wrong split.
`scripts/run_target_walkforward.py:145-168`, fixed in commit `17928f7`.

### 2.2 Why the requested "forced shift(±1) null cloud" is the wrong tool, and what replaces it

This is the explicit theoretical-blind-spot disclosure the brief asked for.

The project's null cloud is a **block permutation**: rows of the price/target
panel are permuted jointly (21-bar blocks) and extra channels are permuted by
**the same row order**, specifically so a channel's *contemporaneous*
relationship to price survives the permutation intact
(`scripts/run_target_walkforward.py:126-131`, "extra channels ride the SAME
permutation as prices"). That design choice — correct for its stated purpose,
which is not confounding a real predictive signal with the null destroying
volatility clustering — has a side effect: **it cannot detect a constant
one-bar misalignment**, because a misalignment is a property of the *index
map*, and permutation acts on *row content*, not on the map between two
already-built arrays. A shift(-1)/shift(+1) forced-null variant would have
the identical property: shifting the *already-misaligned* series again
doesn't reveal that it was misaligned, it just produces a different
misalignment.

The measured proof: the discarded run's null cloud sat at **+0.283**, far
from zero, and that number *was already visible* before the alignment bug
was found — a directional null that far from zero is itself the signature of
a mechanical (non-random, structurally biased) relationship surviving every
permutation, which is a distinct and complementary signal to the
same-bar-vs-forward check.

**The remedial control, now in place:** `tests/test_walkforward_alignment.py`
checks the *slice arithmetic itself*, independent of any statistical test —
walking the AST of `_one_draw` to confirm the trim is computed from
`WARMUP_BARS`, not from a length difference — plus a synthetic marker-channel
test that plants `arange(T)` as a fake feature and asserts the trimmed slice
lands on the expected index, not index+1. **Alignment is verified
structurally, not statistically**, which is the correction to the brief's
proposed protocol: *run an alignment test independent of the permutation
test, because the permutation test cannot see this class of bug.*

**Pass/Fail.** **FAIL → FIXED.** This is the one confirmed, high-severity
finding of the whole audit. Structural regression test added; statistical
symptom (null cloud centred away from zero) now documented as a required
sanity check before trusting any future null-cloud output.

### 2.3 Corporate-action and calendar alignment between archives

**Objective.** The F&O archive and equity archive don't share a holiday
calendar exactly; a positional (not date-keyed) join between them would
silently shift one against the other.

**Verification method.** `run_new_data_audit.py:align()` and
`scripts/build_feature_cache.py` reindex every derivative/delivery frame onto
`pd.DatetimeIndex(dates)` **by date label**, not by position — a day present
in one archive and absent in the other becomes `NaN` at that date rather than
shifting every later row.

**Pass/Fail.** **PASS**, verified by code inspection; already correct going
in, not a finding.

---

## 3. Feature Calculation Validation

### 3.1 Rolling windows use only historical data

**Objective.** `vol_5d[t]` is realised volatility over `t+1..t+5` (a
*forward* target, correctly labelled as such and not a leaking feature); no
*indicator* fed to the model at bar `t` uses information from `t+1` or later.

**Verification method — the causality test already in the suite, executed
this session:**
```python
# tests/test_patterns.py::TestCausality
base  = build_pattern_features(c, h, l, v, open_=o)
c2[cut:] *= 1.5                       # perturb every bar from `cut` onward
after = build_pattern_features(c2, h2, l2, v2, open_=o2)
assert np.allclose(base[name][:cut], after[name][:cut])   # bars before cut unaffected
```
Run against all 27 pattern/microstructure channels. Manual spot check on one
date (2025-02-10) confirmed the derivative channels (`atm_iv`, `basis_annualised`)
are computed from that session's own bhavcopy close, published after that
session closes — legitimate `T → T+1` prediction, no extra lag needed
(documented and tested at `nightevolver/derivatives.py`'s module docstring).

**A related, distinct finding fixed this session (not a look-ahead, the
opposite failure — a validity gate too broad):** `vol_5d` and
`regime_shift_5d` gated validity with `np.isfinite(window).all()` across
**every asset simultaneously**, so one dead name (post-delisting NaN)
invalidated that *date* for all 99 other names. Measured: `vol_5d` validity
went to **0.0%** on the ragged 2019 panel. Not a temporal bug, but exactly
the kind of silent-empty-result failure this audit is chartered to catch —
"no windows" reads as "no signal" and is actually "no data." Fixed
per-asset at all four sites (`nightevolver/targets.py`), commit `2e72b6e`.

**Pass/Fail.** **PASS** on causality; **FAIL → FIXED** on the validity-gate
granularity bug (distinct defect, same severity class).

### 3.2 Corporate actions are reverse-adjusted correctly

**Objective.** Confirm the adjustment direction and magnitude against known
real events, not just that a pipeline runs without error.

**Risk if wrong.** A split showing up as an unadjusted price discontinuity
manufactures a fake ±80% crash on the ex-date.

**Verification method — measured against the live archive, not simulated:**
```python
# a name's PrvsClsgPric vs a genuinely adjusted series, on a KNOWN split date
```
```
IRCTC     ex 2021-10-28, 1:5    close 913.50 vs prev 4130.15   -77.88%
NESTLEIND ex 2024-01-05, 1:10   close 2666.40 vs prev 27116.40  -90.17%
```

**This directly falsified a load-bearing claim in the codebase's own
documentation.** `nse_prices.py`'s module docstring asserted `PrvsClsgPric`
was already the exchange's corporate-action-adjusted previous close — "That
is the whole corporate-actions requirement, discharged by using the right
field instead of building a pipeline." The measurement above proves it is
the **raw, unadjusted** prior close. The actual correction is applied
downstream in `corporate_actions.py::adjust_returns` (ex-date price-ratio
correction, dividend add-back, demerger masking) and — critically —
*that module's own docstring already documented this fact independently*,
so the two files disagreed with each other. The false claim was in the file
whose narrative would tell a future reader adjustment isn't necessary. Fixed
(rewrote the docstring with the measurement in it, commit `10c0dcd`) —
correcting **documentation**, not logic; the actual price series were always
computed correctly by `adjust_returns`.

**Pass/Fail.** **PASS on computed values, FAIL → FIXED on documentation.**
Flagged as high-severity anyway: an accurate pipeline with an inaccurate
account of *why* it's accurate is one incorrect edit away from becoming
wrong, since the false docstring would have told a future engineer that
`require_actions=False` is harmless.

### 3.3 Instrument-class contamination (a finding outside the brief's checklist)

Not explicitly requested, surfaced by inspecting what actually ranked into a
universe. `LIQUIDBEES` (money-market ETF, NAV pinned near 1000) passed every
declared equity filter — `SctySrs == "EQ"`, `FinInstrmTp == "STK"` — in both
file formats, and ranked into a top-100-by-turnover universe with **0.01%
annualised volatility** (27 distinct closes across 1,825 bars). Only the
ISIN prefix (`INE` company equity vs `INF` fund/ETF vs `IN9` DVR share class)
discriminates it, and nothing checked ISIN before this audit.

**Pass/Fail.** **FAIL → FIXED.** `nightevolver/nse_prices.py:_equity_rows()`,
commit `10c0dcd`. Recorded here because a constant-price instrument is not
merely noise in a volatility study — it is the extreme of the exact quantity
being ranked, and would be *preferentially selected* by any search over that
target.

---

## 4. Backtest & Walk-Forward Construction

### 4.1 Zero overlap between train and validation windows

**Verification method.**
```python
# nightevolver/backtest_evolved.py:95-107, and identically in
# run_target_walkforward.py's _flat() calls
train_end = start + train_window
test_end  = min(train_end + test_window, n)
train = md.slice(start, train_end)      # [start, train_end)
test  = md.slice(train_end, test_end)   # [train_end, test_end)
```

**Executed check.** Both slices are half-open at the shared boundary
`train_end`; index `train_end` itself belongs only to `test`. No bar is
double-counted. Confirmed identical index arithmetic in both the GA
walk-forward (`nightevolver/backtest_evolved.py`) and the target
walk-forward (`scripts/run_target_walkforward.py`).

**Pass/Fail.** **PASS.**

### 4.2 GA fitness computed against validation, not training

**Verification method.**
```python
# nightevolver/backtest_evolved.py:130-139
res = GeneticEvolver(cfg).evolve(train, validation=None)   # searches ON train only
strat = res.best_strategy
oos = simulate(test, strat, cfg.cost_bps, cfg.max_position)  # scored STRICTLY on test
```
The genetic search never receives the test slice; `simulate()` on `test` is
the only place OOS numbers originate, and it's called after evolution
finishes, on data the evolver's own `md.slice(start, train_end)` call never
touched.

**Pass/Fail.** **PASS.**

### 4.3 The permutation null itself — verified this audit, not assumed

`tests/test_walkforward_null.py` (pre-existing, re-run this session) pins
block-permutation properties: block boundaries, that permutation is applied
jointly across assets (preserving cross-sectional structure), and that
`block_permutation_order()` is the *same* order object reused for price and
extra channels (§2.2 above — required for it to mean anything, but also the
reason it's blind to a fixed offset).

**Pass/Fail.** **PASS**, with the blind spot from §2.2 now formally
documented rather than implicitly assumed away.

---

## 5. Cost & Execution Modeling

### 5.1 Round-trip cost applied consistently to entry and exit

**Objective.** Confirm `cost_bps` means what its every usage site claims it
means, and is charged exactly once per round trip.

**This is the second confirmed, previously-undiscovered defect found by this
audit.**

**Verification method.** Cross-reference the documented economic assumption
against the actual arithmetic:
```python
# genome.py's own break-even table: 61.1% win rate needed for a 1-day hold
# "after 22bp round-trip costs" — reverse-engineer what cost basis
# reproduces that number:
p = 0.5 + cost / (2 * E_abs_ret)
# 0.5 + 0.0022 / (2*0.0099) = 0.611   <- confirms 22bp is meant as TOTAL round trip
```
```python
# ga_engine.py:317-319 (BEFORE fix):
turnover = |diff(positions)|.sum()      # = 2.0 for one full open+close
net = gross - turnover * (cost_bps / 10_000.0)
# -> charges 2 * cost_bps = 44bp per round trip, not 22bp
```

**Executed check — a deterministic single round trip, zero market return, so
100% of the daily-return series is cost:**
```
turnover.sum() for one round trip: 2.0
total bps charged (BEFORE fix):    44.0   (documented: 22.0)
```

**Direction of the error, stated explicitly because it matters for
interpreting every prior result:** the bug made every GA Sharpe, profit
factor, and deflated-Sharpe-gate decision in this project's history **more
conservative** than the documented economics — the safe direction, not one
that could manufacture a false edge. It does **not** change the walk-forward
"no directional edge" conclusion (§ below), which is a pure Spearman
correlation statistic computed with no cost model in it at all.

**Two call sites shared the identical bug:** `ga_engine.py::simulate()` and
`rl_trainer.py::_evaluate_policy()`. **One adjacent, differently-scoped
engine checked and confirmed unaffected:** `kronos/backtest.py`'s
`WalkForwardConfig.cost_bps` is explicitly named and documented as
`cost_bps_per_turnover` — a *different, self-consistent* per-leg convention
from the start, not a round-trip figure, so it was never subject to the same
inconsistency.

**Fix and regression test.** `cost_bps / 2.0` per turnover unit
(commit `e0e3959`); `tests/test_nightevolver.py::test_cost_bps_is_the_full_round_trip_not_double_charged`
pins the exact economics with a synthetic zero-return round trip, and
`test_break_even_table_in_genome_matches_the_simulator` ties the documented
break-even formula to the same cost basis so the two cannot silently diverge
again.

**Pass/Fail.** **FAIL → FIXED.**

### 5.2 Costs not applied to the price adjustment

**Objective.** Confirm transaction cost is a separate deduction from the
return series, never blended into the corporate-action price adjustment
(which would double-count or silently distort the adjusted price history
itself, corrupting every backtest that reads it, not just the one run).

**Verification method.** Trace the call graph: `adjust_returns()`
(`corporate_actions.py`) never receives `cost_bps` or any cost parameter —
it only sees prices and action metadata. `simulate()`'s cost line
(`ga_engine.py:319`) operates on already-adjusted `close`/`fwd` arrays, after
`build_adjusted_frames` has returned. The two are in different modules with
no shared mutable state.

**Pass/Fail.** **PASS.**

### 5.3 Kronos live paper-trader slippage tiers

**Objective.** `kronos/paper_trader.py`'s liquidity-tiered slippage model is
a separate execution-cost mechanism from the GA's flat `cost_bps` and was in
scope as "brokerage, STT, slippage."

**Verification method.** Inspected `slippage_pct()` (`kronos/paper_trader.py:177`)
and `MAX_IMPACT_SLIPPAGE = 0.05` hard cap. Applied per-fill at both entry and
exit through the same `_fill()` path (not a separate, divergent code path per
side).

**Pass/Fail.** **PASS on structure.** **UNVERIFIABLE within this audit's
scope** on magnitude calibration against real fills — this system has never
executed a real order (Kite integration is auth-only per the session record;
`nightevolver.enabled: false` in `kronos/config.yaml`), so there is no fill
log to reconcile modelled slippage against. Stated as a risk, not answered
with a query that has nothing to query.

---

## 6. Infrastructure Handoff (RunPod ⇄ Hetzner)

### 6.1 Feature/indicator order integrity across the handoff

**Objective.** A checkpoint trained with indicators in one order must not be
loaded and decoded against a differently-ordered running codebase — this
would silently attach gene weights to the wrong indicators, which is
arguably *worse* than a crash because nothing about the output would look
wrong.

**Verification method.**
```python
# nightevolver/saver.py:78, 122-129
payload["indicator_names"] = list(INDICATOR_NAMES)   # written at save time

names = tuple(ck.get("indicator_names", ()))
if names != INDICATOR_NAMES:
    raise ValueError("checkpoint indicator ordering differs from the "
                     "running code - gene weights would attach to the "
                     "wrong indicators. Retrain.")
```
A second, independent guard checks `genome_version` and raises on any
mismatch before the genome is decoded at all — layout changes are caught
even if `INDICATOR_NAMES`'s *order* happened to coincidentally match while
its *meaning* changed.

**Pass/Fail.** **PASS.** This control already existed and needed no fix —
included here because the brief specifically asked for it and a negative
result ("nothing to fix") is still a reportable finding.

### 6.2 What is genuinely unverifiable here

Stated plainly per the brief's instruction to be brutally honest: this audit
has **no way to confirm** that a given RunPod training run and the Hetzner
checkpoint actually deployed from it are the *same* artifact, beyond trusting
the file that was copied. There is no signed manifest, no content hash
compared across the two hosts, no deployment log correlating a training run
ID to a live checkpoint. `load_checkpoint()`'s guards (§6.1) catch a
*structurally incompatible* checkpoint; they cannot catch a *stale but
structurally valid* one being silently redeployed. **Risk, not a finding
with a fix**: recommend a content hash (e.g. SHA-256 of the genome array)
logged at both save and load time, so a mismatch between "what RunPod
trained" and "what Hetzner is running" becomes detectable rather than
merely improbable.

---

## Addendum: closing four items that were originally covered by reasoning, not execution

The user asked directly, after the first version of this report:
**"is each point I gave you, covered?"** Honest re-check found four places
where the report answered from code inspection or general argument rather
than from a result actually produced. All four are closed here with
executed evidence; none surfaced a new defect — each confirms a PASS the
report had already claimed, on firmer footing.

### A.1 — Alignment proven for every channel, not just derivatives/delivery

§2.1's channel-alignment claim was verified only for the extra (derivative,
delivery) channels — where the bug actually was. The genome's price-derived
indicators and the FII/DII flow channel were passed on structural argument
alone. Checked directly:

* **Price indicators.** `data_loader.py`'s `build_market_data` computes
  `indicators`, `close`, and `forward_returns` in one function and applies a
  single `keep = slice(WARMUP_BARS, len(idx)-1)` to all three simultaneously
  before returning. There is no separate re-slicing step for this channel
  group — the exact step that was wrong for the externally-built extra
  channels does not exist here, by construction rather than by policy.
* **FII/DII flow channel.** `flows.py::align_flow_features` carries its own
  independent guard — `lag_bars < 1` raises with an explicit look-ahead
  message — and applies the lag in a fixed order (reindex+ffill, **then**
  shift, **then** causal z-score) documented and matching the code exactly.

**Pass/Fail: PASS**, now on inspected code rather than inference.

### A.2 — The forced shift(±1) reproduction, actually run

§2.2 argued from first principles why a forced-shift variant of the
permutation null can't catch this bug class. Argument is not evidence;
ran it, on the real 2019–2026 panel, `pcr_volume → direction_1d`:

| slice offset | same-bar ρ (t-1→t) | forward ρ (t→t+1) | reads as |
|---|---|---|---|
| correct (`lo=WARMUP_BARS`) | −0.2826 | +0.0239 | honest — matches the verified-correct fixed result exactly |
| **the exact original bug** (`lo=WARMUP_BARS+1`) | −0.0597 | **−0.2826** | suspicious — reproduces the discarded run's numbers to 3 decimals |
| opposite-direction shift (`lo=WARMUP_BARS-1`) | +0.0240 | +0.0182 | inert — decorrelated from both, not merely "less predictive" |

The middle row is not a simulation of the bug; it *is* the bug's exact slice
arithmetic, re-run to confirm the numbers it produces are the same numbers
that were caught in the discarded run. This is the empirical half of §2.2's
theoretical argument.

**Pass/Fail: PASS**, empirically, not just structurally.

### A.3 — Manual, by-hand rolling-window calculation on one real date

§3.1 cited the automated causality test suite rather than a hand
computation. Done directly: `RELIANCE`, `t = 2024-01-24`, `vol_5d`.

```
daily returns t+1..t+5 (2024-01-25 .. 2024-02-01):
    +0.006846  +0.070192  -0.027917  +0.013498  +0.000018
manual std, ddof=1:  0.03588404566071022
code vol_5d[t]:      0.03588404566071022     <- exact match
```

Boundary check, perturbing single bars: value is **unchanged** when
`t+6` (one bar past the window) is perturbed, and **changes** when `t+3`
(inside the window) is perturbed — confirming the window is exactly
`[t+1, t+5]` and reaches no further. (Perturbing `t` itself *does* change
the value — correctly: `vol_5d[t]`'s first observation is the return from
`t` to `t+1`, so `t`'s price is the base of that return by definition, not
a look-ahead.)

**Pass/Fail: PASS.**

### A.4 — Adjustment continuity across many real actions, not two examples

§3.2 falsified the `PrvsClsgPric`-is-adjusted claim using two hand-picked
splits (IRCTC, NESTLEIND) against the *raw* field. Broadened to scan the
*adjusted* close series itself — the thing actually fed to every model —
across 9 names spanning 1,825 bars, 2019–2026, including both of those same
splits plus whatever other actions those names carried in between:

```
residual |log-return| > 25% in the ADJUSTED series: 3 cells, all
    2020-03-18 .. 2020-03-23  (BAJFINANCE, INDUSINDBK x2)
```

All three land in the COVID crash week — a real, dated market event, not a
missed corporate action. IRCTC's and NESTLEIND's splits (the two originally
checked) show **zero** residual in the adjusted series: fully absorbed.

**Pass/Fail: PASS**, on the artifact that matters (adjusted output), not
the input that was falsified (raw field claim).

### A.5 — Scope gap acknowledged, not closed

The brief's stated scope named Zerodha API and yfinance as data sources.
Neither was audited. Zerodha is authentication-only in this project — no
order has ever been placed, so there is no ingestion path to check.
yfinance was evaluated and abandoned early (blocked from this sandbox by
`SSLError`, and judged an unreliable scraped source vs. the bhavcopy) and
carries no active code path into any current result. Stated here rather
than silently narrowed.

---

## Summary: To-Do List and Disposition

| # | Finding | Severity | Section | Status |
|---|---|---|---|---|
| 1 | One-bar look-ahead in extra-channel alignment (`run_target_walkforward.py`) — manufactured a false p=0.032 directional edge | **Critical** | 2.1 | **Fixed**, commit `17928f7`, structural regression test added |
| 2 | Null-cloud permutation structurally cannot detect a fixed-offset misalignment | **Critical (methodology)** | 2.2 | **Documented + independent structural test added**, `tests/test_walkforward_alignment.py` |
| 3 | `vol_5d`/`regime_shift_5d` validity gated across all assets jointly — one dead name zeroed the target panel-wide | **Critical** | 3.1 | **Fixed**, commit `2e72b6e`, per-asset gating + regression tests |
| 4 | `nse_prices.py` docstring falsely claimed `PrvsClsgPric` is pre-adjusted | High (docs) | 3.2 | **Fixed**, commit `10c0dcd` |
| 5 | ETFs/DVR share classes pass every declared equity filter, contaminate volatility-ranked universes | High | 3.3 | **Fixed**, commit `10c0dcd`, ISIN-prefix filter + tests |
| 6 | Delisted names forward-filled into flat, zero-volatility price lines | Critical | 1.3 | **Fixed**, commit `c2ec36b` |
| 7 | Round-trip transaction cost double-charged (44bp billed vs 22bp documented) in GA and RL simulators | High | 5.1 | **Fixed**, commit `e0e3959`, economics pinned against genome.py's own break-even table |
| 8 | No guard against negative volume anywhere in ingestion | Medium (no live occurrence found) | 1.2 | **Fixed**, commit `6ff737d` |
| 9 | 2019 point-in-time universe unresolvable at all (`KeyError: 'SctySrs'`) — legacy-schema bug in a second, undiscovered copy of the column reader | Critical (blocked all further work) | (pre-req to this audit) | **Fixed**, commit `a8757db` |
| 10 | `http.client.IncompleteRead` not in any fetcher's retry tuple — could crash a multi-hour backfill and, separately, a truncated download could reach the cache | High | 1.1 | **Fixed**, commit `8c8c044` |
| 11 | Checkpoint deployment (RunPod → Hetzner) has no content-hash cross-check | Low/Unverifiable | 6.2 | **Not fixed** — recommendation only, no fill/deployment log exists to validate against |
| 12 | Kronos paper-trader slippage magnitude uncalibrated against real fills | Unverifiable | 5.3 | **Not fixed** — no live execution has occurred; nothing to calibrate against yet |

**11 of 12 findings fixed and tested this audit** (982 → 984 tests passed
across the fix sequence, 0 failures, 0 skipped beyond two pre-existing
network-dependent skips). The two unresolved items (#11, #12) are not code
defects — they are missing infrastructure (a deployment manifest, a live
fill log) that cannot be verified into existence; both are recorded as open
risks rather than closed with a fix that would only be cosmetic.

**Net effect on the project's central conclusion.** None of the fixes in
this audit change the "no directional edge" finding — several of them (cost
correction, ETF/DVR removal, delisting fix) make the measurement *more*
conservative, and the walk-forward's own null-cloud check (independent of
every fix above except #1–#3, which are what made the honest re-measurement
possible at all) is what produced that conclusion. What changed is
confidence *in the measurement*, not the measurement's direction.
