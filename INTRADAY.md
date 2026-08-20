# Intraday data: the recorder and the audit path

Two things, built together because they answer the same question from
opposite ends of the calendar.

**The audit path** can run as soon as there are credentials: it asks
whether intraday *day shape* — where the volume sat, how the day trended
against its own VWAP, how much of the range arrived in the first thirty
minutes — carries information that daily OHLCV does not.

**The recorder** produces nothing useful today. Its entire value is that
in three months there is an order-flow dataset to audit. That asymmetry
is the point of starting it now.

---

## Why they are separate

Checked against Kite's live docs rather than recalled, because the split
determines what is worth building:

| | Historical endpoint | WebSocket `full` mode |
|---|---|---|
| Candles (O/H/L/C/V/OI) | ✅ minute … day, "back several years" | — |
| 5-level bid/offer depth | **❌ never** | ✅ live only |

**Market depth cannot be backfilled from Kite at any price on this
plan.** Order-book imbalance, book pressure and signed volume therefore
only exist if something is writing them down. Every day not recorded is
a day that cannot be recovered — which is the whole argument for
starting the recorder before you know whether you need it.

Cost, from Zerodha's own pricing page: **₹500/month** for the full suite
including WebSocket streaming and historical candles. Roughly a third of
the ₹1,500–2,500 quoted for True Data / GDFL.

---

## Running the recorder

```bash
export KITE_API_KEY=...
export KITE_ACCESS_TOKEN=...        # EXPIRES DAILY
python scripts/record_depth.py

# smoke test, no waiting for market hours
python scripts/record_depth.py --duration 60 --any-hours
```

Output is one gzipped JSONL file per IST trading date in `data/depth/`,
roughly 10–20 MB/day for ten symbols — about 1 GB per quarter.

**The daily token expiry is the real operational constraint**, not a
footnote. It is the most common reason an unattended Kite job dies. Two
workable patterns:

- `systemd` with `Restart=on-failure`, plus a morning job that refreshes
  the token before the open; or
- cron at ~09:10 IST with `--duration 24000` (~6h40m) so the process
  exits cleanly after the close.

The recorder **exits** on an auth error rather than retrying, because a
process spinning on a stale token looks alive while recording nothing.
That is asserted by a test.

### What it deliberately does not do

- It does not consult the NSE holiday calendar. On a holiday the stream
  simply carries no data; a recorder that refuses to connect because it
  believes it is a holiday is worse than one that connects and records
  nothing.
- It does not write unchanged books. Kite re-sends the current book on a
  schedule; storing those inflates the file and, worse, makes a stalled
  feed indistinguishable from a quiet one when you look at the data
  months later. The stats line reports `written` alongside `frames` and
  `suppressed` so a stall is visible in the log.

---

## Running the intraday audit

```bash
python scripts/run_intraday_audit.py --start 2026-01-01     # needs credentials
python scripts/run_intraday_audit.py --synthetic            # calibration, none needed
```

Twelve day-shape features per symbol per day, scored against the same
four targets with the same block-permutation null and the same BH-FDR
correction as the daily audit — so the results are directly comparable
with the daily baseline:

> 7 of 104 pairs survived, all on volatility and regime targets, none on
> direction.

Order-flow channels are added automatically once `data/depth/` has
recorded sessions. Until then they are **absent**, and the script says
so explicitly rather than reporting a null result on them — "not yet
recorded" and "carries no information" are very different statements and
conflating them would waste the recorder's whole purpose.

### Why daily-frequency features from minute bars

Minute bars could be used to build a minute-frequency strategy. They are
not used that way here, because the open question is whether intraday
data carries information daily OHLCV does not — and that is answered by
comparing feature sets on the same targets with the same correction, not
by building a minute-level trading system and finding out six weeks
later. If day shape carries nothing about tomorrow, a finer-grained
strategy over the same information will not save it.

If something *does* survive, this tells you which channels to build on.

---

## Calibration

```
python scripts/run_intraday_audit.py --synthetic
  -> NOTHING survives FDR correction.
```

Random-walk minute bars, real feature code, full pipeline: zero
survivors. Any survivor there is a false positive, and this project has
already had one of those slip through (see `NEW_DATA_FINDINGS.md` §3,
bug 3 — a mechanical artefact that scored q=0.04 on a random walk).

---

## Tests

53 in `tests/test_kite_and_intraday.py`. The binary parser gets the most
attention because it is the component whose bugs are invisible: a wrong
byte offset does not raise, it produces plausible numbers that are
wrong, and it would do so for months into a dataset that cannot be
re-recorded.

Specifically covered:

- **The 12-byte depth stride.** Each entry is `quantity(int32) +
  price(int32) + orders(int16) + 2 bytes padding`. A 10-byte stride
  would not raise — it would silently shift every level after the first.
  There is also a test confirming that a wrong stride really does
  produce different numbers, so the first test is not vacuous.
- Heartbeats, truncated frames, and a frame header that lies about its
  packet count — none may raise, because they must not kill a
  mid-session recorder.
- Date rotation, and that reopening **appends** rather than truncates, so
  a mid-session restart does not destroy the morning.
- A reader that survives the truncated final line a `kill -9` leaves.
- **The real async loop**, against a local WebSocket server: subscribe
  ordering, `full` mode, parse, write, reconnect-on-drop, and abort (not
  retry) on auth error.
- No-look-ahead on the intraday features: mutate a later session, assert
  earlier feature rows are unchanged.

That last async test earned its keep immediately — it caught that
`async for message in ws` only checks the deadline when a message
arrives, so `--duration` would not have fired on a quiet feed. Which is
exactly the cron pattern documented above, at exactly the time of day
the feed goes quiet.
