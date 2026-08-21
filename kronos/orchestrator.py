"""
Kronos Orchestrator - the master state machine of the 365-day organism.

Phases (exchange-local time, config: schedule):
    00:00-02:00  DIGESTION    fetch + validate + build DailyMemory
    02:00-04:00  NIGHTMARE    conditional diffusion adversarial futures
    04:00-05:00  EVOLUTION    NEAT variants vs. nightmares -> master model
    05:00-06:00  ADAPTATION   MAML 3-step warm-up on real recent data
    06:00        REPORT       God's Eye markdown
    09:30-16:00  REFLEX       low-latency SNN inference, NO training
    16:00-24:00  LOGGING      close books, persist results

Non-negotiable rules implemented here:
  1. veto.txt overrides take veto_delay_hours (24h) to apply.
  2. Graceful degradation: a phase crash flips degraded=True for heavy
     phases (reduced NEAT population) instead of stopping the year.
  3. Each phase is retried exactly max_retries_per_phase (1) time before
     a fatal log entry - then the day continues with the previous model.
  4. Hourly heartbeat rows in the audit log; model checkpoints daily.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import traceback
from dataclasses import dataclass, field
from datetime import date as ddate, datetime, time as dtime, timedelta, timezone
from enum import Enum
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np
import torch

from kronos.calendar_utils import (
    is_nse_trading_day,
    is_trading_day,
    next_nse_trading_day,
    next_trading_day,
    nse_holidays,
    nyse_holidays,
)

# This deployment trades NSE (India), not NYSE - the live phase-gating
# calendar below is deliberately the NSE one. is_trading_day/nyse_holidays
# stay imported (and still exported from kronos/__init__.py) since
# scripts/train.py's broader training universe and some tests still
# reference the US calendar; they're just not what gates trading here.
from kronos.config import KronosConfig, load_config
from kronos.data_pipeline import DataPipeline, DailyMemory, DataUnavailableError
from kronos.evolver import KronosEvolver, EvolutionResult
from kronos.execution_adapter import build_execution_adapter
from kronos.nightmare_generator import NightmareGenerator, NightmareBuffer
from kronos.notifier import TelegramNotifier
from kronos.paper_trader import PaperTrader
from kronos.reflex import ReflexArc
from kronos.reporter import GodsEyeReporter
from kronos.risk_guard import RiskGuard
from kronos.bias_estimator import compute_daily_bias
from kronos.nightevolver_bridge import NightEvolverBridge
from kronos.runpod_trigger import CHECKPOINT_DIR as RUNPOD_CHECKPOINT_DIR, load_runpod_checkpoint
from kronos.warmer import KronosWarmer, WarmupResult

logger = logging.getLogger(__name__)


class Phase(Enum):
    DIGESTION = "digestion"
    NIGHTMARE = "nightmare"
    EVOLUTION = "evolution"
    ADAPTATION = "adaptation"
    REPORT = "report"
    REFLEX = "reflex"
    LOGGING = "logging"


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


@dataclass
class DayState:
    """Everything produced so far in the current daily cycle."""
    day: int = 0
    memory: Optional[DailyMemory] = None
    nightmare: Optional[NightmareBuffer] = None
    evolution: Optional[EvolutionResult] = None
    warmup: Optional[WarmupResult] = None
    report_path: Optional[str] = None
    phase_failures: Dict[str, str] = field(default_factory=dict)
    degraded: bool = False


class KronosOrchestrator:
    """The clock-driven brain that runs Prometheus through its daily life."""

    def __init__(self, config: Optional[KronosConfig] = None):
        self.cfg = config or load_config()
        sched = self.cfg.schedule
        self._boundaries = [
            (Phase.DIGESTION, _parse_hhmm(sched.digestion_start)),
            (Phase.NIGHTMARE, _parse_hhmm(sched.nightmare_start)),
            (Phase.EVOLUTION, _parse_hhmm(sched.evolution_start)),
            (Phase.ADAPTATION, _parse_hhmm(sched.adaptation_start)),
            (Phase.REPORT, _parse_hhmm(sched.report_time)),
            (Phase.REFLEX, _parse_hhmm(sched.market_open)),
            (Phase.LOGGING, _parse_hhmm(sched.market_close)),
        ]

        self.pipeline = DataPipeline(self.cfg)
        self.nightmare_gen = NightmareGenerator(self.cfg)
        self.evolver = KronosEvolver(self.cfg)
        self.warmer = KronosWarmer(self.cfg)
        self.reflex = ReflexArc(self.cfg)
        self.trader = PaperTrader(self.cfg)
        self.reporter = GodsEyeReporter(self.cfg)
        self.notifier = TelegramNotifier(self.cfg)
        self.risk_guard = RiskGuard(self.cfg)
        # The GA checkpoint consumer. None whenever the config block is
        # absent or disabled, and every call site treats None as "no
        # opinion" - so the default build behaves exactly as it did
        # before this was wired.
        self.nightevolver = NightEvolverBridge.from_config(self.cfg)
        # The execution seam. Until now ExecutionAdapter was dead code -
        # the orchestrator called self.trader.execute() directly in four
        # places, so the abstraction existed, was tested, and routed
        # nothing. Routing through it changes no behaviour in the default
        # `simulated` mode (SimulatedExecutionAdapter is a thin shim over
        # the same PaperTrader) but makes the seam real, so the IBKR path
        # is reachable by configuration rather than by editing the
        # orchestrator.
        self.execution = build_execution_adapter(self.cfg, self.trader)

        self.state = DayState()
        self.master_model: Optional[torch.nn.Module] = None
        self._last_heartbeat: Optional[datetime] = None
        self._pending_veto: Optional[Dict] = None
        self._phase_runs: Set[Tuple[str, str]] = set()
        # AUDIT-2B: today's first-seen price per ticker, used by
        # run_reflex_tick() to compute a live intraday return-so-far each
        # tick without mutating self.state.memory (a frozen DailyMemory -
        # see run_reflex_tick()'s docstring note for why). Reset once per
        # day in run_digestion().
        self._day_open_prices: Dict[str, float] = {}
        self._evolution_progress: Dict = {}   # generation-level resume state
        self._completed_today: Set[str] = set()
        self.skip_trading: bool = False       # set on stale/unavailable data

        os.makedirs(self.cfg.orchestrator.checkpoint_dir, exist_ok=True)
        self._load_persisted_snn()
        self._load_persisted_daily_bias()
        self._load_persisted_size_scale()
        # RunPod training runs entirely in GitHub Actions now (scheduled
        # in .github/workflows/train-runpod.yml), which scp's the result
        # straight onto this box. Kronos's only job is noticing when a
        # newer checkpoint file has appeared - see
        # maybe_adopt_runpod_checkpoint(). Baseline this to "whatever's
        # on disk right now" so a checkpoint that arrived before this
        # process even started isn't treated as new on every restart.
        self._runpod_last_adopted_mtime: Optional[float] = self._current_runpod_checkpoint_mtime()
        self._runpod_adopted_today: bool = False   # reset once per day in run_logging()

    # ------------------------------------------------------------------
    # Phase resolution
    # ------------------------------------------------------------------

    def phase_for(self, now: datetime) -> Phase:
        """Map a wall-clock time to the active phase."""
        t = now.time()
        current = Phase.LOGGING          # 16:00 - midnight wraps around
        for phase, start in self._boundaries:
            if t >= start:
                current = phase
        return current

    # -- calendar awareness (ORC-04 / ORC-05) ---------------------------

    LOW_POWER_PHASES = {Phase.DIGESTION, Phase.NIGHTMARE, Phase.LOGGING}

    def phases_for_day(self, d: ddate) -> Set[Phase]:
        """
        Trading day: the full cycle. Weekend/holiday: low-power mode -
        digestion + nightmare only (stay sharp, save compute, and never
        hit market APIs for data that does not exist).
        """
        if is_nse_trading_day(d):
            return set(Phase)
        return set(self.LOW_POWER_PHASES)

    def should_run_phase(self, phase: Phase, now: datetime) -> bool:
        """
        DST-safe once-per-day phase gating (ORC-02 / ORC-03).

        Each (date, phase) pair runs at most once. During a DST fall-back
        the same wall-clock hour repeats with fold=1 - the duplicate hour is
        detected and skipped, so nothing double-runs. During spring-forward
        the missing hour simply never matches, and the next boundary
        picks up the cycle - nothing crashes.
        """
        if getattr(now, "fold", 0) == 1:
            logger.info("DST Fallback detected. Skipping duplicate hour.")
            return False
        if phase not in self.phases_for_day(now.date()):
            return False
        key = (now.date().isoformat(), phase.value)
        if key in self._phase_runs:
            return False
        self._phase_runs.add(key)
        return True

    # ------------------------------------------------------------------
    # Retry-once wrapper (design principle: idempotency)
    # ------------------------------------------------------------------

    def _run_phase(self, name: str, fn: Callable, *args, **kwargs):
        retries = int(self.cfg.orchestrator.max_retries_per_phase)
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                attempt += 1
                self.trader.audit(
                    self.state.day, name,
                    f"attempt {attempt} failed: {e}",
                )
                if attempt > retries:
                    tb = traceback.format_exc()
                    logger.error(
                        "[orchestrator] FATAL in %s after %d attempts:\n%s",
                        name, attempt, tb,
                    )
                    self.trader.audit(
                        self.state.day, name, f"FATAL - moving on: {e}"
                    )
                    self.state.phase_failures[name] = str(e)
                    self.state.degraded = True
                    self._alert(
                        f"Kronos phase FATAL: {name} (day {self.state.day})\n"
                        f"{e}\nMoving on in degraded mode - see audit log."
                    )
                    return None
                logger.warning(
                    "[orchestrator] %s failed (attempt %d/%d) - retrying: %s",
                    name, attempt, retries + 1, e,
                )

    # ------------------------------------------------------------------
    # Individual phases
    # ------------------------------------------------------------------

    def run_digestion(self, filings: Optional[Dict[str, str]] = None) -> Optional[DailyMemory]:
        self.skip_trading = False
        self._day_open_prices = {}   # AUDIT-2B: fresh intraday reference each day
        memory = self._run_phase("digestion", self.pipeline.run_sync, filings)
        self.state.memory = memory
        if memory is None:
            # E2E-03 / DAT-01: data unavailable -> no trading on guesses
            self.skip_trading = True
            self.trader.audit(
                self.state.day, "digestion",
                "no data available - trading disabled for the day",
            )
            return None
        if any(f.startswith("stale_data") for f in memory.quality_flags):
            self.skip_trading = True
            self.trader.audit(
                self.state.day, "digestion",
                "STALE cached data in use - trading disabled for the day",
            )
        self.trader.audit(
            self.state.day, "digestion",
            f"memory built from {memory.source_used}, "
            f"flags={memory.quality_flags}",
        )
        self._refresh_nightevolver()
        return memory

    def _refresh_nightevolver(self) -> None:
        """Adopt any new GA checkpoint and reload its deep history.

        Runs once per day, in digestion, because both halves are daily
        artifacts: the checkpoint is written by the nightly RunPod job,
        and the history is daily bars. Doing it per tick would re-read a
        rejected checkpoint 375 times a session.

        Wrapped whole: a failure here must never take down digestion,
        which the rest of the day depends on. The bridge stays inert and
        the SNN trades alone.
        """
        if self.nightevolver is None:
            return
        try:
            adopted = self.nightevolver.maybe_reload()
            if adopted:
                hist = self.pipeline.fetch_history(self.nightevolver.history_days)
                if hist is not None:
                    self.nightevolver.set_history(hist[0], hist[1])
                else:
                    self.nightevolver.set_history(None)
            elif self.nightevolver.active:
                # Same checkpoint, new day - the history still needs to
                # move forward or the strategy trades a stale window.
                hist = self.pipeline.fetch_history(self.nightevolver.history_days)
                if hist is not None:
                    self.nightevolver.set_history(hist[0], hist[1])
            self.trader.audit(self.state.day, "nightevolver",
                              self.nightevolver.status)
            logger.info("[orchestrator] nightevolver: %s",
                        self.nightevolver.status)
        except Exception as e:                                   # noqa: BLE001
            logger.warning("[orchestrator] nightevolver refresh failed: %s", e)
            self.trader.audit(self.state.day, "nightevolver",
                              f"refresh failed, staying inert: {e}")

    def run_nightmare(self) -> Optional[NightmareBuffer]:
        if self.state.memory is None:
            self.trader.audit(self.state.day, "nightmare", "skipped - no memory")
            return None
        weights = self._current_weights()
        buffer = self._run_phase(
            "nightmare", self.nightmare_gen.generate,
            self.state.memory, weights,
        )
        self.state.nightmare = buffer
        return buffer

    def run_evolution(
        self, time_budget_seconds: Optional[float] = None
    ) -> Optional[EvolutionResult]:
        if self.state.nightmare is None:
            self.trader.audit(self.state.day, "evolution", "skipped - no nightmares")
            return None

        # ORC-07 / E2E-02: resume at the exact generation that was
        # checkpointed before a crash/preemption.
        resume_pop = self._evolution_progress.get("population")
        resume_gen = self._evolution_progress.get("generation", 0)

        def _on_generation(gen, population):
            self._evolution_progress = {
                "generation": gen, "population": population,
            }
            self.save_checkpoint()

        result = self._run_phase(
            "evolution", self.evolver.evolve,
            self.state.nightmare, self.state.degraded,
            time_budget_seconds, resume_pop, resume_gen, _on_generation,
        )
        if result is None and not self.state.degraded:
            # graceful degradation: one more try at reduced scale
            result = self._run_phase(
                "evolution-degraded", self.evolver.evolve,
                self.state.nightmare, True,
            )
        self.state.evolution = result
        if result is not None:
            self.master_model = result.master_model
            self._evolution_progress = {}   # cycle complete - clear resume state
        return result

    def run_adaptation(self) -> Optional[WarmupResult]:
        if self.master_model is None or self.state.memory is None:
            self.trader.audit(
                self.state.day, "adaptation",
                "skipped - no master model or memory (keeping yesterday's weights)",
            )
            return None
        result = self._run_phase(
            "adaptation", self.warmer.warm, self.master_model, self.state.memory
        )
        self.state.warmup = result
        if result is not None:
            self.master_model = result.adapted_model
        return result

    def run_report(self) -> Optional[str]:
        if self.state.memory is None:
            return None
        signals = self._master_signals()
        evo = self.state.evolution
        warm = self.state.warmup
        path = self._run_phase(
            "report", self.reporter.generate,
            self.state.day, self.state.memory, self.trader,
            signals,
            self.reflex.gate.state.regime,
            self.reflex.gate.position_cap,
            {"population_size": evo.population_size, "degraded": evo.degraded,
             "top_fitness": evo.top_fitness} if evo else None,
            {"regime": warm.regime_estimate,
             "inner_losses": warm.inner_losses} if warm else None,
        )
        self.state.report_path = path
        self._checkpoint_model()
        return path

    def run_reflex_tick(
        self,
        vix_value: float,
        bar_prices: Optional[Dict[str, float]] = None,
        bar_volumes: Optional[Dict[str, float]] = None,
        now: Optional[datetime] = None,
    ):
        """One market-hours tick: infer, gate, trade.

        AUDIT-2B: recent_returns fed to reflex.infer() used to come purely
        from self.state.memory - a frozen DailyMemory snapshotted once
        before market open by run_digestion() and never reassigned for
        the rest of the trading session. Every ~60s tick of the ~375-tick
        NSE session was therefore feeding the SNN an identical, stale
        window all day - it could never see or react to intraday price
        movement, despite running a fresh inference every tick. Fixed via
        _recent_returns_with_intraday(): holds out memory's most recent
        historical bar and replaces it with a live return-so-far row
        (today's bar_prices vs. the first bar_prices seen today), without
        mutating the frozen DailyMemory itself.
        """
        if self.state.memory is None:
            return None
        if self.skip_trading:
            return None   # stale/unavailable data day - observe, never trade
        if bar_prices:
            # Mark-to-market BEFORE the risk check below - RiskGuard reads
            # trader.equity(), which marks off last_prices. Checking risk
            # against last tick's stale prices would let this tick's real
            # move go unnoticed until the tick after.
            self.trader.last_prices.update(bar_prices)
        recent = self._recent_returns_with_intraday(bar_prices)
        decision = self.reflex.infer(
            recent, vix_value, bar_prices, bar_volumes, now=now
        )
        halted = self._enforce_risk()
        # Execute toward signal-derived target weights
        if bar_prices:
            tickers = self.state.memory.tickers
            kelly_fraction = float(self.cfg.trading.kelly_fraction)

            # Mix in the GA checkpoint's opinion, if one is loaded, gated
            # and covering these names. Applied to the RAW signal, before
            # Kelly and the position cap below, so sizing still happens in
            # exactly one place - feeding in the decoder's already-sized
            # target_weight would apply Kelly twice.
            #
            # This sits BEFORE the risk guard and the caps on purpose:
            # nothing here can widen a position beyond what the SNN path
            # was already allowed to take, and every order still passes
            # sanity_check_order and decision.position_cap below.
            signals = list(decision.signals)
            if self.nightevolver is not None:
                try:
                    signals = self.nightevolver.blend(
                        tickers, signals, bar_prices)
                except Exception as e:                           # noqa: BLE001
                    logger.warning("[orchestrator] nightevolver blend failed, "
                                   "using SNN signal alone: %s", e)
                    signals = list(decision.signals)

            for i, ticker in enumerate(tickers):
                if ticker not in bar_prices or i >= len(signals):
                    continue
                price = bar_prices[ticker]
                target = 0.0 if halted else float(signals[i]) \
                    * kelly_fraction * float(self.cfg.trading.max_position_pct)
                if not halted:
                    rejection = self.risk_guard.sanity_check_order(
                        ticker, price, self.trader.last_prices.get(ticker), target,
                    )
                    if rejection:
                        self.trader.audit(self.state.day, "risk",
                                          f"order rejected: {rejection}")
                        self._alert(f"Kronos order rejected (risk guard)\n{rejection}")
                        continue
                asset_cap = min(
                    decision.position_cap,
                    decision.asset_caps.get(ticker, 1.0),
                )
                self.execution.submit_order(
                    self.state.day, ticker, target, price,
                    (bar_volumes or {}).get(ticker, 0.0),
                    position_cap=asset_cap,
                )
        return decision

    def _recent_returns_with_intraday(
        self, bar_prices: Optional[Dict[str, float]]
    ) -> np.ndarray:
        """Build the horizon-bar window fed to reflex.infer(): the last
        (horizon - 1) historical daily bars from self.state.memory, plus a
        final row of today's live return-so-far per ticker - so the
        window actually changes tick to tick instead of being identical
        all day (AUDIT-2B). `self._day_open_prices` (reset once per day in
        run_digestion()) is this session's own reference price per ticker,
        first captured on that ticker's first tick of the day - that
        tick's own live row is 0.0 (no intraday move yet), exactly as a
        return-since-open should read.

        Falls back to the pure-historical window (previous behavior) when
        bar_prices isn't available - e.g. tests or simulations that call
        run_reflex_tick() without live ticks - or when horizon_days is too
        small to hold out a bar at all.
        """
        horizon = self.cfg.nightmare.horizon_days
        if not bar_prices or horizon < 2:
            return self.state.memory.returns_window(horizon)

        tickers = self.state.memory.tickers
        live_row = np.zeros(len(tickers), dtype=np.float32)
        for i, ticker in enumerate(tickers):
            price = bar_prices.get(ticker)
            if price is None:
                continue
            open_price = self._day_open_prices.get(ticker)
            if open_price is None:
                self._day_open_prices[ticker] = price
                continue   # first tick of the day: no return-so-far yet
            if open_price:
                live_row[i] = (price / open_price) - 1.0

        hist = self.state.memory.returns_window(horizon - 1)
        return np.vstack([hist, live_row[np.newaxis, :]]).astype(np.float32)

    def _enforce_risk(self) -> bool:
        """Check the independent risk layer. On any breach - or if already
        halted from a previous check - flattens every position immediately
        (no 24h veto delay, no ML in the loop) and trips the halt file.
        Returns True if trading is halted, so callers skip placing new
        orders this tick."""
        if self.risk_guard.halted:
            return True
        reason = self.risk_guard.check(self.trader)
        if reason is None:
            return False
        self._auto_flatten(reason)
        self.risk_guard.trip(reason)
        self._alert(
            f"KRONOS RISK HALT\n{reason}\nAll positions flattened. "
            f"Trading halted until an operator removes {self.risk_guard.halt_file}."
        )
        return True

    def _auto_flatten(self, reason: str) -> None:
        for ticker, price in list(self.trader.last_prices.items()):
            if self.trader.positions.get(ticker, 0.0) != 0.0:
                self.execution.submit_order(
                    self.state.day, ticker, 0.0, price, bar_volume=1e9,
                )
        self.trader.audit(self.state.day, "risk", f"auto-flatten: {reason}")

    def _alert(self, text: str) -> None:
        """Best-effort out-of-band alert for events that can't wait for the
        daily digest - risk halts, rejected orders, phase FATAL failures.
        Never raises; a notification problem must never affect trading."""
        try:
            if self.notifier.enabled:
                self.notifier.send(text)
        except Exception as e:
            logger.warning("[orchestrator] alert failed: %s", e)

    def run_logging(self, closing_prices: Optional[Dict[str, float]] = None) -> Dict:
        prices = closing_prices or dict(self.trader.last_prices)
        stats = self.trader.close_day(self.state.day, prices)
        self.trader.audit(self.state.day, "logging", json.dumps(stats))
        self._send_daily_notification(stats)
        self._check_large_pnl_move(stats)
        self._backup_trades_db()
        # One digest per day covers it - reset regardless of whether a
        # notification was actually sent (e.g. Telegram not configured),
        # so this always reflects "since the last day boundary."
        self._runpod_adopted_today = False
        return stats

    def _check_large_pnl_move(self, stats: Dict) -> None:
        """A single day's PnL swinging past notifications.large_pnl_alert_pct
        of that day's opening equity gets an immediate out-of-band alert,
        not just a line buried in the daily digest - exactly the kind of
        thing that's easy to miss if you're not reading every digest."""
        equity = stats.get("equity", 0.0)
        pnl = stats.get("pnl", 0.0)
        day_start = equity - pnl
        if day_start <= 0:
            return
        move_pct = abs(pnl) / day_start
        limit = float(self.cfg.notifications.get("large_pnl_alert_pct", 0.10))
        if move_pct >= limit:
            self._alert(
                f"Kronos large PnL move: day {stats.get('day')} "
                f"{'gained' if pnl >= 0 else 'lost'} {move_pct:.1%} of opening "
                f"equity ({pnl:+,.2f}, now ${equity:,.2f})."
            )

    def _backup_trades_db(self) -> None:
        """Copies trades.db into backup.dir once per day, after close_day()
        has already committed - a corrupted or accidentally-deleted live DB
        (this box has already crashed on sqlite errors once) shouldn't be
        able to erase the whole paper-trading history. Best-effort: a
        backup failure must never affect trading. Keeps only the most
        recent backup.max_backups copies."""
        try:
            import shutil
            backup_dir = self.cfg.backup.dir
            os.makedirs(backup_dir, exist_ok=True)
            dest = os.path.join(
                backup_dir, f"trades_day{self.state.day:04d}.db"
            )
            shutil.copyfile(self.trader.db_path, dest)
            max_backups = int(self.cfg.backup.max_backups)
            existing = sorted(
                f for f in os.listdir(backup_dir) if f.startswith("trades_day")
            )
            for stale in existing[:-max_backups] if max_backups > 0 else []:
                os.remove(os.path.join(backup_dir, stale))
        except Exception as e:
            logger.warning("[orchestrator] trades.db backup failed: %s", e)

    def _send_daily_notification(self, stats: Dict) -> None:
        """Best-effort Telegram digest - never let a notification problem
        affect trading. Fully silent no-op if notifications aren't configured."""
        if not (self.notifier.enabled and self.cfg.notifications.send_daily_digest):
            return
        try:
            memory = self.state.memory
            evo = self.state.evolution
            warm = self.state.warmup
            runpod_status = None
            if self._runpod_adopted_today:
                runpod_status = "adopted"
            elif self._runpod_last_adopted_mtime is not None:
                runpod_status = "unchanged"
            text = self.notifier.build_daily_report(
                day=self.state.day,
                stats=stats,
                regime=warm.regime_estimate if warm else None,
                top_fitness=evo.top_fitness if evo else None,
                source_used=memory.source_used if memory else None,
                quality_flags=memory.quality_flags if memory else None,
                phase_failures=self.state.phase_failures or None,
                reflex_regime=self.reflex.gate.state.regime,
                runpod_status=runpod_status,
            )
            self.notifier.send(text)
        except Exception as e:
            logger.warning("[orchestrator] daily notification failed: %s", e)

    # ------------------------------------------------------------------
    # Full-day driver (used by run_kronos.py and the e2e test)
    # ------------------------------------------------------------------

    def run_full_day(
        self,
        day: int,
        filings: Optional[Dict[str, str]] = None,
        market_ticks: Optional[list] = None,
        resume: bool = False,
        completed_phases: Optional[Set[str]] = None,
    ) -> DayState:
        """
        Execute one complete 24h cycle synchronously (compressed time).
        market_ticks: list of (vix, prices, volumes) tuples simulating bars.
        resume=True (ORC-07): reload logs/checkpoint.pkl and skip phases
        already completed - including resuming NEAT at the checkpointed
        generation - without re-downloading data or re-running MAML.
        """
        done: Set[str] = set(completed_phases or set())
        if resume:
            done |= self.load_checkpoint()

        if not (resume and self.state.day == day):
            self.state = DayState(day=day)
        self.state.day = day
        self._completed_today = set(done)
        self._process_veto()

        if "digestion" not in done or self.state.memory is None:
            self.run_digestion(filings)
        self._mark_done(done, "digestion")

        if "nightmare" not in done or self.state.nightmare is None:
            self.run_nightmare()
        self._mark_done(done, "nightmare")

        if "evolution" not in done or self.state.evolution is None:
            self.run_evolution()
        self._mark_done(done, "evolution")

        if "adaptation" not in done:
            self.run_adaptation()
        self._mark_done(done, "adaptation")

        if "report" not in done:
            self.run_report()
        self._mark_done(done, "report")

        for tick in (market_ticks or []):
            vix, prices, volumes = tick
            self.run_reflex_tick(vix, prices, volumes)
            self.heartbeat()

        self.run_logging()
        self.trader.audit(day, "day-complete",
                          f"failures={self.state.phase_failures}")
        self.clear_checkpoint()
        return self.state

    def _mark_done(self, done: Set[str], phase: str) -> None:
        done.add(phase)
        self._completed_today = set(done)
        self.save_checkpoint(completed=done)

    # ------------------------------------------------------------------
    # Crash-recovery checkpoint (ORC-07 / E2E-02)
    # ------------------------------------------------------------------

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.cfg.run.log_dir, "checkpoint.pkl")

    def save_checkpoint(self, completed: Optional[Set[str]] = None) -> None:
        """Persist day progress + generation-level evolution state."""
        try:
            os.makedirs(os.path.dirname(self.checkpoint_path) or ".",
                        exist_ok=True)
            payload = {
                "day": self.state.day,
                "completed_phases": sorted(
                    completed if completed is not None
                    else self._completed_today
                ),
                "evolution_generation": self._evolution_progress.get(
                    "generation", 0),
                "evolution_population": self._evolution_progress.get(
                    "population"),
                "memory": self.state.memory,
                "nightmare": self.state.nightmare,
                "skip_trading": self.skip_trading,
            }
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            logger.warning("[orchestrator] checkpoint save failed: %s", e)

    def load_checkpoint(self) -> Set[str]:
        """Restore state from checkpoint.pkl. Returns completed phase names."""
        if not os.path.exists(self.checkpoint_path):
            return set()
        try:
            with open(self.checkpoint_path, "rb") as f:
                payload = pickle.load(f)
            self.state = DayState(day=payload["day"])
            self.state.memory = payload.get("memory")
            self.state.nightmare = payload.get("nightmare")
            self.skip_trading = payload.get("skip_trading", False)
            self._evolution_progress = {
                "generation": payload.get("evolution_generation", 0),
                "population": payload.get("evolution_population"),
            }
            if self._evolution_progress["population"] is None:
                self._evolution_progress = {}
            completed = set(payload.get("completed_phases", []))
            logger.info(
                "[orchestrator] resumed from checkpoint: day=%d, "
                "completed=%s, evolution_gen=%d",
                payload["day"], sorted(completed),
                payload.get("evolution_generation", 0),
            )
            self.trader.audit(
                payload["day"], "resume",
                f"checkpoint restored: completed={sorted(completed)}, "
                f"gen={payload.get('evolution_generation', 0)}",
            )
            return completed
        except Exception as e:
            logger.error("[orchestrator] checkpoint load failed: %s", e)
            return set()

    def clear_checkpoint(self) -> None:
        self._completed_today = set()
        try:
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Veto processing (non-negotiable #1)
    # ------------------------------------------------------------------

    def _process_veto(self) -> None:
        """
        A veto.txt in the repo root schedules a manual override that only
        takes effect veto_delay_hours later - human panic cannot touch the
        book intraday.
        """
        veto_path = self.cfg.orchestrator.veto_file
        delay = timedelta(hours=int(self.cfg.orchestrator.veto_delay_hours))
        now = datetime.now(timezone.utc)

        if os.path.exists(veto_path):
            mtime = datetime.fromtimestamp(
                os.path.getmtime(veto_path), tz=timezone.utc
            )
            with open(veto_path) as f:
                directive = f.read().strip()
            # VET-02: reject gibberish before it can ever be scheduled
            if not self._is_valid_veto(directive):
                logger.warning("Invalid veto syntax. Ignoring. (%r)", directive)
                self.trader.audit(
                    self.state.day, "veto",
                    f"invalid syntax ignored: {directive!r}",
                )
            elif self._pending_veto is None or \
                    self._pending_veto.get("directive") != directive:
                self._pending_veto = {
                    "directive": directive,
                    "effective_at": (mtime + delay).isoformat(),
                }
                self.trader.audit(
                    self.state.day, "veto",
                    f"scheduled (effective {self._pending_veto['effective_at']}): "
                    f"{directive}",
                )

        if self._pending_veto is not None:
            effective = datetime.fromisoformat(self._pending_veto["effective_at"])
            if now >= effective:
                # VET-03: never execute a veto on a closed market -
                # postpone to the next trading day.
                if not is_nse_trading_day(now.date()):
                    postponed = next_nse_trading_day(now.date())
                    new_effective = datetime.combine(
                        postponed, effective.timetz()
                    )
                    self._pending_veto["effective_at"] = new_effective.isoformat()
                    self.trader.audit(
                        self.state.day, "veto",
                        f"market closed - postponed to {postponed.isoformat()}",
                    )
                    return
                self.trader.audit(
                    self.state.day, "veto",
                    f"APPLYING: {self._pending_veto['directive']}",
                )
                self._apply_veto(self._pending_veto["directive"])
                self._pending_veto = None
                if os.path.exists(veto_path):
                    os.remove(veto_path)

    @staticmethod
    def _is_valid_veto(directive: str) -> bool:
        """
        Accepted grammar (case-insensitive):
          FLATTEN                  close the whole book
          HALT                     zero the position cap
          SELL ALL <TICKER>        close a single position
        """
        import re
        d = directive.strip().upper()
        return bool(
            d in ("FLATTEN", "HALT")
            or re.fullmatch(r"SELL ALL [A-Z][A-Z0-9.\-]{0,9}", d)
        )

    def _apply_veto(self, directive: str) -> None:
        directive = directive.upper()
        if "FLATTEN" in directive:
            for ticker, price in list(self.trader.last_prices.items()):
                self.execution.submit_order(
                    self.state.day, ticker, 0.0, price, bar_volume=1e9
                )
        elif directive.startswith("SELL ALL "):
            ticker = directive.split()[-1]
            price = self.trader.last_prices.get(ticker)
            if price:
                self.execution.submit_order(
                    self.state.day, ticker, 0.0, price, bar_volume=1e9
                )
        elif "HALT" in directive:
            self.reflex.gate.state.position_cap = 0.0
            self.reflex.gate.state.regime = "panic"

    # ------------------------------------------------------------------
    # Heartbeat + checkpointing (non-negotiable #4)
    # ------------------------------------------------------------------

    def heartbeat(self, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        interval = timedelta(minutes=int(self.cfg.orchestrator.heartbeat_minutes))
        if self._last_heartbeat and now - self._last_heartbeat < interval:
            return
        self._last_heartbeat = now
        snapshot = {
            "equity": self.trader.equity(),
            "positions": dict(self.trader.positions),
            "gate": self.reflex.gate.state.regime,
            "position_cap": self.reflex.gate.position_cap,
            "vix_last": self.reflex.gate.state.last_vix,
        }
        self.trader.audit(self.state.day, "heartbeat", json.dumps(snapshot))

    def _checkpoint_model(self) -> None:
        if self.master_model is None:
            return
        path = os.path.join(
            self.cfg.orchestrator.checkpoint_dir,
            f"master_day{self.state.day:03d}.pt",
        )
        try:
            torch.save(self.master_model.state_dict(), path)
            self.trader.audit(self.state.day, "checkpoint", path)
        except Exception as e:
            logger.warning("[orchestrator] checkpoint failed: %s", e)

    # ------------------------------------------------------------------
    # RunPod nightly training
    #
    # RunPod pod orchestration (create/train/pull/delete) runs entirely
    # in GitHub Actions now - .github/workflows/train-runpod.yml is
    # scheduled nightly and scp's the resulting checkpoint straight onto
    # this box. Kronos no longer talks to the RunPod API or holds an SSH
    # key for it at all; its only job is noticing a newer checkpoint file
    # has appeared locally and adopting it.
    #
    # maybe_adopt_runpod_checkpoint() is polled every iteration of the
    # main realtime loop (scripts/run_kronos.py) - a cheap os.path.getmtime()
    # check, never a blocking wait, since there's nothing to wait ON
    # anymore: by the time this box sees the file, training already
    # finished on GitHub's infrastructure.
    # ------------------------------------------------------------------

    def _active_snn_path(self) -> str:
        return os.path.join(self.cfg.orchestrator.checkpoint_dir, "reflex_snn_active.pt")

    def _load_persisted_snn(self) -> None:
        """Best-effort restore of the last successfully-adopted RunPod SNN
        checkpoint, so a service restart doesn't quietly lose it and fall
        back to a fresh random init. Never raises."""
        path = self._active_snn_path()
        if not os.path.exists(path):
            return
        try:
            state_dict = torch.load(path, map_location="cpu")
            self.reflex.snn.load_state_dict(state_dict)
            logger.info("[orchestrator] restored last-adopted RunPod SNN weights from %s", path)
        except Exception as e:
            logger.warning("[orchestrator] could not restore persisted SNN weights (%s) - starting from scratch", e)

    def _persist_active_snn(self) -> None:
        try:
            torch.save(self.reflex.snn.state_dict(), self._active_snn_path())
        except Exception as e:
            logger.warning("[orchestrator] failed to persist active SNN weights: %s", e)

    def _daily_bias_path(self) -> str:
        return os.path.join(self.cfg.orchestrator.checkpoint_dir, "reflex_daily_bias.npy")

    def _load_persisted_daily_bias(self) -> None:
        """Mirrors _load_persisted_snn() - a restart (deploy-hetzner.yml
        redeploys frequently) would otherwise silently drop the confidence
        blend until the next checkpoint adoption, which can be up to a day
        away. Never raises."""
        path = self._daily_bias_path()
        if not os.path.exists(path):
            return
        try:
            self.reflex.set_daily_bias(np.load(path))
            logger.info("[orchestrator] restored last-computed daily bias from %s", path)
        except Exception as e:
            logger.warning("[orchestrator] could not restore persisted daily bias (%s) - none until next adoption", e)

    def _persist_daily_bias(self, bias: Optional[np.ndarray]) -> None:
        path = self._daily_bias_path()
        try:
            if bias is None:
                if os.path.exists(path):
                    os.remove(path)
                return
            np.save(path, bias)
        except Exception as e:
            logger.warning("[orchestrator] failed to persist daily bias: %s", e)

    def _size_scale_path(self) -> str:
        return os.path.join(self.cfg.orchestrator.checkpoint_dir, "reflex_size_scale.txt")

    def _load_persisted_size_scale(self) -> None:
        """Mirrors _load_persisted_daily_bias() - a restart shouldn't
        silently drop the calibrated size scale and fall back to the
        default until the next adoption. Never raises."""
        path = self._size_scale_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                self.reflex._size_scale = float(f.read().strip())
            logger.info("[orchestrator] restored last-calibrated size scale (%.3f) from %s",
                        self.reflex._size_scale, path)
        except Exception as e:
            logger.warning("[orchestrator] could not restore persisted size scale (%s) - using default", e)

    def _persist_size_scale(self) -> None:
        try:
            with open(self._size_scale_path(), "w") as f:
                f.write(str(self.reflex._size_scale))
        except Exception as e:
            logger.warning("[orchestrator] failed to persist size scale: %s", e)

    def _current_runpod_checkpoint_mtime(self) -> Optional[float]:
        path = os.path.join(RUNPOD_CHECKPOINT_DIR, "meta", "snn.pt")
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    def maybe_adopt_runpod_checkpoint(self) -> None:
        """Non-blocking - safe to call every iteration of the main loop.
        If checkpoints/runpod/meta/snn.pt exists and is newer than the
        last one adopted, loads it into self.reflex.snn. No-op otherwise
        (including "file hasn't shown up yet" and "same file as last
        time" - GitHub Actions overwrites it in place each night)."""
        mtime = self._current_runpod_checkpoint_mtime()
        if mtime is None or mtime == self._runpod_last_adopted_mtime:
            return
        if load_runpod_checkpoint(self.reflex.snn, checkpoint_dir=RUNPOD_CHECKPOINT_DIR):
            self._runpod_last_adopted_mtime = mtime
            self._runpod_adopted_today = True
            self._persist_active_snn()
            logger.info("[orchestrator] adopted a new RunPod-trained SNN checkpoint")
            # Recompute the causal_transformer "second opinion" ReflexArc
            # checks its own signal against - tied to this exact checkpoint,
            # so it must be refreshed whenever the SNN is (compute_daily_bias
            # fails closed to None on any missing/mismatched file, which
            # set_daily_bias(None) correctly reads as "no confidence
            # blending until the next successful computation").
            bias = None
            if self.state.memory is not None:
                bias = compute_daily_bias(
                    self.state.memory.returns_window(self.cfg.data.lookback_days),
                    checkpoint_dir=RUNPOD_CHECKPOINT_DIR,
                )
            self.reflex.set_daily_bias(bias)
            self._persist_daily_bias(bias)
            # Recalibrate the raw-pred -> position-size scale against this
            # checkpoint's own realized track record (kronos/reflex.py's
            # calibrate_size_scale - see its docstring for why this replaces
            # a naive "normalize by the prediction's own volatility"
            # approach, which was tested and rejected). Tied to this exact
            # checkpoint for the same reason the daily bias is.
            if self.state.memory is not None:
                self.reflex.calibrate_size_scale(
                    self.state.memory.returns_window(self.cfg.data.lookback_days)
                )
                self._persist_size_scale()
        else:
            # Don't retry the same broken/mismatched file every 30s -
            # remember it as "seen" so the warning (and this alert) fires
            # once per distinct bad file, not forever.
            self._runpod_last_adopted_mtime = mtime
            self._alert(
                "Kronos: RunPod checkpoint adoption FAILED - a new "
                "checkpoints/runpod/meta/snn.pt appeared but failed to "
                "load (shape mismatch or corrupt file). Keeping "
                "yesterday's weights; see the service log for the traceback."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_weights(self) -> Optional[np.ndarray]:
        if self.state.memory is None:
            return None
        tickers = self.state.memory.tickers
        eq = self.trader.equity()
        if eq <= 0 or not self.trader.positions:
            return None
        weights = np.array([
            self.trader.positions.get(t, 0.0)
            * self.trader.last_prices.get(t, 0.0) / eq
            for t in tickers
        ])
        return weights if np.abs(weights).sum() > 0 else None

    def _master_signals(self) -> Optional[np.ndarray]:
        if self.master_model is None or self.state.memory is None:
            return None
        try:
            # AUDIT-1A: master_model's input width is (horizon - 1) bars,
            # not horizon - see kronos/evolver.py's KronosEvolver.__init__.
            horizon = self.cfg.nightmare.horizon_days - 1
            window = self.state.memory.returns_window(horizon)
            x = torch.tensor(window.reshape(1, -1), dtype=torch.float32)
            with torch.no_grad():
                out = self.master_model(x)
            return torch.tanh(out.squeeze(0)).numpy()
        except Exception as e:
            logger.warning("[orchestrator] master signal failed: %s", e)
            return None
