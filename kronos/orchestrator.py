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

from kronos.config import KronosConfig, load_config
from kronos.data_pipeline import DataPipeline, DailyMemory, DataUnavailableError
from kronos.evolver import KronosEvolver, EvolutionResult
from kronos.nightmare_generator import NightmareGenerator, NightmareBuffer
from kronos.notifier import TelegramNotifier
from kronos.paper_trader import PaperTrader
from kronos.reflex import ReflexArc
from kronos.reporter import GodsEyeReporter
from kronos.warmer import KronosWarmer, WarmupResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NYSE trading calendar (ORC-04 / ORC-05 / VET-03)
# ---------------------------------------------------------------------------

def _easter(year: int) -> ddate:
    """Anonymous Gregorian computus."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return ddate(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> ddate:
    """n-th (1-based) given weekday of a month; n=-1 for the last."""
    if n > 0:
        d = ddate(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + timedelta(days=offset + 7 * (n - 1))
    d = ddate(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def _observed(d: ddate) -> ddate:
    """Sat -> Fri, Sun -> Mon observance."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> Set[ddate]:
    return {
        _observed(ddate(year, 1, 1)),                 # New Year's Day
        _nth_weekday(year, 1, 0, 3),                  # MLK Day
        _nth_weekday(year, 2, 0, 3),                  # Presidents' Day
        _easter(year) - timedelta(days=2),            # Good Friday
        _nth_weekday(year, 5, 0, -1),                 # Memorial Day
        _observed(ddate(year, 6, 19)),                # Juneteenth
        _observed(ddate(year, 7, 4)),                 # Independence Day
        _nth_weekday(year, 9, 0, 1),                  # Labor Day
        _nth_weekday(year, 11, 3, 4),                 # Thanksgiving
        _observed(ddate(year, 12, 25)),               # Christmas
    }


def is_trading_day(d: ddate) -> bool:
    if d.weekday() >= 5:
        return False
    return d not in nyse_holidays(d.year)


def next_trading_day(d: ddate) -> ddate:
    nxt = d + timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt


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

        self.state = DayState()
        self.master_model: Optional[torch.nn.Module] = None
        self._last_heartbeat: Optional[datetime] = None
        self._pending_veto: Optional[Dict] = None
        self._phase_runs: Set[Tuple[str, str]] = set()
        self._evolution_progress: Dict = {}   # generation-level resume state
        self._completed_today: Set[str] = set()
        self.skip_trading: bool = False       # set on stale/unavailable data

        os.makedirs(self.cfg.orchestrator.checkpoint_dir, exist_ok=True)

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
        if is_trading_day(d):
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
        return memory

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
        """One market-hours tick: infer, gate, trade."""
        if self.state.memory is None:
            return None
        if self.skip_trading:
            return None   # stale/unavailable data day - observe, never trade
        recent = self.state.memory.returns_window(
            self.cfg.nightmare.horizon_days
        )
        decision = self.reflex.infer(
            recent, vix_value, bar_prices, bar_volumes, now=now
        )
        # Execute toward signal-derived target weights
        if bar_prices:
            tickers = self.state.memory.tickers
            kelly_fraction = float(self.cfg.trading.kelly_fraction)
            for i, ticker in enumerate(tickers):
                if ticker not in bar_prices or i >= len(decision.signals):
                    continue
                target = float(decision.signals[i]) * kelly_fraction \
                    * float(self.cfg.trading.max_position_pct)
                asset_cap = min(
                    decision.position_cap,
                    decision.asset_caps.get(ticker, 1.0),
                )
                self.trader.execute(
                    self.state.day, ticker, target,
                    bar_prices[ticker],
                    (bar_volumes or {}).get(ticker, 0.0),
                    position_cap=asset_cap,
                )
        return decision

    def run_logging(self, closing_prices: Optional[Dict[str, float]] = None) -> Dict:
        prices = closing_prices or dict(self.trader.last_prices)
        stats = self.trader.close_day(self.state.day, prices)
        self.trader.audit(self.state.day, "logging", json.dumps(stats))
        self._send_daily_notification(stats)
        return stats

    def _send_daily_notification(self, stats: Dict) -> None:
        """Best-effort Telegram digest - never let a notification problem
        affect trading. Fully silent no-op if notifications aren't configured."""
        if not (self.notifier.enabled and self.cfg.notifications.send_daily_digest):
            return
        try:
            memory = self.state.memory
            evo = self.state.evolution
            warm = self.state.warmup
            text = self.notifier.build_daily_report(
                day=self.state.day,
                stats=stats,
                regime=warm.regime_estimate if warm else None,
                top_fitness=evo.top_fitness if evo else None,
                source_used=memory.source_used if memory else None,
                quality_flags=memory.quality_flags if memory else None,
                phase_failures=self.state.phase_failures or None,
                reflex_regime=self.reflex.gate.state.regime,
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
                if not is_trading_day(now.date()):
                    postponed = next_trading_day(now.date())
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
                self.trader.execute(
                    self.state.day, ticker, 0.0, price, bar_volume=1e9
                )
        elif directive.startswith("SELL ALL "):
            ticker = directive.split()[-1]
            price = self.trader.last_prices.get(ticker)
            if price:
                self.trader.execute(
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
            horizon = self.cfg.nightmare.horizon_days
            window = self.state.memory.returns_window(horizon)
            x = torch.tensor(window.reshape(1, -1), dtype=torch.float32)
            with torch.no_grad():
                out = self.master_model(x)
            return torch.tanh(out.squeeze(0)).numpy()
        except Exception as e:
            logger.warning("[orchestrator] master signal failed: %s", e)
            return None
