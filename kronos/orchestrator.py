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
import traceback
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from enum import Enum
from typing import Callable, Dict, Optional

import numpy as np
import torch

from kronos.config import KronosConfig, load_config
from kronos.data_pipeline import DataPipeline, DailyMemory
from kronos.evolver import KronosEvolver, EvolutionResult
from kronos.nightmare_generator import NightmareGenerator, NightmareBuffer
from kronos.paper_trader import PaperTrader
from kronos.reflex import ReflexArc
from kronos.reporter import GodsEyeReporter
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

        self.state = DayState()
        self.master_model: Optional[torch.nn.Module] = None
        self._last_heartbeat: Optional[datetime] = None
        self._pending_veto: Optional[Dict] = None

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
        memory = self._run_phase("digestion", self.pipeline.run_sync, filings)
        self.state.memory = memory
        if memory is not None:
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

    def run_evolution(self) -> Optional[EvolutionResult]:
        if self.state.nightmare is None:
            self.trader.audit(self.state.day, "evolution", "skipped - no nightmares")
            return None
        result = self._run_phase(
            "evolution", self.evolver.evolve,
            self.state.nightmare, self.state.degraded,
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
                self.trader.execute(
                    self.state.day, ticker, target,
                    bar_prices[ticker],
                    (bar_volumes or {}).get(ticker, 0.0),
                    position_cap=decision.position_cap,
                )
        return decision

    def run_logging(self, closing_prices: Optional[Dict[str, float]] = None) -> Dict:
        prices = closing_prices or dict(self.trader.last_prices)
        stats = self.trader.close_day(self.state.day, prices)
        self.trader.audit(self.state.day, "logging", json.dumps(stats))
        return stats

    # ------------------------------------------------------------------
    # Full-day driver (used by run_kronos.py and the e2e test)
    # ------------------------------------------------------------------

    def run_full_day(
        self,
        day: int,
        filings: Optional[Dict[str, str]] = None,
        market_ticks: Optional[list] = None,
    ) -> DayState:
        """
        Execute one complete 24h cycle synchronously (compressed time).
        market_ticks: list of (vix, prices, volumes) tuples simulating bars.
        """
        self.state = DayState(day=day)
        self._process_veto()

        self.run_digestion(filings)
        self.run_nightmare()
        self.run_evolution()
        self.run_adaptation()
        self.run_report()

        for tick in (market_ticks or []):
            vix, prices, volumes = tick
            self.run_reflex_tick(vix, prices, volumes)
            self.heartbeat()

        self.run_logging()
        self.trader.audit(day, "day-complete",
                          f"failures={self.state.phase_failures}")
        return self.state

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
            if self._pending_veto is None or \
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
                self.trader.audit(
                    self.state.day, "veto",
                    f"APPLYING: {self._pending_veto['directive']}",
                )
                self._apply_veto(self._pending_veto["directive"])
                self._pending_veto = None
                if os.path.exists(veto_path):
                    os.remove(veto_path)

    def _apply_veto(self, directive: str) -> None:
        """Supported directives: FLATTEN (close everything), HALT (cap=0)."""
        directive = directive.upper()
        if "FLATTEN" in directive:
            for ticker, price in list(self.trader.last_prices.items()):
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
