"""
The consumer half of the NightEvolver checkpoint contract.

WHAT WAS MISSING. `nightevolver/saver.py` writes a verified checkpoint and
the deploy path delivers it to Hetzner; `nightevolver/strategy_decoder.py`
can turn one into live signals. Nothing in `kronos/` imported either -
`grep -n "nightevolver" kronos/*.py` returned zero matches. The training
ran, the artifact shipped, and the trading loop never looked at it. This
module is that missing link, and it is deliberately the only place the
two halves touch.

THREE WAYS THIS WIRING SILENTLY DOES NOTHING, all measured, all guarded:

1. TICKER NAMING. NightEvolver names assets from the NSE bhavcopy -
   RELIANCE. Kronos names them for yfinance - RELIANCE.NS. A join on the
   raw strings matches zero rows, and the tempting "fix" of zipping the
   two lists by INDEX is far worse: it attaches every gene weight to the
   wrong asset while looking like it works. Names are normalised and the
   overlap is ASSERTED non-empty; an empty overlap disables the bridge
   loudly rather than trading a scrambled book.

2. WARMUP DEPTH. EvolvedStrategy.signal() needs WARMUP_BARS + 2 = 62 bars
   and returns all-flat below that, with only a warning. Kronos's
   pipeline fetches `data.lookback_days` = 30. Wiring the bridge to
   DailyMemory would therefore have logged "loaded checkpoint" once and
   then returned zeros forever. The bridge keeps its OWN deeper history
   and refuses to contribute at all until it has the depth, instead of
   contributing zeros that look like a flat opinion.

3. DOUBLE KELLY. LiveSignal.target_weight is already sized - conviction
   floor, Kelly fraction and position cap applied. Kronos's tick loop
   takes a RAW signal in [-1, +1] and applies its own kelly_fraction and
   max_position_pct downstream. Feeding target_weight in would square
   both. The bridge emits the raw score instead, so sizing stays in
   exactly one place.

FAILING CLOSED. Every error path here returns "no opinion", never a
guess, and the SNN's own signal continues untouched. That includes a
checkpoint below the deflated-Sharpe gate: `load_checkpoint` raises and
the bridge stays dark. This is the mechanism that stops an overfit
nightly run reaching the account for being the newest file on disk.

ON WHETHER TO ENABLE IT. Default off, and that default is a finding, not
caution. This project's own information audit could not establish a
directional edge in this data by three independent methods, and the cost
arithmetic puts the break-even win rate above what the measured
correlations can deliver. Wiring it makes the checkpoint REACHABLE and
auditable; it does not make it profitable. Turning `enabled` on is a
decision the evidence does not currently support - take it deliberately,
in paper mode, and read `blend_weight` as how much of the book you are
handing to a strategy that has not beaten its own noise benchmark.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger("kronos.nightevolver")

# EvolvedStrategy.signal() goes flat below WARMUP_BARS + 2. Imported
# lazily in _load so a missing nightevolver package degrades to "bridge
# disabled" rather than breaking kronos's import graph.
DEFAULT_HISTORY_DAYS = 180
DEFAULT_BLEND_WEIGHT = 0.0
DEFAULT_MAX_POSITION = 0.10


def normalise_ticker(t: str) -> str:
    """RELIANCE.NS / reliance / RELIANCE -> RELIANCE.

    Only the exchange suffix is stripped. Anything cleverer (fuzzy
    matching, prefix search) would risk pairing two genuinely different
    listings, which is the failure this function exists to prevent.
    """
    s = str(t).strip().upper()
    for suffix in (".NS", ".BO", ".NSE", ".BSE"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


class NightEvolverBridge:
    """Loads a gated checkpoint and maps it onto Kronos's ticker space."""

    def __init__(self, checkpoint_path: Path, blend_weight: float = DEFAULT_BLEND_WEIGHT,
                 history_days: int = DEFAULT_HISTORY_DAYS,
                 max_position: float = DEFAULT_MAX_POSITION,
                 require_gate: bool = True):
        self.checkpoint_path = Path(checkpoint_path)
        self.blend_weight = float(min(max(blend_weight, 0.0), 1.0))
        self.history_days = int(history_days)
        self.max_position = float(max_position)
        self.require_gate = bool(require_gate)

        self._strategy = None                  # EvolvedStrategy | None
        self._ck_tickers: List[str] = []       # normalised
        self._mtime: Optional[float] = None
        self._close: Optional[pd.DataFrame] = None      # [date x NORMALISED]
        self._volume: Optional[pd.DataFrame] = None
        self._status = "not loaded"
        self._min_bars = 62                    # refined from the package on load

    # -- status ---------------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    @property
    def active(self) -> bool:
        """True only when this bridge can actually produce an opinion."""
        return (self._strategy is not None
                and self._close is not None
                and len(self._close) >= self._min_bars
                and self.blend_weight > 0.0)

    # -- loading --------------------------------------------------------

    def maybe_reload(self) -> bool:
        """Load the checkpoint if it is new. Returns True if adopted.

        Mtime-gated like the RunPod SNN adoption, so a broken checkpoint
        is not re-read and re-reported on every tick.
        """
        try:
            mtime = os.path.getmtime(self.checkpoint_path)
        except OSError:
            if self._status != "no checkpoint file":
                self._status = "no checkpoint file"
                logger.info("[nightevolver] no checkpoint at %s - bridge idle",
                            self.checkpoint_path)
            return False

        if self._mtime is not None and mtime <= self._mtime:
            return False
        self._mtime = mtime
        return self._load()

    def _load(self) -> bool:
        try:
            from nightevolver.data_loader import WARMUP_BARS
            from nightevolver.strategy_decoder import EvolvedStrategy
        except Exception as e:                                   # noqa: BLE001
            self._status = f"nightevolver package unavailable: {e}"
            logger.warning("[nightevolver] %s", self._status)
            return False

        self._min_bars = int(WARMUP_BARS) + 2
        try:
            strat = EvolvedStrategy.from_checkpoint(
                self.checkpoint_path, require_gate=self.require_gate,
                max_position=self.max_position)
        except Exception as e:                                   # noqa: BLE001
            # Covers the statistical gate, a genome_version mismatch and a
            # corrupt file alike. All three mean the same thing here:
            # do not trade this, keep whatever we had.
            self._strategy = None
            self._status = f"checkpoint REJECTED: {e}"
            logger.warning("[nightevolver] %s", self._status)
            return False

        self._strategy = strat
        self._ck_tickers = [normalise_ticker(t) for t in strat.tickers]
        self._status = f"loaded ({len(self._ck_tickers)} tickers)"
        logger.info("[nightevolver] checkpoint adopted: %s", self._status)
        return True

    # -- history --------------------------------------------------------

    def set_history(self, close: pd.DataFrame,
                    volume: Optional[pd.DataFrame] = None) -> None:
        """Install the deep price history the decoder needs.

        Columns are normalised here so every later lookup is on the same
        key space as the checkpoint's tickers.
        """
        if close is None or close.empty:
            self._close, self._volume = None, None
            self._status = "history empty"
            return
        c = close.copy()
        c.columns = [normalise_ticker(t) for t in c.columns]
        c = c.loc[:, ~c.columns.duplicated()]
        self._close = c
        if volume is not None and not volume.empty:
            v = volume.copy()
            v.columns = [normalise_ticker(t) for t in v.columns]
            self._volume = v.loc[:, ~v.columns.duplicated()]
        else:
            self._volume = None

        if len(c) < self._min_bars:
            self._status = (f"history too short: {len(c)} bars < {self._min_bars} "
                            f"needed - contributing nothing")
            logger.warning("[nightevolver] %s", self._status)

    def covered(self, tickers: Sequence[str]) -> List[str]:
        """Kronos tickers this checkpoint can actually speak about."""
        if self._strategy is None:
            return []
        have = set(self._ck_tickers)
        if self._close is not None:
            have &= set(self._close.columns)
        return [t for t in tickers if normalise_ticker(t) in have]

    # -- signals --------------------------------------------------------

    def raw_signals(self, tickers: Sequence[str],
                    intraday_prices: Optional[Dict[str, float]] = None
                    ) -> Dict[str, float]:
        """RAW scores in [-1, +1], keyed by the caller's ticker strings.

        Raw on purpose: Kronos applies kelly_fraction and max_position_pct
        downstream, so returning the decoder's already-sized
        target_weight would apply Kelly twice.

        An empty dict means "no opinion" and every caller must treat it
        that way - never as "go flat".
        """
        if not self.active:
            return {}
        cov = self.covered(tickers)
        if not cov:
            self._status = ("checkpoint covers none of the live tickers - "
                            "bridge inert")
            logger.warning("[nightevolver] %s (live=%s, checkpoint=%s)",
                           self._status, list(tickers)[:4], self._ck_tickers[:4])
            return {}

        cols = [normalise_ticker(t) for t in cov]
        close = self._close.loc[:, cols]

        # Append today's live bar so the newest row is the current price
        # rather than yesterday's close. Without this the strategy trades
        # a one-day-stale view all session.
        if intraday_prices:
            live = {normalise_ticker(k): v for k, v in intraday_prices.items()}
            row = {c: live.get(c) for c in cols}
            if all(v is not None and v > 0 for v in row.values()):
                close = pd.concat(
                    [close, pd.DataFrame([row], index=[pd.Timestamp.utcnow()])])

        try:
            sigs = self._strategy.signal(close)
        except Exception as e:                                   # noqa: BLE001
            self._status = f"signal computation failed: {e}"
            logger.warning("[nightevolver] %s", self._status)
            return {}

        by_norm = {}
        for s in sigs:
            # direction is 0 when the score is under the conviction
            # floor; that is an abstention, and it must reach the caller
            # as 0.0 rather than as a small unconvinced score.
            by_norm[normalise_ticker(s.ticker)] = (
                float(s.score) if s.direction != 0 else 0.0)

        out = {}
        for t in cov:
            n = normalise_ticker(t)
            if n in by_norm:
                out[t] = by_norm[n]
        self._status = f"active on {len(out)}/{len(tickers)} tickers"
        return out

    def blend(self, tickers: Sequence[str], base: Sequence[float],
              intraday_prices: Optional[Dict[str, float]] = None) -> List[float]:
        """Mix this checkpoint's opinion into the SNN's signal vector.

        Tickers the checkpoint does not cover keep the base signal
        UNCHANGED - a checkpoint trained on 48 names must not drag the
        other assets toward zero just by being silent about them.
        """
        out = [float(b) for b in base]
        if not self.active:
            return out
        ne = self.raw_signals(tickers, intraday_prices)
        if not ne:
            return out
        w = self.blend_weight
        for i, t in enumerate(tickers):
            if i >= len(out) or t not in ne:
                continue
            out[i] = (1.0 - w) * out[i] + w * ne[t]
        return out

    # -- construction ---------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> Optional["NightEvolverBridge"]:
        """Build from the `nightevolver:` config block, or None if off."""
        try:
            block = cfg.get("nightevolver", None)
        except Exception:                                        # noqa: BLE001
            block = None
        if not block:
            return None
        get = block.get if hasattr(block, "get") else (lambda k, d=None: d)
        if not bool(get("enabled", False)):
            logger.info("[nightevolver] bridge disabled in config")
            return None
        return cls(
            checkpoint_path=Path(get("checkpoint_path",
                                     "checkpoints/nightevolver/nightevolver_best.json")),
            blend_weight=float(get("blend_weight", DEFAULT_BLEND_WEIGHT)),
            history_days=int(get("history_days", DEFAULT_HISTORY_DAYS)),
            max_position=float(get("max_position", DEFAULT_MAX_POSITION)),
            require_gate=bool(get("require_gate", True)),
        )
