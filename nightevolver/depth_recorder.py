"""
Forward-recorder for Kite 5-level market depth.

WHY THIS IS A RECORDER AND NOT A FETCHER
----------------------------------------
Kite's historical endpoint serves candles only. Market depth exists
exclusively in the live WebSocket stream, so order-book imbalance,
book pressure and signed volume **cannot be backfilled at any price**.
The only way to ever have this dataset is to start writing it down, and
the only bad day to start is a later one - every day not recorded is a
day that cannot be recovered.

That makes this module unusual for this codebase: it produces nothing
useful today. Its entire value is that in three months there is
something to run the information audit against.

DESIGN NOTES
------------
Storage is gzipped JSONL, one file per IST trading date. Not a binary
format, and not Parquet:

  - inspectable with `zcat | head` when something looks wrong at 3am,
    which matters for a dataset nobody is watching accumulate;
  - append-only and crash-safe at record granularity - a kill -9 costs
    the last line, not the file;
  - no schema migration problem when a field is added later;
  - ~10-20 MB/day gzipped for 10 symbols, so ~1 GB per quarter. The
    space saved by a binary layout is not worth the class of bug it
    invites (see the 12-byte depth padding in kite.py).

Keys are short (`t`, `b`, `a`) because they repeat on every record and
this is the one place terseness actually buys something.

THE FAILURE MODE THAT MATTERS is not a crash - it is a recorder that
looks alive and is writing nothing, or writing stale repeats. So:
records are only written when the book actually CHANGES (see
`min_interval_ms` and change detection), and the stats line reports
records-written alongside frames-received so a silent stall is visible
in the log rather than discovered months later in the data.
"""

from __future__ import annotations

import gzip
import json
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from nightevolver.kite import (
    KiteAuthError, Tick, load_credentials, parse_binary_message, resolve_tokens,
    websocket_url,
)

logger = logging.getLogger("nightevolver.recorder")

IST = timezone(timedelta(hours=5, minutes=30))

DEFAULT_DIR = Path(__file__).parent.parent / "data" / "depth"

# NSE cash equity continuous session.
MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)


def ist_now() -> datetime:
    return datetime.now(IST)


def is_market_open(now: Optional[datetime] = None) -> bool:
    """Weekday and inside 09:15-15:30 IST.

    Deliberately does NOT consult the NSE holiday calendar: on a holiday
    the stream simply carries no data, and a recorder that refuses to
    connect because it believes it is a holiday is worse than one that
    connects and records nothing. kronos/calendar_utils.py has the
    calendar if a caller wants it.
    """
    now = now or ist_now()
    if now.weekday() >= 5:
        return False
    hm = (now.hour, now.minute)
    return MARKET_OPEN <= hm <= MARKET_CLOSE


def tick_to_record(tick: Tick, recv_ns: int) -> Dict:
    """One depth snapshot, as the dict that becomes a JSONL line.

    Both timestamps are kept on purpose. `ts` is our receive time and is
    what orders events for research; `xts` is the exchange timestamp and
    is the only way to detect that our clock or the network drifted.
    Keeping one and discarding the other makes latency effects
    permanently invisible.
    """
    return {
        "ts": recv_ns,
        "xts": tick.exchange_timestamp,
        "tk": tick.instrument_token,
        "ltp": tick.last_price,
        "lq": tick.last_quantity,
        "v": tick.volume,
        "tbq": tick.total_buy_quantity,
        "tsq": tick.total_sell_quantity,
        "b": [[e.quantity, e.price, e.orders] for e in tick.bids],
        "a": [[e.quantity, e.price, e.orders] for e in tick.asks],
    }


def _book_signature(tick: Tick) -> Tuple:
    """What counts as 'the book changed'. Used to suppress duplicates."""
    return (tick.last_price, tick.volume,
            tuple((e.quantity, e.price) for e in tick.bids),
            tuple((e.quantity, e.price) for e in tick.asks))


@dataclass
class RecorderStats:
    frames: int = 0
    ticks: int = 0
    written: int = 0
    suppressed: int = 0
    reconnects: int = 0
    bytes_written: int = 0
    started: float = field(default_factory=time.monotonic)

    def line(self) -> str:
        mins = (time.monotonic() - self.started) / 60.0
        return (f"[recorder] {mins:.1f}m | frames={self.frames} ticks={self.ticks} "
                f"written={self.written} suppressed={self.suppressed} "
                f"reconnects={self.reconnects} "
                f"~{self.bytes_written / 1e6:.1f}MB")


class DepthWriter:
    """Gzipped JSONL sink with per-IST-date rotation.

    Rotation is by IST date rather than by size so that one file is one
    trading session - the unit every downstream analysis actually wants.
    """

    def __init__(self, directory: Path = DEFAULT_DIR, compresslevel: int = 6):
        self.directory = Path(directory)
        self.compresslevel = compresslevel
        self._date: Optional[str] = None
        self._fh: Optional[gzip.GzipFile] = None
        self.bytes_written = 0

    def path_for(self, date_str: str) -> Path:
        return self.directory / f"depth_{date_str}.jsonl.gz"

    def _rotate(self, date_str: str) -> None:
        self.close()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(date_str)
        # Append, never truncate: a restart mid-session must not destroy
        # the morning's recording.
        self._fh = gzip.open(path, "at", compresslevel=self.compresslevel,
                             encoding="utf-8")
        self._date = date_str
        logger.info("[recorder] writing to %s", path)

    def write(self, record: Dict, now: Optional[datetime] = None) -> None:
        date_str = (now or ist_now()).strftime("%Y%m%d")
        if date_str != self._date:
            self._rotate(date_str)
        assert self._fh is not None
        line = json.dumps(record, separators=(",", ":")) + "\n"
        self._fh.write(line)
        self.bytes_written += len(line)

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    def __enter__(self) -> "DepthWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class DepthRecorder:
    """Long-running WebSocket recorder.

    Reconnects on transport errors with backoff, but exits immediately
    and loudly on KiteAuthError: a stale daily token cannot be fixed by
    retrying, and a process that spins on it looks alive while recording
    nothing - the exact silent failure this module is meant to avoid.
    """

    def __init__(self, symbols: Sequence[str],
                 directory: Path = DEFAULT_DIR,
                 min_interval_ms: int = 0,
                 market_hours_only: bool = True,
                 stats_every: int = 20000,
                 tokens: Optional[Dict[str, int]] = None,
                 url: Optional[str] = None):
        """`tokens` and `url` exist so the connect/subscribe/parse/write
        loop can be exercised against a local WebSocket server without
        credentials or network. An async loop that has only ever been
        tested by running it in production is not tested."""
        self.symbols = list(symbols)
        self.directory = Path(directory)
        self.min_interval_ms = min_interval_ms
        self.market_hours_only = market_hours_only
        self.stats_every = stats_every
        self._tokens_override = dict(tokens) if tokens else None
        self._url_override = url
        self.stats = RecorderStats()
        self._stop = False
        self._last_sig: Dict[int, Tuple] = {}
        self._last_write_ns: Dict[int, int] = {}
        self.tokens: Dict[str, int] = {}

    def request_stop(self, *_) -> None:
        logger.info("[recorder] stop requested - finishing current frame")
        self._stop = True

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):
                pass          # not the main thread; caller handles shutdown

    def should_record(self, tick: Tick, recv_ns: int) -> bool:
        """Drop repeats and (optionally) throttle per instrument.

        Kite re-sends the current book on a schedule even when nothing
        moved. Writing those is pure cost: it inflates the file, and
        worse, it makes a stalled feed indistinguishable from a quiet
        one when you look at the data later.
        """
        tok = tick.instrument_token
        if self.min_interval_ms > 0:
            last = self._last_write_ns.get(tok)
            if last is not None and (recv_ns - last) < self.min_interval_ms * 1_000_000:
                return False
        sig = _book_signature(tick)
        if self._last_sig.get(tok) == sig:
            return False
        self._last_sig[tok] = sig
        self._last_write_ns[tok] = recv_ns
        return True

    def _handle_frame(self, data: bytes, writer: DepthWriter) -> None:
        self.stats.frames += 1
        for tick in parse_binary_message(data):
            self.stats.ticks += 1
            if tick.mode != "full":
                continue          # depth is the point; ignore ltp/quote/index
            recv_ns = time.time_ns()
            if not self.should_record(tick, recv_ns):
                self.stats.suppressed += 1
                continue
            writer.write(tick_to_record(tick, recv_ns))
            self.stats.written += 1
            if self.stats_every and self.stats.written % self.stats_every == 0:
                writer.flush()
                self.stats.bytes_written = writer.bytes_written
                logger.info("%s", self.stats.line())

    async def run(self, max_seconds: Optional[float] = None) -> RecorderStats:
        import asyncio

        import websockets

        if self._url_override:
            url = self._url_override
            self.tokens = self._tokens_override or {}
        else:
            creds = load_credentials()
            url = websocket_url(creds)
            self.tokens = self._tokens_override or resolve_tokens(self.symbols)
        logger.info("[recorder] resolved %d symbols: %s", len(self.tokens),
                    ", ".join(f"{k}={v}" for k, v in sorted(self.tokens.items())))
        token_list = sorted(self.tokens.values())

        deadline = None if max_seconds is None else time.monotonic() + max_seconds
        backoff = 1.0

        with DepthWriter(self.directory) as writer:
            while not self._stop:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if self.market_hours_only and not is_market_open():
                    # Idle politely outside the session rather than
                    # holding a connection open for 17 hours a day.
                    await asyncio.sleep(30)
                    continue
                try:
                    async with websockets.connect(
                            url, ping_interval=20,
                            ping_timeout=20, max_size=None) as ws:
                        await ws.send(json.dumps(
                            {"a": "subscribe", "v": token_list}))
                        await ws.send(json.dumps(
                            {"a": "mode", "v": ["full", token_list]}))
                        logger.info("[recorder] connected, subscribed %d "
                                    "instruments in full mode", len(token_list))
                        backoff = 1.0
                        while True:
                            # recv with a timeout rather than `async for`.
                            # `async for` only yields when a message
                            # arrives, so on a quiet feed the stop flag
                            # and the deadline would not be checked for
                            # as long as the market stayed silent - and
                            # --duration exists precisely so a cron job
                            # exits after the close, when the feed IS
                            # silent. Caught by a test.
                            try:
                                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            except asyncio.TimeoutError:
                                message = None
                            if message is not None:
                                if isinstance(message, bytes):
                                    self._handle_frame(message, writer)
                                else:
                                    self._on_text(message)
                            if self._stop:
                                break
                            if deadline is not None and time.monotonic() >= deadline:
                                break
                            if self.market_hours_only and not is_market_open():
                                logger.info("[recorder] session closed - "
                                            "disconnecting until next open")
                                break
                except KiteAuthError:
                    raise
                except Exception as e:                    # transport-level
                    self.stats.reconnects += 1
                    logger.warning("[recorder] connection lost (%s: %s) - "
                                   "reconnecting in %.1fs", type(e).__name__, e,
                                   backoff)
                    writer.flush()
                    await asyncio.sleep(backoff)
                    backoff = min(60.0, backoff * 2)
            writer.flush()
            self.stats.bytes_written = writer.bytes_written

        logger.info("[recorder] stopped. %s", self.stats.line())
        return self.stats

    def _on_text(self, message: str) -> None:
        """Non-binary frames: errors, order postbacks, broker messages."""
        try:
            payload = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return
        if payload.get("type") == "error":
            detail = str(payload.get("data", ""))
            logger.error("[recorder] stream error: %s", detail)
            if "token" in detail.lower() or "session" in detail.lower():
                raise KiteAuthError(
                    f"Kite closed the stream with an auth error: {detail}. "
                    f"The daily access token has expired; re-run the login "
                    f"flow and export a fresh KITE_ACCESS_TOKEN.")


def read_depth_file(path: Path) -> Iterable[Dict]:
    """Stream records back out of a recorded session file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A kill -9 can leave one truncated final line. Skip it
                # rather than losing the session.
                logger.warning("[recorder] skipping malformed line in %s", path)
