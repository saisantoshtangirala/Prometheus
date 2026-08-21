"""
Zerodha Kite Connect: credentials, instruments, historical candles, and
the binary tick parser.

WHAT KITE CAN AND CANNOT GIVE YOU
---------------------------------
Checked against the live docs rather than recalled, because the split
determines what is worth building:

  Historical endpoint  -> candles ONLY: (Timestamp, O, H, L, C, Volume, OI)
                          at minute / 3 / 5 / 10 / 15 / 30 / 60minute / day.
  WebSocket 'full'     -> 5 levels of bid/offer depth, LIVE ONLY.

There is no historical depth endpoint at any price on this plan. Order
imbalance, signed volume and book pressure therefore **cannot be
backfilled** - they have to be recorded forward, which is why
depth_recorder.py exists and why it is worth starting immediately rather
than when it is needed.

NO CREDENTIALS IN THE REPO. Everything is read from the environment:

    KITE_API_KEY        the app's API key
    KITE_ACCESS_TOKEN   the daily token from the login flow

The access token EXPIRES DAILY (Kite invalidates it each morning), which
is the single most common reason an unattended Kite job breaks. This
module surfaces that as a specific, named exception rather than as a
generic HTTP error, so a nightly cron can report "the token expired"
instead of "something went wrong".

Deliberately NOT using the `kiteconnect` SDK: it is not installed, and
this codebase already carries a measured preference for direct HTTP over
system dependencies that can fail on an unattended box. The pieces used
here are a handful of GETs and one binary layout.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from .nethttp import TRANSIENT_NET_ERRORS

logger = logging.getLogger("nightevolver.kite")

API_ROOT = "https://api.kite.trade"
WS_ROOT = "wss://ws.kite.trade"
INSTRUMENTS_URL = f"{API_ROOT}/instruments"

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "kite"

# Kite documents a rate limit around 3 requests/second on historical.
HISTORICAL_MIN_INTERVAL = 0.34

# Prices arrive as integers. The divisor is segment-dependent; NSE cash
# equity (NSE_CM) is paise, so 100. Currency segments use different
# divisors - if this module is ever pointed at those, this must change.
PRICE_DIVISOR_NSE_CM = 100.0

# Packet sizes, from the docs.
PACKET_LTP = 8
PACKET_QUOTE = 44
PACKET_FULL = 184
PACKET_INDEX_QUOTE = 28
PACKET_INDEX_FULL = 32

DEPTH_LEVELS = 5


class KiteAuthError(RuntimeError):
    """Credentials missing, or the daily access token has expired.

    Separated from other failures on purpose: this one has a specific
    and boring remedy (log in again and re-export KITE_ACCESS_TOKEN),
    and an unattended job should say so rather than retrying forever.
    """


@dataclass(frozen=True)
class Credentials:
    api_key: str
    access_token: str

    @property
    def auth_header(self) -> str:
        return f"token {self.api_key}:{self.access_token}"


def load_credentials(api_key: Optional[str] = None,
                     access_token: Optional[str] = None) -> Credentials:
    """Read credentials from arguments or the environment.

    Never reads from a file in the repo and never logs the token.
    """
    api_key = api_key or os.environ.get("KITE_API_KEY", "")
    access_token = access_token or os.environ.get("KITE_ACCESS_TOKEN", "")
    missing = [n for n, v in (("KITE_API_KEY", api_key),
                              ("KITE_ACCESS_TOKEN", access_token)) if not v]
    if missing:
        raise KiteAuthError(
            f"missing {', '.join(missing)}. Kite access tokens are issued by "
            f"the interactive login flow and expire daily, so they belong in "
            f"the environment, not in the repo. Export them before running."
        )
    return Credentials(api_key=api_key, access_token=access_token)


# --------------------------------------------------------------------------
# Binary tick parsing. Pure functions - no network, fully unit-testable.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DepthEntry:
    quantity: int
    price: float
    orders: int


@dataclass(frozen=True)
class Tick:
    instrument_token: int
    mode: str                       # ltp | quote | full | index
    last_price: float
    last_quantity: int = 0
    average_price: float = 0.0
    volume: int = 0
    total_buy_quantity: int = 0
    total_sell_quantity: int = 0
    ohlc_open: float = 0.0
    ohlc_high: float = 0.0
    ohlc_low: float = 0.0
    ohlc_close: float = 0.0
    exchange_timestamp: Optional[int] = None
    bids: Tuple[DepthEntry, ...] = ()
    asks: Tuple[DepthEntry, ...] = ()

    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return float("nan")
        return self.asks[0].price - self.bids[0].price

    @property
    def depth_imbalance(self) -> float:
        """(bid_qty - ask_qty) / (bid_qty + ask_qty) over all 5 levels.

        This is the headline order-flow quantity and the reason the
        recorder exists. In [-1, 1]; positive means more resting size on
        the bid.
        """
        b = sum(e.quantity for e in self.bids)
        a = sum(e.quantity for e in self.asks)
        return (b - a) / (b + a) if (b + a) > 0 else float("nan")


def _i32(buf: bytes, off: int) -> int:
    return struct.unpack(">i", buf[off:off + 4])[0]


def _i16(buf: bytes, off: int) -> int:
    return struct.unpack(">h", buf[off:off + 2])[0]


def parse_depth(buf: bytes, divisor: float = PRICE_DIVISOR_NSE_CM
                ) -> Tuple[Tuple[DepthEntry, ...], Tuple[DepthEntry, ...]]:
    """Bytes [64, 184) of a `full` packet -> (bids, asks).

    Layout, verbatim from the docs: each entry is quantity(int32),
    price(int32), orders(int16) plus 2 bytes of padding to skip = 12
    bytes. Ten entries in succession, five bid [64-124] then five offer
    [124-184].

    The padding is the easy thing to get wrong: reading 10 bytes per
    entry instead of 12 would still "work" and silently shift every
    field after the first level.
    """
    if len(buf) < PACKET_FULL:
        raise ValueError(f"depth needs a {PACKET_FULL}-byte packet, got {len(buf)}")
    entries: List[DepthEntry] = []
    for i in range(2 * DEPTH_LEVELS):
        off = 64 + i * 12
        entries.append(DepthEntry(quantity=_i32(buf, off),
                                  price=_i32(buf, off + 4) / divisor,
                                  orders=_i16(buf, off + 8)))
        # bytes [off+10, off+12) are padding and are deliberately skipped
    return tuple(entries[:DEPTH_LEVELS]), tuple(entries[DEPTH_LEVELS:])


def parse_packet(buf: bytes, divisor: float = PRICE_DIVISOR_NSE_CM) -> Optional[Tick]:
    """One quote packet -> Tick. Returns None for an unrecognised length."""
    n = len(buf)
    if n < PACKET_LTP:
        return None
    token = _i32(buf, 0)

    if n == PACKET_LTP:
        return Tick(token, "ltp", _i32(buf, 4) / divisor)

    if n in (PACKET_INDEX_QUOTE, PACKET_INDEX_FULL):
        # Index packets have their own, shorter layout.
        return Tick(token, "index", _i32(buf, 4) / divisor,
                    ohlc_high=_i32(buf, 8) / divisor,
                    ohlc_low=_i32(buf, 12) / divisor,
                    ohlc_open=_i32(buf, 16) / divisor,
                    ohlc_close=_i32(buf, 20) / divisor,
                    exchange_timestamp=_i32(buf, 28) if n == PACKET_INDEX_FULL else None)

    if n < PACKET_QUOTE:
        return None

    common = dict(
        instrument_token=token,
        last_price=_i32(buf, 4) / divisor,
        last_quantity=_i32(buf, 8),
        average_price=_i32(buf, 12) / divisor,
        volume=_i32(buf, 16),
        total_buy_quantity=_i32(buf, 20),
        total_sell_quantity=_i32(buf, 24),
        ohlc_open=_i32(buf, 28) / divisor,
        ohlc_high=_i32(buf, 32) / divisor,
        ohlc_low=_i32(buf, 36) / divisor,
        ohlc_close=_i32(buf, 40) / divisor,
    )
    if n < PACKET_FULL:
        return Tick(mode="quote", **common)

    bids, asks = parse_depth(buf, divisor)
    return Tick(mode="full", exchange_timestamp=_i32(buf, 60),
                bids=bids, asks=asks, **common)


def parse_binary_message(data: bytes,
                         divisor: float = PRICE_DIVISOR_NSE_CM) -> List[Tick]:
    """A whole WebSocket binary frame -> list of Ticks.

    Frame layout: int16 packet count, then for each packet an int16
    length followed by that many bytes.

    A 1-byte frame is the heartbeat Kite sends when there is nothing to
    stream; it is not an error and yields no ticks.
    """
    if len(data) < 2:
        return []                      # heartbeat
    count = _i16(data, 0)
    if count <= 0:
        return []
    out: List[Tick] = []
    off = 2
    for _ in range(count):
        if off + 2 > len(data):
            logger.warning("[kite] truncated frame: header claims %d packets", count)
            break
        length = _i16(data, off)
        off += 2
        if length <= 0 or off + length > len(data):
            logger.warning("[kite] truncated packet (len=%d) - stopping frame", length)
            break
        tick = parse_packet(data[off:off + length], divisor)
        if tick is not None:
            out.append(tick)
        off += length
    return out


# --------------------------------------------------------------------------
# HTTP: instruments and historical candles
# --------------------------------------------------------------------------

def _get(url: str, creds: Optional[Credentials] = None,
         timeout: int = 30, max_attempts: int = 5) -> bytes:
    headers = {"X-Kite-Version": "3", "User-Agent": "nightevolver/1.0"}
    if creds is not None:
        headers["Authorization"] = creds.auth_header
    last = None
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as f:
                return f.read()
        except urllib.error.HTTPError as e:
            body = b""
            try:
                body = e.read()[:400]
            except Exception:
                pass
            if e.code in (401, 403):
                raise KiteAuthError(
                    f"Kite returned HTTP {e.code}. The daily access token has "
                    f"almost certainly expired - Kite invalidates it each "
                    f"morning. Re-run the login flow and export a fresh "
                    f"KITE_ACCESS_TOKEN. Response: {body!r}") from e
            last = e
            if e.code == 429:                     # too many requests
                time.sleep(1.0 + attempt)
                continue
            if 400 <= e.code < 500:
                raise RuntimeError(f"Kite HTTP {e.code}: {body!r}") from e
        except TRANSIENT_NET_ERRORS as e:
            last = e
        time.sleep(min(8.0, 0.5 * (2 ** attempt)))
    raise RuntimeError(f"Kite request failed after {max_attempts} attempts: {last}")


def fetch_instruments(exchange: str = "NSE", use_cache: bool = True,
                      max_age_hours: float = 24.0) -> List[Dict[str, str]]:
    """The instrument dump. Public - needs no credentials.

    Cached for a day: it is ~9 MB and changes at most daily.
    """
    cp = CACHE_DIR / f"instruments_{exchange}.csv"
    if use_cache and cp.exists():
        age_h = (time.time() - cp.stat().st_mtime) / 3600.0
        if age_h < max_age_hours:
            return list(csv.DictReader(io.StringIO(cp.read_text(encoding="utf-8"))))

    raw = _get(f"{INSTRUMENTS_URL}/{urllib.parse.quote(exchange)}")
    text = raw.decode("utf-8", "replace")
    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cp.write_text(text, encoding="utf-8")
        except OSError:
            pass
    return list(csv.DictReader(io.StringIO(text)))


def resolve_tokens(symbols: Sequence[str], exchange: str = "NSE",
                   use_cache: bool = True) -> Dict[str, int]:
    """NSE trading symbols -> instrument_token, equity series only.

    Raises on any symbol that cannot be resolved. A silently dropped
    symbol would mean recording nine names while believing you recorded
    ten, and the gap would only surface months later in the data.
    """
    wanted = {s[:-3] if s.upper().endswith(".NS") else s.upper() for s in symbols}
    out: Dict[str, int] = {}
    for row in fetch_instruments(exchange, use_cache=use_cache):
        sym = (row.get("tradingsymbol") or "").strip().upper()
        if sym in wanted and (row.get("instrument_type") or "").strip() == "EQ":
            try:
                out[sym] = int(row["instrument_token"])
            except (KeyError, ValueError):
                continue
    missing = wanted - set(out)
    if missing:
        raise RuntimeError(
            f"could not resolve instrument tokens for {sorted(missing)} on "
            f"{exchange}. Recording a subset while believing otherwise would "
            f"leave a gap that only shows up months later.")
    return out


@dataclass
class RateLimiter:
    min_interval: float = HISTORICAL_MIN_INTERVAL
    _last: float = field(default=0.0, repr=False)

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()


# Kite caps how much history one historical request may span, and the
# cap is tighter for finer intervals. These are conservative chunk sizes
# used to split a long range into several requests.
_CHUNK_DAYS = {
    "minute": 55, "3minute": 85, "5minute": 85, "10minute": 85,
    "15minute": 180, "30minute": 180, "60minute": 365, "day": 1800,
}


def fetch_candles(token: int, interval: str, start: datetime, end: datetime,
                  creds: Optional[Credentials] = None,
                  limiter: Optional[RateLimiter] = None,
                  oi: bool = False) -> List[Dict]:
    """Historical candles for one instrument over an arbitrary range.

    Splits the range into chunks the API will accept and concatenates.
    Returns dicts with keys date/open/high/low/close/volume (+oi).
    """
    if interval not in _CHUNK_DAYS:
        raise ValueError(f"unknown interval {interval!r}; "
                         f"expected one of {sorted(_CHUNK_DAYS)}")
    creds = creds or load_credentials()
    limiter = limiter or RateLimiter()

    step = timedelta(days=_CHUNK_DAYS[interval])
    rows: List[Dict] = []
    seen: set = set()
    cur = start
    while cur < end:
        chunk_end = min(cur + step, end)
        q = urllib.parse.urlencode({
            "from": cur.strftime("%Y-%m-%d %H:%M:%S"),
            "to": chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
            "oi": 1 if oi else 0,
        })
        url = (f"{API_ROOT}/instruments/historical/{token}/"
               f"{urllib.parse.quote(interval)}?{q}")
        limiter.wait()
        payload = _get(url, creds)
        import json
        data = json.loads(payload)
        candles = (data.get("data") or {}).get("candles") or []
        for c in candles:
            # [timestamp, o, h, l, c, volume, (oi)]
            ts = c[0]
            if ts in seen:
                continue          # chunk boundaries overlap by one bar
            seen.add(ts)
            row = {"date": ts, "open": c[1], "high": c[2], "low": c[3],
                   "close": c[4], "volume": c[5]}
            if oi and len(c) > 6:
                row["oi"] = c[6]
            rows.append(row)
        cur = chunk_end
    rows.sort(key=lambda r: r["date"])
    logger.info("[kite] token=%s %s: %d candles %s..%s", token, interval,
                len(rows), start.date(), end.date())
    return rows


def websocket_url(creds: Optional[Credentials] = None) -> str:
    creds = creds or load_credentials()
    return (f"{WS_ROOT}?api_key={urllib.parse.quote(creds.api_key)}"
            f"&access_token={urllib.parse.quote(creds.access_token)}")
