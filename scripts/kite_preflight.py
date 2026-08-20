"""
Check every link in the Kite chain before relying on it.

    export KITE_API_KEY=... KITE_ACCESS_TOKEN=...
    python scripts/kite_preflight.py

Run this the first time credentials exist, and again any morning the
recorder misbehaves. It exists because the alternative is discovering a
missing subscription or an unresolvable symbol at 09:15 IST, and losing
a session that cannot be re-recorded.

Each check reports PASS / FAIL / SKIP with a specific remedy. The one
most likely to fail on a new account is HISTORICAL: Kite bills the
historical-candle API as part of the Connect subscription, and a fresh
app sometimes has streaming working while historical calls still 403.
Streaming is what the recorder needs, so a historical failure does NOT
block recording - the script says so rather than reporting one red line
and letting you assume everything is broken.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                   "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK"]

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


class Report:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, str]] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.rows.append((name, status, detail))
        mark = {PASS: "  ok  ", FAIL: " FAIL ", SKIP: " skip ", WARN: " warn "}[status]
        print(f"[{mark}] {name}" + (f"\n         {detail}" if detail else ""))

    @property
    def failed(self) -> bool:
        return any(s == FAIL for _, s, _ in self.rows)


def check_credentials(rep: Report):
    from nightevolver.kite import KiteAuthError, load_credentials
    try:
        creds = load_credentials()
    except KiteAuthError as e:
        rep.add("credentials", FAIL, str(e))
        return None
    rep.add("credentials", PASS,
            f"api_key={creds.api_key[:4]}… token={creds.access_token[:4]}…")
    return creds


def check_profile(rep: Report, creds):
    from nightevolver.kite import API_ROOT, KiteAuthError, _get
    try:
        data = json.loads(_get(f"{API_ROOT}/user/profile", creds))
        d = data.get("data") or {}
        rep.add("auth / user profile", PASS,
                f"{d.get('user_name', '?')} ({d.get('user_id', '?')}), "
                f"products={','.join(d.get('products', []) or [])}, "
                f"exchanges={','.join(d.get('exchanges', []) or [])}")
        return d
    except KiteAuthError as e:
        rep.add("auth / user profile", FAIL,
                f"{e}\n         -> re-run scripts/kite_login.py")
    except Exception as e:
        rep.add("auth / user profile", FAIL, f"{type(e).__name__}: {e}")
    return None


def check_cnc_available(rep: Report, profile):
    """CNC is delivery. MIS auto-squares off intraday, which would close
    every position the same day - the genome holds for 2-60 days, so MIS
    would make live trading unrelated to the backtest."""
    if not profile:
        rep.add("CNC product enabled", SKIP, "no profile")
        return
    products = [p.upper() for p in (profile.get("products") or [])]
    if "CNC" in products:
        rep.add("CNC product enabled", PASS, "delivery orders available")
    else:
        rep.add("CNC product enabled", WARN,
                f"CNC not in {products}. Only relevant once orders are placed, "
                f"but MIS would square off positions the same day and the "
                f"strategy holds for 2-60 days.")


def check_instruments(rep: Report, symbols):
    from nightevolver.kite import resolve_tokens
    try:
        tokens = resolve_tokens(symbols)
        rep.add("instrument resolution", PASS,
                f"{len(tokens)}/{len(symbols)} symbols resolved")
        return tokens
    except Exception as e:
        rep.add("instrument resolution", FAIL, f"{type(e).__name__}: {e}")
        return {}


def check_historical(rep: Report, creds, tokens):
    from nightevolver.kite import KiteAuthError, fetch_candles
    if not tokens:
        rep.add("historical candles", SKIP, "no tokens")
        return
    sym, tok = sorted(tokens.items())[0]
    end = datetime.now()
    start = end - timedelta(days=5)
    try:
        rows = fetch_candles(tok, "minute", start, end, creds=creds)
        if rows:
            rep.add("historical candles", PASS,
                    f"{sym}: {len(rows)} minute bars over 5 days "
                    f"(latest {rows[-1]['date']})")
        else:
            rep.add("historical candles", WARN,
                    f"{sym}: request succeeded but returned no bars "
                    f"(a holiday stretch would do this)")
    except KiteAuthError as e:
        rep.add("historical candles", FAIL, str(e))
    except Exception as e:
        rep.add("historical candles", FAIL,
                f"{type(e).__name__}: {e}\n"
                f"         -> historical is billed with the Connect "
                f"subscription and can lag app creation. This does NOT block "
                f"the depth recorder, which needs streaming only.")


def check_stream(rep: Report, creds, tokens, seconds: float):
    import asyncio

    from nightevolver.depth_recorder import is_market_open

    if not tokens:
        rep.add("live depth stream", SKIP, "no tokens")
        return

    open_now = is_market_open()

    async def probe():
        import websockets

        from nightevolver.kite import parse_binary_message, websocket_url
        got, frames = [], 0
        async with websockets.connect(websocket_url(creds), max_size=None,
                                      ping_interval=20) as ws:
            toks = sorted(tokens.values())
            await ws.send(json.dumps({"a": "subscribe", "v": toks}))
            await ws.send(json.dumps({"a": "mode", "v": ["full", toks]}))
            deadline = asyncio.get_event_loop().time() + seconds
            while asyncio.get_event_loop().time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(msg, bytes):
                    frames += 1
                    got.extend(t for t in parse_binary_message(msg)
                               if t.mode == "full")
                else:
                    payload = json.loads(msg)
                    if payload.get("type") == "error":
                        raise RuntimeError(f"stream error: {payload.get('data')}")
        return frames, got

    try:
        frames, ticks = asyncio.run(probe())
    except Exception as e:
        rep.add("live depth stream", FAIL, f"{type(e).__name__}: {e}")
        return

    if ticks:
        t = ticks[0]
        rep.add("live depth stream", PASS,
                f"{frames} frames, {len(ticks)} full-mode ticks; sample "
                f"spread={t.spread:.2f} imbalance={t.depth_imbalance:+.3f}, "
                f"{len(t.bids)} bid / {len(t.asks)} ask levels")
    elif not open_now:
        rep.add("live depth stream", SKIP,
                "connected and subscribed, but the market is closed so there "
                "is nothing to stream. Re-run inside 09:15-15:30 IST to "
                "confirm data actually arrives.")
    else:
        rep.add("live depth stream", FAIL,
                f"market is OPEN, connected and subscribed, but received "
                f"{frames} frames and zero depth ticks in {seconds:.0f}s. "
                f"That is a real failure, not a quiet market.")


def check_disk(rep: Report, path: Path, need_gb: float = 5.0):
    try:
        path.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(path).free / 1e9
    except OSError as e:
        rep.add("disk space", FAIL, f"cannot use {path}: {e}")
        return
    detail = (f"{free_gb:.1f} GB free at {path} "
              f"(~10-20 MB/day for 10 symbols, ~1 GB/quarter)")
    rep.add("disk space", PASS if free_gb >= need_gb else WARN, detail)


def main() -> int:
    p = argparse.ArgumentParser(description="Kite preflight checks")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--stream-seconds", type=float, default=8.0)
    p.add_argument("--skip-stream", action="store_true")
    p.add_argument("--depth-dir",
                   default=str(Path(__file__).parent.parent / "data" / "depth"))
    args = p.parse_args()
    symbols: List[str] = []
    for chunk in args.symbols:
        symbols.extend(s.strip() for s in str(chunk).split(",") if s.strip())

    print("\nKite preflight\n" + "-" * 66)
    rep = Report()

    creds = check_credentials(rep)
    if creds is None:
        print("\n" + "-" * 66)
        print("Cannot continue without credentials. Run scripts/kite_login.py.")
        return 1

    profile = check_profile(rep, creds)
    check_cnc_available(rep, profile)
    tokens = check_instruments(rep, symbols)
    check_historical(rep, creds, tokens)
    if args.skip_stream:
        rep.add("live depth stream", SKIP, "--skip-stream")
    else:
        check_stream(rep, creds, tokens, args.stream_seconds)
    check_disk(rep, Path(args.depth_dir))

    print("-" * 66)
    if rep.failed:
        print("PREFLIGHT FAILED - see the FAIL lines above.")
        print("Note: a historical-candles failure alone does not block the")
        print("depth recorder, which needs streaming only.")
        return 1
    print("All checks passed. Start recording with:")
    print("  python scripts/record_depth.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
