"""
Does intraday structure carry information that daily OHLCV does not?

    export KITE_API_KEY=... KITE_ACCESS_TOKEN=...
    python scripts/run_intraday_audit.py --start 2026-01-01

    # no credentials needed - exercises the whole pipeline on synthetic
    # minute bars and checks the audit finds nothing in them
    python scripts/run_intraday_audit.py --synthetic

This is the cheap decisive test described in nightevolver/intraday.py:
day-shape features from minute candles, scored against the same four
targets with the same block-permutation null and the same BH-FDR
correction as the daily audit, so the two are directly comparable.

The daily audit's result is the baseline to beat: 7 of 104 pairs
survived, all on volatility and regime targets, none on direction. The
question here is whether intraday adds anything to that - especially on
direction, where nothing has ever survived.

ORDER FLOW is included automatically once `data/depth/` contains
recorded sessions. Until then those channels are absent and the script
says so rather than reporting a null result on them, because "we have
not recorded it yet" and "it carries no information" are very different
statements.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from nightevolver.information_audit import (
    DEFAULT_BLOCK_BARS, DEFAULT_FDR_ALPHA, DEFAULT_N_PERMUTATIONS, audit_features,
)
from nightevolver.intraday import (
    INTRADAY_FEATURE_NAMES, ORDERFLOW_FEATURE_NAMES, load_orderflow_features,
    minutes_to_daily_features, normalise_features,
)
from nightevolver.targets import build_targets

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("intraday.audit")

DEFAULT_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                   "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK"]

DEPTH_DIR = Path(__file__).parent.parent / "data" / "depth"


def parse_args():
    p = argparse.ArgumentParser(description="Intraday information audit")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--start", default="2026-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="minute",
                   choices=["minute", "3minute", "5minute", "10minute",
                            "15minute", "30minute", "60minute"])
    p.add_argument("--permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    p.add_argument("--block-bars", type=int, default=DEFAULT_BLOCK_BARS)
    p.add_argument("--fdr-alpha", type=float, default=DEFAULT_FDR_ALPHA)
    p.add_argument("--depth-dir", default=str(DEPTH_DIR))
    p.add_argument("--no-orderflow", action="store_true")
    p.add_argument("--synthetic", action="store_true",
                   help="synthetic minute bars, no credentials. Calibration: "
                        "expect ZERO survivors.")
    p.add_argument("--out", default="checkpoints/nightevolver/intraday_audit.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    flat: List[str] = []
    for chunk in args.symbols:
        flat.extend(s.strip() for s in str(chunk).split(",") if s.strip())
    args.symbols = flat or DEFAULT_SYMBOLS
    return args


def synthetic_minutes(symbols, n_days: int = 180, seed: int = 0
                      ) -> Dict[str, pd.DataFrame]:
    """Random-walk minute bars with a realistic session shape.

    Deliberately structureless: the calibration run must find nothing.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, pd.DataFrame] = {}
    days = pd.bdate_range("2026-01-01", periods=n_days)
    for sym in symbols:
        frames = []
        level = 100.0
        for d in days:
            idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=15),
                                d + pd.Timedelta(hours=15, minutes=29), freq="1min")
            r = rng.normal(0, 0.0008, size=len(idx))
            c = level * np.cumprod(1 + r)
            level = float(c[-1])
            o = np.concatenate([[c[0]], c[:-1]])
            spread = np.abs(rng.normal(0, 0.0004, size=len(idx))) * c
            frames.append(pd.DataFrame({
                "open": o, "high": np.maximum(o, c) + spread,
                "low": np.minimum(o, c) - spread, "close": c,
                "volume": rng.lognormal(8, 0.6, size=len(idx)),
            }, index=idx))
        out[sym] = pd.concat(frames)
    return out


def fetch_minutes(symbols, start: str, end, interval: str) -> Dict[str, pd.DataFrame]:
    from nightevolver.kite import (
        RateLimiter, fetch_candles, load_credentials, resolve_tokens,
    )
    creds = load_credentials()
    tokens = resolve_tokens(symbols)
    limiter = RateLimiter()
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end) if end else datetime.now()

    out: Dict[str, pd.DataFrame] = {}
    for sym, tok in sorted(tokens.items()):
        rows = fetch_candles(tok, interval, start_dt, end_dt,
                             creds=creds, limiter=limiter)
        if not rows:
            logger.warning("no candles for %s", sym)
            continue
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed"
                                    ).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        out[sym] = df.set_index("date")[["open", "high", "low", "close", "volume"]]
    return out


def main() -> int:
    args = parse_args()

    if args.synthetic:
        logger.info("SYNTHETIC minute bars - calibration run. Expect ZERO "
                    "survivors; any survivor is a false positive.")
        minutes = synthetic_minutes(args.symbols, seed=args.seed)
    else:
        minutes = fetch_minutes(args.symbols, args.start, args.end, args.interval)
    if not minutes:
        logger.error("no minute data - nothing to audit")
        return 1

    symbols = sorted(minutes)
    per_symbol = {s: minutes_to_daily_features(minutes[s]) for s in symbols}

    # Daily close matrix, from the minute bars themselves, so features
    # and targets come from one source and cannot disagree about what a
    # trading day was.
    closes = pd.DataFrame({
        s: minutes[s]["close"].groupby(
            pd.DatetimeIndex(minutes[s].index).normalize()).last()
        for s in symbols}).dropna(how="any")
    dates = pd.DatetimeIndex(closes.index)
    logger.info("data: %d days x %d symbols (%s .. %s)", len(dates), len(symbols),
                dates[0].date(), dates[-1].date())

    names = list(INTRADAY_FEATURE_NAMES)
    feats = normalise_features(per_symbol, symbols, dates, INTRADAY_FEATURE_NAMES)

    if not args.no_orderflow and not args.synthetic:
        from nightevolver.kite import resolve_tokens
        tok = resolve_tokens(symbols)
        of = load_orderflow_features(Path(args.depth_dir),
                                     {v: k for k, v in tok.items()}, symbols)
        if of:
            of_arr = normalise_features(of, symbols, dates, ORDERFLOW_FEATURE_NAMES)
            feats = np.concatenate([feats, of_arr], axis=2)
            names += list(ORDERFLOW_FEATURE_NAMES)
            logger.info("order-flow channels included from %d symbols", len(of))
        else:
            logger.warning(
                "NO recorded depth in %s, so order-flow channels are ABSENT "
                "from this audit. That is 'not yet recorded', NOT 'carries no "
                "information' - depth cannot be backfilled from Kite, so this "
                "stays true until scripts/record_depth.py has run for a while.",
                args.depth_dir)

    close_arr = closes.to_numpy(dtype=float)
    targets = build_targets(close_arr)
    result = audit_features(feats, names, targets, close_arr,
                            block_bars=args.block_bars,
                            n_permutations=args.permutations,
                            fdr_alpha=args.fdr_alpha, seed=args.seed)
    print("\n" + result.summary())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "n_days": len(dates), "n_symbols": len(symbols),
            "symbols": symbols, "interval": args.interval,
            "synthetic": args.synthetic,
            "orderflow_included": any(n in names for n in ORDERFLOW_FEATURE_NAMES),
            "n_pairs": len(result.pairs), "n_survivors": len(result.survivors),
            "pairs": [{"feature": p.feature, "target": p.target,
                       "spearman": p.spearman,
                       "spearman_incremental": p.spearman_incremental,
                       "p_value_incremental": p.p_value_incremental,
                       "q_value": p.q_value, "significant": p.significant}
                      for p in result.pairs],
        }, f, indent=2)
    logger.info("written: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
