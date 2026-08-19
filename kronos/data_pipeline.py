"""
Kronos Data Digestion - phase 1 of the daily cycle (midnight - 02:00).

Fetches the last 24h (plus rolling lookback) of market data from multiple
sources in parallel, cross-validates them against each other, repairs gaps
with the existing Kalman filter, scores sentiment with the existing
LegalBERT analyzer, and emits a single immutable DailyMemory object that
every downstream phase consumes.

Fallback chain (config: data.sources): yfinance -> polygon -> alphavantage.
A source is skipped silently if its API key is missing; the pipeline only
fails if ALL sources fail.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from prometheus.data.data_validator import (
    KalmanFilter1D,
    detect_flash_crash,
    detect_illiquid_periods,
    floor_nanoseconds_to_microseconds,
    standardize_timezone,
)
from prometheus.data.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities shared by the pipeline and the reflex arc
# ---------------------------------------------------------------------------

MIN_SPREAD_PCT = 0.001   # 0.1% minimum spread when clamping corrupt quotes


def clamp_spread(bid: float, ask: float) -> tuple:
    """
    Repair a corrupt (negative) bid-ask spread: if ask < bid, clamp ask to
    bid * (1 + MIN_SPREAD_PCT). Returns (bid, ask, was_corrupt).
    Prevents the paper trader from "arbitraging" a data error.
    """
    if ask < bid:
        return bid, bid * (1.0 + MIN_SPREAD_PCT), True
    return bid, ask, False


class Throttle:
    """Minimum-interval rate limiter for API calls (DAT-07)."""

    def __init__(self, min_interval_seconds: float = 0.5):
        self.min_interval = float(min_interval_seconds)
        self._last_call: Optional[float] = None

    def wait(self) -> float:
        """Block until the interval has elapsed. Returns seconds slept."""
        import time as _time
        now = _time.monotonic()
        slept = 0.0
        if self._last_call is not None:
            remaining = self.min_interval - (now - self._last_call)
            if remaining > 0:
                _time.sleep(remaining)
                slept = remaining
        self._last_call = _time.monotonic()
        return slept


# ---------------------------------------------------------------------------
# DailyMemory - the structured output of digestion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DailyMemory:
    """Immutable snapshot of everything Kronos knows about today's market."""

    as_of: datetime
    prices: pd.DataFrame            # [date x ticker] close prices
    volumes: pd.DataFrame           # [date x ticker] volumes
    returns: pd.DataFrame           # [date x ticker] simple returns
    vix: pd.Series                  # VIX close series
    sentiment: Dict[str, float]     # ticker -> sentiment score [-1, 1]
    macro: Dict[str, float]         # macro indicators
    source_used: str                # which data source won
    quality_flags: List[str] = field(default_factory=list)

    @property
    def tickers(self) -> List[str]:
        return list(self.prices.columns)

    @property
    def latest_returns(self) -> np.ndarray:
        return self.returns.iloc[-1].values.astype(np.float32)

    def returns_window(self, days: int) -> np.ndarray:
        """Last N days of returns as [days, n_tickers] float32."""
        return self.returns.iloc[-days:].fillna(0.0).values.astype(np.float32)

    def volumes_window(self, days: int) -> np.ndarray:
        """Last N days of volumes as [days, n_tickers] float32, column-
        aligned to self.returns's ticker order (not necessarily
        self.volumes' own) so it lines up with returns_window() for
        kronos/features.py.build_features() without a silent ticker-order
        mismatch. Missing/unresolvable columns become 0.0, matching
        returns_window()'s own NaN handling."""
        aligned = self.volumes.reindex(columns=self.returns.columns)
        return aligned.iloc[-days:].fillna(0.0).values.astype(np.float32)


# ---------------------------------------------------------------------------
# Source adapters - uniform interface: fetch(tickers, lookback) -> DataFrame
# ---------------------------------------------------------------------------

class SourceError(Exception):
    """A data source failed or returned unusable data."""


class DataUnavailableError(SourceError):
    """Every source AND the local cache failed - skip trading, never guess."""


class YFinanceSource:
    name = "yfinance"

    def fetch(self, tickers: List[str], lookback_days: int) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:
            raise SourceError("yfinance not installed") from e
        data = yf.download(
            tickers, period=f"{lookback_days}d", interval="1d",
            progress=False, auto_adjust=True, group_by="column",
        )
        if data is None or data.empty:
            raise SourceError("yfinance returned empty frame")
        return data


class PolygonSource:
    name = "polygon"

    def fetch(self, tickers: List[str], lookback_days: int) -> pd.DataFrame:
        api_key = os.environ.get("POLYGON_API_KEY")
        if not api_key:
            raise SourceError("POLYGON_API_KEY not set")
        try:
            import requests
        except ImportError as e:
            raise SourceError("requests not installed") from e

        frames: Dict[str, pd.DataFrame] = {}
        end = datetime.now(timezone.utc).date()
        start = end - pd.Timedelta(days=lookback_days)
        for ticker in tickers:
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
                f"{start}/{end}?adjusted=true&apiKey={api_key}"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                raise SourceError(f"polygon HTTP {resp.status_code} for {ticker}")
            rows = resp.json().get("results") or []
            if not rows:
                raise SourceError(f"polygon empty for {ticker}")
            df = pd.DataFrame(rows)
            df.index = pd.to_datetime(df["t"], unit="ms")
            frames[ticker] = df.rename(
                columns={"c": "Close", "v": "Volume", "o": "Open",
                         "h": "High", "l": "Low"}
            )[["Open", "High", "Low", "Close", "Volume"]]
        merged = pd.concat(frames, axis=1)
        merged.columns = merged.columns.swaplevel(0, 1)
        return merged.sort_index(axis=1, level=0)


class AlphaVantageSource:
    name = "alphavantage"

    def fetch(self, tickers: List[str], lookback_days: int) -> pd.DataFrame:
        api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
        if not api_key:
            raise SourceError("ALPHAVANTAGE_API_KEY not set")
        try:
            import requests
        except ImportError as e:
            raise SourceError("requests not installed") from e

        frames: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            url = (
                "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
                f"&symbol={ticker}&apikey={api_key}&outputsize=compact"
            )
            resp = requests.get(url, timeout=15)
            series = resp.json().get("Time Series (Daily)") or {}
            if not series:
                raise SourceError(f"alphavantage empty for {ticker}")
            df = pd.DataFrame(series).T.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index().tail(lookback_days)
            frames[ticker] = df.rename(columns={
                "1. open": "Open", "2. high": "High", "3. low": "Low",
                "5. adjusted close": "Close", "6. volume": "Volume",
            })[["Open", "High", "Low", "Close", "Volume"]]
        merged = pd.concat(frames, axis=1)
        merged.columns = merged.columns.swaplevel(0, 1)
        return merged.sort_index(axis=1, level=0)


SOURCE_REGISTRY = {
    "yfinance": YFinanceSource,
    "polygon": PolygonSource,
    "alphavantage": AlphaVantageSource,
}


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """
    Parallel multi-source fetch with cross-validation and Kalman repair.

    Absolute data integrity rules (non-negotiable #3):
      - corrupted timestamps are floored/standardized
      - missing values repaired via Kalman filter, never forward-guessed
      - a day is flagged if sources disagree beyond tolerance
    """

    def __init__(self, config):
        self.cfg = config
        self.kalman = KalmanFilter1D(
            process_noise=config.data.kalman_process_noise,
            obs_noise=config.data.kalman_obs_noise,
        )
        self.sentiment_analyzer = SentimentAnalyzer()
        self._sources = [
            SOURCE_REGISTRY[name]()
            for name in config.data.sources
            if name in SOURCE_REGISTRY
        ]

    # -- fetching -----------------------------------------------------------

    async def _fetch_one(self, source, tickers, lookback) -> Optional[pd.DataFrame]:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, source.fetch, tickers, lookback
            )
        except SourceError as e:
            logger.warning("[digestion] source %s failed: %s", source.name, e)
            return None
        except Exception as e:
            logger.warning("[digestion] source %s crashed: %s", source.name, e)
            return None

    async def fetch_parallel(self) -> Dict[str, pd.DataFrame]:
        """Fetch from all configured sources concurrently."""
        tickers = list(self.cfg.data.tickers) + [self.cfg.data.vix_ticker]
        lookback = self.cfg.data.lookback_days
        results = await asyncio.gather(*[
            self._fetch_one(s, tickers, lookback) for s in self._sources
        ])
        return {
            s.name: frame
            for s, frame in zip(self._sources, results)
            if frame is not None
        }

    # -- cross-validation ---------------------------------------------------

    def cross_validate(
        self, frames: Dict[str, pd.DataFrame]
    ) -> tuple:
        """
        Pick the primary (first successful in priority order) frame and flag
        any close-price disagreement with secondary sources.
        Returns (primary_name, primary_frame, quality_flags).
        """
        if not frames:
            raise SourceError("ALL data sources failed - cannot build DailyMemory")

        flags: List[str] = []
        priority = [s.name for s in self._sources]
        primary_name = next(n for n in priority if n in frames)
        primary = frames[primary_name]

        tolerance = self.cfg.data.cross_validation_tolerance_pct / 100.0
        primary_close = primary["Close"] if "Close" in primary else primary

        for other_name, other in frames.items():
            if other_name == primary_name:
                continue
            try:
                other_close = other["Close"] if "Close" in other else other
                common_cols = primary_close.columns.intersection(other_close.columns)
                common_idx = primary_close.index.intersection(other_close.index)
                if len(common_cols) == 0 or len(common_idx) == 0:
                    continue
                a = primary_close.loc[common_idx, common_cols]
                b = other_close.loc[common_idx, common_cols]
                rel_diff = ((a - b).abs() / (a.abs() + 1e-9)).max().max()
                if rel_diff > tolerance:
                    flags.append(
                        f"cross_validation:{primary_name}_vs_{other_name}"
                        f":max_diff={rel_diff:.4%}"
                    )
            except Exception as e:
                flags.append(f"cross_validation_error:{other_name}:{e}")

        return primary_name, primary, flags

    # -- repair & assembly --------------------------------------------------

    def _clean(self, frame: pd.DataFrame) -> tuple:
        """Timestamp hygiene + Kalman gap repair. Returns (prices, volumes, flags)."""
        flags: List[str] = []
        # Indices (e.g. ^VIX) are not traded directly and legitimately report
        # zero volume - that's a category difference from stocks, not a
        # liquidity problem. Without this exemption the illiquid-volume check
        # below drops VIX every time, which then silently forces the
        # downstream vix_missing:synthetic_fallback path (a hardcoded 20.0)
        # instead of real volatility - the exact input the reflex arc's
        # panic gate depends on.
        volume_exempt = {self.cfg.data.vix_ticker}
        try:
            frame = standardize_timezone(frame)
        except Exception:
            flags.append("timezone_standardization_failed")
        try:
            # DAT-05: nanosecond-precision timestamps floored to microseconds
            frame = floor_nanoseconds_to_microseconds(frame)
        except ValueError:
            pass  # non-datetime index: nothing to floor

        closes = frame["Close"].copy() if "Close" in frame else frame.copy()
        volumes = (
            frame["Volume"].copy() if "Volume" in frame
            else pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        )

        max_missing = self.cfg.data.max_missing_pct / 100.0
        for col in list(closes.columns):
            series = closes[col]
            missing_frac = series.isna().mean()
            if missing_frac > max_missing:
                flags.append(f"dropped:{col}:missing={missing_frac:.0%}")
                closes = closes.drop(columns=[col])
                volumes = volumes.drop(columns=[col], errors="ignore")
                continue
            if missing_frac > 0:
                closes[col] = self.kalman.fill(series)
                flags.append(f"kalman_repaired:{col}")

            # DAT-04: zero-volume (illiquid) assets are dropped for the day so
            # they cannot distort systemic-risk attention weights. Index
            # tickers are exempt - see volume_exempt above.
            if col in volumes.columns and col not in volume_exempt:
                vol_series = volumes[col].fillna(0.0)
                if (vol_series == 0).all() and len(vol_series) > 0:
                    flags.append(f"illiquid:{col}:dropped")
                    closes = closes.drop(columns=[col], errors="ignore")
                    volumes = volumes.drop(columns=[col], errors="ignore")
                    continue
                _, is_illiquid = detect_illiquid_periods(vol_series)
                if is_illiquid:
                    flags.append(f"illiquid_periods:{col}")

        return closes, volumes, flags

    def build_memory(
        self,
        frames: Dict[str, pd.DataFrame],
        filings: Optional[Dict[str, str]] = None,
    ) -> DailyMemory:
        """Assemble the DailyMemory from validated frames."""
        source_name, primary, flags = self.cross_validate(frames)
        closes, volumes, clean_flags = self._clean(primary)
        flags.extend(clean_flags)

        vix_ticker = self.cfg.data.vix_ticker
        if vix_ticker in closes.columns:
            vix = closes[vix_ticker].rename("VIX")
            closes = closes.drop(columns=[vix_ticker])
            volumes = volumes.drop(columns=[vix_ticker], errors="ignore")
        else:
            flags.append("vix_missing:synthetic_fallback")
            vix = pd.Series(20.0, index=closes.index, name="VIX")

        returns = closes.pct_change().fillna(0.0)

        # Flash-crash flag on the most recent bar
        for col in returns.columns:
            try:
                if detect_flash_crash(returns[col].iloc[-5:]):
                    flags.append(f"flash_crash:{col}")
            except Exception:
                pass

        sentiment = self._score_sentiment(filings or {}, list(closes.columns))

        macro = {
            "vix_last": float(vix.iloc[-1]),
            "vix_mean_20d": float(vix.tail(20).mean()),
            "market_return_1d": float(returns.iloc[-1].mean()),
            "market_vol_20d": float(returns.tail(20).std().mean()),
        }

        return DailyMemory(
            as_of=datetime.now(timezone.utc),
            prices=closes,
            volumes=volumes,
            returns=returns,
            vix=vix,
            sentiment=sentiment,
            macro=macro,
            source_used=source_name,
            quality_flags=flags,
        )

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove tags/entities so LegalBERT sees prose, not markup."""
        import re
        text = re.sub(r"<script.*?</script>", " ", text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?</style>", " ", text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _score_sentiment(
        self, filings: Dict[str, str], tickers: List[str]
    ) -> Dict[str, float]:
        """
        Score SEC filings via the existing LegalBERT analyzer; 0.0 if none.
        DAT-06: a filing that is pure HTML (no recognizable text) degrades
        gracefully to score 0.0 / confidence 0.1 instead of crashing.
        """
        scores: Dict[str, float] = {t: 0.0 for t in tickers}
        for ticker, text in filings.items():
            try:
                clean = self._strip_html(text or "")
                if len(clean) < 10:
                    logger.warning(
                        "[digestion] filing for %s has no usable text - "
                        "neutral sentiment (score=0.0, confidence=0.1)", ticker,
                    )
                    scores[ticker] = 0.0
                    continue
                result = self.sentiment_analyzer.analyze_sec_filing(clean, ticker)
                scores[ticker] = float(result.get("score", 0.0))
            except Exception as e:
                logger.warning("[digestion] sentiment failed for %s: %s", ticker, e)
        return scores

    # -- local cache (last line of defence, DAT-01 / E2E-03) ---------------

    @property
    def cache_path(self) -> str:
        return self.cfg.data.get("cache_path", "logs/data_cache.pkl")

    def save_cache(self, memory: DailyMemory) -> None:
        """Persist the latest good memory so an outage can serve stale data."""
        import pickle
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump({
                "prices": memory.prices, "volumes": memory.volumes,
                "returns": memory.returns, "vix": memory.vix,
                "sentiment": memory.sentiment, "macro": memory.macro,
                "as_of": memory.as_of,
            }, f)

    def load_cache(self) -> Optional[DailyMemory]:
        import pickle
        if not os.path.exists(self.cache_path):
            return None
        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            return DailyMemory(
                as_of=data["as_of"],
                prices=data["prices"], volumes=data["volumes"],
                returns=data["returns"], vix=data["vix"],
                sentiment=data["sentiment"], macro=data["macro"],
                source_used="cache",
                quality_flags=[f"stale_data:cached_at_{data['as_of'].isoformat()}"],
            )
        except Exception as e:
            logger.error("[digestion] cache load failed: %s", e)
            return None

    # -- entry point --------------------------------------------------------

    async def run(self, filings: Optional[Dict[str, str]] = None) -> DailyMemory:
        """
        Full digestion: parallel fetch -> cross-validate -> clean -> memory.

        Fallback ladder: live sources (priority order) -> local cache
        (flagged stale) -> DataUnavailableError. Stale data is served for
        situational awareness only; the orchestrator must not trade on it.
        """
        frames = await self.fetch_parallel()
        try:
            memory = self.build_memory(frames, filings)
        except SourceError:
            cached = self.load_cache()
            if cached is not None:
                logger.error(
                    "[digestion] ALL live sources failed - serving STALE cache "
                    "from %s. Trading must be skipped.", cached.as_of,
                )
                return cached
            raise DataUnavailableError(
                "All data sources failed and no local cache exists. "
                "Skipping the trading day - stale guesses are worse than no trades."
            )
        self.save_cache(memory)
        logger.info(
            "[digestion] DailyMemory built: source=%s, tickers=%d, flags=%s",
            memory.source_used, len(memory.tickers), memory.quality_flags,
        )
        return memory

    def run_sync(self, filings: Optional[Dict[str, str]] = None) -> DailyMemory:
        return asyncio.run(self.run(filings))
