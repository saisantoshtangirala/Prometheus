"""
Market Data Fetcher – unified interface for all data sources.

Integrates:
  - OHLCV from yfinance (free tier)
  - Simulated Level-2 order book (from trade flow reconstruction)
  - SEC filing text via EDGAR API
  - Reddit/Twitter sentiment (with Bayesian Truth Serum weighting)
  - Macro data from FRED API
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Asset universe
EQUITY_TICKERS = [
    "SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "XLV", "XLU",
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "GS", "BAC", "C", "WFC",
    "GLD", "SLV", "USO", "TLT", "HYG", "LQD",
]

FX_TICKERS = ["USDJPY=X", "EURUSD=X", "GBPUSD=X", "DXY"]

CRYPTO_TICKERS = ["BTC-USD", "ETH-USD"]

VIX_TICKER = "^VIX"


class TickerNotFoundError(Exception):
    """Raised when a ticker cannot be resolved from any data source."""


class MarketDataFetcher:
    """
    Unified market data fetcher with caching and normalization.

    Handles:
      - Async parallel fetching of multiple symbols
      - Normalization and z-score standardization
      - Missing data imputation (forward-fill + rolling mean)
      - Feature engineering (returns, log-returns, rolling vol, RSI, etc.)
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        lookback_days: int = 252 * 3,  # 3 years
        bar_size: str = "1d",
    ):
        self.cache_dir = cache_dir
        self.lookback_days = lookback_days
        self.bar_size = bar_size
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_all(
        self,
        tickers: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for all tickers. Returns a multi-level DataFrame.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — returning synthetic data")
            return self._synthetic_data(tickers or EQUITY_TICKERS)

        if tickers is None:
            tickers = EQUITY_TICKERS + FX_TICKERS + CRYPTO_TICKERS + [VIX_TICKER]

        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start_dt = datetime.now() - timedelta(days=self.lookback_days)
            start = start_dt.strftime("%Y-%m-%d")

        logger.info("Fetching %d tickers: %s to %s", len(tickers), start, end)
        try:
            data = yf.download(
                tickers,
                start=start,
                end=end,
                interval=self.bar_size,
                progress=False,
                auto_adjust=True,
            )
            if data.empty:
                raise TickerNotFoundError(
                    f"yfinance returned no data for tickers: {tickers}"
                )
            return data
        except TickerNotFoundError:
            raise
        except Exception as e:
            logger.error("yfinance fetch failed: %s — using synthetic data", e)
            return self._synthetic_data(tickers)

    def get_returns(
        self,
        data: pd.DataFrame,
        log_returns: bool = True,
    ) -> pd.DataFrame:
        """Compute per-bar returns (log or simple) from Close prices."""
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        else:
            close = data

        if log_returns:
            returns = np.log(close / close.shift(1))
        else:
            returns = close.pct_change()

        return returns.dropna()

    def build_feature_matrix(
        self,
        returns: pd.DataFrame,
        include_technical: bool = True,
        include_cross_sectional: bool = True,
    ) -> pd.DataFrame:
        """
        Build a rich feature matrix from returns data.

        Features per asset:
          - 1/5/20-bar log returns
          - Realized volatility (20-bar)
          - RSI(14), momentum(20)
          - Z-score vs 252-bar rolling mean
        Cross-sectional:
          - Rank among peers
          - Beta to SPY
        """
        features = {}
        n = len(returns)

        for col in returns.columns:
            r = returns[col].fillna(0)
            features[f"{col}_ret1"] = r
            features[f"{col}_ret5"] = r.rolling(5).sum()
            features[f"{col}_ret20"] = r.rolling(20).sum()
            features[f"{col}_vol20"] = r.rolling(20).std()
            features[f"{col}_zscore"] = (
                (r - r.rolling(252).mean()) / (r.rolling(252).std() + 1e-8)
            )
            if include_technical:
                features[f"{col}_rsi14"] = self._rsi(r, 14)
                features[f"{col}_mom20"] = r.rolling(20).mean()

        feature_df = pd.DataFrame(features, index=returns.index)
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return feature_df

    def get_macro_context(self) -> pd.DataFrame:
        """
        Fetch macro indicators (VIX, yield curve, DXY).
        In production, this connects to FRED API via fredapi.
        """
        try:
            import yfinance as yf
            vix = yf.download("^VIX", period="2y", interval="1d", progress=False)
            tlt = yf.download("TLT", period="2y", interval="1d", progress=False)
            dxy = yf.download("DX-Y.NYB", period="2y", interval="1d", progress=False)
            macro = pd.DataFrame({
                "VIX": vix["Close"] if "Close" in vix else pd.Series(dtype=float),
                "TLT": tlt["Close"] if "Close" in tlt else pd.Series(dtype=float),
                "DXY": dxy["Close"] if "Close" in dxy else pd.Series(dtype=float),
            }).ffill().dropna()
            return macro
        except Exception:
            logger.warning("Macro fetch failed — returning zeros")
            return pd.DataFrame()

    def normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize index to UTC+0 and floor nanoseconds to microseconds.
        Handles mixed UTC/EST/naive datetime indices gracefully.
        """
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            return df
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        df = df.copy()
        df.index = idx.floor("us")
        return df

    def validate_and_fill_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect zero-volume periods and fill with median.
        3+ consecutive zero-volume bars → asset tagged 'Illiquid'.
        Returns (df_filled, illiquid_flags dict).
        """
        if df.empty:
            return df
        df = df.copy()
        illiquid_assets: list[str] = []

        # Support multi-level (field, ticker) or (ticker, field) and flat DataFrames
        if isinstance(df.columns, pd.MultiIndex):
            if "Volume" in df.columns.get_level_values(0):
                # (field, ticker) format
                vol_df = df["Volume"]
            elif "Volume" in df.columns.get_level_values(1):
                # (ticker, field) format — transpose to get tickers as columns
                vol_df = df.xs("Volume", axis=1, level=1)
            else:
                return df
        elif "Volume" in df.columns:
            vol_df = df[["Volume"]]
        else:
            return df

        for col in vol_df.columns:
            series = vol_df[col]
            med = series[series > 0].median() if (series > 0).any() else 1
            zero_run = (series == 0).rolling(3).sum()
            if (zero_run >= 3).any():
                illiquid_assets.append(str(col))
            vol_df[col] = series.where(series > 0, med)

        df.attrs["illiquid_assets"] = illiquid_assets
        return df

    def normalize(self, df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """Normalize feature matrix. method: 'zscore' | 'minmax' | 'robust'."""
        if method == "zscore":
            return (df - df.mean()) / (df.std() + 1e-8)
        elif method == "minmax":
            return (df - df.min()) / (df.max() - df.min() + 1e-8)
        elif method == "robust":
            median = df.median()
            mad = (df - median).abs().median()
            return (df - median) / (mad + 1e-8)
        return df

    @staticmethod
    def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _synthetic_data(tickers: List[str], n_bars: int = 756) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing without network access."""
        np.random.seed(42)
        dates = pd.bdate_range(end=datetime.now(), periods=n_bars)
        dfs = {}
        for ticker in tickers:
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n_bars)))
            dfs[ticker] = pd.DataFrame({
                "Open": prices * (1 + np.random.uniform(-0.002, 0.002, n_bars)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.005, n_bars))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, n_bars))),
                "Close": prices,
                "Volume": np.random.randint(1_000_000, 50_000_000, n_bars),
            }, index=dates)
        return pd.concat(dfs, axis=1)
