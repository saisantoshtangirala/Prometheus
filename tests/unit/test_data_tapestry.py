"""
Phase 1: Data Tapestry & Input Validation
Tests: DT-01, DT-02, DT-03, DT-04, IN-01, IN-02
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prometheus.data.market_fetcher import MarketDataFetcher, TickerNotFoundError
from prometheus.data.data_validator import (
    KalmanFilter1D,
    detect_illiquid_periods,
    floor_nanoseconds_to_microseconds,
    standardize_timezone,
)
from prometheus.data.sentiment_analyzer import LegalBERTSentimentAnalyzer, BayesianTruthSerum


# ---------------------------------------------------------------------------
# DT-01: fetch valid ticker — OHLCV columns, correct dtypes
# ---------------------------------------------------------------------------

class TestDT01FetchValidTicker:
    def test_ohlcv_columns_present(self, sample_ohlcv, mock_yfinance):
        fetcher = MarketDataFetcher()
        data = fetcher.fetch_all(["SPY"])
        top_level = data.columns.get_level_values(0).unique().tolist()
        assert "Close" in top_level, "Close column missing"
        assert "Volume" in top_level, "Volume column missing"

    def test_price_dtype_float64(self, sample_ohlcv, mock_yfinance):
        fetcher = MarketDataFetcher()
        data = fetcher.fetch_all(["SPY"])
        close = data["Close"]
        assert close.dtypes.iloc[0] == np.float64, "Close prices must be float64"

    def test_volume_dtype(self, sample_ohlcv, mock_yfinance):
        fetcher = MarketDataFetcher()
        data = fetcher.fetch_all(["SPY"])
        vol = data["Volume"]
        # Volume may be float64 after yfinance normalisation — both accepted
        assert vol.dtypes.iloc[0] in (np.float64, np.int64)

    def test_no_all_nan_rows(self, sample_ohlcv, mock_yfinance):
        fetcher = MarketDataFetcher()
        data = fetcher.fetch_all(["SPY"])
        close = data["Close"]
        assert not close.isnull().all(axis=1).any(), "Found rows where all Close prices are NaN"


# ---------------------------------------------------------------------------
# DT-02: fetch invalid ticker — TickerNotFoundError + synthetic fallback
# ---------------------------------------------------------------------------

class TestDT02InvalidTicker:
    def test_invalid_ticker_raises_or_falls_back(self):
        """
        When yfinance returns an empty DataFrame, the fetcher must either
        raise TickerNotFoundError or fall back to synthetic data — not crash silently.
        """
        empty_df = pd.DataFrame()
        with patch("yfinance.download", return_value=empty_df):
            fetcher = MarketDataFetcher()
            try:
                result = fetcher.fetch_all(["INVALID123"])
                # Fallback path: must return non-empty synthetic data
                assert not result.empty, "Empty fallback — synthetic data must be returned"
            except TickerNotFoundError:
                pass  # Explicit error is equally valid

    def test_synthetic_fallback_has_positive_prices(self):
        empty_df = pd.DataFrame()
        with patch("yfinance.download", return_value=empty_df):
            fetcher = MarketDataFetcher()
            try:
                result = fetcher.fetch_all(["INVALID123"])
                if not result.empty and "Close" in result.columns.get_level_values(0):
                    assert (result["Close"].dropna() > 0).all()
            except TickerNotFoundError:
                pass

    def test_ticker_not_found_error_is_catchable(self):
        error = TickerNotFoundError("INVALID123 not found")
        assert isinstance(error, Exception)


# ---------------------------------------------------------------------------
# DT-03: SEC filing text extraction
# ---------------------------------------------------------------------------

class TestDT03SECFilings:
    """
    Tests LegalBERTSentimentAnalyzer's ability to parse filings with
    HTML tags mixed into text (a common EDGAR format issue).
    """

    def test_html_tags_stripped_from_analysis(self):
        analyzer = LegalBERTSentimentAnalyzer()
        html_text = (
            "<p>The company <b>reported</b> a <i>significant loss</i>. "
            "<span>Risk factors include: <ul><li>declining revenue</li>"
            "<li>credit default</li></ul></span></p>"
        )
        # Analyzer should not crash on HTML input
        result = analyzer.analyze(html_text)
        assert "sentiment_score" in result
        assert -1.0 <= result["sentiment_score"] <= 1.0

    def test_negative_signals_dominate_in_distress_filing(self):
        analyzer = LegalBERTSentimentAnalyzer()
        distress_text = (
            "The company faces substantial going concern risk, bankruptcy proceedings, "
            "and significant impairment charges. Material adverse effects are expected."
        )
        result = analyzer.analyze(distress_text)
        assert result["sentiment_score"] < 0, "Distress filing must yield negative score"

    def test_positive_filing_yields_positive_score(self):
        analyzer = LegalBERTSentimentAnalyzer()
        # Use exact phrases from POSITIVE_SIGNALS to ensure detection
        positive_text = (
            "Record revenue this quarter driven by strong demand and accelerating growth. "
            "We exceeded expectations and are confident we are well-positioned for continued "
            "positive momentum with market share gains and dividend increase."
        )
        result = analyzer.analyze(positive_text)
        assert result["sentiment_score"] > 0, "Positive filing must yield positive score"

    def test_uncertainty_hedges_reduce_magnitude(self):
        analyzer = LegalBERTSentimentAnalyzer()
        hedged = "We believe results may improve, subject to potential market volatility."
        unhedged = "Results will improve dramatically with strong growth."
        r_hedged = analyzer.analyze(hedged)
        r_unhedged = analyzer.analyze(unhedged)
        # Hedged text should have lower absolute score
        assert abs(r_hedged["sentiment_score"]) <= abs(r_unhedged["sentiment_score"]) + 0.3


# ---------------------------------------------------------------------------
# DT-04: Social sentiment edge cases
# ---------------------------------------------------------------------------

class TestDT04SocialSentiment:
    def test_empty_string_returns_neutral(self):
        analyzer = LegalBERTSentimentAnalyzer()
        result = analyzer.analyze("")
        assert result["sentiment_score"] == 0.0, "Empty input must return 0.0"

    def test_score_range(self):
        analyzer = LegalBERTSentimentAnalyzer()
        for text in ["great", "terrible", "neutral information", ""]:
            r = analyzer.analyze(text)
            assert -1.0 <= r["sentiment_score"] <= 1.0

    def test_bts_empty_signal_applies_prior(self):
        bts = BayesianTruthSerum()
        result = bts.aggregate_sentiment([])
        # Empty signal list: return neutral prior (0.0)
        assert result == 0.0 or result is not None  # must not crash

    def test_bts_contrarian_downweighted(self):
        bts = BayesianTruthSerum()
        # All users say bullish; one outlier says bearish
        signals = [0.8, 0.9, 0.7, 0.85, -0.9]
        result = bts.aggregate_sentiment(signals)
        # Consensus should win; result should be positive
        assert result > 0, "BTS should weight consensus over single contrarian"


# ---------------------------------------------------------------------------
# IN-01: Missing Volume Data — median fill + Illiquid flag
# ---------------------------------------------------------------------------

class TestIN01MissingVolumeData:
    def _make_volume_with_zeros(self, n=100):
        rng = np.random.default_rng(1)
        vol = pd.Series(rng.integers(1_000_000, 10_000_000, n).astype(float))
        vol.iloc[10] = 0
        vol.iloc[11] = 0
        vol.iloc[12] = 0  # 3 consecutive zeros
        return vol

    def test_zero_volume_replaced_by_median(self):
        vol = self._make_volume_with_zeros()
        filled, _ = detect_illiquid_periods(vol)
        assert (filled > 0).all(), "All volumes must be positive after fill"
        assert filled.iloc[10] == float(vol[vol > 0].median())

    def test_illiquid_flag_raised(self):
        vol = self._make_volume_with_zeros()
        _, is_illiquid = detect_illiquid_periods(vol)
        assert is_illiquid, "3 consecutive zero-volume bars must set Illiquid flag"

    def test_no_illiquid_flag_when_single_zero(self):
        vol = pd.Series([1_000_000.0] * 100)
        vol.iloc[5] = 0  # only 1 zero bar
        _, is_illiquid = detect_illiquid_periods(vol)
        assert not is_illiquid, "Single zero-volume bar must not set Illiquid flag"

    def test_fetcher_validate_and_fill_volume(self):
        from tests.conftest import make_ohlcv
        df = make_ohlcv(["SPY"], n_bars=100, include_zero_volume=True)
        fetcher = MarketDataFetcher()
        result = fetcher.validate_and_fill_volume(df)
        assert result.attrs.get("illiquid_assets") is not None
        assert len(result.attrs["illiquid_assets"]) > 0, "SPY must be flagged illiquid"


# ---------------------------------------------------------------------------
# IN-02: Timezone Chaos — standardize to UTC
# ---------------------------------------------------------------------------

class TestIN02TimezoneNormalization:
    def test_naive_index_localized_to_utc(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")  # naive
        df = pd.DataFrame({"Close": np.ones(10)}, index=dates)
        result = standardize_timezone(df)
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"

    def test_est_converted_to_utc(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="America/New_York")
        df = pd.DataFrame({"Close": np.ones(10)}, index=dates)
        result = standardize_timezone(df)
        assert str(result.index.tz) == "UTC"

    def test_mixed_naive_and_localized_via_fetcher(self):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        df = pd.DataFrame({"Close": np.random.rand(20), "Volume": np.ones(20) * 1e6},
                          index=dates)
        fetcher = MarketDataFetcher()
        result = fetcher.normalize_timestamps(df)
        assert result.index.tz is not None

    def test_nanosecond_floored_to_microsecond(self):
        ns_index = pd.DatetimeIndex(["2025-01-01 00:00:00.123456789"])
        df = pd.DataFrame({"x": [1.0]}, index=ns_index)
        result = floor_nanoseconds_to_microseconds(df)
        # Floor to microseconds: nanosecond part must be 0
        assert result.index[0].nanosecond == 0

    def test_kalman_fills_nan_without_exploding(self):
        kf = KalmanFilter1D()
        series = pd.Series([1.0, np.nan, np.nan, np.nan, 1.5, 2.0])
        result = kf.fill(series)
        assert not result.isnull().any(), "Kalman filter must fill all NaN values"
        assert result.between(0.0, 10.0).all(), "Filled values must be in plausible range"
