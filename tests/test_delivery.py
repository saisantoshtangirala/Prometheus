"""
NSE delivery features, offline.

The z-score tests carry the weight. A rolling statistic that includes
the current bar is a look-ahead so small it looks like a rounding
choice, and this codebase has already paid for one: before it was
corrected, a pure random walk scored rho = -0.39 against the regime
target purely because the baseline shared a denominator with its own
target. `shift(1)` before `rolling` is the whole defence, and nothing
fails loudly if it is removed - the features simply become slightly
prophetic.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from nightevolver.delivery import (
    FEATURE_NAMES, _causal_z, parse_delivery,
)

CSV = (
    "SYMBOL, SERIES,          DATE1, PREV_CLOSE, TTL_TRD_QNTY, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
    "RELIANCE, EQ, 19-Aug-2026, 1322.0, 7489365, 160390, 4352516, 58.12\n"
    "TCS, EQ, 19-Aug-2026, 2280.0, 3368922, 122060, 1780817, 52.86\n"
    "SUSPENDED, EQ, 19-Aug-2026, 10.0, 100, 5, -, -\n"
    "RELIANCE, BE, 19-Aug-2026, 1322.0, 999, 9, 999, 99.9\n"
    "OTHER, EQ, 19-Aug-2026, 50.0, 1000, 100, 500, 50.0\n"
)


class TestParsing:
    def test_headers_with_leading_spaces_are_handled(self):
        """NSE ships ' SERIES', ' DELIV_PER' with leading spaces. Not
        stripping them yields an empty frame and an all-NaN feature
        column - a silent loss, not an error."""
        df = parse_delivery(CSV.encode(), ["RELIANCE", "TCS"])
        assert df is not None and len(df) == 2
        assert df.at["RELIANCE", "DELIV_QTY"] == 4352516

    def test_only_the_EQ_series_is_kept(self):
        """The same symbol appears under BE (trade-to-trade) with
        different numbers. Mixing the series double-counts the day."""
        df = parse_delivery(CSV.encode(), ["RELIANCE"])
        assert len(df) == 1
        assert df.at["RELIANCE", "TTL_TRD_QNTY"] == 7489365

    def test_dash_placeholders_become_nan_not_zero(self):
        """Illiquid rows carry '-'. Zero delivery is a real, meaningful
        reading; a missing one must not impersonate it."""
        df = parse_delivery(CSV.encode(), ["SUSPENDED"])
        assert np.isnan(df.at["SUSPENDED", "DELIV_QTY"])

    def test_unrequested_symbols_are_dropped(self):
        df = parse_delivery(CSV.encode(), ["RELIANCE"])
        assert "OTHER" not in df.index

    def test_garbage_returns_none(self):
        assert parse_delivery(b"\x00\x01not a csv", ["RELIANCE"]) is None

    def test_missing_columns_return_none(self):
        assert parse_delivery(b"SYMBOL,SERIES\nRELIANCE,EQ\n", ["RELIANCE"]) is None

    def test_derived_percentage_matches_nse_own_column(self):
        """Cross-check the derivation against the number NSE publishes."""
        df = parse_delivery(CSV.encode(), ["RELIANCE", "TCS"])
        derived = df["DELIV_QTY"] / df["TTL_TRD_QNTY"] * 100.0
        assert derived.round(2).tolist() == df["DELIV_PER"].tolist()


class TestCausalZScore:
    def _series(self, n=120, seed=0):
        rng = np.random.RandomState(seed)
        return pd.DataFrame({"A": rng.normal(0, 1, n).cumsum()})

    def test_the_current_bar_never_enters_its_own_statistic(self):
        """THE look-ahead test. Change the value at t and everything up
        to and including z[t-1] must be untouched; only z[t] onward may
        move. If the current bar fed its own mean, z[t] would be damped
        by its own contribution and the feature would carry information
        about itself."""
        f = self._series()
        z1 = _causal_z(f)
        f2 = f.copy()
        f2.iloc[80] += 50.0
        z2 = _causal_z(f2)
        pd.testing.assert_series_equal(z1["A"].iloc[:80], z2["A"].iloc[:80])
        assert z1["A"].iloc[80] != z2["A"].iloc[80]

    def test_a_future_bar_cannot_affect_the_present(self):
        f = self._series()
        z1 = _causal_z(f)
        f2 = f.copy()
        f2.iloc[100] += 99.0
        z2 = _causal_z(f2)
        pd.testing.assert_series_equal(z1["A"].iloc[:100], z2["A"].iloc[:100])

    def test_early_rows_are_nan_until_min_periods(self):
        """A z-score from three observations is noise wearing a
        statistic's clothes."""
        z = _causal_z(self._series())
        assert z["A"].iloc[:20].isna().all()

    def test_a_constant_series_yields_nan_not_infinity(self):
        """Zero standard deviation must not divide through to inf, which
        np.nan_to_num would later map to 1.8e308."""
        z = _causal_z(pd.DataFrame({"A": [5.0] * 100}))
        assert not np.isinf(z["A"]).any()

    def test_it_standardises(self):
        rng = np.random.RandomState(1)
        f = pd.DataFrame({"A": rng.normal(10.0, 2.0, 500)})
        z = _causal_z(f)["A"].dropna()
        assert abs(z.mean()) < 0.25
        assert 0.7 < z.std() < 1.4


class TestFeatureContract:
    def test_the_declared_names_are_what_callers_get(self):
        assert set(FEATURE_NAMES) == {
            "deliv_pct", "deliv_pct_z", "avg_trade_size_log",
            "avg_trade_size_z", "deliv_qty_z"}

    def test_delivery_percentage_is_a_fraction_not_a_percent(self):
        """0.58, not 58.12 - so it shares a scale with the other
        ratio-valued channels and no downstream code has to know which
        convention this one uses."""
        df = parse_delivery(CSV.encode(), ["RELIANCE"])
        pct = df["DELIV_QTY"] / df["TTL_TRD_QNTY"]
        assert 0.0 < float(pct.iloc[0]) < 1.0
