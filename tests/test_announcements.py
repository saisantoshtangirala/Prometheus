"""
Corporate-announcement features, offline.

`days_since_last` gets the most attention because it is the one feature
here that carries state across bars, and a stateful feature is where
look-ahead hides most comfortably: it is easy to write a version that
resets on a filing it has not seen yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nightevolver.announcements import (
    FEATURE_NAMES, _causal_z, _record_date, build_features,
    days_since_last, to_daily_counts,
)

RECS = [
    {"symbol": "RELIANCE", "an_dt": "21-Aug-2026 15:57:43", "desc": "Board"},
    {"symbol": "RELIANCE", "an_dt": "21-Aug-2026 09:10:00", "desc": "Update"},
    {"symbol": "TCS", "an_dt": "20-Aug-2026 11:00:00", "desc": "Results"},
    {"symbol": "NOTOURS", "an_dt": "21-Aug-2026 10:00:00", "desc": "x"},
    {"symbol": "TCS", "an_dt": "garbage", "desc": "unparseable"},
]


class TestDateParsing:
    def test_the_real_an_dt_format(self):
        assert _record_date({"an_dt": "21-Aug-2026 15:57:43"}) == pd.Timestamp("2026-08-21")

    def test_date_only_variant(self):
        assert _record_date({"an_dt": "21-Aug-2026"}) == pd.Timestamp("2026-08-21")

    def test_unparseable_returns_none_not_today(self):
        """Defaulting to today concentrates every malformed record onto
        the current bar - the one bar a live feature is read from."""
        assert _record_date({"an_dt": "garbage"}) is None

    def test_missing_date_field_returns_none(self):
        assert _record_date({"symbol": "X"}) is None


class TestDailyCounts:
    def test_same_day_filings_are_summed(self):
        c = to_daily_counts(RECS, ["RELIANCE", "TCS"])
        assert c.at[pd.Timestamp("2026-08-21"), "RELIANCE"] == 2.0

    def test_unrequested_symbols_are_excluded(self):
        c = to_daily_counts(RECS, ["RELIANCE", "TCS"])
        assert "NOTOURS" not in c.columns

    def test_undated_records_are_dropped(self):
        """The TCS 'garbage' row must not become a filing on some day."""
        c = to_daily_counts(RECS, ["TCS"])
        assert c["TCS"].sum() == 1.0

    def test_no_matching_records_gives_empty_not_an_error(self):
        """Measured live: the endpoint returned 20 recent records, none
        of which were the requested large caps. That is a normal
        outcome, not a failure."""
        c = to_daily_counts(RECS, ["ZZZZ"])
        assert list(c.columns) == ["ZZZZ"]
        assert len(c) == 0


class TestDaysSinceLast:
    def test_a_filing_day_reads_zero(self):
        counts = pd.DataFrame({"A": [0.0, 1.0, 0.0, 0.0]})
        assert days_since_last(counts)["A"].tolist()[1] == 0.0

    def test_it_counts_up_between_filings(self):
        counts = pd.DataFrame({"A": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]})
        got = days_since_last(counts)["A"].tolist()
        assert np.isnan(got[0])                      # nothing seen yet
        assert got[1:] == [0.0, 1.0, 2.0, 0.0, 1.0]  # resets on each filing

    def test_before_any_filing_is_nan_not_zero(self):
        """Zero means 'filed today'. A name we have never seen file is
        not filing today - it is unknown, and must not impersonate the
        most eventful possible reading."""
        got = days_since_last(pd.DataFrame({"A": [0.0, 0.0, 1.0]}))["A"].tolist()
        assert np.isnan(got[0]) and np.isnan(got[1]) and got[2] == 0.0

    def test_a_later_filing_cannot_change_an_earlier_value(self):
        """The look-ahead test for the one stateful feature here."""
        base = pd.DataFrame({"A": [0.0, 1.0, 0.0, 0.0, 0.0]})
        later = base.copy()
        later.iloc[4] = 1.0
        a, b = days_since_last(base)["A"], days_since_last(later)["A"]
        assert a.iloc[:4].equals(b.iloc[:4])

    def test_counted_in_bars_not_calendar_days(self):
        """The panel is indexed by trading session. A weekend is not two
        days of silence in a market that was closed."""
        idx = pd.to_datetime(["2026-08-20", "2026-08-21", "2026-08-24"])
        counts = pd.DataFrame({"A": [1.0, 0.0, 0.0]}, index=idx)
        assert days_since_last(counts)["A"].tolist() == [0.0, 1.0, 2.0]


class TestFeatureBuild:
    def test_all_declared_features_are_returned(self):
        f = build_features(RECS, ["RELIANCE", "TCS"])
        assert set(f) == set(FEATURE_NAMES)

    def test_reindexing_to_the_price_panel_fills_quiet_days_with_zero(self):
        """A day with no filing is a real zero, not missing data."""
        dates = pd.date_range("2026-08-19", "2026-08-21")
        f = build_features(RECS, ["RELIANCE"], dates=dates)
        assert f["ann_count"].loc[pd.Timestamp("2026-08-19"), "RELIANCE"] == 0.0

    def test_z_score_has_no_look_ahead(self):
        rng = np.random.RandomState(0)
        f = pd.DataFrame({"A": rng.poisson(0.5, 120).astype(float)})
        z1 = _causal_z(f)
        f2 = f.copy()
        f2.iloc[95] += 20
        z2 = _causal_z(f2)
        pd.testing.assert_series_equal(z1["A"].iloc[:95], z2["A"].iloc[:95])

    def test_a_never_filing_name_does_not_divide_by_zero(self):
        z = _causal_z(pd.DataFrame({"A": [0.0] * 100}))
        assert not np.isinf(z["A"]).any()
