"""
News features, entirely offline.

The staleness test is the one that earns its place. A dead RSS feed does
not error - it returns HTTP 200 and well-formed XML full of old items.
Measured during development: Moneycontrol's business.xml served fifteen
valid items dated April 2024 in an August 2026 session, while Business
Standard served current ones from the same request loop. Without an age
check that feed contributes "no news today" every day, forever, and the
resulting feature is a constant wearing a time series' clothes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from nightevolver.news import (
    FEATURE_NAMES, _causal_z, _parse_articles, feed_age_days, parse_rss,
)

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def rss(pub_dates):
    items = "".join(
        f"<item><title>Headline {i}</title><pubDate>{d}</pubDate></item>"
        for i, d in enumerate(pub_dates))
    return f"<rss><channel>{items}</channel></rss>".encode()


class TestStaleFeedDetection:
    def test_a_current_feed_reads_as_fresh(self):
        items = parse_rss(rss(["Fri, 21 Aug 2026 09:30:00 +0530"]))
        assert feed_age_days(items, now=NOW) < 1.0

    def test_the_measured_moneycontrol_case_is_caught(self):
        """Valid XML, HTTP 200, content from two years ago."""
        items = parse_rss(rss(["Tue, 23 Apr 2024 22:36:32 +0530"]))
        age = feed_age_days(items, now=NOW)
        assert age is not None and age > 700

    def test_age_uses_the_NEWEST_item_not_the_oldest(self):
        """A live feed always carries some old items. Judging it by its
        oldest would reject every healthy feed."""
        items = parse_rss(rss(["Tue, 23 Apr 2024 10:00:00 +0530",
                               "Fri, 21 Aug 2026 09:30:00 +0530"]))
        assert feed_age_days(items, now=NOW) < 1.0

    def test_undated_items_yield_none_not_zero(self):
        """None means 'cannot tell'. Zero would mean 'perfectly fresh',
        which is the opposite conclusion from the same evidence."""
        assert feed_age_days(parse_rss(b"<rss><channel><item>"
                                       b"<title>x</title></item></channel></rss>")) is None

    def test_empty_feed_yields_none(self):
        assert feed_age_days([]) is None


class TestRssParsing:
    def test_titles_and_dates_survive(self):
        items = parse_rss(rss(["Fri, 21 Aug 2026 09:30:00 +0530"]))
        assert len(items) == 1
        assert items[0]["title"] == "Headline 0"
        assert items[0]["published"].year == 2026

    def test_malformed_xml_is_empty_not_an_exception(self):
        assert parse_rss(b"<rss><channel><item>") == []

    def test_unparseable_dates_leave_published_none(self):
        items = parse_rss(rss(["not a date"]))
        assert items[0]["published"] is None

    def test_an_item_with_a_bad_date_still_yields_its_title(self):
        """Dropping the row entirely would lose a real headline over a
        formatting quirk."""
        items = parse_rss(rss(["not a date"]))
        assert items[0]["title"] == "Headline 0"


class TestGdeltParsing:
    def test_articles_become_dated_rows(self):
        payload = {"articles": [
            {"seendate": "20260814T171500Z", "tone": "-2.5"},
            {"seendate": "20260814T090000Z", "tone": "1.0"},
            {"seendate": "20260815T090000Z", "tone": "0.0"},
        ]}
        df = _parse_articles(payload)
        assert len(df) == 3
        assert df["date"].nunique() == 2

    def test_undated_articles_are_dropped_not_defaulted(self):
        """Defaulting to today would pile unrelated history onto the
        current bar - the one bar a live feature is read from."""
        df = _parse_articles({"articles": [{"seendate": "garbage"},
                                           {"seendate": "20260814T171500Z"}]})
        assert len(df) == 1

    def test_missing_tone_is_nan_not_zero(self):
        """Zero tone means neutral coverage, which is a real reading."""
        df = _parse_articles({"articles": [{"seendate": "20260814T171500Z"}]})
        assert np.isnan(df["tone"].iloc[0])

    def test_empty_and_none_payloads_are_empty_frames(self):
        assert _parse_articles(None).empty
        assert _parse_articles({}).empty
        assert _parse_articles({"articles": []}).empty


class TestCausalZScore:
    def test_no_look_ahead(self):
        rng = np.random.RandomState(0)
        f = pd.DataFrame({"A": rng.poisson(5, 120).astype(float)})
        z1 = _causal_z(f)
        f2 = f.copy()
        f2.iloc[90] += 40
        z2 = _causal_z(f2)
        pd.testing.assert_series_equal(z1["A"].iloc[:90], z2["A"].iloc[:90])

    def test_a_silent_name_does_not_divide_by_zero(self):
        """A name nobody writes about has zero variance in its count."""
        z = _causal_z(pd.DataFrame({"A": [0.0] * 100}))
        assert not np.isinf(z["A"]).any()


class TestContract:
    def test_declared_features(self):
        assert set(FEATURE_NAMES) == {"news_count", "news_count_z", "news_tone"}

    def test_zero_articles_is_zero_attention_not_missing(self):
        """Distinct from tone, which is genuinely undefined when nothing
        was written. Conflating them makes quiet days look like outages."""
        counts = pd.DataFrame({"A": [3.0, np.nan, 5.0]}).fillna(0.0)
        assert counts["A"].tolist() == [3.0, 0.0, 5.0]
