"""
Cross-exchange features, offline.

THE TEST THAT MATTERS is the adjustment-convention one. The obvious
implementation - difference the two closing PRICES - was written, ran
without error, and produced:

    RELIANCE 2026-08-20   NSE panel 107.56   BSE bhavcopy 1307.50
    spread = -16,977 bps, stable across every day and every name

nse_prices.py back-adjusts for corporate actions so returns stay
continuous; the BSE bhavcopy carries the actual traded price. Neither is
wrong, and differencing them yields a large, stable, precise-looking
number that measures nothing but the conventions.

It is a good example of a failure this codebase keeps meeting: the code
did not crash, the output had the right shape and units, and only the
MAGNITUDE gave it away. A -170% arbitrage between two Indian exchanges
on the same stock is not a subtle error, but nothing in the pipeline was
positioned to notice it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nightevolver.bse_prices import (
    BSE_EQUITY_GROUPS, EX_DATE_MASK_BPS, FEATURE_NAMES, cross_exchange_features,
    parse_bse,
)

CSV = (
    "TradDt,FinInstrmTp,TckrSymb,SctySrs,ClsPric,TtlTradgVol\n"
    "2026-08-19,STK,RELIANCE,A,1307.50,572053\n"
    "2026-08-19,STK,TCS,A,2287.00,181811\n"
    "2026-08-19,STK,SMECO,M,10.00,100\n"
    "2026-08-19,STK,T2TCO,T,50.00,900\n"
    "2026-08-19,STK,RELIANCE,B,1300.00,10\n"
)


class TestSeriesFiltering:
    def test_only_bse_equity_groups_are_kept(self):
        """NSE marks equity 'EQ'; BSE uses group letters. Filtering for
        'EQ' here returns an empty frame, which downstream looks like a
        quiet day rather than a bug."""
        df = parse_bse(CSV.encode(), ["RELIANCE", "SMECO", "T2TCO"])
        assert "SMECO" not in df.index      # SME platform
        assert "T2TCO" not in df.index      # trade-to-trade
        assert "RELIANCE" in df.index

    def test_groups_are_A_and_B(self):
        assert set(BSE_EQUITY_GROUPS) == {"A", "B"}

    def test_the_most_traded_row_wins_a_duplicate(self):
        """RELIANCE appears in both A and B here. Taking an arbitrary
        first row could quote the illiquid one."""
        df = parse_bse(CSV.encode(), ["RELIANCE"])
        assert df.at["RELIANCE", "ClsPric"] == 1307.50

    def test_garbage_returns_none(self):
        assert parse_bse(b"\x00not a csv", ["RELIANCE"]) is None

    def test_missing_columns_return_none(self):
        assert parse_bse(b"TckrSymb\nRELIANCE\n", ["RELIANCE"]) is None


class TestAdjustmentConventionBug:
    """The measured failure, pinned so it cannot come back."""

    def _panel(self, nse_level, bse_level, n=6):
        """Two series with IDENTICAL returns at different price levels -
        exactly the adjusted-vs-unadjusted situation."""
        idx = pd.date_range("2026-08-10", periods=n, freq="B")
        rng = np.random.RandomState(0)
        rets = rng.normal(0, 0.01, n)
        path = np.cumprod(1 + rets)
        return (pd.DataFrame({"AAA": nse_level * path}, index=idx),
                pd.DataFrame({"AAA": bse_level * path}, index=idx))

    def test_identical_returns_at_different_levels_give_zero_spread(self,
                                                                   monkeypatch):
        """The whole point. A back-adjusted series and a raw series for
        the same stock have the same returns and wildly different levels.
        The feature must see zero divergence, not -17,000 bps."""
        import nightevolver.bse_prices as B
        nse, bse = self._panel(107.56, 1307.50)

        monkeypatch.setattr(B, "fetch_bse_raw", lambda *a, **k: (b"x", "ok"))
        monkeypatch.setattr(
            B, "parse_bse",
            lambda raw, syms: pd.DataFrame(
                {"ClsPric": [np.nan], "TtlTradgVol": [1.0]}, index=["AAA"]))

        # Feed BSE levels directly through a stubbed parse keyed by date.
        seq = iter(bse["AAA"].tolist())
        monkeypatch.setattr(
            B, "parse_bse",
            lambda raw, syms: pd.DataFrame(
                {"ClsPric": [next(seq)], "TtlTradgVol": [1.0]}, index=["AAA"]))

        out = B.cross_exchange_features(nse, None, ["AAA"])
        s = out["nse_bse_spread"]["AAA"].dropna()
        assert len(s) > 0, "no spreads computed"
        assert s.abs().max() < 1.0, (
            f"levels leaked into the spread: max |{s.abs().max():.1f}| bps")

    def test_the_mask_threshold_is_far_below_the_measured_artifact(self):
        """The broken version produced ~17,000 bps. Any mask that would
        have let that through is not a guard."""
        assert EX_DATE_MASK_BPS < 1000.0

    def test_the_measured_failure_is_recorded_in_the_source(self):
        """A future reader must be able to find out WHY this is a return
        difference and not the obvious price difference, without
        rediscovering it. The specific measured numbers are the evidence,
        so they have to survive in the source."""
        import inspect

        import nightevolver.bse_prices as B
        src = inspect.getsource(B)
        assert "1307.50" in src and "107.56" in src, \
            "the measured level mismatch is not recorded"
        assert "16,977" in src or "16977" in src, \
            "the magnitude of the artifact is not recorded"


class TestMissingData:
    def test_no_sessions_yields_nan_frames_not_an_exception(self, monkeypatch):
        import nightevolver.bse_prices as B
        idx = pd.date_range("2026-08-10", periods=4, freq="B")
        nse = pd.DataFrame({"AAA": [100.0, 101.0, 102.0, 103.0]}, index=idx)
        monkeypatch.setattr(B, "fetch_bse_raw", lambda *a, **k: (None, "absent"))
        out = B.cross_exchange_features(nse, None, ["AAA"])
        assert set(out) == set(FEATURE_NAMES)
        assert out["nse_bse_spread"].isna().all().all()

    def test_a_name_absent_from_bse_is_nan_not_zero(self, monkeypatch):
        """Zero spread means 'the venues agreed exactly', a real and
        informative reading. A missing listing must not impersonate it."""
        import nightevolver.bse_prices as B
        idx = pd.date_range("2026-08-10", periods=3, freq="B")
        nse = pd.DataFrame({"ZZZ": [100.0, 101.0, 102.0]}, index=idx)
        monkeypatch.setattr(B, "fetch_bse_raw", lambda *a, **k: (b"x", "ok"))
        monkeypatch.setattr(B, "parse_bse", lambda raw, syms: None)
        out = B.cross_exchange_features(nse, None, ["ZZZ"])
        assert out["nse_bse_spread"]["ZZZ"].isna().all()
