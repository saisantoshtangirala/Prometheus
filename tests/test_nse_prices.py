"""
Bhavcopy reading across the 2024 schema change.

THE SEAM THIS COVERS. NSE publishes equity bhavcopies in two entirely
different layouts: the legacy one (SYMBOL, SERIES, CLOSE, TOTTRDVAL...)
up to the end of 2023, and UDiFF (TckrSymb, SctySrs, ClsPric,
TtlTrfVal...) from 2024. Supporting both is what makes a ~7-year panel
reachable instead of ~2.5, which is the binding constraint on the one
live result in this project.

THE MEASURED FAILURE. The legacy->UDiFF rename was inlined in
_parse_bhav. top_liquid_symbols opens the same zip for a different
purpose and carried its OWN copy of the column handling, written before
legacy support existed. Resolving a universe as of 2019 therefore died:

    KeyError: 'SctySrs'

That is the loud version. The quiet version - a second reader that
renames some columns and silently drops the rest - is the one worth
guarding against, because it returns a plausible universe built from
whichever names happened to survive the mangling.

So the normalisation now lives in _read_bhav_csv and nowhere else, and
the tests below run every reader against BOTH layouts rather than
against whichever one was current when the reader was written.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from nightevolver.nse_prices import (
    _parse_bhav, _read_bhav_csv, _zip_is_intact, top_liquid_symbols,
)

# Turnover is deliberately NOT in name order, so a reader that ignores it
# and returns the first N rows fails the ranking tests.
ROWS = [
    # symbol,      close,  volume,  turnover
    ("SMALLCO",     50.0,   1_000,   50_000.0),
    ("MEGACORP",  1000.0, 100_000,  100_000_000.0),
    ("MIDCO",      200.0,  10_000,   2_000_000.0),
    ("PENNYCO",      2.0, 500_000,   1_000_000.0),
]


def udiff_csv():
    head = ("TradDt,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,"
            "PrvsClsgPric,TtlTradgVol,TtlTrfVal,FinInstrmTp\n")
    body = "".join(
        f"2025-01-02,{s},EQ,{c},{c},{c},{c},{c},{v},{t},STK\n"
        for s, c, v, t in ROWS)
    return head + body


def legacy_csv():
    head = ("SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,"
            "TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN\n")
    body = "".join(
        f"{s},EQ,{c},{c},{c},{c},{c},{c},{v},{t},02-JAN-2019,10,INE000A01000\n"
        for s, c, v, t in ROWS)
    return head + body


def zipped(csv_text, name="bhav.csv"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, csv_text)
    return buf.getvalue()


@pytest.fixture(params=["udiff", "legacy"])
def era(request):
    """Every reader is tested against BOTH layouts. Parametrising rather
    than writing two suites is the point: a test that only exists for one
    era is how the second reader drifted out of sync in the first
    place."""
    return request.param


@pytest.fixture
def raw(era):
    return zipped(udiff_csv() if era == "udiff" else legacy_csv())


class TestSchemaNormalisation:
    def test_both_layouts_yield_udiff_column_names(self, raw):
        df = _read_bhav_csv(raw)
        for col in ("TckrSymb", "SctySrs", "ClsPric", "PrvsClsgPric",
                    "TtlTradgVol", "TtlTrfVal", "FinInstrmTp"):
            assert col in df.columns, f"{col} missing after normalisation"

    def test_turnover_survives_the_rename(self, raw):
        """TOTTRDVAL -> TtlTrfVal is the column the universe is RANKED by.
        Losing it does not raise - top_liquid_symbols falls back to
        close x volume, which ranks PENNYCO above MIDCO. A silently
        different universe is the worst outcome here."""
        df = _read_bhav_csv(raw)
        got = dict(zip(df["TckrSymb"], df["TtlTrfVal"]))
        assert got["MEGACORP"] == pytest.approx(100_000_000.0)
        assert got["MIDCO"] == pytest.approx(2_000_000.0)

    def test_legacy_rows_are_marked_as_stock(self, raw):
        """Legacy files carry no FinInstrmTp. Downstream filters on
        == 'STK', so an absent column means an empty frame."""
        df = _read_bhav_csv(raw)
        assert (df["FinInstrmTp"] == "STK").all()

    def test_values_are_identical_across_layouts(self):
        """The two files describe the same market. If normalisation
        changed a number, every cross-era comparison would be measuring
        the file format."""
        u = _read_bhav_csv(zipped(udiff_csv())).set_index("TckrSymb")
        l = _read_bhav_csv(zipped(legacy_csv())).set_index("TckrSymb")
        for sym, close, vol, turn in ROWS:
            assert u.at[sym, "ClsPric"] == l.at[sym, "ClsPric"] == close
            assert u.at[sym, "TtlTrfVal"] == l.at[sym, "TtlTrfVal"] == turn

    def test_garbage_returns_none_not_an_exception(self):
        assert _read_bhav_csv(b"\x00not a zip") is None
        assert _read_bhav_csv(b"") is None

    def test_a_zip_with_no_members_returns_none(self):
        """namelist()[0] on an empty archive raises IndexError, which is
        not in the usual (BadZipFile, ValueError, OSError) tuple."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w"):
            pass
        assert _read_bhav_csv(buf.getvalue()) is None


class TestParseBhav:
    def test_requested_tickers_are_extracted_in_both_eras(self, raw):
        df = _parse_bhav(raw, ["MEGACORP", "MIDCO"])
        assert set(df["TckrSymb"]) == {"MEGACORP", "MIDCO"}

    def test_a_ticker_not_in_the_file_is_absent_not_zero(self, raw):
        df = _parse_bhav(raw, ["MEGACORP", "NOTLISTED"])
        assert set(df["TckrSymb"]) == {"MEGACORP"}

    def test_no_requested_ticker_present_yields_none(self, raw):
        assert _parse_bhav(raw, ["NOTLISTED"]) is None

    def test_non_equity_series_is_excluded(self):
        """The file also carries SGBs, ETFs and debt. Mixing them into an
        equity universe is a silent contamination."""
        csv = udiff_csv() + ("2025-01-02,GOLDBEES,GB,1,1,1,1,1,1,1,STK\n")
        df = _parse_bhav(zipped(csv), ["GOLDBEES", "MEGACORP"])
        assert set(df["TckrSymb"]) == {"MEGACORP"}


class TestUniverseResolution:
    """top_liquid_symbols is the reader that carried the stale copy."""

    def _patch(self, monkeypatch, raw):
        import nightevolver.nse_prices as P
        monkeypatch.setattr(P, "_fetch_raw", lambda *a, **k: (raw, "ok"))

    def test_it_resolves_in_both_eras(self, monkeypatch, raw):
        """The regression. This raised KeyError: 'SctySrs' on any
        pre-2024 date."""
        self._patch(monkeypatch, raw)
        syms = top_liquid_symbols("2019-01-01", n=3, min_price=0.0)
        assert len(syms) == 3

    def test_ranking_is_by_turnover_not_by_volume(self, monkeypatch, raw):
        """PENNYCO has 5x MIDCO's share volume and half its turnover. A
        reader that lost TtlTrfVal falls back to close x volume and
        silently returns a different, more illiquid universe."""
        self._patch(monkeypatch, raw)
        syms = top_liquid_symbols("2019-01-01", n=4, min_price=0.0)
        assert syms[0] == "MEGACORP"
        assert syms.index("MIDCO") < syms.index("PENNYCO")

    def test_the_price_floor_excludes_penny_stocks(self, monkeypatch, raw):
        self._patch(monkeypatch, raw)
        syms = top_liquid_symbols("2019-01-01", n=10, min_price=10.0)
        assert "PENNYCO" not in syms
        assert "MEGACORP" in syms

    def test_an_unreadable_file_raises_rather_than_returning_a_partial(
            self, monkeypatch):
        """A universe silently built from a mangled file is worse than no
        universe: every downstream result would be about the wrong
        names."""
        import nightevolver.nse_prices as P
        monkeypatch.setattr(P, "_fetch_raw", lambda *a, **k: (b"junk", "ok"))
        with pytest.raises(RuntimeError):
            top_liquid_symbols("2019-01-01", n=5)


class TestZipIntegrityGuard:
    def test_a_truncated_bhavcopy_is_rejected(self):
        good = zipped(udiff_csv())
        assert _zip_is_intact(good)
        assert not _zip_is_intact(good[:len(good) // 2])


class TestPrvsClsgPricIsNotAdjusted:
    """THE FALSE PREMISE, pinned against the archive.

    The module docstring used to assert that PrvsClsgPric is the
    exchange's ADJUSTED previous close, and used that as the reason not
    to build a corporate-actions pipeline at all. Measured on two clean
    splits it is simply the previous session's traded close:

        IRCTC     ex 2021-10-28, 1:5    -77.88%
        NESTLEIND ex 2024-01-05, 1:10   -90.17%

    A wrong RATIONALE is more dangerous than a wrong line of code here.
    Nothing breaks while corporate_actions.py is doing the work, but a
    reader who believes the docstring concludes that require_actions
    =False is harmless - and it is the difference between a real session
    and an -80% bar.
    """

    def test_the_docstring_no_longer_claims_the_field_is_adjusted(self):
        import inspect

        import nightevolver.nse_prices as P
        doc = inspect.getdoc(P) or ""
        assert "already corporate-action-adjusted" not in doc, \
            "the refuted claim is back in the module docstring"
        assert "-77.88%" in doc or "77.88" in doc, \
            "the measurement that refuted it is not recorded"

    def test_a_split_bar_is_a_raw_ratio_not_an_adjusted_one(self):
        """Synthetic, so it runs offline: a 1:10 split with an unadjusted
        previous close reads as -90%, which is what the archive shows."""
        csv = ("TradDt,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,"
               "PrvsClsgPric,TtlTradgVol,TtlTrfVal,FinInstrmTp\n"
               "2024-01-05,SPLITCO,EQ,2700,2700,2600,2666.40,27116.40,"
               "100,1000,STK\n")
        df = _parse_bhav(zipped(csv), ["SPLITCO"]).set_index("TckrSymb")
        r = df.at["SPLITCO", "ClsPric"] / df.at["SPLITCO", "PrvsClsgPric"] - 1
        assert r < -0.85, (
            "an unadjusted split bar must read as a large negative return - "
            "if this passes at ~0 the field really is adjusted and the "
            "pipeline is double-correcting")

    def _split_panel(self):
        idx = pd.date_range("2024-01-02", periods=3, freq="B")
        return (idx,
                pd.DataFrame({"SPLITCO": [100.0, 100.0, 10.0]}, index=idx),
                pd.DataFrame({"SPLITCO": [100.0, 100.0, 100.0]}, index=idx))

    def test_a_failed_fetch_refuses_rather_than_warns(self):
        """The guard that the false docstring would have talked someone
        out of. None means the fetch failed, so we do not know whether
        that -90% is a split or a crash."""
        from nightevolver.corporate_actions import adjust_returns
        _, close, prev = self._split_panel()
        with pytest.raises(RuntimeError, match="fetch FAILED"):
            adjust_returns(close, prev, {"SPLITCO": None},
                           require_actions=True)

    def test_a_genuinely_actionless_symbol_does_NOT_block_a_run(self):
        """[] is an answer, not a failure. NSE's endpoint serves only
        currently-listed symbols under their current name, so a 2019
        top-100 always contains names it answers emptily for - NIITTECH
        (now COFORGE), MCDOWELL-N (UNITDSPR), SRTRANSFIN (SHRIRAMFIN).
        Refusing those would drop exactly the renamed and delisted names,
        reintroducing the survivorship bias that as-of universe selection
        exists to remove."""
        from nightevolver.corporate_actions import adjust_returns
        _, close, prev = self._split_panel()
        rets, masked = adjust_returns(close, prev, {"SPLITCO": []},
                                      require_actions=True)
        assert rets is not None

    def test_an_unknown_action_is_masked_not_kept(self):
        """What protects the actionless names: a move still beyond 25%
        after adjustment is dropped as an action we do not know about."""
        import numpy as np

        from nightevolver.corporate_actions import adjust_returns
        idx, close, prev = self._split_panel()
        rets, masked = adjust_returns(close, prev, {"SPLITCO": []},
                                      require_actions=True)
        assert bool(masked.at[idx[2], "SPLITCO"]), \
            "an unexplained -90% move was neither adjusted nor masked"
        assert not np.isfinite(rets.at[idx[2], "SPLITCO"])

    def test_missing_entirely_is_treated_as_a_failure(self):
        """A symbol absent from the dict is 'we never asked', which is a
        failure, not an empty answer."""
        from nightevolver.corporate_actions import adjust_returns
        _, close, prev = self._split_panel()
        with pytest.raises(RuntimeError, match="fetch FAILED"):
            adjust_returns(close, prev, {}, require_actions=True)


class TestETFsAndDVRsAreNotEquity:
    """LIQUIDBEES ranked into a 2019 top-100 by turnover and sat in the
    panel with 27 distinct closes across 1,825 bars at 0.01% annualised
    volatility. Nothing rejected it: a money-market ETF is EQ series, is
    FinInstrmTp STK, and is enormously traded. Only the ISIN prefix
    separates it.

    In a volatility study a constant series is not merely useless, it is
    SELECTABLE - the lowest-vol 'stock' in the universe by a factor of
    2,000, and a search told to find low volatility will find it.
    """

    ISIN_CSV = (
        "TradDt,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,"
        "PrvsClsgPric,TtlTradgVol,TtlTrfVal,FinInstrmTp,ISIN\n"
        # a real company
        "2025-02-10,RELIANCE,EQ,1250,1260,1240,1253.65,1250,1000,"
        "1000000,STK,INE002A01018\n"
        # a money-market ETF, pinned at 1000, with HUGE turnover
        "2025-02-10,LIQUIDBEES,EQ,1000,1000,1000,1000.00,1000,900000,"
        "900000000,STK,INF732E01037\n"
        # a DVR class of a company already in the panel
        "2025-02-10,TATAMTRDVR,EQ,89,90,88,89.05,89,5000,"
        "500000,STK,IN9155A01020\n"
    )

    def test_an_etf_is_excluded_from_parsed_prices(self):
        df = _parse_bhav(zipped(self.ISIN_CSV),
                         ["RELIANCE", "LIQUIDBEES", "TATAMTRDVR"])
        assert set(df["TckrSymb"]) == {"RELIANCE"}

    def test_an_etf_cannot_win_the_universe_on_turnover(self, monkeypatch):
        """LIQUIDBEES has 900x RELIANCE's turnover here. Ranking without
        the ISIN filter puts it first."""
        import nightevolver.nse_prices as P
        monkeypatch.setattr(P, "_fetch_raw",
                            lambda *a, **k: (zipped(self.ISIN_CSV), "ok"))
        syms = top_liquid_symbols("2025-02-10", n=10, min_price=0.0)
        assert "LIQUIDBEES" not in syms
        assert "TATAMTRDVR" not in syms
        assert syms == ["RELIANCE"]

    def test_the_filter_runs_BEFORE_the_column_subset(self):
        """_COLS does not carry ISIN. Narrowing the frame first would drop
        the only discriminating column and turn the filter into a no-op
        that still looked like it was filtering - which is exactly how it
        was first written."""
        from nightevolver.nse_prices import _COLS
        assert "ISIN" not in _COLS
        df = _parse_bhav(zipped(self.ISIN_CSV), ["LIQUIDBEES"])
        assert df is None, "the ISIN filter was bypassed by the subset"

    def test_a_file_without_isins_is_not_emptied(self):
        """Defensive: an era or a file that omits ISIN must fall back to
        the series/type filter rather than silently returning nothing."""
        df = _parse_bhav(zipped(udiff_csv()), ["MEGACORP"])
        assert df is not None and len(df) == 1

    def test_legacy_files_are_filtered_too(self):
        """The legacy layout carries ISIN in its own column, and
        LIQUIDBEES was in the 2019 EQ series as well."""
        csv = ("SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,"
               "TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN\n"
               "RELIANCE,EQ,1090,1095,1085,1092.75,1092,1090,1000,1000000,"
               "03-JAN-2019,10,INE002A01018\n"
               "LIQUIDBEES,EQ,1000,1000,1000,1000,1000,1000,900000,900000000,"
               "03-JAN-2019,10,INF732E01037\n")
        df = _parse_bhav(zipped(csv), ["RELIANCE", "LIQUIDBEES"])
        assert set(df["TckrSymb"]) == {"RELIANCE"}
