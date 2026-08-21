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
