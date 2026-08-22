"""
The sector mapping used only by the edge-search pairs idea.

Not an authoritative classification - see nightevolver/sectors.py's
module docstring. These tests pin coverage and structure, not the
"correctness" of any individual label, since there is no ground truth
being checked against.
"""

from __future__ import annotations

from nightevolver.sectors import SECTOR, sector_groups, sector_of


class TestSectorMapping:
    def test_known_bank_pair_is_the_same_sector(self):
        assert sector_of("HDFCBANK") == sector_of("ICICIBANK")

    def test_a_bank_and_an_it_name_are_different_sectors(self):
        assert sector_of("HDFCBANK") != sector_of("TCS")

    def test_an_unmapped_ticker_is_none_not_other(self):
        assert sector_of("NOT_A_REAL_TICKER") is None

    def test_lookup_is_case_insensitive(self):
        assert sector_of("reliance") == sector_of("RELIANCE")

    def test_no_duplicate_keys_collided_during_editing(self):
        """A dict literal silently keeps the LAST assignment on a
        duplicate key - this catches a copy-paste accident that would
        otherwise misclassify one name with no error."""
        import ast
        import inspect

        import nightevolver.sectors as S
        tree = ast.parse(inspect.getsource(S))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "SECTOR"
                    for t in node.targets):
                keys = [ast.literal_eval(k) for k in node.value.keys]
                dupes = {k for k in keys if keys.count(k) > 1}
                assert not dupes, f"duplicate ticker key(s) in SECTOR: {dupes}"

    def test_sector_groups_drops_groups_below_min_size(self):
        groups = sector_groups(["HDFCBANK", "ICICIBANK", "AXISBANK", "TCS"],
                               min_size=3)
        assert "bank_private" in groups
        assert "it_services" not in groups   # only 1 IT name here

    def test_sector_groups_restricted_to_the_requested_tickers(self):
        """Passing a subset must not leak in other SECTOR members."""
        groups = sector_groups(["HDFCBANK", "ICICIBANK"], min_size=1)
        assert set(groups["bank_private"]) == {"HDFCBANK", "ICICIBANK"}
        assert "AXISBANK" not in groups.get("bank_private", [])

    def test_the_2019_hundred_name_universe_is_almost_fully_covered(self):
        """Documents the real coverage gap rather than asserting a
        specific number that would go stale as the universe list is
        re-derived. Failing here means someone should extend SECTOR."""
        # A static snapshot of the ids this project's top-100-as-of-2019
        # universe actually contained, captured once rather than
        # re-fetched (this test must not need network or the archive
        # cache to run).
        universe = [
            "AXISBANK", "INDUSINDBK", "DREDGECORP", "RELIANCE", "MARUTI",
            "YESBANK", "SUNPHARMA", "SBIN", "HDFCBANK", "BANKBARODA",
            "ICICIBANK", "M&M", "IBULHSGFIN", "HDFC", "BAJFINANCE",
            "ESCORTS", "JETAIRWAYS", "TATASTEEL", "TCS", "INFY", "TITAN",
            "BHARTIARTL", "SRF", "CANBK", "KOTAKBANK", "BEML", "PNB",
            "HINDUNILVR", "DHFL", "ASHOKLEY", "ACC", "BANKINDIA", "UPL",
            "TATAMOTORS", "SIEMENS", "LT", "DLF", "HINDALCO", "JUBLFOOD",
            "JSWSTEEL", "HEROMOTOCO", "ZEEL", "EICHERMOT", "JINDALSTEL",
            "AUROPHARMA", "WOCKPHARMA", "LUPIN", "BAJAJFINSV",
            "ASIANPAINT", "INDIGO", "DRREDDY", "DMART", "RELCAPITAL",
            "SUNTV", "PAGEIND", "BAJAJ-AUTO", "IGL", "UNIONBANK",
            "MCDOWELL-N", "VEDL", "GAIL", "PFC", "ADANIENT", "HINDPETRO",
            "CIPLA", "COLPAL", "ITC", "TATAELXSI", "DABUR", "PEL",
            "SRTRANSFIN", "ONGC", "RECLTD", "PCJEWELLER", "RELINFRA",
            "TECHM", "INDIANB", "BPCL", "IOC", "FEDERALBNK", "BALKRISIND",
            "SAIL", "NCC", "COALINDIA", "WIPRO", "NTPC", "JAICORPLTD",
            "MGL", "ADANIPORTS", "BHEL", "BIOCON", "JUSTDIAL", "L&TFH",
            "POWERGRID", "BRITANNIA", "UBL", "ORIENTBANK", "GRASIM",
            "NIITTECH", "CEATLTD",
        ]
        missing = [t for t in universe if sector_of(t) is None]
        assert len(missing) <= 2, f"coverage regressed: {missing}"
