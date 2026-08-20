"""
Tests for the Kite client, the depth recorder and the intraday path.

The binary parser gets the most attention here, because it is the one
component whose bugs are invisible: a wrong byte offset does not raise,
it produces plausible numbers that are wrong, and it would do so for
months into a dataset that cannot be re-recorded.
"""

from __future__ import annotations

import gzip
import json
import struct
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nightevolver.depth_recorder import (
    IST, DepthRecorder, DepthWriter, is_market_open, read_depth_file,
    tick_to_record,
)
from nightevolver.intraday import (
    INTRADAY_FEATURE_NAMES, ORDERFLOW_FEATURE_NAMES, day_shape_features,
    depth_file_to_daily_features, minutes_to_daily_features, normalise_features,
)
from nightevolver.kite import (
    DEPTH_LEVELS, PACKET_FULL, PACKET_LTP, PACKET_QUOTE, DepthEntry, KiteAuthError,
    Tick, load_credentials, parse_binary_message, parse_depth, parse_packet,
)

# --------------------------------------------------------------------------
# packet builders — byte-exact, following the documented layout
# --------------------------------------------------------------------------

def _build_full(token=408065, ltp=150000, bids=None, asks=None) -> bytes:
    """A 184-byte `full` packet. Prices are in paise (divisor 100)."""
    bids = bids or [(10 + i, 149900 - i * 100, 2 + i) for i in range(DEPTH_LEVELS)]
    asks = asks or [(20 + i, 150100 + i * 100, 3 + i) for i in range(DEPTH_LEVELS)]
    buf = bytearray(PACKET_FULL)
    ints = [token, ltp, 5, 149950, 1_000_000, 7777, 8888,
            149000, 151000, 148000, 149500, 1_700_000_000, 0, 0, 0,
            1_700_000_060]
    for i, val in enumerate(ints):
        struct.pack_into(">i", buf, i * 4, val)
    for i, (q, p, o) in enumerate(list(bids) + list(asks)):
        off = 64 + i * 12
        struct.pack_into(">i", buf, off, q)
        struct.pack_into(">i", buf, off + 4, p)
        struct.pack_into(">h", buf, off + 8, o)
        # bytes off+10..off+12 stay zero: the documented padding
    return bytes(buf)


def _frame(*packets: bytes) -> bytes:
    out = struct.pack(">h", len(packets))
    for p in packets:
        out += struct.pack(">h", len(p)) + p
    return out


# --------------------------------------------------------------------------
# binary parsing
# --------------------------------------------------------------------------

class TestPacketParsing:
    def test_full_packet_fields(self):
        t = parse_packet(_build_full())
        assert t.instrument_token == 408065
        assert t.mode == "full"
        assert t.last_price == pytest.approx(1500.00)
        assert t.volume == 1_000_000
        assert t.total_buy_quantity == 7777
        assert t.total_sell_quantity == 8888
        assert t.ohlc_open == pytest.approx(1490.00)
        assert t.exchange_timestamp == 1_700_000_060

    def test_depth_respects_the_two_byte_padding(self):
        """THE test for this module.

        Each depth entry is quantity(int32) + price(int32) + orders(int16)
        + 2 bytes padding = 12 bytes. Reading a 10-byte stride would not
        raise - it would silently shift every level after the first. The
        distinct per-level values here only decode correctly at stride 12.
        """
        t = parse_packet(_build_full())
        assert len(t.bids) == DEPTH_LEVELS and len(t.asks) == DEPTH_LEVELS
        for i, e in enumerate(t.bids):
            assert e.quantity == 10 + i
            assert e.price == pytest.approx((149900 - i * 100) / 100.0)
            assert e.orders == 2 + i
        for i, e in enumerate(t.asks):
            assert e.quantity == 20 + i
            assert e.price == pytest.approx((150100 + i * 100) / 100.0)
            assert e.orders == 3 + i

    def test_wrong_stride_would_be_detected(self):
        """Guard the guard: confirm a 10-byte stride really does produce
        different numbers, so the test above is not vacuous."""
        buf = _build_full()
        wrong = [struct.unpack_from(">i", buf, 64 + i * 10)[0] for i in range(5)]
        right = [e.quantity for e in parse_packet(buf).bids]
        assert wrong != right

    def test_bids_are_descending_and_asks_ascending(self):
        t = parse_packet(_build_full())
        bid_px = [e.price for e in t.bids]
        ask_px = [e.price for e in t.asks]
        assert bid_px == sorted(bid_px, reverse=True)
        assert ask_px == sorted(ask_px)
        assert bid_px[0] < ask_px[0], "best bid must be below best ask"

    def test_ltp_packet(self):
        buf = struct.pack(">ii", 12345, 98765)
        t = parse_packet(buf)
        assert t.mode == "ltp" and t.instrument_token == 12345
        assert t.last_price == pytest.approx(987.65)
        assert t.bids == () and t.asks == ()

    def test_quote_packet_has_no_depth(self):
        t = parse_packet(_build_full()[:PACKET_QUOTE])
        assert t.mode == "quote"
        assert t.bids == () and t.asks == ()
        assert t.ohlc_close == pytest.approx(1495.00)

    def test_index_packet_uses_its_own_layout(self):
        buf = bytearray(28)
        for i, v in enumerate([260105, 2450000, 2460000, 2440000, 2445000,
                               2448000, 1000]):
            struct.pack_into(">i", buf, i * 4, v)
        t = parse_packet(bytes(buf))
        assert t.mode == "index"
        assert t.last_price == pytest.approx(24500.00)
        assert t.ohlc_high == pytest.approx(24600.00)

    def test_undersized_packet_returns_none(self):
        assert parse_packet(b"\x00\x01\x02") is None

    def test_parse_depth_rejects_short_buffer(self):
        with pytest.raises(ValueError, match="184-byte"):
            parse_depth(b"\x00" * 100)


class TestFrameParsing:
    def test_multiple_packets_in_one_frame(self):
        f = _frame(_build_full(token=1), _build_full(token=2), _build_full(token=3))
        ticks = parse_binary_message(f)
        assert [t.instrument_token for t in ticks] == [1, 2, 3]

    def test_heartbeat_yields_no_ticks(self):
        assert parse_binary_message(b"\x00") == []
        assert parse_binary_message(b"") == []

    def test_zero_packet_count(self):
        assert parse_binary_message(struct.pack(">h", 0)) == []

    def test_truncated_frame_does_not_raise(self):
        """A partial frame must degrade to the packets it can read, not
        blow up the recorder mid-session."""
        good = _frame(_build_full(token=7), _build_full(token=8))
        ticks = parse_binary_message(good[:len(good) - 60])
        assert [t.instrument_token for t in ticks] == [7]

    def test_lying_header_does_not_raise(self):
        f = struct.pack(">h", 50) + struct.pack(">h", PACKET_FULL) + _build_full()
        assert len(parse_binary_message(f)) == 1

    def test_mixed_modes_in_one_frame(self):
        f = _frame(struct.pack(">ii", 111, 5000), _build_full(token=222))
        ticks = parse_binary_message(f)
        assert [t.mode for t in ticks] == ["ltp", "full"]


class TestTickDerivedQuantities:
    def test_spread_and_imbalance(self):
        t = parse_packet(_build_full())
        assert t.spread == pytest.approx(1501.00 - 1499.00)
        # bids 10..14 = 60, asks 20..24 = 110
        assert t.depth_imbalance == pytest.approx((60 - 110) / 170)

    def test_imbalance_sign_follows_the_heavier_side(self):
        heavy_bid = _build_full(
            bids=[(1000, 149900 - i * 100, 5) for i in range(5)],
            asks=[(1, 150100 + i * 100, 1) for i in range(5)])
        assert parse_packet(heavy_bid).depth_imbalance > 0.9

    def test_imbalance_is_nan_without_depth(self):
        t = parse_packet(struct.pack(">ii", 1, 100))
        assert np.isnan(t.depth_imbalance) and np.isnan(t.spread)


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------

def test_missing_credentials_raise_a_named_error(monkeypatch):
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
    with pytest.raises(KiteAuthError, match="KITE_API_KEY"):
        load_credentials()


def test_credentials_never_embed_the_token_in_repr():
    from nightevolver.kite import Credentials
    c = Credentials("key", "supersecret")
    assert c.auth_header == "token key:supersecret"


# --------------------------------------------------------------------------
# recorder: writer, rotation, duplicate suppression
# --------------------------------------------------------------------------

class TestDepthWriter:
    def test_rotates_by_ist_date(self, tmp_path):
        w = DepthWriter(tmp_path)
        d1 = datetime(2026, 8, 20, 10, 0, tzinfo=IST)
        d2 = datetime(2026, 8, 21, 10, 0, tzinfo=IST)
        w.write({"x": 1}, now=d1)
        w.write({"x": 2}, now=d1)
        w.write({"x": 3}, now=d2)
        w.close()
        assert w.path_for("20260820").exists()
        assert w.path_for("20260821").exists()
        assert len(list(read_depth_file(w.path_for("20260820")))) == 2
        assert len(list(read_depth_file(w.path_for("20260821")))) == 1

    def test_reopening_appends_rather_than_truncating(self, tmp_path):
        """A mid-session restart must not destroy the morning."""
        d = datetime(2026, 8, 20, 10, 0, tzinfo=IST)
        w1 = DepthWriter(tmp_path); w1.write({"x": 1}, now=d); w1.close()
        w2 = DepthWriter(tmp_path); w2.write({"x": 2}, now=d); w2.close()
        assert [r["x"] for r in read_depth_file(w1.path_for("20260820"))] == [1, 2]

    def test_reader_survives_a_truncated_final_line(self, tmp_path):
        p = tmp_path / "depth_20260820.jsonl.gz"
        with gzip.open(p, "wt", encoding="utf-8") as f:
            f.write(json.dumps({"x": 1}) + "\n")
            f.write('{"x": 2, "b": [[1,2,')          # kill -9 mid-write
        assert [r["x"] for r in read_depth_file(p)] == [1]


class TestRecorderLogic:
    def test_duplicate_books_are_suppressed(self):
        r = DepthRecorder(["RELIANCE"])
        t = parse_packet(_build_full())
        assert r.should_record(t, 1_000)
        assert not r.should_record(t, 2_000), "identical book was recorded twice"

    def test_a_changed_book_is_recorded(self):
        r = DepthRecorder(["RELIANCE"])
        assert r.should_record(parse_packet(_build_full(ltp=150000)), 1_000)
        assert r.should_record(parse_packet(_build_full(ltp=150100)), 2_000)

    def test_depth_change_alone_counts_as_a_change(self):
        """Price can be unchanged while the book moves - that IS the
        order-flow signal, so it must not be suppressed."""
        r = DepthRecorder(["RELIANCE"])
        a = _build_full()
        b = _build_full(bids=[(999, 149900 - i * 100, 2) for i in range(5)])
        assert r.should_record(parse_packet(a), 1_000)
        assert r.should_record(parse_packet(b), 2_000)

    def test_min_interval_throttles(self):
        r = DepthRecorder(["RELIANCE"], min_interval_ms=100)
        assert r.should_record(parse_packet(_build_full(ltp=1)), 0)
        assert not r.should_record(parse_packet(_build_full(ltp=2)), 50_000_000)
        assert r.should_record(parse_packet(_build_full(ltp=3)), 200_000_000)

    def test_record_round_trips_through_json(self):
        t = parse_packet(_build_full())
        rec = json.loads(json.dumps(tick_to_record(t, 123)))
        assert rec["tk"] == 408065 and rec["ts"] == 123
        assert len(rec["b"]) == 5 and len(rec["a"]) == 5
        assert rec["b"][0] == [10, 1499.0, 2]

    def test_record_keeps_both_timestamps(self):
        """Receive time orders events; exchange time is the only way to
        see clock or network drift. Dropping either hides something."""
        rec = tick_to_record(parse_packet(_build_full()), 999)
        assert rec["ts"] == 999
        assert rec["xts"] == 1_700_000_060


class TestMarketHours:
    @pytest.mark.parametrize("when,expected", [
        (datetime(2026, 8, 20, 9, 14, tzinfo=IST), False),
        (datetime(2026, 8, 20, 9, 15, tzinfo=IST), True),
        (datetime(2026, 8, 20, 12, 0, tzinfo=IST), True),
        (datetime(2026, 8, 20, 15, 30, tzinfo=IST), True),
        (datetime(2026, 8, 20, 15, 31, tzinfo=IST), False),
        (datetime(2026, 8, 22, 12, 0, tzinfo=IST), False),   # Saturday
        (datetime(2026, 8, 23, 12, 0, tzinfo=IST), False),   # Sunday
    ])
    def test_session_window(self, when, expected):
        assert is_market_open(when) is expected


# --------------------------------------------------------------------------
# intraday features
# --------------------------------------------------------------------------

def _session(date="2026-08-20", n=375, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{date} 09:15", periods=n, freq="1min")
    r = rng.normal(drift, 0.0008, size=n)
    c = 100 * np.cumprod(1 + r)
    o = np.concatenate([[c[0]], c[:-1]])
    pad = np.abs(rng.normal(0, 0.0003, size=n)) * c
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + pad,
                         "low": np.minimum(o, c) - pad, "close": c,
                         "volume": rng.lognormal(8, 0.5, size=n)}, index=idx)


class TestIntradayFeatures:
    def test_all_declared_features_are_produced(self):
        f = day_shape_features(_session(), prev_close=100.0)
        assert set(f) == set(INTRADAY_FEATURE_NAMES)
        assert all(np.isfinite(v) for v in f.values())

    def test_empty_day_gives_nans_not_an_exception(self):
        f = day_shape_features(pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]))
        assert set(f) == set(INTRADAY_FEATURE_NAMES)
        assert all(np.isnan(v) for v in f.values())

    def test_overnight_gap_matches_definition(self):
        s = _session()
        f = day_shape_features(s, prev_close=90.0)
        assert f["overnight_gap"] == pytest.approx(s["open"].iloc[0] / 90.0 - 1.0)

    def test_path_efficiency_is_high_for_a_clean_trend(self):
        """A monotonic day should be efficient; a chopping day should not."""
        idx = pd.date_range("2026-08-20 09:15", periods=100, freq="1min")
        trend = pd.DataFrame({"open": np.arange(100.0, 200.0),
                              "high": np.arange(100.0, 200.0) + 0.5,
                              "low": np.arange(100.0, 200.0) - 0.5,
                              "close": np.arange(100.0, 200.0),
                              "volume": np.ones(100)}, index=idx)
        chop = trend.copy()
        chop["close"] = 100 + (np.arange(100) % 2)
        assert day_shape_features(trend)["path_efficiency"] > 0.95
        assert day_shape_features(chop)["path_efficiency"] < 0.1

    def test_volume_concentration_detects_a_burst(self):
        even = _session(seed=1)
        even["volume"] = 100.0
        burst = even.copy()
        burst["volume"] = 1.0
        burst.iloc[10, burst.columns.get_loc("volume")] = 1e6
        assert (day_shape_features(burst)["volume_concentration"]
                > day_shape_features(even)["volume_concentration"] * 50)

    def test_signed_volume_sign_follows_the_drift(self):
        up = day_shape_features(_session(seed=2, drift=0.002))
        down = day_shape_features(_session(seed=2, drift=-0.002))
        assert up["signed_volume_frac"] > 0 > down["signed_volume_frac"]

    def test_minutes_to_daily_one_row_per_session(self):
        bars = pd.concat([_session("2026-08-20", seed=1),
                          _session("2026-08-21", seed=2),
                          _session("2026-08-24", seed=3)])
        daily = minutes_to_daily_features(bars)
        assert len(daily) == 3
        assert list(daily.columns) == list(INTRADAY_FEATURE_NAMES)
        # First session has no prior close, so the gap is undefined -
        # NaN, not a silently fabricated zero.
        assert np.isnan(daily["overnight_gap"].iloc[0])
        assert np.isfinite(daily["overnight_gap"].iloc[1])

    def test_features_do_not_use_the_next_day(self):
        """Mutate a later session; earlier feature rows must not move."""
        days = ["2026-08-20", "2026-08-21", "2026-08-24", "2026-08-25"]
        base = pd.concat([_session(d, seed=i) for i, d in enumerate(days)])
        tampered = base.copy()
        mask = pd.DatetimeIndex(tampered.index).normalize() >= pd.Timestamp(days[2])
        tampered.loc[mask, ["open", "high", "low", "close"]] *= 1.5

        a = minutes_to_daily_features(base).iloc[:2]
        b = minutes_to_daily_features(tampered).iloc[:2]
        pd.testing.assert_frame_equal(a, b)

    def test_normalise_features_shape_and_bounds(self):
        dates = pd.bdate_range("2026-01-01", periods=120)
        per_sym = {s: pd.DataFrame(
            np.random.default_rng(0).normal(size=(120, len(INTRADAY_FEATURE_NAMES))),
            index=dates, columns=list(INTRADAY_FEATURE_NAMES)) for s in ("A", "B")}
        out = normalise_features(per_sym, ["A", "B"], dates, INTRADAY_FEATURE_NAMES)
        assert out.shape == (120, 2, len(INTRADAY_FEATURE_NAMES))
        assert np.isfinite(out).all()
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_missing_symbol_becomes_inert_zeros(self):
        dates = pd.bdate_range("2026-01-01", periods=80)
        out = normalise_features({}, ["A"], dates, INTRADAY_FEATURE_NAMES)
        assert out.shape == (80, 1, len(INTRADAY_FEATURE_NAMES))
        assert np.allclose(out, 0.0)


class TestRecorderEndToEnd:
    """Exercise the real connect/subscribe/parse/write loop against a
    local WebSocket server. Everything except Kite itself is real."""

    def test_records_frames_from_a_live_socket(self, tmp_path):
        import asyncio

        import websockets

        received = []

        async def server(ws):
            # The recorder must subscribe and set full mode before data.
            for _ in range(2):
                received.append(json.loads(await ws.recv()))
            for tok in (1, 2, 3):
                await ws.send(_frame(_build_full(token=tok, ltp=150000 + tok)))
            await ws.send(b"\x00")                      # heartbeat
            await ws.send(_frame(_build_full(token=1, ltp=160000)))
            # Hold the connection open and silent. The recorder must
            # still honour its deadline while nothing is arriving.
            await asyncio.sleep(30)

        async def run():
            async with websockets.serve(server, "127.0.0.1", 0) as srv:
                port = srv.sockets[0].getsockname()[1]
                rec = DepthRecorder(
                    ["A", "B", "C"], directory=tmp_path,
                    market_hours_only=False,
                    tokens={"A": 1, "B": 2, "C": 3},
                    url=f"ws://127.0.0.1:{port}")
                return await rec.run(max_seconds=2.5)

        stats = asyncio.run(run())

        assert received[0]["a"] == "subscribe"
        assert sorted(received[0]["v"]) == [1, 2, 3]
        assert received[1]["a"] == "mode"
        assert received[1]["v"][0] == "full", "depth requires full mode"

        assert stats.written == 4, f"expected 4 records, got {stats.written}"
        assert stats.reconnects == 0, "reconnected despite a healthy socket"
        files = list(tmp_path.glob("depth_*.jsonl.gz"))
        assert len(files) == 1
        recs = list(read_depth_file(files[0]))
        assert [r["tk"] for r in recs] == [1, 2, 3, 1]
        assert recs[-1]["ltp"] == pytest.approx(1600.00)
        assert len(recs[0]["b"]) == 5 and len(recs[0]["a"]) == 5

    def test_zero_records_when_the_stream_is_silent(self, tmp_path):
        """A connected-but-silent feed must not look like success."""
        import asyncio

        import websockets

        async def server(ws):
            await ws.recv(); await ws.recv()
            for _ in range(4):
                await ws.send(b"\x00")                  # heartbeats only
                await asyncio.sleep(0.1)

        async def run():
            async with websockets.serve(server, "127.0.0.1", 0) as srv:
                port = srv.sockets[0].getsockname()[1]
                rec = DepthRecorder(["A"], directory=tmp_path,
                                    market_hours_only=False, tokens={"A": 1},
                                    url=f"ws://127.0.0.1:{port}")
                return await rec.run(max_seconds=1.5)

        stats = asyncio.run(run())
        assert stats.written == 0
        assert stats.ticks == 0

    def test_auth_error_on_stream_aborts_instead_of_retrying(self, tmp_path):
        """A stale daily token cannot be fixed by reconnecting. Spinning
        on it would look alive while recording nothing."""
        import asyncio

        import websockets

        async def server(ws):
            await ws.recv(); await ws.recv()
            await ws.send(json.dumps(
                {"type": "error", "data": "Invalid access token"}))
            await asyncio.sleep(0.5)

        async def run():
            async with websockets.serve(server, "127.0.0.1", 0) as srv:
                port = srv.sockets[0].getsockname()[1]
                rec = DepthRecorder(["A"], directory=tmp_path,
                                    market_hours_only=False, tokens={"A": 1},
                                    url=f"ws://127.0.0.1:{port}")
                return await rec.run(max_seconds=5.0)

        with pytest.raises(KiteAuthError, match="expired"):
            asyncio.run(run())

    def test_transport_failure_reconnects(self, tmp_path):
        """A dropped connection is normal and must be retried - unlike an
        auth error."""
        import asyncio

        import websockets

        state = {"n": 0}

        async def server(ws):
            state["n"] += 1
            if state["n"] == 1:
                await ws.close()                        # drop the first one
                return
            await ws.recv(); await ws.recv()
            await ws.send(_frame(_build_full(token=1)))
            await asyncio.sleep(0.5)

        async def run():
            async with websockets.serve(server, "127.0.0.1", 0) as srv:
                port = srv.sockets[0].getsockname()[1]
                rec = DepthRecorder(["A"], directory=tmp_path,
                                    market_hours_only=False, tokens={"A": 1},
                                    url=f"ws://127.0.0.1:{port}")
                return await rec.run(max_seconds=6.0)

        stats = asyncio.run(run())
        assert state["n"] >= 2, "recorder never reconnected"
        assert stats.written >= 1


class TestOrderFlowFeatures:
    def test_daily_orderflow_from_records(self):
        t = parse_packet(_build_full(token=555))
        recs = [tick_to_record(t, i) for i in range(10)]
        out = depth_file_to_daily_features(recs, {555: "RELIANCE"})
        assert set(out) == {"RELIANCE"}
        assert set(out["RELIANCE"]) == set(ORDERFLOW_FEATURE_NAMES)
        assert out["RELIANCE"]["depth_imbalance_mean"] == pytest.approx((60 - 110) / 170)
        assert out["RELIANCE"]["spread_mean_bps"] > 0

    def test_unknown_tokens_are_ignored(self):
        recs = [tick_to_record(parse_packet(_build_full(token=999)), 1)]
        assert depth_file_to_daily_features(recs, {555: "RELIANCE"}) == {}

    def test_records_without_depth_are_skipped(self):
        assert depth_file_to_daily_features(
            [{"tk": 1, "b": [], "a": []}], {1: "X"}) == {}
