"""
The retry-handler exception gap, pinned.

MEASURED FAILURE. A 2019-2023 equity backfill cached 1,615 sessions over
about an hour and then died:

    http.client.IncompleteRead: IncompleteRead(73728 bytes read,
                                               12597 more expected)

Every fetcher caught `(urllib.error.URLError, OSError, TimeoutError)`,
which reads as exhaustive. IncompleteRead is not an OSError - its MRO is
IncompleteRead -> HTTPException -> Exception - so the most likely
transport failure when pulling thousands of files from a throttling
archive was precisely the one the retry loop could not catch.

Two things are asserted here, and the second is the one that will
actually catch a regression: not just that the shared tuple is right, but
that NO fetcher in the package has quietly gone back to writing its own
narrower tuple.
"""

from __future__ import annotations

import ast
import http.client
import inspect
import pkgutil
import socket
import ssl
import urllib.error

import pytest

import nightevolver
from nightevolver.nethttp import TRANSIENT_NET_ERRORS

# Modules that reach the network and therefore must retry, not die.
FETCHERS = ["nse_prices", "derivatives", "delivery", "bse_prices",
            "flows", "news", "announcements", "kite"]


class TestTheMeasuredCrash:
    def test_incomplete_read_is_caught(self):
        """The exact exception that killed the backfill."""
        err = http.client.IncompleteRead(b"partial", 12597)
        assert isinstance(err, TRANSIENT_NET_ERRORS)

    def test_incomplete_read_is_not_an_oserror(self):
        """Why the old tuple missed it. If this ever becomes true in some
        future CPython the gap closes on its own - but the test above is
        what guarantees the behaviour either way."""
        assert not issubclass(http.client.IncompleteRead, OSError)

    @pytest.mark.parametrize("exc", [
        http.client.BadStatusLine("garbage"),
        http.client.LineTooLong("header line"),
        http.client.RemoteDisconnected("closed"),
        urllib.error.URLError("dns"),
        socket.timeout("timed out"),
        ssl.SSLError("handshake"),
        ConnectionResetError("reset by peer"),
        TimeoutError("read timeout"),
    ])
    def test_every_transient_transport_failure_is_caught(self, exc):
        assert isinstance(exc, TRANSIENT_NET_ERRORS)


class TestWhatMustStayOutside:
    def test_httperror_is_still_reachable_but_handled_first(self):
        """HTTPError IS a URLError, so the shared tuple would swallow it.
        Every call site must keep its `except urllib.error.HTTPError`
        clause BEFORE the transient one - the status code is what decides
        403-retry from 404-genuine-holiday, and collapsing that is what
        manufactured phantom holidays in flows.py."""
        assert issubclass(urllib.error.HTTPError, urllib.error.URLError)

        # Checked per try-block via the AST, not by string position: a
        # module has several independent try blocks and only some catch
        # HTTPError, so comparing offsets across the whole file compares
        # handlers that never see each other's exceptions.
        for name in FETCHERS:
            mod = __import__(f"nightevolver.{name}", fromlist=["x"])
            for node in ast.walk(ast.parse(inspect.getsource(mod))):
                if not isinstance(node, ast.Try):
                    continue
                kinds = [ast.unparse(h.type) if h.type else "bare"
                         for h in node.handlers]
                if "urllib.error.HTTPError" not in kinds:
                    continue
                trans = [i for i, k in enumerate(kinds)
                         if "TRANSIENT_NET_ERRORS" in k]
                if not trans:
                    continue
                assert kinds.index("urllib.error.HTTPError") < trans[0], (
                    f"{name} line {node.lineno}: the transient handler "
                    "precedes the HTTPError handler and will swallow 404s "
                    "as retryable")

    def test_keyboardinterrupt_and_memoryerror_are_not_swallowed(self):
        """A retry loop that catches these turns ctrl-C into a sleep."""
        for exc in (KeyboardInterrupt(), MemoryError(), SystemExit()):
            assert not isinstance(exc, TRANSIENT_NET_ERRORS)


class TestNoFetcherRollsItsOwn:
    def test_no_module_uses_the_old_narrow_tuple(self):
        """The regression guard. The gap existed in six places at once
        because each was written independently; a seventh copy would
        reintroduce it silently."""
        offenders = []
        for m in pkgutil.iter_modules(nightevolver.__path__):
            mod = __import__(f"nightevolver.{m.name}", fromlist=["x"])
            src = inspect.getsource(mod)
            if m.name == "nethttp":
                continue                      # quotes the old tuple in prose
            if "except (urllib.error.URLError" in src:
                offenders.append(m.name)
        assert not offenders, (
            f"{offenders} catch a hand-rolled network tuple; use "
            "TRANSIENT_NET_ERRORS or IncompleteRead will kill the run")

    def test_every_fetcher_imports_the_shared_tuple(self):
        for name in FETCHERS:
            mod = __import__(f"nightevolver.{name}", fromlist=["x"])
            assert hasattr(mod, "TRANSIENT_NET_ERRORS"), \
                f"{name} does not use the shared retry tuple"


class TestTruncatedDownloadsNeverReachTheCache:
    """The second half of the failure. IncompleteRead is the LOUD version
    of a short read; a chunked response can also end cleanly on a
    truncated body, in which case read() returns fewer bytes and raises
    nothing. Writing that to the cache converts one bad download into a
    permanent hole, because every later run reads the corrupt file from
    cache and never re-fetches."""

    def test_a_truncated_zip_is_rejected(self):
        import io
        import zipfile

        from nightevolver.nse_prices import _zip_is_intact

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("bhav.csv", "TckrSymb,ClsPric\n" + "AAA,100\n" * 500)
        good = buf.getvalue()

        assert _zip_is_intact(good)
        assert not _zip_is_intact(good[:len(good) // 2])
        assert not _zip_is_intact(b"")
        assert not _zip_is_intact(b"<html>403 Forbidden</html>")

    def test_derivatives_uses_the_same_guard(self):
        from nightevolver.derivatives import _zip_is_intact as d
        from nightevolver.nse_prices import _zip_is_intact as e
        assert d(b"") is False and e(b"") is False

    def test_a_valid_file_with_no_requested_tickers_is_still_cached(self,
                                                                   tmp_path,
                                                                   monkeypatch):
        """Deliberate distinction. The guard tests the ZIP, not whether
        parsing found anything: a perfect bhavcopy that happens to carry
        none of the requested names must still be cached, or every run
        re-downloads a good file forever."""
        import io
        import zipfile

        import pandas as pd

        import nightevolver.nse_prices as P

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("bhav.csv",
                       "TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,"
                       "PrvsClsgPric,TtlTradgVol,FinInstrmTp\n"
                       "SOMEONEELSE,EQ,1,2,0.5,1.5,1,100,STK\n")
        raw = buf.getvalue()

        monkeypatch.setattr(P, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(P, "_fetch_raw", lambda *a, **k: (raw, "ok"))
        date = pd.Timestamp("2019-06-03")

        parsed, reason = P.fetch_bhav_day(date, ["RELIANCE"], use_cache=True)
        assert parsed is None                      # none of ours in the file
        assert (tmp_path / "20190603.zip").exists(), \
            "a good bhavcopy was not cached and will be re-fetched forever"

    def test_a_truncated_body_is_not_cached(self, tmp_path, monkeypatch):
        import io
        import zipfile

        import pandas as pd

        import nightevolver.nse_prices as P

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("bhav.csv", "TckrSymb,ClsPric\n" + "AAA,100\n" * 500)
        whole = buf.getvalue()
        truncated = whole[:len(whole) // 2]

        monkeypatch.setattr(P, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(P, "_fetch_raw", lambda *a, **k: (truncated, "ok"))

        P.fetch_bhav_day(pd.Timestamp("2019-06-04"), ["AAA"], use_cache=True)
        assert not (tmp_path / "20190604.zip").exists(), \
            "a truncated download reached the cache as a permanent hole"
