"""
The Kite login redirect walk, against a local server that mimics it.

WHY THIS EXISTS. Three Hetzner runs failed at "no request_token after
2FA" with the password and the TOTP both working. The token was real
every time; the redirect walk kept throwing it away, and each attempt
cost a round trip through GitHub Actions to a box in another country to
find out. Two opposite bugs, in sequence:

  1. urllib follows redirects by default, so reading f.geturl() gave the
     END of the chain - the app's registered redirect URL, which is
     recommended to be http://127.0.0.1/ and where nothing listens. The
     fetch failed and the token in the previous Location header was lost.

  2. Stopping at the FIRST hop lost it too: /connect/login redirects to
     /connect/finish, an intermediate Kite page carrying sess_id and no
     token. /connect/finish is what issues the token-bearing redirect.

Both are invisible to a unit test that mocks the HTTP layer, because
both are about what urllib does with a real 302. So this runs a real
HTTP server on loopback and shapes it like Kite's, including the dead
final hop - a port deliberately closed. If the code ever goes back to
fetching that hop, the connection is refused, the token is lost, and
these tests fail instead of a workflow run in the morning.
"""

from __future__ import annotations

import http.cookiejar
import importlib.util
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def _load():
    spec = importlib.util.spec_from_file_location(
        "kite_auto_login",
        Path(__file__).parent.parent / "scripts" / "kite_auto_login.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _dead_port() -> int:
    """A port with nothing listening on it.

    Bound and immediately released, so the number is real and the
    connection to it is refused rather than hanging.
    """
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Server:
    """A tiny HTTP server whose routes the test defines."""

    def __init__(self, routes):
        self.routes = routes            # path -> (status, headers, body)
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?")[0]
                status, headers, body = outer.routes.get(
                    path, (404, {}, b"not found"))
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass                    # keep pytest output clean

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def mod():
    return _load()


def _run(mod, routes, base_override=None):
    with _Server(routes) as srv:
        mod.KITE_WEB = base_override or srv.base
        return mod.collect_request_token(
            "testapikey", cookiejar=http.cookiejar.CookieJar())


class TestRedirectWalk:
    def test_two_hop_chain_ending_at_a_dead_port(self, mod):
        """The shape Kite actually produces, measured on run #4.

        login -> /connect/finish -> registered redirect URL?request_token
        with nothing listening on that last address.
        """
        dead = _dead_port()
        with _Server({}) as srv:
            srv.routes = {
                "/connect/login": (
                    302, {"Location": f"{srv.base}/connect/finish?sess_id=s1"},
                    b""),
                "/connect/finish": (
                    302,
                    {"Location": f"http://127.0.0.1:{dead}/"
                                 f"?request_token=TOK123&action=login"},
                    b""),
            }
            mod.KITE_WEB = srv.base
            assert mod.collect_request_token(
                "k", cookiejar=http.cookiejar.CookieJar()) == "TOK123"

    def test_token_on_the_very_first_hop_still_works(self, mod):
        """Kite has shipped both shapes; neither may be assumed."""
        dead = _dead_port()
        routes = {
            "/connect/login": (
                302,
                {"Location": f"http://127.0.0.1:{dead}/?request_token=EARLY9"},
                b""),
        }
        assert _run(mod, routes) == "EARLY9"

    def test_three_hops_are_followed(self, mod):
        dead = _dead_port()
        with _Server({}) as srv:
            srv.routes = {
                "/connect/login": (
                    302, {"Location": f"{srv.base}/connect/finish"}, b""),
                "/connect/finish": (
                    302, {"Location": f"{srv.base}/connect/relay"}, b""),
                "/connect/relay": (
                    302,
                    {"Location": f"http://127.0.0.1:{dead}/?request_token=T3"},
                    b""),
            }
            mod.KITE_WEB = srv.base
            assert mod.collect_request_token(
                "k", cookiejar=http.cookiejar.CookieJar()) == "T3"

    def test_token_in_a_javascript_redirect_body(self, mod):
        """Some 200-page flows carry it in the body, not in Location."""
        routes = {
            "/connect/login": (
                200, {"Content-Type": "text/html"},
                b"<script>location='http://x/?request_token=BODYTOK'</script>"),
        }
        assert _run(mod, routes) == "BODYTOK"


class TestDiagnostics:
    def test_a_chain_with_no_token_names_where_it_landed(self, mod):
        routes = {
            "/connect/login": (200, {"Content-Type": "text/html"},
                               b"<html>login page</html>"),
        }
        with pytest.raises(mod.KiteAuthError) as e:
            _run(mod, routes)
        assert "/connect/login" in str(e.value)

    def test_the_consent_screen_is_named_by_path_not_body_text(self, mod):
        """A brand-new Connect app needs one manual Authorise click.

        Run #5 landed exactly here, and the previous check - scanning the
        page BODY for "authoriz" - missed it, because /connect/authorize
        is a JS shell whose HTML does not contain the word. The path
        does. This test therefore serves a body with NO such text, so it
        fails again if anyone reverts to body sniffing.
        """
        with _Server({}) as srv:
            srv.routes = {
                "/connect/login": (
                    302, {"Location": f"{srv.base}/connect/finish"}, b""),
                "/connect/finish": (
                    302, {"Location": f"{srv.base}/connect/authorize"}, b""),
                "/connect/authorize": (
                    200, {"Content-Type": "text/html"},
                    b"<html><div id=app></div></html>"),
            }
            mod.KITE_WEB = srv.base
            with pytest.raises(mod.KiteAuthError, match="not been authorised"):
                mod.collect_request_token(
                    "k", cookiejar=http.cookiejar.CookieJar())

    def test_the_consent_error_says_it_is_one_time_and_manual(self, mod):
        """The two facts that stop someone re-running this five times."""
        with _Server({}) as srv:
            srv.routes = {
                "/connect/login": (
                    302, {"Location": f"{srv.base}/connect/authorize"}, b""),
                "/connect/authorize": (200, {}, b"<html></html>"),
            }
            mod.KITE_WEB = srv.base
            with pytest.raises(mod.KiteAuthError) as e:
                mod.collect_request_token(
                    "k", cookiejar=http.cookiejar.CookieJar())
            msg = str(e.value)
            assert "never appears again" in msg
            assert "console.zerodha.com" in msg

    def test_a_consent_screen_that_still_yields_a_token_is_not_hijacked(self, mod):
        """The consent branch must not pre-empt an actual token.

        If Kite ever routes through /connect/authorize AND still issues
        the redirect, the token wins - the error path is for when there
        is genuinely nothing to return.
        """
        dead = _dead_port()
        with _Server({}) as srv:
            srv.routes = {
                "/connect/login": (
                    302, {"Location": f"{srv.base}/connect/authorize"}, b""),
                "/connect/authorize": (
                    302,
                    {"Location": f"http://127.0.0.1:{dead}/?request_token=OK7"},
                    b""),
            }
            mod.KITE_WEB = srv.base
            assert mod.collect_request_token(
                "k", cookiejar=http.cookiejar.CookieJar()) == "OK7"

    def test_the_query_string_is_never_in_the_error(self, mod):
        """Hops carry sess_id, and errors get shipped to CI logs."""
        with _Server({}) as srv:
            srv.routes = {
                "/connect/login": (
                    302,
                    {"Location": f"{srv.base}/connect/finish?sess_id=SEKRIT"},
                    b""),
                "/connect/finish": (200, {}, b"<html>dead end</html>"),
            }
            mod.KITE_WEB = srv.base
            with pytest.raises(mod.KiteAuthError) as e:
                mod.collect_request_token(
                    "k", cookiejar=http.cookiejar.CookieJar())
            assert "SEKRIT" not in str(e.value)
            assert "/connect/finish" in str(e.value)
