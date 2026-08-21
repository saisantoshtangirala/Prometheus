"""
Unattended Kite login: password + generated TOTP -> daily access_token.

    python scripts/kite_auto_login.py                       # refresh in place
    python scripts/kite_auto_login.py --deploy-to root@host # and push it
    python scripts/kite_auto_login.py --check               # validate only

Replaces the manual `kite_login.py` flow for a box that has to refresh
its own token every morning with nobody watching.

READ THIS BEFORE DEPLOYING IT
-----------------------------
This collapses both authentication factors onto one filesystem. The
password and the TOTP seed now live on the same machine, so the second
factor stops being a second factor: anyone who can read those files can
mint codes indefinitely. That is a genuine reduction in account
protection and it is a deliberate operator decision, taken here because
it was asked for explicitly.

What follows from that, and is worth acting on rather than nodding at:

  * Treat the credentials file exactly like the trading password itself.
    Mode 600, owned by the service user, never in the repo, never in a
    backup that leaves the box.
  * Assume compromise of the host means compromise of the account. The
    existing RiskGuard limits (max_daily_loss_pct, max_drawdown_pct) and
    the file-backed halt are the controls that still bite in that case -
    they are worth setting conservatively BECAUSE of this change.
  * Zerodha's 2FA exists to establish a human is present. Automating it
    is against that intent even though the account is your own.

CREDENTIALS FILE - /etc/nightevolver/kite_auth.env, mode 600:

    KITE_API_KEY=...
    KITE_API_SECRET=...
    KITE_USER_ID=AB1234
    KITE_PASSWORD=...
    KITE_TOTP_SECRET=<base32 seed from the authenticator setup screen>

Note this file holds api_secret, which the manual flow deliberately kept
off the server. That is unavoidable here: the token exchange needs it.
It is a second reason the host is now as sensitive as the account.

THE FLOW (Kite's own web login, three requests):
  1. POST /api/login          user_id + password        -> request_id
  2. POST /api/twofa          request_id + TOTP         -> session cookies
  3. GET  /connect/login      api_key                   -> request_token
  4. POST /session/token      checksum                  -> access_token
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import logging
import os
import re
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nightevolver.kite import API_ROOT, KiteAuthError
from nightevolver.totp import totp_with_headroom

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kite.autologin")

KITE_WEB = "https://kite.zerodha.com"
DEFAULT_AUTH_FILE = "/etc/nightevolver/kite_auth.env"
DEFAULT_OUT_FILE = "/etc/nightevolver/kite.env"

REQUIRED = ("KITE_API_KEY", "KITE_API_SECRET", "KITE_USER_ID",
            "KITE_PASSWORD", "KITE_TOTP_SECRET")


def load_auth(path: str) -> Dict[str, str]:
    """Read the credential file, refusing loose permissions.

    A file holding the password AND the TOTP seed AND api_secret that
    any local user can read is a single point of total compromise, so
    this raises rather than warns.
    """
    p = Path(path)
    creds: Dict[str, str] = {k: os.environ[k] for k in REQUIRED if k in os.environ}

    if p.exists():
        mode = p.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
            raise PermissionError(
                f"{p} is readable or writable beyond its owner "
                f"(mode {stat.filemode(mode)}). It holds the password, the "
                f"TOTP seed and api_secret. chmod 600 {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            creds.setdefault(k.strip(), v.strip())

    missing = [k for k in REQUIRED if not creds.get(k)]
    if missing:
        raise KiteAuthError(
            f"missing {', '.join(missing)} - set them in {path} (mode 600) "
            f"or in the environment. See this script's docstring for the "
            f"file format.")
    return creds


# The 2FA step authenticates a COOKIE JAR, and step 3 must reuse it -
# a fresh jar would arrive at /connect/login unauthenticated and be sent
# back to the login page instead of the redirect carrying the token.
opener_cookiejar: http.cookiejar.CookieJar = http.cookiejar.CookieJar()


def _opener() -> urllib.request.OpenerDirector:
    global opener_cookiejar
    opener_cookiejar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(opener_cookiejar))


def _post(opener, url: str, data: Dict[str, str]) -> Dict:
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "X-Kite-Version": "3",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0",
    })
    try:
        with opener.open(req, timeout=30) as f:
            return json.loads(f.read())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise KiteAuthError(f"HTTP {e.code} from {url.split('/api/')[-1]}: "
                            f"{detail}") from e


def web_login(creds: Dict[str, str]) -> str:
    """Password + TOTP -> request_token."""
    opener = _opener()

    logger.info("step 1/3: password login for %s", creds["KITE_USER_ID"])
    r1 = _post(opener, f"{KITE_WEB}/api/login", {
        "user_id": creds["KITE_USER_ID"],
        "password": creds["KITE_PASSWORD"],
    })
    request_id = (r1.get("data") or {}).get("request_id")
    if not request_id:
        raise KiteAuthError(f"no request_id in login response: {r1}")

    # Generated only after step 1 succeeds, and only when it has enough
    # life left to survive the round trip - see totp_with_headroom.
    code = totp_with_headroom(creds["KITE_TOTP_SECRET"], min_seconds=3.0)
    logger.info("step 2/3: submitting generated 2FA code")
    r2 = _post(opener, f"{KITE_WEB}/api/twofa", {
        "user_id": creds["KITE_USER_ID"],
        "request_id": request_id,
        "twofa_value": code,
        "twofa_type": "totp",
    })
    # Kite can report a REJECTED code with HTTP 200 and status="error".
    # Without this check a bad seed looks like a step-3 problem and sends
    # you hunting through redirect-URL configuration instead.
    if str(r2.get("status", "success")).lower() == "error":
        raise KiteAuthError(
            f"2FA rejected: {r2.get('message', r2)}. The TOTP seed is most "
            f"likely for a different account, or the server clock has "
            f"drifted more than 30s.")

    logger.info("step 3/3: collecting request_token")
    return collect_request_token(creds["KITE_API_KEY"])


def _token_in(url: str) -> Optional[str]:
    """`request_token` out of a URL's query, or None."""
    if not url:
        return None
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    return (qs.get("request_token") or [None])[0]


def _strip_query(url: str) -> str:
    """A URL safe to log: host and path only.

    The path is the diagnostic fact and is not a secret; the query on
    these hops carries sess_id and eventually request_token.
    """
    return urllib.parse.urlunparse(
        urllib.parse.urlparse(url)._replace(query="", fragment=""))


def collect_request_token(api_key: str, cookiejar=None) -> str:
    """Walk /connect/login's redirect chain to the token.

    FOLLOW THE KITE HOPS, STOP AT THE TOKEN. Two failure modes bracket
    this, and only doing both halves gets past them:

      * Following everything (urllib's default) reads f.geturl() at the
        END of the chain. The last hop is the app's registered redirect
        URL, and the recommended registration is http://127.0.0.1/ where
        nothing listens - so the fetch fails and the token, which was in
        the Location header one hop earlier, is lost.

      * Stopping at the FIRST hop misses it too. Measured: /connect/login
        302s to /connect/finish, an intermediate page on kite.zerodha.com
        that carries sess_id and no token; /connect/finish is what issues
        the token-bearing redirect.

    So: keep following while the hop is still on Kite and has no token,
    and stop the moment a hop carries request_token - before fetching a
    redirect target that is probably a dead loopback address.
    """
    seen: List[str] = []

    class _Capture(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            seen.append(newurl)
            if _token_in(newurl):
                return None        # arrived; do not fetch the dead target
            return super().redirect_request(req, fp, code, msg, headers,
                                            newurl)

    jar = opener_cookiejar if cookiejar is None else cookiejar
    cap_opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar), _Capture())

    url = f"{KITE_WEB}/connect/login?v=3&api_key={urllib.parse.quote(api_key)}"
    final, body = "", ""
    try:
        with cap_opener.open(urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as f:
            final = f.geturl()
            body = f.read(200_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        # redirect_request returning None lands here, with Location intact.
        if e.headers and e.headers.get("Location"):
            seen.append(e.headers["Location"])
    except urllib.error.URLError:
        pass                       # unreachable redirect target is expected

    for candidate in seen + [final]:
        got = _token_in(candidate)
        if got:
            return got

    # Last resort: a 200 page that carries the token in a JS redirect
    # rather than a Location header. Matched narrowly, and only the
    # token is extracted - the page body is never logged.
    m = re.search(r"request_token=([A-Za-z0-9]+)", body)
    if m:
        return m.group(1)

    hops = [_strip_query(u) for u in seen if u] or ["(no redirect issued)"]
    landed = _strip_query(final) if final else hops[-1]

    # THE CONSENT SCREEN, detected by PATH not by body text.
    #
    # A Connect app that the account has never approved gets served
    # /connect/authorize - Zerodha's "this app wants access to your
    # account" page - and the chain stops there with a 200. Measured on
    # run #5: login -> /connect/finish -> /connect/authorize.
    #
    # The first version of this check scanned the page BODY for
    # "authoriz", and missed: the page is a JS shell whose HTML does not
    # contain the word. The path does, always, and it is the one part of
    # the response that cannot be re-templated out from under us.
    #
    # Deliberately NOT auto-submitted. Approving an application for full
    # account access - orders and funds - is the account holder's
    # consent to give, and automating a consent screen is exactly the
    # step that should stay manual. It is also a ONE-TIME cost: once
    # approved, this hop disappears and every later login is unattended.
    if "/connect/authorize" in landed or any(
            "/connect/authorize" in h for h in hops):
        raise KiteAuthError(
            "the Kite app has not been authorised for this account yet. "
            "Kite stopped at its consent screen (/connect/authorize), which "
            "needs ONE manual approval and then never appears again. On any "
            "browser, phone included, open "
            "https://kite.zerodha.com/connect/login?v=3&api_key=YOUR_API_KEY "
            "(the login URL shown on the app's page at console.zerodha.com), "
            "sign in, press Authorise, and re-run this. The page failing to "
            "load AFTER you press it is expected and means it worked. "
            "This step is not automated on purpose: granting an app access "
            "to orders and funds is your consent to give, not this script's.")

    raise KiteAuthError(
        f"no request_token after 2FA. Chain: {' -> '.join(hops)}; landed on "
        f"{landed}. Causes: the app's redirect URL on the Kite developer "
        f"console does not match, the TOTP seed is for a different account, "
        f"or the password changed.")


def exchange(creds: Dict[str, str], request_token: str) -> Dict:
    """request_token -> access_token. Reuses the manual flow's checksum."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "kite_login", Path(__file__).parent / "kite_login.py")
    kl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kl)
    return kl.exchange(creds["KITE_API_KEY"], creds["KITE_API_SECRET"],
                       request_token)


def validate(api_key: str, token: str) -> bool:
    from nightevolver.kite import Credentials, _get
    try:
        data = json.loads(_get(f"{API_ROOT}/user/profile",
                               Credentials(api_key, token)))
        d = data.get("data") or {}
        logger.info("validated: %s (%s)", d.get("user_name", "?"),
                    d.get("user_id", "?"))
        return True
    except Exception as e:                                   # noqa: BLE001
        logger.error("token failed validation: %s", e)
        return False


def write_env(path: str, api_key: str, token: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Create with restrictive perms BEFORE writing, so the token is never
    # briefly world-readable between creation and chmod.
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"KITE_API_KEY={api_key}\nKITE_ACCESS_TOKEN={token}\n")
    os.chmod(p, 0o600)
    logger.info("wrote %s (mode 600)", p)


def deploy(api_key: str, token: str, target: str, remote_path: str) -> bool:
    import subprocess
    payload = f"KITE_API_KEY={api_key}\nKITE_ACCESS_TOKEN={token}\n"
    remote = (f"sudo install -d -m 755 $(dirname {remote_path}) && "
              f"umask 077 && sudo tee {remote_path} >/dev/null && "
              f"sudo chmod 600 {remote_path}")
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", target, remote],
                           input=payload, text=True, capture_output=True,
                           timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.error("deploy failed: %s: %s", type(e).__name__, e)
        return False
    if r.returncode != 0:
        logger.error("deploy failed (exit %d): %s", r.returncode,
                     r.stderr.strip()[:300])
        return False
    logger.info("deployed to %s:%s (mode 600)", target, remote_path)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Unattended Kite token refresh")
    ap.add_argument("--auth-file", default=DEFAULT_AUTH_FILE)
    ap.add_argument("--out", default=DEFAULT_OUT_FILE,
                    help="where to write the refreshed token")
    ap.add_argument("--deploy-to", default=None, metavar="USER@HOST")
    ap.add_argument("--remote-path", default=DEFAULT_OUT_FILE)
    ap.add_argument("--check", action="store_true",
                    help="validate the EXISTING token and exit; refresh only "
                         "if it has expired")
    args = ap.parse_args()

    try:
        creds = load_auth(args.auth_file)
    except (KiteAuthError, PermissionError) as e:
        logger.error("%s", e)
        return 2

    # A token that still works is not worth replacing: every login is an
    # extra chance to trip Zerodha's rate limiting for no gain.
    if args.check and Path(args.out).exists():
        existing = {}
        for line in Path(args.out).read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()
        tok = existing.get("KITE_ACCESS_TOKEN", "")
        if tok and validate(creds["KITE_API_KEY"], tok):
            logger.info("existing token is still valid - nothing to do")
            return 0
        logger.info("existing token is stale - refreshing")

    try:
        request_token = web_login(creds)
        data = exchange(creds, request_token)
    except KiteAuthError as e:
        logger.error("login failed: %s", e)
        return 2
    except SystemExit as e:            # exchange() raises SystemExit on HTTP error
        logger.error("token exchange failed: %s", e)
        return 2

    token = data["access_token"]
    if not validate(creds["KITE_API_KEY"], token):
        return 2

    write_env(args.out, creds["KITE_API_KEY"], token)
    if args.deploy_to and not deploy(creds["KITE_API_KEY"], token,
                                     args.deploy_to, args.remote_path):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
