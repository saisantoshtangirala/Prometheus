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
    # DO NOT FOLLOW THE REDIRECT.
    #
    # Kite answers /connect/login with a 302 to the app's registered
    # redirect URL, and `?request_token=...` rides in that Location
    # header. urllib follows redirects by default, so the first version
    # read f.geturl() - the end of the chain - and lost the token
    # whenever the redirect URL was unreachable (http://127.0.0.1/ is
    # the recommended registration and nothing listens there) or itself
    # redirected onward. Capturing Location directly makes this work
    # regardless of what the redirect URL points at.
    seen: List[str] = []

    class _Capture(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            seen.append(newurl)
            return None            # stop here; do not fetch newurl

    cap_opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(opener_cookiejar), _Capture())

    url = f"{KITE_WEB}/connect/login?v=3&api_key={urllib.parse.quote(creds['KITE_API_KEY'])}"
    final = ""
    try:
        with cap_opener.open(urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as f:
            final = f.geturl()
    except urllib.error.HTTPError as e:
        if e.headers and e.headers.get("Location"):
            seen.append(e.headers["Location"])
    except urllib.error.URLError:
        pass                       # unreachable redirect target is fine

    token: Optional[str] = None
    for candidate in seen + [final]:
        if not candidate:
            continue
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(candidate).query)
        got = (qs.get("request_token") or [None])[0]
        if got:
            token = got
            break

    if not token:
        # Report WHERE it went, with the query stripped - the redirect
        # target is the single most diagnostic fact here and it is not a
        # secret, while the query could carry one.
        hops = [urllib.parse.urlunparse(
                    urllib.parse.urlparse(u)._replace(query="", fragment=""))
                for u in seen if u] or ["(no redirect issued)"]
        raise KiteAuthError(
            f"no request_token after 2FA. Redirect chain went to: "
            f"{' -> '.join(hops)}. Usual causes: the app's redirect URL on "
            f"the Kite developer console does not match, the TOTP seed is "
            f"for a different account, or the password changed.")
    return token


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
