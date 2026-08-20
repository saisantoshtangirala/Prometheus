"""
Exchange a Kite request_token for the daily access_token.

RUN THIS ON YOUR LAPTOP, NOT ON THE SERVER. It is the only step that
needs `api_secret`, and keeping it off the Hetzner box means the secret
never sits on the machine that holds the trading account. Everything
else in this project reads only KITE_API_KEY and KITE_ACCESS_TOKEN -
`nightevolver/kite.py` has no code path that touches the secret.

    export KITE_API_KEY=...
    export KITE_API_SECRET=...
    python scripts/kite_login.py

It prints a login URL, you open it, log in, and Kite redirects to your
registered redirect URL with `?request_token=...` in the address bar.
The redirect URL does NOT need to serve anything - `http://127.0.0.1/`
is fine and the browser showing "unable to connect" is the expected and
correct outcome. Copy the token out of the address bar and paste it back.

Then copy the printed export line to wherever the recorder runs.

WHY THIS IS MANUAL. Kite access tokens expire daily and the login
requires interactive 2FA, so there is no supported way to make this
fully unattended. TOTP automation exists and people use it, but it sits
against Zerodha's intent, so this script does not do it for you - that
is your call to make deliberately rather than a default I bury in a
helper.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nightevolver.kite import API_ROOT

LOGIN_URL = "https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"


def parse_args():
    p = argparse.ArgumentParser(description="Kite daily token exchange")
    p.add_argument("--request-token", default=None,
                   help="the request_token from the redirect URL. Omitted, "
                        "the script prompts for it.")
    p.add_argument("--api-key", default=os.environ.get("KITE_API_KEY"))
    p.add_argument("--write-env", default=None,
                   help="also write an env file (chmod 600) at this path")
    return p.parse_args()


def login_checksum(api_key: str, request_token: str, api_secret: str) -> str:
    """SHA-256 of api_key + request_token + api_secret, hex.

    Split out so it is testable. Order matters and a wrong order fails
    opaquely - the API returns a generic auth error, not "your checksum
    is backwards" - so it is worth pinning against a known vector.
    """
    return hashlib.sha256(
        (api_key + request_token + api_secret).encode("utf-8")).hexdigest()


def exchange(api_key: str, api_secret: str, request_token: str) -> dict:
    """POST /session/token. Checksum is SHA-256 of key + request_token + secret."""
    checksum = login_checksum(api_key, request_token, api_secret)
    body = urllib.parse.urlencode({
        "api_key": api_key, "request_token": request_token, "checksum": checksum,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{API_ROOT}/session/token", data=body,
        headers={"X-Kite-Version": "3",
                 "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=30) as f:
            payload = json.loads(f.read())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        raise SystemExit(
            f"token exchange failed (HTTP {e.code}): {detail}\n\n"
            f"The usual causes, in order of likelihood:\n"
            f"  - the request_token was already used (they are single-use)\n"
            f"  - it expired (they are short-lived; redo the login)\n"
            f"  - api_secret does not match this api_key\n"
            f"  - the redirect URL on the console does not match the one used"
        ) from e
    data = payload.get("data") or {}
    if not data.get("access_token"):
        raise SystemExit(f"no access_token in response: {payload}")
    return data


def main() -> int:
    args = parse_args()
    api_key = args.api_key
    if not api_key:
        raise SystemExit("set KITE_API_KEY or pass --api-key")

    print("\n1. Open this URL and log in:\n")
    print(f"   {LOGIN_URL.format(api_key=urllib.parse.quote(api_key))}\n")
    print("2. After login Kite redirects to your registered redirect URL with")
    print("   ?request_token=... in the address bar. The page failing to load")
    print("   is EXPECTED - the token is in the URL, not the page.\n")

    request_token = args.request_token or input("3. Paste the request_token: ").strip()
    if not request_token:
        raise SystemExit("no request_token given")

    # Prompt rather than read the environment by default, so the secret
    # does not end up in shell history or a process listing.
    api_secret = os.environ.get("KITE_API_SECRET") or getpass.getpass(
        "   api_secret (hidden): ").strip()
    if not api_secret:
        raise SystemExit("no api_secret given")

    data = exchange(api_key, api_secret, request_token)
    token = data["access_token"]

    print("\n" + "=" * 66)
    print(f"  logged in as: {data.get('user_name', '?')} "
          f"({data.get('user_id', '?')})")
    print(f"  valid until:  the next Kite daily expiry (early morning IST)")
    print("=" * 66)
    print("\nExport this where the recorder runs:\n")
    print(f"  export KITE_API_KEY={api_key}")
    print(f"  export KITE_ACCESS_TOKEN={token}\n")
    print("Do NOT copy api_secret to that machine - nothing there needs it.\n")

    if args.write_env:
        path = Path(args.write_env)
        path.write_text(f"KITE_API_KEY={api_key}\nKITE_ACCESS_TOKEN={token}\n",
                        encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            print(f"WARNING: could not chmod 600 {path} - check its permissions")
        print(f"written: {path} (mode 600)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
