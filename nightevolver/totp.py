"""
RFC 6238 time-based one-time passwords, stdlib only.

No `pyotp` dependency: the algorithm is ~20 lines of HMAC and this
codebase already carries a measured preference for not taking system
dependencies that can fail on an unattended box. Correctness is pinned
against the RFC's own published test vectors rather than assumed.

WHAT AUTOMATING 2FA ACTUALLY COSTS - stated once, in the place someone
reading this code will look.

Two-factor authentication is two factors because they are meant to live
in different places: something you know (password) and something you
hold (a phone). Storing the TOTP seed on the same server as the password
collapses both onto one filesystem. From that moment the second factor
adds NO security against anyone who can read those files - it is a
deterministic function of a secret sitting beside the first factor.

That is a real reduction in protection for the account, and it is the
operator's decision to make, not this module's. What this module can do
is make the remaining surface as small as possible:

  * the seed is read from a mode-600 file OUTSIDE the repository, and
    load_totp_secret() REFUSES to read a world- or group-readable file
    rather than warning about it;
  * nothing here logs, prints or returns the seed;
  * the generated code is valid for 30 seconds and is never persisted.

If the box that holds these files is compromised, assume the trading
account is compromised. Size positions accordingly.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import stat
import struct
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nightevolver.totp")

DEFAULT_PERIOD = 30
DEFAULT_DIGITS = 6
DEFAULT_ALGORITHM = "sha1"      # what Zerodha (and almost everyone) uses


def _normalise_secret(secret: str) -> bytes:
    """Base32 seed -> raw bytes.

    Authenticator apps show the seed in base32, usually without padding
    and often with spaces for readability. Both are accepted; a
    lowercase seed is upper-cased because base32 decoding is
    case-sensitive and a lowercase paste is the most common way this
    silently produces wrong codes.
    """
    cleaned = "".join(secret.split()).upper().replace("-", "")
    if not cleaned:
        raise ValueError("empty TOTP secret")
    pad = (-len(cleaned)) % 8
    try:
        return base64.b32decode(cleaned + "=" * pad, casefold=True)
    except Exception as e:                                   # noqa: BLE001
        raise ValueError(
            "TOTP secret is not valid base32. Copy the key Zerodha showed "
            "when you enabled the authenticator, not the 6-digit code."
        ) from e


def totp_at(secret: str, timestamp: float,
            period: int = DEFAULT_PERIOD,
            digits: int = DEFAULT_DIGITS,
            algorithm: str = DEFAULT_ALGORITHM) -> str:
    """The code valid at `timestamp` (unix seconds).

    HOTP over the counter floor(t / period), per RFC 4226/6238.
    """
    key = _normalise_secret(secret)
    counter = int(timestamp) // period
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, getattr(hashlib, algorithm)).digest()

    # Dynamic truncation: the low nibble of the last byte selects a
    # 4-byte window; the top bit is masked off so the result is a
    # positive 31-bit integer regardless of platform.
    offset = digest[-1] & 0x0F
    code = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(code % (10 ** digits)).zfill(digits)


def totp_now(secret: str, period: int = DEFAULT_PERIOD,
             digits: int = DEFAULT_DIGITS,
             algorithm: str = DEFAULT_ALGORITHM) -> str:
    return totp_at(secret, time.time(), period, digits, algorithm)


def seconds_remaining(period: int = DEFAULT_PERIOD,
                      now: Optional[float] = None) -> float:
    """Seconds until the current code expires.

    Used to avoid submitting a code that is about to roll over: a login
    that takes two seconds with one second of validity left fails with a
    generic "invalid 2FA" that looks like a wrong seed.
    """
    now = time.time() if now is None else now
    return period - (now % period)


def totp_with_headroom(secret: str, min_seconds: float = 3.0,
                       period: int = DEFAULT_PERIOD,
                       digits: int = DEFAULT_DIGITS,
                       algorithm: str = DEFAULT_ALGORITHM) -> str:
    """A code guaranteed to stay valid for at least `min_seconds`.

    Waits for the next window rather than submitting one about to
    expire. The alternative is an intermittent, unreproducible auth
    failure roughly `min_seconds/period` of the time - which reads
    exactly like a bad secret and wastes a morning to diagnose.
    """
    remaining = seconds_remaining(period)
    if remaining < min_seconds:
        logger.info("[totp] code expires in %.1fs - waiting %.1fs for the next "
                    "window rather than risking a rollover mid-login",
                    remaining, remaining)
        time.sleep(remaining + 0.25)
    return totp_now(secret, period, digits, algorithm)


def load_totp_secret(path: os.PathLike | str,
                     env_var: str = "KITE_TOTP_SECRET") -> str:
    """Read the seed from the environment or a mode-600 file.

    REFUSES a file readable by group or others. A TOTP seed in a
    world-readable file is not a second factor, it is a public constant,
    and a warning that scrolls past in a cron log is not a control.
    """
    from_env = os.environ.get(env_var, "").strip()
    if from_env:
        return from_env

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"no TOTP secret: {env_var} is unset and {p} does not exist")

    mode = p.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError(
            f"{p} is readable or writable beyond its owner "
            f"(mode {stat.filemode(mode)}). A TOTP seed stored like that is "
            f"not a second factor. Fix with: chmod 600 {p}")

    secret = p.read_text(encoding="utf-8").strip()
    if not secret:
        raise ValueError(f"{p} is empty")
    # Validate now rather than at 3am inside a cron job.
    _normalise_secret(secret)
    return secret
