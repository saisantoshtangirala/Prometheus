"""
TOTP correctness and the guards around the seed.

Pinned against RFC 6238's own published test vectors. A TOTP
implementation that is subtly wrong fails as a generic "invalid 2FA"
from the server, which looks identical to a wrong seed, a clock skew, or
a changed password - so this is worth testing against an authority
rather than against itself.
"""

from __future__ import annotations

import base64
import stat

import pytest

from nightevolver.totp import (
    _normalise_secret, load_totp_secret, seconds_remaining, totp_at,
    totp_with_headroom,
)

# RFC 6238 Appendix B uses the ASCII seed "12345678901234567890".
RFC_SEED = base64.b32encode(b"12345678901234567890").decode()

RFC_VECTORS_SHA1 = [
    (59, "94287082"),
    (1111111109, "07081804"),
    (1111111111, "14050471"),
    (1234567890, "89005924"),
    (2000000000, "69279037"),
    (20000000000, "65353130"),
]


class TestRFCVectors:
    @pytest.mark.parametrize("t,expected", RFC_VECTORS_SHA1)
    def test_matches_rfc6238_appendix_b(self, t, expected):
        assert totp_at(RFC_SEED, t, digits=8) == expected

    def test_six_digit_codes_are_the_last_six(self):
        """Zerodha uses 6 digits; the RFC publishes 8. The 6-digit code
        is the 8-digit one truncated, so this cross-checks the modulus."""
        for t, expected8 in RFC_VECTORS_SHA1:
            assert totp_at(RFC_SEED, t, digits=6) == expected8[-6:]


class TestSecretHandling:
    def test_lowercase_seed_is_accepted(self):
        """A pasted seed is often lowercase, and base32 is case-sensitive
        - getting this wrong yields valid-looking but wrong codes."""
        assert totp_at(RFC_SEED.lower(), 59, digits=8) == "94287082"

    def test_spaces_and_dashes_are_stripped(self):
        spaced = " ".join(RFC_SEED[i:i + 4] for i in range(0, len(RFC_SEED), 4))
        assert totp_at(spaced, 59, digits=8) == "94287082"
        assert totp_at(spaced.replace(" ", "-"), 59, digits=8) == "94287082"

    def test_unpadded_base32_is_accepted(self):
        assert _normalise_secret(RFC_SEED.rstrip("=")) == b"12345678901234567890"

    def test_empty_secret_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _normalise_secret("   ")

    def test_a_six_digit_code_is_rejected_as_a_seed(self):
        """The most likely operator error: pasting the CODE from the
        phone instead of the SEED from the setup screen."""
        with pytest.raises(ValueError, match="base32"):
            _normalise_secret("492871")


class TestWindowing:
    def test_code_is_stable_within_a_period(self):
        base = 1_700_000_000 - (1_700_000_000 % 30)
        assert totp_at(RFC_SEED, base) == totp_at(RFC_SEED, base + 29)

    def test_code_changes_across_a_period_boundary(self):
        base = 1_700_000_000 - (1_700_000_000 % 30)
        assert totp_at(RFC_SEED, base) != totp_at(RFC_SEED, base + 30)

    def test_seconds_remaining_is_within_the_period(self):
        for now in (0.0, 1.5, 29.9, 30.0, 1_700_000_123.4):
            r = seconds_remaining(30, now=now)
            assert 0 < r <= 30

    def test_headroom_never_returns_an_expiring_code(self, monkeypatch):
        """Submitting a code with under a second of life produces an
        intermittent auth failure that reads exactly like a bad seed."""
        import nightevolver.totp as T

        fake = {"t": 1_700_000_000 - (1_700_000_000 % 30) + 29.5}
        monkeypatch.setattr(T.time, "time", lambda: fake["t"])
        monkeypatch.setattr(T.time, "sleep",
                            lambda s: fake.__setitem__("t", fake["t"] + s))

        code = totp_with_headroom(RFC_SEED, min_seconds=3.0)
        assert T.seconds_remaining(30, now=fake["t"]) >= 3.0
        assert code == totp_at(RFC_SEED, fake["t"])

    def test_headroom_returns_immediately_when_there_is_time(self, monkeypatch):
        import nightevolver.totp as T
        base = 1_700_000_000 - (1_700_000_000 % 30) + 2.0
        monkeypatch.setattr(T.time, "time", lambda: base)
        slept = []
        monkeypatch.setattr(T.time, "sleep", lambda s: slept.append(s))
        assert totp_with_headroom(RFC_SEED, min_seconds=3.0) == totp_at(RFC_SEED, base)
        assert not slept, "waited despite having 28s of headroom"


class TestFilePermissions:
    def test_loose_permissions_are_refused_not_warned(self, tmp_path):
        """A TOTP seed any local user can read is not a second factor,
        it is a public constant. A warning in a cron log is not a
        control, so this raises."""
        p = tmp_path / "totp"
        p.write_text(RFC_SEED)
        p.chmod(0o644)
        with pytest.raises(PermissionError, match="beyond its owner"):
            load_totp_secret(p, env_var="_ABSENT_")

    def test_group_writable_is_also_refused(self, tmp_path):
        p = tmp_path / "totp"
        p.write_text(RFC_SEED)
        p.chmod(0o620)
        with pytest.raises(PermissionError):
            load_totp_secret(p, env_var="_ABSENT_")

    def test_mode_600_is_accepted(self, tmp_path):
        p = tmp_path / "totp"
        p.write_text(RFC_SEED)
        p.chmod(0o600)
        assert load_totp_secret(p, env_var="_ABSENT_") == RFC_SEED

    def test_environment_wins_and_skips_the_file_check(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KITE_TOTP_SECRET", RFC_SEED)
        assert load_totp_secret(tmp_path / "missing") == RFC_SEED

    def test_missing_file_raises_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no TOTP secret"):
            load_totp_secret(tmp_path / "nope", env_var="_ABSENT_")

    def test_invalid_seed_in_file_fails_at_load_not_at_3am(self, tmp_path):
        """Validate on read, so a bad seed surfaces when it is installed
        rather than inside an unattended cron run."""
        p = tmp_path / "totp"
        p.write_text("not-base32-!!!")
        p.chmod(0o600)
        with pytest.raises(ValueError, match="base32"):
            load_totp_secret(p, env_var="_ABSENT_")

    def test_secret_is_never_echoed_in_an_error(self, tmp_path):
        p = tmp_path / "totp"
        p.write_text(RFC_SEED)
        p.chmod(0o644)
        try:
            load_totp_secret(p, env_var="_ABSENT_")
        except PermissionError as e:
            assert RFC_SEED not in str(e), "the seed leaked into an exception"


class TestAutoLoginCredentialGuards:
    def _mod(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "kite_auto_login",
            Path(__file__).parent.parent / "scripts" / "kite_auto_login.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_loose_credential_file_is_refused(self, tmp_path, monkeypatch):
        for k in ("KITE_API_KEY", "KITE_API_SECRET", "KITE_USER_ID",
                  "KITE_PASSWORD", "KITE_TOTP_SECRET"):
            monkeypatch.delenv(k, raising=False)
        m = self._mod()
        p = tmp_path / "auth.env"
        p.write_text("KITE_API_KEY=k\nKITE_API_SECRET=s\nKITE_USER_ID=u\n"
                     "KITE_PASSWORD=p\nKITE_TOTP_SECRET=" + RFC_SEED + "\n")
        p.chmod(0o644)
        with pytest.raises(PermissionError, match="beyond its owner"):
            m.load_auth(str(p))

    def test_missing_fields_are_named(self, tmp_path, monkeypatch):
        for k in ("KITE_API_KEY", "KITE_API_SECRET", "KITE_USER_ID",
                  "KITE_PASSWORD", "KITE_TOTP_SECRET"):
            monkeypatch.delenv(k, raising=False)
        m = self._mod()
        p = tmp_path / "auth.env"
        p.write_text("KITE_API_KEY=k\n")
        p.chmod(0o600)
        from nightevolver.kite import KiteAuthError
        with pytest.raises(KiteAuthError, match="KITE_PASSWORD"):
            m.load_auth(str(p))

    def test_complete_mode_600_file_loads(self, tmp_path, monkeypatch):
        for k in ("KITE_API_KEY", "KITE_API_SECRET", "KITE_USER_ID",
                  "KITE_PASSWORD", "KITE_TOTP_SECRET"):
            monkeypatch.delenv(k, raising=False)
        m = self._mod()
        p = tmp_path / "auth.env"
        p.write_text("KITE_API_KEY=k\nKITE_API_SECRET=s\nKITE_USER_ID=u\n"
                     "KITE_PASSWORD=p\nKITE_TOTP_SECRET=" + RFC_SEED + "\n")
        p.chmod(0o600)
        creds = m.load_auth(str(p))
        assert creds["KITE_USER_ID"] == "u"

    def test_written_token_file_is_mode_600(self, tmp_path):
        m = self._mod()
        out = tmp_path / "kite.env"
        m.write_env(str(out), "key", "tok")
        assert stat.S_IMODE(out.stat().st_mode) == 0o600
        assert "KITE_API_SECRET" not in out.read_text(), \
            "api_secret must never reach the token file"
