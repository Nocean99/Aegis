from __future__ import annotations

"""Session authentication for the analyst dashboard.

Auth is disabled by default so local development stays frictionless. It turns
on when ANALYST_PASSWORD or ANALYST_PASSWORD_HASH is set (or ANALYST_AUTH=1,
which refuses to start without a credential). Passwords are stored as salted
PBKDF2-SHA256 hashes; sessions are random tokens in an in-memory store with a
TTL; failed logins are rate-limited per client address.

Generate a hash for deployment (never put a plain password in a unit file):

    python3 -m autonomy.dashboard_auth hash 'your-password-here'

Environment variables:
    ANALYST_AUTH=1                 force auth on (fails closed without a credential)
    ANALYST_USERNAME               login username (default: analyst)
    ANALYST_PASSWORD               plain password, hashed at startup (dev convenience)
    ANALYST_PASSWORD_HASH          pbkdf2_sha256$<iter>$<salt_hex>$<hash_hex>
    ANALYST_SESSION_TTL_S          session lifetime, default 28800 (8 hours)
    ANALYST_COOKIE_SECURE=1        mark the session cookie Secure (behind HTTPS)
"""

import hashlib
import hmac
import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass


PBKDF2_ITERATIONS = 310_000
SESSION_COOKIE = "analyst_session"
DEFAULT_SESSION_TTL_S = 8 * 60 * 60
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_ATTEMPT_WINDOW_S = 300.0


@dataclass(frozen=True)
class AuthConfig:
    enabled: bool
    username: str = "analyst"
    password_hash: str = ""
    session_ttl_s: int = DEFAULT_SESSION_TTL_S
    cookie_secure: bool = False


def load_auth_config(env: dict | None = None) -> AuthConfig:
    env = dict(os.environ if env is None else env)
    forced = env.get("ANALYST_AUTH", "").strip() == "1"
    password = env.get("ANALYST_PASSWORD", "")
    password_hash = env.get("ANALYST_PASSWORD_HASH", "").strip()
    if password and not password_hash:
        password_hash = hash_password(password)
    enabled = forced or bool(password_hash)
    if forced and not password_hash:
        raise ValueError(
            "ANALYST_AUTH=1 but no credential is configured. "
            "Set ANALYST_PASSWORD_HASH (preferred) or ANALYST_PASSWORD."
        )
    return AuthConfig(
        enabled=enabled,
        username=env.get("ANALYST_USERNAME", "analyst").strip() or "analyst",
        password_hash=password_hash,
        session_ttl_s=int(env.get("ANALYST_SESSION_TTL_S", str(DEFAULT_SESSION_TTL_S))),
        cookie_secure=env.get("ANALYST_COOKIE_SECURE", "").strip() == "1",
    )


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    if not password:
        raise ValueError("Password must not be empty.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations_text, salt_hex, hash_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(digest, expected)


class SessionStore:
    """In-memory session tokens with TTL. Thread-safe.

    In-memory is a deliberate choice for a single-process dashboard: restart
    invalidates all sessions, which is acceptable (analysts just log in
    again) and removes a persistence attack surface.
    """

    def __init__(self, *, ttl_s: int = DEFAULT_SESSION_TTL_S, clock=time.monotonic) -> None:
        self.ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[str, tuple[str, float]] = {}  # token -> (username, expires)

    def create(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[token] = (username, self._clock() + self.ttl_s)
        return token

    def validate(self, token: str | None) -> str | None:
        """Return the username for a live session, or None."""
        if not token:
            return None
        now = self._clock()
        with self._lock:
            entry = self._sessions.get(token)
            if entry is None:
                return None
            username, expires = entry
            if now >= expires:
                del self._sessions[token]
                return None
            return username

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def active_count(self) -> int:
        now = self._clock()
        with self._lock:
            self._sessions = {
                token: entry for token, entry in self._sessions.items() if entry[1] > now
            }
            return len(self._sessions)


class LoginRateLimiter:
    """Per-client failed-login limiter: max_attempts failures per window."""

    def __init__(
        self,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        window_s: float = DEFAULT_ATTEMPT_WINDOW_S,
        clock=time.monotonic,
    ) -> None:
        self.max_attempts = max_attempts
        self.window_s = window_s
        self._clock = clock
        self._lock = threading.Lock()
        self._failures: dict[str, list[float]] = {}

    def allow(self, client: str) -> bool:
        now = self._clock()
        with self._lock:
            recent = [t for t in self._failures.get(client, []) if now - t < self.window_s]
            self._failures[client] = recent
            return len(recent) < self.max_attempts

    def record_failure(self, client: str) -> None:
        with self._lock:
            self._failures.setdefault(client, []).append(self._clock())

    def reset(self, client: str) -> None:
        with self._lock:
            self._failures.pop(client, None)


def parse_cookie_header(header: str | None) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for part in (header or "").split(";"):
        name, _, value = part.strip().partition("=")
        if name and value:
            cookies[name] = value
    return cookies


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "hash":
        if len(sys.argv) >= 3:
            password = sys.argv[2]
        else:
            import getpass

            password = getpass.getpass("Password to hash: ")
        print(hash_password(password))
        return
    print("Usage: python3 -m autonomy.dashboard_auth hash [password]")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
