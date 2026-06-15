from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.dashboard_auth import (
    LoginRateLimiter,
    SessionStore,
    hash_password,
    load_auth_config,
    parse_cookie_header,
    verify_password,
)


def test_password_hash_round_trip() -> None:
    stored = hash_password("correct horse battery staple", iterations=1000)
    assert stored.startswith("pbkdf2_sha256$1000$")
    assert verify_password("correct horse battery staple", stored)
    assert not verify_password("wrong password", stored)


def test_two_hashes_of_same_password_differ() -> None:
    a = hash_password("secret", iterations=1000)
    b = hash_password("secret", iterations=1000)
    assert a != b  # unique salt per hash
    assert verify_password("secret", a) and verify_password("secret", b)


def test_verify_rejects_garbage_stored_values() -> None:
    for bad in ("", "plaintext", "md5$1$aa$bb", "pbkdf2_sha256$x$zz$qq", None):
        assert not verify_password("anything", bad)


def test_hash_rejects_empty_password() -> None:
    try:
        hash_password("")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for empty password")


def test_auth_disabled_by_default() -> None:
    config = load_auth_config(env={})
    assert config.enabled is False


def test_auth_enabled_by_password_env() -> None:
    config = load_auth_config(env={"ANALYST_PASSWORD": "hunter2", "ANALYST_USERNAME": "noah"})
    assert config.enabled is True
    assert config.username == "noah"
    assert verify_password("hunter2", config.password_hash)


def test_forced_auth_without_credential_fails_closed() -> None:
    try:
        load_auth_config(env={"ANALYST_AUTH": "1"})
    except ValueError:
        return
    raise AssertionError("ANALYST_AUTH=1 without a credential must refuse to start")


def test_session_create_validate_revoke() -> None:
    store = SessionStore(ttl_s=60)
    token = store.create("analyst")
    assert store.validate(token) == "analyst"
    store.revoke(token)
    assert store.validate(token) is None
    assert store.validate(None) is None
    assert store.validate("never-issued") is None


def test_session_expires_after_ttl() -> None:
    now = [0.0]
    store = SessionStore(ttl_s=10, clock=lambda: now[0])
    token = store.create("analyst")
    now[0] = 9.9
    assert store.validate(token) == "analyst"
    now[0] = 10.1
    assert store.validate(token) is None
    assert store.active_count() == 0


def test_rate_limiter_blocks_after_max_failures() -> None:
    now = [0.0]
    limiter = LoginRateLimiter(max_attempts=3, window_s=300, clock=lambda: now[0])
    for _ in range(3):
        assert limiter.allow("10.0.0.1")
        limiter.record_failure("10.0.0.1")
    assert not limiter.allow("10.0.0.1")
    assert limiter.allow("10.0.0.2")  # other clients unaffected
    now[0] = 301.0  # window expires
    assert limiter.allow("10.0.0.1")


def test_rate_limiter_reset_on_success() -> None:
    limiter = LoginRateLimiter(max_attempts=1, window_s=300)
    limiter.record_failure("10.0.0.1")
    assert not limiter.allow("10.0.0.1")
    limiter.reset("10.0.0.1")
    assert limiter.allow("10.0.0.1")


def test_parse_cookie_header() -> None:
    cookies = parse_cookie_header("a=1; analyst_session=tok; b=2")
    assert cookies["analyst_session"] == "tok"
    assert parse_cookie_header(None) == {}
    assert parse_cookie_header("malformed") == {}


if __name__ == "__main__":
    tests = [
        test_password_hash_round_trip,
        test_two_hashes_of_same_password_differ,
        test_verify_rejects_garbage_stored_values,
        test_hash_rejects_empty_password,
        test_auth_disabled_by_default,
        test_auth_enabled_by_password_env,
        test_forced_auth_without_credential_fails_closed,
        test_session_create_validate_revoke,
        test_session_expires_after_ttl,
        test_rate_limiter_blocks_after_max_failures,
        test_rate_limiter_reset_on_success,
        test_parse_cookie_header,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
