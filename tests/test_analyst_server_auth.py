from __future__ import annotations

"""Integration tests: boot the real dashboard with auth enabled and exercise
the login flow, protected routes, health, and metrics over actual HTTP."""

import http.client
import importlib
import os
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Configure auth BEFORE importing the server module (config loads at import).
os.environ["ANALYST_PASSWORD"] = "test-password-123"
os.environ["ANALYST_USERNAME"] = "tester"

import analyst_server  # noqa: E402

analyst_server = importlib.reload(analyst_server)


def start_server() -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), analyst_server.AnalystHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def request(
    port: int,
    method: str,
    path: str,
    *,
    body: str | None = None,
    headers: dict | None = None,
):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response, payload


def login(port: int, username: str, password: str):
    response, _ = request(
        port,
        "POST",
        "/login",
        body=f"username={username}&password={password}",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    cookie = response.getheader("Set-Cookie") or ""
    return response, cookie.split(";")[0]


def test_auth_flow() -> None:
    assert analyst_server.AUTH_CONFIG.enabled, "Auth must be enabled for this test module"
    server, port = start_server()
    try:
        # Unauthenticated browser requests redirect to /login.
        response, _ = request(port, "GET", "/")
        assert response.status == 302
        assert response.getheader("Location") == "/login"

        # Unauthenticated API requests get 401 JSON, not a redirect.
        response, payload = request(port, "GET", "/api/reports")
        assert response.status == 401
        assert b"Authentication required" in payload

        # /healthz stays public for load balancers.
        response, payload = request(port, "GET", "/healthz")
        assert response.status == 200
        assert b'"ok": true' in payload

        # Wrong password is rejected and counted.
        response, _ = login(port, "tester", "wrong")
        assert response.status == 401

        # Correct login sets a hardened session cookie and redirects home.
        response, cookie = login(port, "tester", "test-password-123")
        assert response.status == 302
        set_cookie = response.getheader("Set-Cookie")
        assert "HttpOnly" in set_cookie and "SameSite=Strict" in set_cookie

        # The session opens protected routes.
        response, _ = request(port, "GET", "/api/reports", headers={"Cookie": cookie})
        assert response.status == 200

        # Metrics is protected, reports the failed attempt, and renders.
        response, payload = request(port, "GET", "/metrics")
        assert response.status == 401
        response, payload = request(port, "GET", "/metrics", headers={"Cookie": cookie})
        assert response.status == 200
        text = payload.decode("utf-8")
        assert "analyst_auth_failures_total" in text
        assert "analyst_requests_total" in text

        # Logout revokes the session.
        response, _ = request(port, "POST", "/logout", headers={"Cookie": cookie})
        assert response.status == 302
        response, _ = request(port, "GET", "/api/reports", headers={"Cookie": cookie})
        assert response.status == 401
    finally:
        server.shutdown()
        server.server_close()


def test_login_rate_limiting() -> None:
    server, port = start_server()
    try:
        last_status = None
        for _ in range(8):
            response, _ = login(port, "tester", "definitely-wrong")
            last_status = response.status
        assert last_status == 429  # locked out after repeated failures
    finally:
        server.shutdown()
        server.server_close()
        # Other tests in this process shouldn't inherit the lockout.
        analyst_server.RATE_LIMITER.reset("127.0.0.1")


if __name__ == "__main__":
    tests = [
        test_auth_flow,
        test_login_rate_limiting,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
