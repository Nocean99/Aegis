from __future__ import annotations

"""Request metrics for the analyst dashboard.

A small, dependency-free registry exposing Prometheus text format at
/metrics: request counts by method/route/status, latency sum+count per route
(enough for an average; histograms can come later if anything ever needs
them), failed logins, and process uptime. Thread-safe because the dashboard
serves from a ThreadingHTTPServer.
"""

import threading
import time


# Known route prefixes, longest match first. Everything else is "other"
# so unbounded user-supplied paths can never explode label cardinality.
ROUTE_PREFIXES = [
    "/api/reports",
    "/api/report",
    "/api/mission-plan",
    "/api/mission-memory",
    "/api/review",
    "/api/file",
    "/healthz",
    "/metrics",
    "/login",
    "/logout",
    "/static",
]


def route_group(path: str) -> str:
    clean = (path or "/").split("?")[0]
    if clean == "/":
        return "/"
    for prefix in ROUTE_PREFIXES:
        if clean == prefix or clean.startswith(prefix + "/"):
            return prefix
    return "other"


class MetricsRegistry:
    def __init__(self, *, clock=time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._started_at = clock()
        self._requests: dict[tuple[str, str, int], int] = {}
        self._latency_sum_ms: dict[tuple[str, str], float] = {}
        self._latency_count: dict[tuple[str, str], int] = {}
        self._auth_failures = 0

    def observe_request(self, method: str, path: str, status: int, duration_ms: float) -> None:
        route = route_group(path)
        with self._lock:
            key = (method, route, int(status))
            self._requests[key] = self._requests.get(key, 0) + 1
            latency_key = (method, route)
            self._latency_sum_ms[latency_key] = (
                self._latency_sum_ms.get(latency_key, 0.0) + duration_ms
            )
            self._latency_count[latency_key] = self._latency_count.get(latency_key, 0) + 1

    def record_auth_failure(self) -> None:
        with self._lock:
            self._auth_failures += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_s": round(self._clock() - self._started_at, 1),
                "requests_total": sum(self._requests.values()),
                "auth_failures_total": self._auth_failures,
            }

    def render(self, *, active_sessions: int | None = None) -> str:
        lines = [
            "# HELP analyst_uptime_seconds Seconds since the dashboard started.",
            "# TYPE analyst_uptime_seconds gauge",
        ]
        with self._lock:
            lines.append(f"analyst_uptime_seconds {self._clock() - self._started_at:.1f}")
            lines += [
                "# HELP analyst_requests_total HTTP requests by method, route, status.",
                "# TYPE analyst_requests_total counter",
            ]
            for (method, route, status), count in sorted(self._requests.items()):
                lines.append(
                    f'analyst_requests_total{{method="{method}",route="{route}",status="{status}"}} {count}'
                )
            lines += [
                "# HELP analyst_request_duration_ms_sum Total request time by method, route.",
                "# TYPE analyst_request_duration_ms_sum counter",
            ]
            for (method, route), total in sorted(self._latency_sum_ms.items()):
                lines.append(
                    f'analyst_request_duration_ms_sum{{method="{method}",route="{route}"}} {total:.1f}'
                )
            lines += [
                "# HELP analyst_request_duration_ms_count Request count by method, route.",
                "# TYPE analyst_request_duration_ms_count counter",
            ]
            for (method, route), count in sorted(self._latency_count.items()):
                lines.append(
                    f'analyst_request_duration_ms_count{{method="{method}",route="{route}"}} {count}'
                )
            lines += [
                "# HELP analyst_auth_failures_total Failed login attempts.",
                "# TYPE analyst_auth_failures_total counter",
                f"analyst_auth_failures_total {self._auth_failures}",
            ]
        if active_sessions is not None:
            lines += [
                "# HELP analyst_active_sessions Live authenticated sessions.",
                "# TYPE analyst_active_sessions gauge",
                f"analyst_active_sessions {active_sessions}",
            ]
        return "\n".join(lines) + "\n"
