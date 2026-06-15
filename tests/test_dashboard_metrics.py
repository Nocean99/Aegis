from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.dashboard_metrics import MetricsRegistry, route_group


def test_route_group_maps_known_routes() -> None:
    assert route_group("/") == "/"
    assert route_group("/api/reports") == "/api/reports"
    assert route_group("/api/report?path=logs/x/vision_report.json") == "/api/report"
    assert route_group("/static/analyst.html") == "/static"
    assert route_group("/healthz") == "/healthz"


def test_route_group_collapses_unknown_paths() -> None:
    # User-controlled paths must not create unbounded label values.
    assert route_group("/etc/passwd") == "other"
    assert route_group("/anything/else/at/all") == "other"


def test_observe_and_render_prometheus_text() -> None:
    registry = MetricsRegistry(clock=lambda: 100.0)
    registry.observe_request("GET", "/api/reports", 200, 12.5)
    registry.observe_request("GET", "/api/reports", 200, 7.5)
    registry.observe_request("POST", "/api/review", 400, 3.0)
    registry.record_auth_failure()
    text = registry.render(active_sessions=2)
    assert 'analyst_requests_total{method="GET",route="/api/reports",status="200"} 2' in text
    assert 'analyst_requests_total{method="POST",route="/api/review",status="400"} 1' in text
    assert 'analyst_request_duration_ms_sum{method="GET",route="/api/reports"} 20.0' in text
    assert "analyst_auth_failures_total 1" in text
    assert "analyst_active_sessions 2" in text


def test_snapshot_totals() -> None:
    now = [50.0]
    registry = MetricsRegistry(clock=lambda: now[0])
    registry.observe_request("GET", "/", 200, 1.0)
    now[0] = 60.0
    snapshot = registry.snapshot()
    assert snapshot["requests_total"] == 1
    assert snapshot["auth_failures_total"] == 0
    assert snapshot["uptime_s"] == 10.0


if __name__ == "__main__":
    tests = [
        test_route_group_maps_known_routes,
        test_route_group_collapses_unknown_paths,
        test_observe_and_render_prometheus_text,
        test_snapshot_totals,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
