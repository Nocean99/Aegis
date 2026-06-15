from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from autonomy.contextual_search_plan import create_contextual_search_plan
from autonomy.dashboard_auth import (
    SESSION_COOKIE,
    LoginRateLimiter,
    SessionStore,
    load_auth_config,
    parse_cookie_header,
    verify_password,
)
from autonomy.dashboard_metrics import MetricsRegistry
from autonomy.mission_command import create_mission_command
from autonomy.mission_memory import build_mission_memory
from autonomy.mission_vision_plan import create_mission_vision_plan


ROOT = Path(__file__).parent.resolve()
STATIC = ROOT / "static"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPORT_NAMES = {"vision_report.json", "acoustic_report.json"}

AUTH_CONFIG = load_auth_config()
SESSIONS = SessionStore(ttl_s=AUTH_CONFIG.session_ttl_s)
RATE_LIMITER = LoginRateLimiter()
METRICS = MetricsRegistry()

# Paths that never require a session: health checks must work for load
# balancers, and the login flow must be reachable to authenticate at all.
PUBLIC_PATHS = {"/healthz", "/login", "/favicon.ico"}

LOGIN_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Aegis Analyst Login</title>
<style>
body{font-family:system-ui,sans-serif;background:#0e1116;color:#e6e6e6;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
form{background:#161b22;padding:2rem;border-radius:8px;min-width:280px}
h1{font-size:1.1rem;margin:0 0 1rem}
input{display:block;width:100%;box-sizing:border-box;margin:.5rem 0;
padding:.55rem;border-radius:4px;border:1px solid #30363d;
background:#0e1116;color:#e6e6e6}
button{width:100%;padding:.6rem;margin-top:.5rem;border:0;border-radius:4px;
background:#2f81f7;color:#fff;font-weight:600;cursor:pointer}
.err{color:#f85149;font-size:.85rem;min-height:1.2em}
</style></head><body>
<form method="post" action="/login">
<h1>Aegis Mission Intelligence — Analyst Login</h1>
<div class="err">__ERROR__</div>
<input name="username" placeholder="Username" autocomplete="username" required>
<input name="password" type="password" placeholder="Password" autocomplete="current-password" required>
<button type="submit">Sign in</button>
</form></body></html>"""


class AnalystHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def _handle(self, method: str) -> None:
        start = time.monotonic()
        self._status_code = 0
        try:
            if method == "GET":
                self._route_get()
            else:
                self._route_post()
        finally:
            duration_ms = (time.monotonic() - start) * 1000.0
            METRICS.observe_request(method, self.path, self._status_code or 0, duration_ms)
            self._log_json(method, duration_ms)

    def _route_get(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._send_json({"ok": True, **METRICS.snapshot()})
            return
        if parsed.path == "/login":
            self._send_login_page()
            return
        if not self._authorized(parsed.path):
            return
        if parsed.path == "/metrics":
            self._send_text(METRICS.render(active_sessions=SESSIONS.active_count()))
            return
        if parsed.path == "/api/reports":
            self._send_json({"reports": list_reports()})
            return
        if parsed.path == "/api/report":
            report_path = first_query_value(parsed.query, "path")
            self._send_json(load_report_payload(report_path))
            return
        if parsed.path == "/api/mission-memory":
            self._send_json({"ok": True, "memory": build_mission_memory(ROOT)})
            return
        if parsed.path == "/api/file":
            file_path = first_query_value(parsed.query, "path")
            self._send_file(file_path)
            return
        if parsed.path == "/":
            self.path = "/static/analyst.html"
        super().do_GET()

    def _route_post(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/login":
            self._handle_login()
            return
        if parsed.path == "/logout":
            self._handle_logout()
            return
        if not self._authorized(parsed.path):
            return
        if parsed.path == "/api/mission-plan":
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                result = create_mission_plan_payload(payload)
                self._send_json(result, 200 if result.get("ok") else 400)
            except json.JSONDecodeError:
                self._send_json({"ok": False, "error": "Invalid JSON"}, 400)
            return
        if parsed.path != "/api/review":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            result = save_review(payload)
            self._send_json(result, 200 if result.get("ok") else 400)
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "Invalid JSON"}, 400)

    # -- auth -----------------------------------------------------------------

    def _authorized(self, path: str) -> bool:
        """True when the request may proceed. Sends the response when not."""
        if not AUTH_CONFIG.enabled or path in PUBLIC_PATHS:
            return True
        token = parse_cookie_header(self.headers.get("Cookie")).get(SESSION_COOKIE)
        if SESSIONS.validate(token):
            return True
        if path.startswith("/api/") or path == "/metrics":
            self._send_json({"ok": False, "error": "Authentication required"}, 401)
        else:
            self.send_response(302)
            self.send_header("Location", "/login")
            self.send_header("Content-Length", "0")
            self.end_headers()
        return False

    def _handle_login(self) -> None:
        client = self.client_address[0]
        if not RATE_LIMITER.allow(client):
            METRICS.record_auth_failure()
            self._send_login_page(error="Too many attempts. Try again in a few minutes.", status=429)
            return
        length = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        username = (form.get("username") or [""])[0]
        password = (form.get("password") or [""])[0]
        if (
            AUTH_CONFIG.enabled
            and username == AUTH_CONFIG.username
            and verify_password(password, AUTH_CONFIG.password_hash)
        ):
            RATE_LIMITER.reset(client)
            token = SESSIONS.create(username)
            cookie = (
                f"{SESSION_COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/"
                f"; Max-Age={AUTH_CONFIG.session_ttl_s}"
            )
            if AUTH_CONFIG.cookie_secure:
                cookie += "; Secure"
            self.send_response(302)
            self.send_header("Set-Cookie", cookie)
            self.send_header("Location", "/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        RATE_LIMITER.record_failure(client)
        METRICS.record_auth_failure()
        self._send_login_page(error="Invalid username or password.", status=401)

    def _handle_logout(self) -> None:
        token = parse_cookie_header(self.headers.get("Cookie")).get(SESSION_COOKIE)
        SESSIONS.revoke(token)
        self.send_response(302)
        self.send_header("Set-Cookie", f"{SESSION_COOKIE}=; Max-Age=0; Path=/")
        self.send_header("Location", "/login")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_login_page(self, *, error: str = "", status: int = 200) -> None:
        if not AUTH_CONFIG.enabled:
            self.send_response(302)
            self.send_header("Location", "/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = LOGIN_PAGE.replace("__ERROR__", error).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- observability ----------------------------------------------------------

    def send_response(self, code: int, message: str | None = None) -> None:
        self._status_code = code
        super().send_response(code, message)

    def log_message(self, format: str, *args) -> None:
        # Default per-line logging is replaced by the structured log in _handle.
        pass

    def _log_json(self, method: str, duration_ms: float) -> None:
        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": "info" if (self._status_code or 0) < 500 else "error",
            "method": method,
            "path": self.path.split("?")[0],
            "status": self._status_code or 0,
            "duration_ms": round(duration_ms, 1),
            "client": self.client_address[0],
        }
        print(json.dumps(record), flush=True)

    def _send_text(self, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def translate_path(self, path: str) -> str:
        clean_path = urlparse(path).path.lstrip("/")
        candidate = (ROOT / clean_path).resolve()
        if not candidate.is_relative_to(ROOT):
            # Refuse traversal outside the project root; resolves to a 404.
            return str(ROOT / "static" / "__forbidden__")
        return str(candidate)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, body: dict, status: int = 200) -> None:
        encoded = json.dumps(body, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, file_path: str | None) -> None:
        path = resolve_local_path(file_path)
        if path is None or not path.exists() or path.suffix.lower() not in IMAGE_SUFFIXES:
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def list_reports() -> list[dict]:
    reports = []
    for report_path in sorted((ROOT / "logs").glob("**/vision_report.json"), reverse=True):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = data.get("summary") or {}
        evaluation = data.get("evaluation") or {}
        analyst_capture = evaluation.get("analyst_capture") or {}
        reports.append(
            {
                "type": "vision",
                "path": str(report_path.relative_to(ROOT)),
                "timestamp": data.get("timestamp"),
                "mission_request": data.get("mission_request"),
                "proposal_mode": data.get("proposal_mode"),
                "scorer": data.get("scorer"),
                "processed": summary.get("processed"),
                "detections": summary.get("detections"),
                "shortlist_count": summary.get("shortlist_count"),
                "precision": evaluation.get("precision"),
                "recall": evaluation.get("recall"),
                "f1": evaluation.get("f1"),
                "capture_recall": analyst_capture.get("recall"),
            }
        )
    for report_path in sorted((ROOT / "logs").glob("**/acoustic_report.json"), reverse=True):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = data.get("summary") or {}
        evaluation = data.get("evaluation") or {}
        reports.append(
            {
                "type": "acoustic",
                "path": str(report_path.relative_to(ROOT)),
                "timestamp": report_path.parent.name,
                "mission_request": data.get("mission_request"),
                "proposal_mode": "acoustic",
                "scorer": "local-acoustic-proposal-v1",
                "processed": summary.get("processed"),
                "detections": summary.get("candidate_count"),
                "shortlist_count": summary.get("candidate_count"),
                "precision": evaluation.get("capture_precision"),
                "recall": evaluation.get("capture_recall"),
                "f1": evaluation.get("capture_f1"),
                "capture_recall": evaluation.get("capture_recall"),
            }
        )
    return reports


def create_mission_plan_payload(payload: dict) -> dict:
    request = str(payload.get("mission_request") or "").strip()
    if not request:
        return {"ok": False, "error": "mission_request is required"}
    mode = str(payload.get("operating_mode") or "connected-supervised")
    try:
        command = create_mission_command(request, operating_mode=mode)
        vision_plan = create_mission_vision_plan(command.objective)
        contextual_search_plan = create_contextual_search_plan(command.objective)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "mission_request": request,
        "command": asdict(command),
        "vision_plan": asdict(vision_plan),
        "contextual_search_plan": asdict(contextual_search_plan),
        "next_actions": [
            "Run a vision-only benchmark against image or video evidence.",
            "Review shortlisted candidates in the analyst dashboard.",
            "Use PX4/Gazebo only when validating a moving sensor platform.",
        ],
    }


def load_report_payload(report_path: str | None) -> dict:
    path = resolve_local_path(report_path)
    if path is None or path.name not in REPORT_NAMES or not path.exists():
        return {"ok": False, "error": "Report not found"}
    data = json.loads(path.read_text(encoding="utf-8"))
    reviews = load_reviews(path)
    return {"ok": True, "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "report": data, "reviews": reviews}


def save_review(payload: dict) -> dict:
    report_path = resolve_local_path(str(payload.get("report_path", "")))
    if report_path is None or report_path.name not in REPORT_NAMES or not report_path.exists():
        return {"ok": False, "error": "Report not found"}
    candidate_key = str(payload.get("candidate_key") or "")
    if not candidate_key:
        return {"ok": False, "error": "candidate_key is required"}
    decision = normalize_decision(payload.get("decision") or payload.get("status") or "investigate")
    reason_tag = normalize_reason_tag(payload.get("reason_tag"))
    reason = str(payload.get("reason") or "").strip()
    if not reason and reason_tag:
        reason = reason_tag.replace("_", " ")
    notes = str(payload.get("notes") or "")
    reviews = load_reviews(report_path)
    reviews[candidate_key] = {
        "candidate_id": str(payload.get("candidate_id") or candidate_key),
        "decision": decision,
        "status": decision,
        "reason_tag": reason_tag,
        "reason": reason,
        "notes": notes,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    review_path(report_path).write_text(json.dumps(reviews, indent=2), encoding="utf-8")
    return {"ok": True, "reviews": reviews}


def load_reviews(report_path: Path) -> dict:
    path = review_path(report_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def review_path(report_path: Path) -> Path:
    return report_path.with_name("candidate_reviews.json")


def normalize_decision(value) -> str:
    decision = str(value or "").strip().lower()
    aliases = {
        "approved": "approve",
        "confirmed": "approve",
        "rejected": "reject",
        "needs_closer_look": "investigate",
        "needs closer look": "investigate",
    }
    decision = aliases.get(decision, decision)
    if decision not in {"approve", "reject", "investigate"}:
        return "investigate"
    return decision


def normalize_reason_tag(value) -> str:
    tag = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    allowed = {
        "person_visible",
        "vehicle_visible",
        "too_small",
        "vegetation",
        "shadow",
        "debris",
        "rooftop",
        "road_marking",
        "building",
        "hot_object",
        "thermal_clutter",
        "wave_noise",
        "dock_machinery",
        "wind_noise",
        "low_snr",
        "overlapping_signals",
        "vessel_sound",
        "acoustic_clutter",
        "false_alarm",
        "uncertain_vehicle",
    }
    return tag if tag in allowed else ""


def allowed_roots() -> list[Path]:
    """Directories the dashboard is allowed to serve files from.

    Defaults to the project root. Extra dataset directories can be allowed via
    the ANALYST_ALLOWED_DIRS environment variable (os.pathsep-separated).
    """
    roots = [ROOT]
    for part in os.environ.get("ANALYST_ALLOWED_DIRS", "").split(os.pathsep):
        cleaned = part.strip()
        if cleaned:
            roots.append(Path(cleaned).expanduser())
    return roots


def resolve_local_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    decoded = unquote(path_value)
    path = Path(decoded)
    if not path.is_absolute():
        path = ROOT / decoded
    try:
        resolved = path.resolve()
    except OSError:
        return None
    for root in allowed_roots():
        try:
            if resolved.is_relative_to(root.resolve()):
                return resolved
        except OSError:
            continue
    return None


def first_query_value(query: str, key: str) -> str | None:
    values = parse_qs(query).get(key)
    return values[0] if values else None


def _port_from_args() -> int:
    if "--port" in sys.argv:
        index = sys.argv.index("--port")
        if index + 1 >= len(sys.argv):
            raise SystemExit("--port requires a value")
        return int(sys.argv[index + 1])
    return int(os.environ.get("ANALYST_DASHBOARD_PORT", "8010"))


def main() -> None:
    port = _port_from_args()
    host = os.environ.get("ANALYST_BIND", "127.0.0.1")
    server = ThreadingHTTPServer((host, port), AnalystHandler)
    auth_state = "enabled" if AUTH_CONFIG.enabled else "disabled (set ANALYST_PASSWORD_HASH to enable)"
    print(f"Analyst dashboard running at http://{host}:{port} | auth: {auth_state}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
