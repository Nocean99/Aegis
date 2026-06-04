from __future__ import annotations

import json
import mimetypes
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).parent.resolve()
STATIC = ROOT / "static"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class AnalystHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/reports":
            self._send_json({"reports": list_reports()})
            return
        if parsed.path == "/api/report":
            report_path = first_query_value(parsed.query, "path")
            self._send_json(load_report_payload(report_path))
            return
        if parsed.path == "/api/file":
            file_path = first_query_value(parsed.query, "path")
            self._send_file(file_path)
            return
        if parsed.path == "/":
            self.path = "/static/analyst.html"
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
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

    def translate_path(self, path: str) -> str:
        clean_path = urlparse(path).path.lstrip("/")
        return str(ROOT / clean_path)

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
        reports.append(
            {
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
            }
        )
    return reports


def load_report_payload(report_path: str | None) -> dict:
    path = resolve_local_path(report_path)
    if path is None or path.name != "vision_report.json" or not path.exists():
        return {"ok": False, "error": "Report not found"}
    data = json.loads(path.read_text(encoding="utf-8"))
    reviews = load_reviews(path)
    return {"ok": True, "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "report": data, "reviews": reviews}


def save_review(payload: dict) -> dict:
    report_path = resolve_local_path(str(payload.get("report_path", "")))
    if report_path is None or report_path.name != "vision_report.json" or not report_path.exists():
        return {"ok": False, "error": "Report not found"}
    candidate_key = str(payload.get("candidate_key") or "")
    if not candidate_key:
        return {"ok": False, "error": "candidate_key is required"}
    status = str(payload.get("status") or "unreviewed")
    notes = str(payload.get("notes") or "")
    reviews = load_reviews(report_path)
    reviews[candidate_key] = {"status": status, "notes": notes}
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


def resolve_local_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    decoded = unquote(path_value)
    path = Path(decoded)
    if not path.is_absolute():
        path = ROOT / decoded
    return path.resolve()


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
    server = ThreadingHTTPServer(("127.0.0.1", port), AnalystHandler)
    print(f"Analyst dashboard running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
