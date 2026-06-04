from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyst_server


def test_load_report_payload_and_save_review() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        logs = root / "logs" / "vision_lab" / "run"
        logs.mkdir(parents=True)
        report_path = logs / "vision_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "timestamp": "test",
                    "mission_request": "Search for people",
                    "summary": {"processed": 1, "detections": 1, "shortlist_count": 1},
                    "evaluation": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                    "results": [],
                }
            ),
            encoding="utf-8",
        )
        old_root = analyst_server.ROOT
        try:
            analyst_server.ROOT = root
            payload = analyst_server.load_report_payload("logs/vision_lab/run/vision_report.json")
            assert payload["ok"] is True
            result = analyst_server.save_review(
                {
                    "report_path": "logs/vision_lab/run/vision_report.json",
                    "candidate_key": "image.jpg::",
                    "status": "approved",
                    "notes": "Looks relevant",
                }
            )
            assert result["ok"] is True
            reviews = analyst_server.load_reviews(report_path)
            assert reviews["image.jpg::"]["status"] == "approved"
        finally:
            analyst_server.ROOT = old_root


if __name__ == "__main__":
    tests = [test_load_report_payload_and_save_review]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
