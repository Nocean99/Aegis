from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.cascade_ranking_benchmark import (
    CascadeClaim,
    CascadeTrial,
    load_ranked_candidates,
    parse_cascade_claim,
    run_cascade_benchmark,
)


class FakeReviewer:
    model_name = "fake-cascade-reviewer"

    def __init__(self, claims_by_candidate: dict[str, bool]) -> None:
        self.claims_by_candidate = claims_by_candidate
        self.calls: list[str] = []

    def review(self, *, target_description: str, image_path: str, candidate_id: str, rank: int) -> CascadeClaim:
        self.calls.append(candidate_id)
        return CascadeClaim(
            claimed_find=self.claims_by_candidate.get(candidate_id, False),
            confidence=0.9 if self.claims_by_candidate.get(candidate_id, False) else 0.1,
            explanation="fake claim",
        )


def write_report(path: Path) -> None:
    results = [
        candidate("rank-1-negative", path.parent / "negative_1.jpg", 0.95, False),
        candidate("rank-2-target-a", path.parent / "target_a.jpg", 0.90, True),
        candidate("rank-3-target-b", path.parent / "target_b.jpg", 0.80, True),
        candidate("rank-4-negative", path.parent / "negative_2.jpg", 0.70, False),
    ]
    for result in results:
        Path(result["image_path"]).write_bytes(b"not-real-image")
    path.write_text(json.dumps({"results": results}), encoding="utf-8")


def candidate(candidate_id: str, image_path: Path, priority: float, expected: bool) -> dict:
    return {
        "candidate_id": candidate_id,
        "image_path": str(image_path),
        "review_priority": priority,
        "final_score": priority,
        "label": {"expected_match": expected, "label": "positive" if expected else "negative"},
    }


def test_ranked_candidates_sort_by_review_priority() -> None:
    with TemporaryDirectory() as tmp:
        report = Path(tmp) / "vision_report.json"
        write_report(report)
        ranked = load_ranked_candidates(report)
        assert [item["candidate_id"] for item in ranked] == [
            "rank-1-negative",
            "rank-2-target-a",
            "rank-3-target-b",
            "rank-4-negative",
        ]


def test_cascade_reports_false_stop_and_true_rank() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        report = root / "vision_report.json"
        write_report(report)
        reviewer = FakeReviewer({"rank-1-negative": True, "rank-2-target-a": True})
        output = run_cascade_benchmark(
            report_path=report,
            trials=[
                CascadeTrial(
                    target_id="target-a",
                    target_image_path=str(root / "target_a.jpg"),
                    target_description="the target vehicle",
                )
            ],
            reviewer=reviewer,
            output_dir=root / "cascade",
        )
        data = json.loads(output.read_text(encoding="utf-8"))
        trial = data["trials"][0]
        assert trial["claimed_rank"] == 1
        assert trial["true_target_rank"] == 2
        assert trial["claim_correct"] is False
        assert trial["stopped_on_false_positive"] is True
        assert trial["api_calls_made"] == 1
        assert data["summary"]["false_positive_stops"] == 1


def test_cascade_reports_correct_late_find_and_api_calls() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        report = root / "vision_report.json"
        write_report(report)
        reviewer = FakeReviewer({"rank-3-target-b": True})
        output = run_cascade_benchmark(
            report_path=report,
            trials=[
                CascadeTrial(
                    target_id="target-b",
                    target_image_path=str(root / "target_b.jpg"),
                    target_description="another target vehicle",
                )
            ],
            reviewer=reviewer,
            output_dir=root / "cascade",
        )
        trial = json.loads(output.read_text(encoding="utf-8"))["trials"][0]
        assert trial["claimed_rank"] == 3
        assert trial["true_target_rank"] == 3
        assert trial["claim_correct"] is True
        assert trial["api_calls_made"] == 3


def test_parse_cascade_claim_defaults_invalid_json_to_no_claim() -> None:
    claim = parse_cascade_claim("not json")
    assert claim.claimed_find is False
    assert claim.confidence == 0.0


if __name__ == "__main__":
    tests = [
        test_ranked_candidates_sort_by_review_priority,
        test_cascade_reports_false_stop_and_true_rank,
        test_cascade_reports_correct_late_find_and_api_calls,
        test_parse_cascade_claim_defaults_invalid_json_to_no_claim,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
