from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.contact_tracker import ContactTracker, track_video_results


def test_overlapping_detections_share_a_track() -> None:
    tracker = ContactTracker()
    first = tracker.update(
        frame_index=0, timestamp_s=0.0, bbox=(100, 100, 50, 50), candidate_id="c1", score=0.4
    )
    second = tracker.update(
        frame_index=1, timestamp_s=1.0, bbox=(105, 102, 50, 50), candidate_id="c2", score=0.7
    )
    assert first == second
    assert len(tracker.tracks) == 1


def test_distant_detections_get_separate_tracks() -> None:
    tracker = ContactTracker()
    first = tracker.update(
        frame_index=0, timestamp_s=0.0, bbox=(0, 0, 30, 30), candidate_id="c1", score=0.4
    )
    second = tracker.update(
        frame_index=1, timestamp_s=1.0, bbox=(600, 400, 30, 30), candidate_id="c2", score=0.5,
        frame_size=(640, 480),
    )
    assert first != second
    assert len(tracker.tracks) == 2


def test_centroid_fallback_links_fast_small_target() -> None:
    # No IoU overlap, but centroid moved only ~6% of the frame diagonal.
    tracker = ContactTracker()
    first = tracker.update(
        frame_index=0, timestamp_s=0.0, bbox=(100, 100, 20, 20), candidate_id="c1", score=0.4,
        frame_size=(640, 480),
    )
    second = tracker.update(
        frame_index=1, timestamp_s=1.0, bbox=(140, 120, 20, 20), candidate_id="c2", score=0.5,
        frame_size=(640, 480),
    )
    assert first == second


def test_track_expires_after_max_gap_frames() -> None:
    tracker = ContactTracker(max_gap_frames=3)
    first = tracker.update(
        frame_index=0, timestamp_s=0.0, bbox=(100, 100, 50, 50), candidate_id="c1", score=0.4
    )
    second = tracker.update(
        frame_index=10, timestamp_s=10.0, bbox=(100, 100, 50, 50), candidate_id="c2", score=0.5
    )
    assert first != second


def test_track_summary_keeps_best_observation() -> None:
    tracker = ContactTracker()
    tracker.update(
        frame_index=0, timestamp_s=0.0, bbox=(100, 100, 50, 50), candidate_id="weak",
        score=0.3, decision="NEEDS_REVIEW",
    )
    tracker.update(
        frame_index=1, timestamp_s=1.0, bbox=(102, 101, 50, 50), candidate_id="strong",
        score=0.8, decision="LIKELY_MATCH",
    )
    summary = tracker.summaries()[0]
    assert summary["observations"] == 2
    assert summary["best_candidate_id"] == "strong"
    assert summary["best_score"] == 0.8
    assert summary["best_decision"] == "LIKELY_MATCH"
    assert summary["duration_s"] == 1.0


def test_track_video_results_annotates_and_summarizes() -> None:
    results = [
        {
            "detected": True, "frame_index": 0, "timestamp_s": 0.0,
            "bbox": [100, 100, 50, 50], "candidate_id": "c1",
            "final_score": 0.4, "final_decision": "NEEDS_REVIEW",
        },
        {
            "detected": True, "frame_index": 1, "timestamp_s": 1.0,
            "bbox": [104, 102, 50, 50], "candidate_id": "c2",
            "final_score": 0.7, "final_decision": "POSSIBLE_MATCH",
        },
        {"detected": False, "frame_index": 2, "bbox": None},
    ]
    summary = track_video_results(results)
    assert summary is not None
    assert summary["tracked_candidates"] == 2
    assert summary["track_count"] == 1
    assert results[0]["track_id"] == results[1]["track_id"]
    assert "track_id" not in results[2]


def test_track_video_results_returns_none_without_detections() -> None:
    assert track_video_results([{"detected": False, "frame_index": 0, "bbox": None}]) is None


if __name__ == "__main__":
    tests = [
        test_overlapping_detections_share_a_track,
        test_distant_detections_get_separate_tracks,
        test_centroid_fallback_links_fast_small_target,
        test_track_expires_after_max_gap_frames,
        test_track_summary_keeps_best_observation,
        test_track_video_results_annotates_and_summarizes,
        test_track_video_results_returns_none_without_detections,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
