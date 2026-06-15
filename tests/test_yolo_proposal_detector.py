from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.yolo_proposal_detector import YoloProposalDetector


class FakeBoxes:
    def __init__(self, rows: list[tuple[tuple[float, float, float, float], float, int]]) -> None:
        self.xywh = np.array([row[0] for row in rows], dtype=np.float32)
        self.conf = np.array([row[1] for row in rows], dtype=np.float32)
        self.cls = np.array([row[2] for row in rows], dtype=np.float32)


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None, names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


class FakeModel:
    """Mimics an ultralytics model: model(frame, verbose=False) -> [result]."""

    def __init__(self, results: list[FakeResult]) -> None:
        self.results = results
        self.calls = 0

    def __call__(self, frame, verbose=False):
        self.calls += 1
        return self.results


NAMES = {0: "person", 1: "boat", 2: "traffic light"}


def frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_detect_returns_best_class_filtered_detection() -> None:
    boxes = FakeBoxes([
        ((100.0, 100.0, 40.0, 30.0), 0.9, 1),   # boat, allowed
        ((300.0, 200.0, 20.0, 20.0), 0.95, 2),  # traffic light, filtered out
        ((50.0, 50.0, 10.0, 10.0), 0.5, 1),     # weaker boat
    ])
    detector = YoloProposalDetector(categories=["boat"], model=FakeModel([FakeResult(boxes, NAMES)]))
    detection = detector.detect(frame())
    assert detection.detected
    assert abs(detection.confidence - 0.9) < 1e-6  # float32 round-trip
    assert detection.bbox == (80, 85, 40, 30)  # xywh center converted to corner
    assert detection.center_px == (100, 100)
    assert "yolo:boat" in (detection.proposal_reason or "")


def test_low_confidence_detections_are_dropped() -> None:
    boxes = FakeBoxes([((100.0, 100.0, 40.0, 30.0), 0.1, 1)])
    detector = YoloProposalDetector(
        categories=["boat"], confidence_threshold=0.2, model=FakeModel([FakeResult(boxes, NAMES)])
    )
    assert detector.detect_all(frame()) == []


def test_fallback_keeps_frame_in_review_when_nothing_clears_threshold() -> None:
    detector = YoloProposalDetector(categories=["boat"], model=FakeModel([FakeResult(None, NAMES)]))
    detection = detector.detect(frame())
    assert detection.detected
    assert detection.bbox is None
    assert detection.center_px == (320, 240)
    assert "fallback" in (detection.proposal_reason or "")


def test_fallback_can_be_disabled() -> None:
    detector = YoloProposalDetector(categories=["boat"], model=FakeModel([FakeResult(None, NAMES)]))
    detection = detector.detect(frame(), allow_fallback=False)
    assert not detection.detected


def test_unmapped_category_accepts_any_class() -> None:
    # "debris" has no COCO mapping, so even a traffic light may reach scoring.
    boxes = FakeBoxes([((300.0, 200.0, 20.0, 20.0), 0.8, 2)])
    detector = YoloProposalDetector(categories=["debris"], model=FakeModel([FakeResult(boxes, NAMES)]))
    detections = detector.detect_all(frame())
    assert len(detections) == 1
    assert "traffic light" in (detections[0].proposal_reason or "")


def test_empty_frame_returns_no_detection() -> None:
    detector = YoloProposalDetector(model=FakeModel([]))
    assert not detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)).detected


def test_mask_fills_detection_boxes() -> None:
    boxes = FakeBoxes([((100.0, 100.0, 40.0, 30.0), 0.9, 1)])
    detector = YoloProposalDetector(categories=["boat"], model=FakeModel([FakeResult(boxes, NAMES)]))
    mask = detector.mask(frame())
    assert mask.shape == (480, 640)
    assert mask[100, 100] == 255
    assert mask[0, 0] == 0


if __name__ == "__main__":
    tests = [
        test_detect_returns_best_class_filtered_detection,
        test_low_confidence_detections_are_dropped,
        test_fallback_keeps_frame_in_review_when_nothing_clears_threshold,
        test_fallback_can_be_disabled,
        test_unmapped_category_accepts_any_class,
        test_empty_frame_returns_no_detection,
        test_mask_fills_detection_boxes,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
