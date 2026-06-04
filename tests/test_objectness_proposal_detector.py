from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.objectness_proposal_detector import ObjectnessProposalDetector


def test_objectness_detector_finds_non_background_region() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (35, 115, 35)
    cv2.rectangle(image, (125, 85), (205, 145), (220, 220, 225), -1)
    detection = ObjectnessProposalDetector().detect(image)
    assert detection.detected
    assert detection.bbox is not None


def test_objectness_detector_ignores_uniform_background() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (35, 115, 35)
    detection = ObjectnessProposalDetector().detect(image)
    assert not detection.detected


if __name__ == "__main__":
    tests = [
        test_objectness_detector_finds_non_background_region,
        test_objectness_detector_ignores_uniform_background,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
