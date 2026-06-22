from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.config_loader import load_search_mission_config
from autonomy.color_proposal_detector import MissionColorProposalDetector
from autonomy.mission_vision_plan import create_mission_vision_plan
from autonomy.objectness_proposal_detector import ObjectnessProposalDetector
from autonomy.red_block_detector import RedBlockDetector
from autonomy.vehicle_proposal_detector import VehicleProposalDetector, infer_sensor_modality
from autonomy.vision_lab import detect_with_mode, run_vision_lab
from autonomy.semantic_vision import LocalSemanticVisionScorer
from autonomy.mission_objective import parse_mission_request
from autonomy.types import SemanticDecision, TargetDetection


def rgb_vehicle_like_image() -> np.ndarray:
    image = np.zeros((220, 260, 3), dtype=np.uint8)
    image[:, :] = (96, 124, 112)
    cv2.rectangle(image, (98, 88), (142, 108), (32, 32, 32), -1)
    cv2.rectangle(image, (105, 92), (135, 104), (225, 225, 225), 1)
    return image


def ir_hot_blob_image() -> np.ndarray:
    image = np.full((220, 260, 3), 32, dtype=np.uint8)
    cv2.rectangle(image, (115, 96), (148, 113), (238, 238, 238), -1)
    return image


def blank_image() -> np.ndarray:
    return np.full((220, 260, 3), 80, dtype=np.uint8)


def test_vehicle_mission_request_activates_vehicle_proposal_mode() -> None:
    plan = create_mission_vision_plan("Search aerial imagery for vehicles near the road")
    config = load_search_mission_config("config/autonomy.yaml")
    detection = detect_with_mode(
        RedBlockDetector(config.target),
        MissionColorProposalDetector(plan),
        ObjectnessProposalDetector(),
        plan,
        rgb_vehicle_like_image(),
        "mission-color",
        sensor_modality="rgb",
        vehicle_detector=VehicleProposalDetector(),
    )
    assert detection.detected
    assert detection.proposal_reason in {"small high-contrast object", "rectangle-like aerial object"}
    assert detection.sensor_modality == "rgb"


def test_rgb_vehicle_proposal_returns_candidate_on_synthetic_vehicle() -> None:
    detection = VehicleProposalDetector().detect(rgb_vehicle_like_image(), modality="rgb", allow_fallback=False)
    assert detection.detected
    assert detection.bbox is not None
    assert detection.confidence > 0.4
    assert detection.proposal_reason in {"small high-contrast object", "rectangle-like aerial object"}


def test_ir_vehicle_proposal_returns_candidate_on_hot_blob() -> None:
    detection = VehicleProposalDetector().detect(ir_hot_blob_image(), modality="infrared", allow_fallback=False)
    assert detection.detected
    assert detection.bbox is not None
    assert detection.sensor_modality == "infrared"
    assert detection.proposal_reason == "hot IR blob"


def test_ir_hot_blob_semantic_threshold_is_stricter_than_rgb() -> None:
    objective = parse_mission_request("Search infrared aerial imagery for vehicles")
    scorer = LocalSemanticVisionScorer()
    frame = ir_hot_blob_image()
    detection = TargetDetection(
        True,
        confidence=0.7,
        bbox=(115, 96, 33, 17),
        center_px=(131, 104),
        area_px=561,
        area_ratio=561 / float(frame.shape[0] * frame.shape[1]),
        sensor_modality="infrared",
        proposal_reason="hot IR blob",
    )
    result = scorer.score(objective=objective, frame_bgr=frame, crop_bgr=frame[90:120, 108:155], detection=detection)
    assert result.decision == SemanticDecision.POSSIBLE_MATCH
    assert result.score < 0.82
    assert "thermal_triage_stricter_threshold" in result.tags

    weak_detection = TargetDetection(
        True,
        confidence=0.45,
        bbox=(115, 96, 33, 17),
        center_px=(131, 104),
        area_px=561,
        area_ratio=561 / float(frame.shape[0] * frame.shape[1]),
        sensor_modality="infrared",
        proposal_reason="hot IR blob",
    )
    weak = scorer.score(objective=objective, frame_bgr=frame, crop_bgr=frame[90:120, 108:155], detection=weak_detection)
    assert weak.decision == SemanticDecision.NEEDS_REVIEW


def test_vehicle_fallback_creates_reviewable_candidate() -> None:
    detection = VehicleProposalDetector().detect(blank_image(), modality="infrared", allow_fallback=True)
    assert detection.detected
    assert detection.bbox is None
    assert detection.confidence >= 0.3
    assert detection.proposal_reason == "full-frame fallback"


def test_vehicle_proposal_layer_makes_no_api_calls() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        image_path = root / "vehicle.jpg"
        cv2.imwrite(str(image_path), rgb_vehicle_like_image())
        labels = root / "labels.csv"
        labels.write_text("image_path,expected_match,label\nvehicle.jpg,true,positive\n", encoding="utf-8")
        report_path = run_vision_lab(
            mission_request="Search aerial imagery for vehicles",
            image_paths=[image_path],
            output_dir=root / "out",
            labels_csv=labels,
            semantic_vision="local",
            proposal_mode="mission-color",
            full_frame_semantic="misses",
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = report["results"][0]
        assert report["scorer"] == "local-visual-triage-v2"
        assert result["detected"] is True
        assert result["sensor_modality"] == "rgb"
        assert result["proposal_reason"] in {"small high-contrast object", "rectangle-like aerial object"}
        assert "openai" not in json.dumps(report).lower()


def test_modality_inference_uses_ir_path_and_grayscale() -> None:
    assert infer_sensor_modality("logs/benchmark_samples/dronevehicle_ir_local_500/images/one.jpg") == "infrared"
    assert infer_sensor_modality("one.jpg", ir_hot_blob_image()) == "infrared"


if __name__ == "__main__":
    tests = [
        test_vehicle_mission_request_activates_vehicle_proposal_mode,
        test_rgb_vehicle_proposal_returns_candidate_on_synthetic_vehicle,
        test_ir_vehicle_proposal_returns_candidate_on_hot_blob,
        test_ir_hot_blob_semantic_threshold_is_stricter_than_rgb,
        test_vehicle_fallback_creates_reviewable_candidate,
        test_vehicle_proposal_layer_makes_no_api_calls,
        test_modality_inference_uses_ir_path_and_grayscale,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
