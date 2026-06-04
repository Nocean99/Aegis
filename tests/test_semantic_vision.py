from __future__ import annotations

import sys
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.config_loader import load_search_mission_config
from autonomy.mission_objective import parse_mission_request
from autonomy.red_block_detector import RedBlockDetector
from autonomy.semantic_vision import (
    LocalSemanticVisionScorer,
    OpenAIVisionLanguageScorer,
    crop_detection,
    image_to_data_url,
    parse_semantic_json,
    save_candidate_crop,
)
from autonomy.types import SemanticDecision, TargetDetection


def red_block_image(width: int = 640, height: int = 480) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (35, 55, 35)
    cv2.rectangle(image, (250, 170), (370, 260), (0, 0, 255), -1)
    return image


def test_semantic_scorer_prioritizes_requested_visual_color() -> None:
    config = load_search_mission_config("config/autonomy.yaml")
    frame = red_block_image()
    detection = RedBlockDetector(config.target).detect(frame)
    crop = crop_detection(frame, detection)
    objective = parse_mission_request("Search the field for a red vehicle-like object")
    result = LocalSemanticVisionScorer().score(
        objective=objective,
        frame_bgr=frame,
        crop_bgr=crop,
        detection=detection,
    )
    assert result.score >= 0.65
    assert result.decision in {SemanticDecision.POSSIBLE_MATCH, SemanticDecision.LIKELY_MATCH}
    assert "visual_color:red" in result.tags
    assert result.needs_human_review


def test_semantic_scorer_rejects_missing_candidate() -> None:
    frame = red_block_image()
    objective = parse_mission_request("Search the field for anything unusual")
    result = LocalSemanticVisionScorer().score(
        objective=objective,
        frame_bgr=frame,
        crop_bgr=None,
        detection=TargetDetection(False),
    )
    assert result.score == 0.0
    assert result.decision == SemanticDecision.REJECT


def test_detection_crop_can_be_saved_for_later_model_review() -> None:
    config = load_search_mission_config("config/autonomy.yaml")
    frame = red_block_image()
    detection = RedBlockDetector(config.target).detect(frame)
    crop = crop_detection(frame, detection)
    assert crop is not None
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.png"
        saved = save_candidate_crop(crop, path)
        assert saved == str(path)
        assert path.exists()


def test_high_recall_finds_darker_red_region() -> None:
    config = load_search_mission_config("config/autonomy.yaml")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (45, 65, 45)
    cv2.rectangle(image, (260, 180), (360, 250), (80, 80, 130), -1)
    detector = RedBlockDetector(config.target)
    precise = detector.detect(image)
    high_recall = detector.detect_high_recall(image)
    assert not precise.detected
    assert high_recall.detected


def test_local_full_frame_scan_returns_review_result() -> None:
    objective = parse_mission_request("Search for a person in a green shirt")
    result = LocalSemanticVisionScorer().score_full_frame(objective=objective, frame_bgr=red_block_image())
    assert result.score > 0
    assert result.needs_human_review
    assert "full_frame_scan" in result.tags


def test_semantic_json_parser_accepts_model_response() -> None:
    parsed = parse_semantic_json(
        '{"score": 0.81, "decision": "LIKELY_MATCH", "explanation": "Looks plausible", "tags": ["vehicle"], "needs_human_review": true}'
    )
    assert parsed["score"] == 0.81
    assert parsed["decision"] == SemanticDecision.LIKELY_MATCH
    assert parsed["tags"] == ["vehicle"]


def test_image_to_data_url_encodes_jpeg() -> None:
    data_url = image_to_data_url(red_block_image())
    assert data_url.startswith("data:image/jpeg;base64,")


def test_openai_scorer_initializes_with_explicit_settings() -> None:
    scorer = OpenAIVisionLanguageScorer(
        model="test-vision-model",
        api_key="test-key",
        detail="low",
        timeout_s=12.0,
    )
    assert scorer.model_name == "test-vision-model"
    assert scorer.detail == "low"
    assert scorer.timeout_s == 12.0


def test_openai_scorer_rejects_invalid_detail() -> None:
    try:
        OpenAIVisionLanguageScorer(model="test-vision-model", api_key="test-key", detail="maximum")
    except ValueError as exc:
        assert "detail" in str(exc)
    else:
        raise AssertionError("Expected invalid OpenAI detail setting to fail")


def test_openai_scorer_requires_api_key_and_model() -> None:
    previous_key = os.environ.pop("OPENAI_API_KEY", None)
    previous_model = os.environ.pop("OPENAI_VISION_MODEL", None)
    try:
        try:
            OpenAIVisionLanguageScorer()
        except ValueError as exc:
            assert "model" in str(exc)
        else:
            raise AssertionError("Expected missing OpenAI model to fail")
    finally:
        if previous_key is not None:
            os.environ["OPENAI_API_KEY"] = previous_key
        if previous_model is not None:
            os.environ["OPENAI_VISION_MODEL"] = previous_model


if __name__ == "__main__":
    tests = [
        test_semantic_scorer_prioritizes_requested_visual_color,
        test_semantic_scorer_rejects_missing_candidate,
        test_detection_crop_can_be_saved_for_later_model_review,
        test_high_recall_finds_darker_red_region,
        test_local_full_frame_scan_returns_review_result,
        test_semantic_json_parser_accepts_model_response,
        test_image_to_data_url_encodes_jpeg,
        test_openai_scorer_initializes_with_explicit_settings,
        test_openai_scorer_rejects_invalid_detail,
        test_openai_scorer_requires_api_key_and_model,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
