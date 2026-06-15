from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.clip_semantic_scorer import (
    BACKGROUND_PROMPTS,
    ClipSemanticVisionScorer,
    mission_prompts,
)
from autonomy.mission_objective import parse_mission_request
from autonomy.types import SemanticDecision, TargetDetection


def fake_backend(*, positive_similarity: float, background_similarity: float):
    """Build injectable encoders: positives get one embedding, backgrounds another.

    Embeddings live on a 2D unit circle so cosine similarity is exact and
    controllable without torch or open_clip installed.
    """

    def encode_image(image_bgr: np.ndarray) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode_text(prompts: list[str]) -> np.ndarray:
        rows = []
        for prompt in prompts:
            similarity = background_similarity if prompt in BACKGROUND_PROMPTS else positive_similarity
            rows.append([similarity, float(np.sqrt(max(0.0, 1.0 - similarity**2)))])
        return np.array(rows, dtype=np.float32)

    return encode_image, encode_text


def make_scorer(*, positive_similarity: float, background_similarity: float) -> ClipSemanticVisionScorer:
    encode_image, encode_text = fake_backend(
        positive_similarity=positive_similarity, background_similarity=background_similarity
    )
    return ClipSemanticVisionScorer(encode_image=encode_image, encode_text=encode_text)


def red_detection() -> TargetDetection:
    return TargetDetection(True, confidence=0.6, bbox=(10, 10, 40, 40), center_px=(30, 30))


def test_strong_positive_match_scores_high() -> None:
    objective = parse_mission_request("find a red boat near the shoreline")
    scorer = make_scorer(positive_similarity=0.9, background_similarity=0.1)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = scorer.score(objective=objective, frame_bgr=frame, crop_bgr=frame, detection=red_detection())
    assert result.score >= 0.75
    assert result.decision == SemanticDecision.LIKELY_MATCH
    assert result.needs_human_review is True
    assert result.model_name.startswith("clip-local:")
    assert any(tag == "clip_positive_top" for tag in result.tags)


def test_background_lookalike_scores_low() -> None:
    objective = parse_mission_request("find a red boat near the shoreline")
    scorer = make_scorer(positive_similarity=0.1, background_similarity=0.9)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = scorer.score(objective=objective, frame_bgr=frame, crop_bgr=frame, detection=red_detection())
    assert result.score < 0.2
    assert result.decision == SemanticDecision.REJECT


def test_missing_candidate_is_rejected_without_encoding() -> None:
    objective = parse_mission_request("find a red boat")

    def explode(*args, **kwargs):
        raise AssertionError("encoder must not be called for a missing candidate")

    scorer = ClipSemanticVisionScorer(encode_image=explode, encode_text=explode)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = scorer.score(
        objective=objective, frame_bgr=frame, crop_bgr=None, detection=TargetDetection(False)
    )
    assert result.decision == SemanticDecision.REJECT
    assert result.score == 0.0


def test_full_frame_scan_tags_and_keeps_review() -> None:
    objective = parse_mission_request("find a red boat")
    scorer = make_scorer(positive_similarity=0.9, background_similarity=0.1)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = scorer.score_full_frame(objective=objective, frame_bgr=frame)
    assert "full_frame_scan" in result.tags
    assert result.needs_human_review is True


def test_text_embeddings_are_cached() -> None:
    objective = parse_mission_request("find a red boat")
    calls = {"count": 0}
    encode_image, encode_text = fake_backend(positive_similarity=0.9, background_similarity=0.1)

    def counting_encode_text(prompts: list[str]) -> np.ndarray:
        calls["count"] += 1
        return encode_text(prompts)

    scorer = ClipSemanticVisionScorer(encode_image=encode_image, encode_text=counting_encode_text)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    scorer.score(objective=objective, frame_bgr=frame, crop_bgr=frame, detection=red_detection())
    scorer.score(objective=objective, frame_bgr=frame, crop_bgr=frame, detection=red_detection())
    assert calls["count"] == 1


def test_mission_prompts_cover_description_categories_and_colors() -> None:
    objective = parse_mission_request("find a red boat near the shoreline")
    prompts = mission_prompts(objective)
    assert prompts, "Expected at least one prompt"
    assert len(prompts) == len(set(prompts)), "Prompts must be unique"
    assert any("boat" in prompt for prompt in prompts)


def test_mission_prompts_fall_back_to_raw_request() -> None:
    from autonomy.types import MissionObjective

    objective = MissionObjective(raw_request="locate the anomaly")
    prompts = mission_prompts(objective)
    assert prompts == ["an aerial photo of locate the anomaly"]


if __name__ == "__main__":
    tests = [
        test_strong_positive_match_scores_high,
        test_background_lookalike_scores_low,
        test_missing_candidate_is_rejected_without_encoding,
        test_full_frame_scan_tags_and_keeps_review,
        test_text_embeddings_are_cached,
        test_mission_prompts_cover_description_categories_and_colors,
        test_mission_prompts_fall_back_to_raw_request,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
