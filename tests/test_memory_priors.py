from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.candidate_ranking import rank_candidate
from autonomy.memory_priors import (
    MAX_ADJUSTMENT,
    MemoryPriors,
    candidate_terms,
    memory_adjustment,
    update_priors_from_mission,
)
from autonomy.types import SemanticDecision, SemanticVisionResult, TargetDetection


def make_result(candidate_id: str, *, bbox, image_path="logs/run/boat_dock_001.png", reason="mission color match (red)"):
    return {
        "candidate_id": candidate_id,
        "image_path": image_path,
        "proposal_reason": reason,
        "bbox": bbox,
        "frame_size": [320, 240],
        "semantic": {"tags": []},
    }


def reviewed_priors() -> MemoryPriors:
    """One mission: vessel at (230,95) confirmed, clutter at (64,179) rejected."""
    priors = MemoryPriors()
    results = [
        make_result("c1", bbox=(230, 95, 34, 14), image_path="logs/run/vessel_pass.png"),
        make_result("c2", bbox=(58, 173, 24, 24), image_path="logs/run/shore_rocks.png"),
    ]
    reviews = {
        "c1": {"decision": "approve", "reason_tag": "vehicle_visible"},
        "c2": {"decision": "reject", "reason_tag": "false_alarm"},
    }
    return update_priors_from_mission(priors, results=results, reviews=reviews)


def test_candidate_terms_extracts_stable_tokens() -> None:
    terms = candidate_terms(make_result("c1", bbox=(0, 0, 5, 5), image_path="x/boat_dock_001.png", reason="yolo:boat 0.91"))
    assert "boat" in terms and "dock" in terms
    assert "001" not in terms  # digits dropped


def test_priors_build_from_reviews() -> None:
    priors = reviewed_priors()
    assert priors.confirmed_terms["vessel"] == 1
    assert priors.rejected_terms["rocks"] == 1
    assert len(priors.confirmed_locations) == 1
    assert len(priors.rejected_locations) == 1
    assert priors.missions_observed == 1


def test_adjustment_boosts_confirmed_location_and_terms() -> None:
    priors = reviewed_priors()
    delta, reasons = memory_adjustment(
        priors,
        terms={"vessel", "pass"},
        location_norm=((230 + 17) / 320, (95 + 7) / 240),
    )
    assert delta > 0
    assert any("confirmed" in reason for reason in reasons)


def test_adjustment_penalizes_dismissed_clutter() -> None:
    priors = reviewed_priors()
    delta, reasons = memory_adjustment(
        priors,
        terms={"rocks", "shore"},
        location_norm=((58 + 12) / 320, (173 + 12) / 240),
    )
    assert delta < 0
    assert any("dismissed" in reason for reason in reasons)


def test_adjustment_is_bounded() -> None:
    priors = MemoryPriors()
    for _ in range(10):  # pile on confirmations to push past the cap
        results = [make_result("c1", bbox=(100, 100, 30, 30), image_path="x/vessel_boat_dock.png")]
        update_priors_from_mission(priors, results=results, reviews={"c1": {"decision": "approve"}})
    delta, _ = memory_adjustment(
        priors, terms={"vessel", "boat", "dock"}, location_norm=(115 / 320, 115 / 240)
    )
    assert 0 < delta <= MAX_ADJUSTMENT


def test_empty_priors_are_neutral() -> None:
    delta, reasons = memory_adjustment(MemoryPriors(), terms={"vessel"}, location_norm=(0.5, 0.5))
    assert delta == 0.0 and reasons == []
    delta, reasons = memory_adjustment(None, terms={"vessel"}, location_norm=(0.5, 0.5))
    assert delta == 0.0 and reasons == []


def test_repeated_nearby_locations_merge() -> None:
    priors = MemoryPriors()
    for offset in (0, 2, -2):
        results = [make_result("c1", bbox=(100 + offset, 100, 30, 30))]
        update_priors_from_mission(priors, results=results, reviews={"c1": {"decision": "approve"}})
    assert len(priors.confirmed_locations) == 1
    assert priors.confirmed_locations[0][2] == 3  # merged count


def test_priors_persist_round_trip() -> None:
    priors = reviewed_priors()
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "priors.json"
        priors.save(path)
        loaded = MemoryPriors.load(path)
    assert loaded.confirmed_terms == priors.confirmed_terms
    assert loaded.rejected_locations == priors.rejected_locations
    assert loaded.missions_observed == 1


def test_rank_candidate_applies_memory_adjustment() -> None:
    detection = TargetDetection(True, confidence=0.5, bbox=(10, 10, 30, 30), area_ratio=0.01)
    semantic = SemanticVisionResult(
        score=0.4, decision=SemanticDecision.NEEDS_REVIEW, explanation="", model_name="test"
    )
    base = rank_candidate(
        detection=detection,
        semantic=semantic,
        full_frame_result=None,
        final_score=0.4,
        final_decision=SemanticDecision.NEEDS_REVIEW,
    )
    boosted = rank_candidate(
        detection=detection,
        semantic=semantic,
        full_frame_result=None,
        final_score=0.4,
        final_decision=SemanticDecision.NEEDS_REVIEW,
        memory_adjustment=0.1,
        memory_reasons=["memory: near a previously confirmed contact location (seen 2x)"],
    )
    assert boosted.review_priority > base.review_priority
    assert boosted.memory_adjustment == 0.1
    assert any("memory" in reason for reason in boosted.reasons)
    # Memory never decides alone: candidate stays in the queue either way.
    penalized = rank_candidate(
        detection=detection,
        semantic=semantic,
        full_frame_result=None,
        final_score=0.4,
        final_decision=SemanticDecision.NEEDS_REVIEW,
        memory_adjustment=-0.15,
    )
    assert penalized.review_priority >= 0.0


if __name__ == "__main__":
    tests = [
        test_candidate_terms_extracts_stable_tokens,
        test_priors_build_from_reviews,
        test_adjustment_boosts_confirmed_location_and_terms,
        test_adjustment_penalizes_dismissed_clutter,
        test_adjustment_is_bounded,
        test_empty_priors_are_neutral,
        test_repeated_nearby_locations_merge,
        test_priors_persist_round_trip,
        test_rank_candidate_applies_memory_adjustment,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
