from __future__ import annotations

from dataclasses import asdict, dataclass

from autonomy.types import SemanticDecision, SemanticVisionResult, TargetDetection


@dataclass(frozen=True)
class CandidateRank:
    proposal_score: float
    semantic_score: float
    uncertainty_score: float
    mission_relevance_score: float
    review_priority: float
    reasons: list[str]
    memory_adjustment: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def rank_candidate(
    *,
    detection: TargetDetection,
    semantic: SemanticVisionResult,
    full_frame_result: SemanticVisionResult | None,
    final_score: float,
    final_decision: SemanticDecision,
    semantic_error: str | None = None,
    memory_adjustment: float = 0.0,
    memory_reasons: list[str] | None = None,
) -> CandidateRank:
    proposal_score = proposal_component(detection)
    semantic_score = semantic_component(final_score, final_decision)
    uncertainty_score = uncertainty_component(
        detection=detection,
        semantic=semantic,
        full_frame_result=full_frame_result,
        final_decision=final_decision,
        semantic_error=semantic_error,
    )
    mission_relevance_score = mission_relevance_component(final_decision, semantic_score, full_frame_result)
    reasons = ranking_reasons(
        detection=detection,
        final_decision=final_decision,
        full_frame_result=full_frame_result,
        semantic_error=semantic_error,
    )

    priority = (
        proposal_score * 0.22
        + semantic_score * 0.32
        + uncertainty_score * 0.18
        + mission_relevance_score * 0.28
    )
    if final_decision == SemanticDecision.LIKELY_MATCH:
        priority += 0.18
    elif final_decision == SemanticDecision.POSSIBLE_MATCH:
        priority += 0.04
    elif final_decision == SemanticDecision.REJECT:
        priority -= 0.12
    if semantic_error:
        priority = max(priority, 0.28)
    if full_frame_result is not None and full_frame_result.score >= semantic.score:
        priority += 0.04
    # Mission-memory read-back: a bounded nudge from prior analyst decisions.
    # It reorders attention only — candidates are never dropped by memory.
    priority += memory_adjustment
    if memory_reasons:
        reasons = reasons + list(memory_reasons)

    return CandidateRank(
        proposal_score=round(proposal_score, 3),
        semantic_score=round(semantic_score, 3),
        uncertainty_score=round(uncertainty_score, 3),
        mission_relevance_score=round(mission_relevance_score, 3),
        review_priority=round(_clamp(priority), 3),
        reasons=reasons[:8],
        memory_adjustment=round(memory_adjustment, 4),
    )


def proposal_component(detection: TargetDetection) -> float:
    if not detection.detected:
        return 0.0
    if detection.area_ratio <= 0:
        return _clamp(detection.confidence)
    area_signal = min(1.0, max(0.0, detection.area_ratio * 18.0))
    return _clamp(detection.confidence * 0.85 + area_signal * 0.15)


def semantic_component(final_score: float, final_decision: SemanticDecision) -> float:
    score = _clamp(final_score)
    if final_decision == SemanticDecision.LIKELY_MATCH:
        return max(score, 0.78)
    if final_decision == SemanticDecision.POSSIBLE_MATCH:
        return max(score, 0.58)
    if final_decision == SemanticDecision.NEEDS_REVIEW:
        return max(score, 0.28)
    return min(score, 0.2)


def uncertainty_component(
    *,
    detection: TargetDetection,
    semantic: SemanticVisionResult,
    full_frame_result: SemanticVisionResult | None,
    final_decision: SemanticDecision,
    semantic_error: str | None,
) -> float:
    if semantic_error:
        return 0.85
    if final_decision == SemanticDecision.NEEDS_REVIEW:
        return 0.78
    if final_decision == SemanticDecision.POSSIBLE_MATCH:
        return 0.5
    if final_decision == SemanticDecision.REJECT and full_frame_result is not None:
        return 0.38
    if detection.detected and semantic.score < 0.25:
        return 0.32
    return 0.16


def mission_relevance_component(
    final_decision: SemanticDecision,
    semantic_score: float,
    full_frame_result: SemanticVisionResult | None,
) -> float:
    if final_decision == SemanticDecision.LIKELY_MATCH:
        return 1.0
    if final_decision == SemanticDecision.POSSIBLE_MATCH:
        return max(0.7, semantic_score)
    if final_decision == SemanticDecision.NEEDS_REVIEW:
        return max(0.42, semantic_score * 0.82)
    if full_frame_result is not None and full_frame_result.decision != SemanticDecision.REJECT:
        return max(0.35, full_frame_result.score)
    return min(0.22, semantic_score)


def ranking_reasons(
    *,
    detection: TargetDetection,
    final_decision: SemanticDecision,
    full_frame_result: SemanticVisionResult | None,
    semantic_error: str | None,
) -> list[str]:
    reasons: list[str] = []
    if detection.detected:
        reasons.append("local proposal exists")
    if final_decision == SemanticDecision.LIKELY_MATCH:
        reasons.append("likely mission match")
    elif final_decision == SemanticDecision.POSSIBLE_MATCH:
        reasons.append("possible mission match")
    elif final_decision == SemanticDecision.NEEDS_REVIEW:
        reasons.append("uncertain evidence kept for review")
    else:
        reasons.append("semantic review rejected match")
    if full_frame_result is not None:
        reasons.append("full-frame semantic pass available")
    if semantic_error:
        reasons.append("semantic scorer error preserved for review")
    return reasons


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
