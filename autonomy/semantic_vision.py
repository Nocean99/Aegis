from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Protocol
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
import numpy as np

from autonomy.types import MissionObjective, SemanticDecision, SemanticVisionResult, TargetDetection


class SemanticVisionScorer(Protocol):
    def score(
        self,
        *,
        objective: MissionObjective,
        frame_bgr: np.ndarray,
        crop_bgr: np.ndarray | None,
        detection: TargetDetection,
    ) -> SemanticVisionResult:
        ...

    def score_full_frame(
        self,
        *,
        objective: MissionObjective,
        frame_bgr: np.ndarray,
    ) -> SemanticVisionResult:
        ...


class LocalSemanticVisionScorer:
    """Deterministic local triage scorer.

    This scorer does not claim make/model recognition. It uses objective text,
    proposal geometry, detector confidence, and simple visual cues to decide
    whether a candidate deserves human review while preserving the same
    interface a real VLM will use.
    """

    model_name = "local-visual-triage-v2"

    def score(
        self,
        *,
        objective: MissionObjective,
        frame_bgr: np.ndarray,
        crop_bgr: np.ndarray | None,
        detection: TargetDetection,
    ) -> SemanticVisionResult:
        tags: list[str] = []
        if not detection.detected:
            return SemanticVisionResult(
                score=0.0,
                decision=SemanticDecision.REJECT,
                explanation="No visual candidate was proposed.",
                model_name=self.model_name,
                tags=tags,
            )

        if _is_vehicle_objective(objective) or _is_vehicle_detection(detection):
            return _score_vehicle_candidate(objective, crop_bgr, detection)

        score = detection.confidence * 0.45
        if objective.extracted_colors and crop_bgr is not None:
            color_hits = _matching_color_tags(crop_bgr, objective.extracted_colors)
            tags.extend(color_hits)
            if color_hits:
                score += 0.25
            else:
                score -= 0.12

        if objective.extracted_categories:
            tags.extend(f"requested:{category}" for category in objective.extracted_categories)
            score += 0.1

        if objective.urgency == "high":
            tags.append("high_urgency")
            score += 0.05

        score = max(0.0, min(1.0, score))
        if score >= 0.72:
            decision = SemanticDecision.LIKELY_MATCH
        elif score >= 0.38:
            decision = SemanticDecision.POSSIBLE_MATCH
        else:
            decision = SemanticDecision.NEEDS_REVIEW

        explanation = (
            "Candidate ranked against the mission text with local visual cues. "
            "This is a review-prioritization score, not true object identity recognition yet."
        )
        return SemanticVisionResult(
            score=round(score, 3),
            decision=decision,
            explanation=explanation,
            model_name=self.model_name,
            tags=sorted(set(tags)),
            needs_human_review=True,
        )

    def score_full_frame(
        self,
        *,
        objective: MissionObjective,
        frame_bgr: np.ndarray,
    ) -> SemanticVisionResult:
        tags = [f"requested:{category}" for category in objective.extracted_categories]
        if objective.extracted_colors:
            color_hits = _matching_color_tags(frame_bgr, objective.extracted_colors)
            tags.extend(color_hits)
        score = 0.08
        if objective.extracted_colors and any(tag.startswith("visual_color:") for tag in tags):
            score += 0.18
        if objective.extracted_categories:
            score += 0.04
        return SemanticVisionResult(
            score=round(min(score, 0.32), 3),
            decision=SemanticDecision.NEEDS_REVIEW,
            explanation=(
                "Local full-frame scan cannot identify category-only targets such as people, vehicles, or boats. "
                "Use a real vision-language scorer for this mission type."
            ),
            model_name=self.model_name,
            tags=sorted(set(tags + ["full_frame_scan", "local_scorer_limited"])),
            needs_human_review=True,
        )


def _score_vehicle_candidate(
    objective: MissionObjective,
    crop_bgr: np.ndarray | None,
    detection: TargetDetection,
) -> SemanticVisionResult:
    tags = [f"requested:{category}" for category in objective.extracted_categories]
    if detection.sensor_modality:
        tags.append(f"sensor:{detection.sensor_modality}")
    if detection.proposal_reason:
        tags.append(f"proposal:{detection.proposal_reason}")
    if detection.bbox:
        _, _, width, height = detection.bbox
        aspect = width / float(max(1, height))
        tags.append(f"aspect:{aspect:.2f}")
    else:
        aspect = 1.0
        tags.append("full_frame_fallback")

    confidence_signal = max(0.0, min(1.0, detection.confidence))
    size_signal = min(1.0, max(0.0, detection.area_ratio * 45.0)) if detection.area_ratio else 0.0
    rectangle_signal = 1.0 if detection.proposal_reason and "rectangle" in detection.proposal_reason else 0.35
    aspect_signal = 1.0 if 0.35 <= aspect <= 3.2 else 0.45
    contrast_signal = _crop_contrast_signal(crop_bgr)
    color_hits = _matching_color_tags(crop_bgr, objective.extracted_colors) if objective.extracted_colors and crop_bgr is not None else []
    tags.extend(color_hits)

    score = (
        0.18
        + confidence_signal * 0.38
        + size_signal * 0.12
        + rectangle_signal * 0.14
        + aspect_signal * 0.08
        + contrast_signal * 0.10
        + (0.08 if color_hits else 0.0)
    )
    if detection.proposal_reason == "full-frame fallback":
        score = min(score, 0.46)
    score = max(0.0, min(1.0, score))
    if score >= 0.76:
        decision = SemanticDecision.LIKELY_MATCH
    elif score >= 0.55:
        decision = SemanticDecision.POSSIBLE_MATCH
    else:
        decision = SemanticDecision.NEEDS_REVIEW
    explanation = (
        "Local vehicle triage scored this candidate from proposal confidence, "
        "bounding-box shape, relative size, and crop contrast. This is evidence "
        "prioritization for analyst review, not exact vehicle identity recognition."
    )
    return SemanticVisionResult(
        score=round(score, 3),
        decision=decision,
        explanation=explanation,
        model_name=LocalSemanticVisionScorer.model_name,
        tags=sorted(set(tags)),
        needs_human_review=True,
    )


def _is_vehicle_objective(objective: MissionObjective) -> bool:
    vehicle_terms = {"vehicle", "car", "truck", "van", "bus", "freight", "freight car"}
    categories = {item.lower() for item in objective.extracted_categories}
    text = objective.raw_request.lower()
    return bool(categories & vehicle_terms) or any(term in text for term in vehicle_terms)


def _is_vehicle_detection(detection: TargetDetection) -> bool:
    reason = (detection.proposal_reason or "").lower()
    return any(term in reason for term in ("vehicle", "aerial object", "hot ir blob", "full-frame fallback"))


def _crop_contrast_signal(crop_bgr: np.ndarray | None) -> float:
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.2
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return max(0.0, min(1.0, float(np.std(gray)) / 64.0))


class OpenAIVisionLanguageScorer:
    """Optional OpenAI-compatible vision-language scorer.

    This scorer is intentionally isolated from flight control. It only scores
    frames/crops and returns review metadata. It requires OPENAI_API_KEY and an
    explicit model name via constructor or OPENAI_VISION_MODEL.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        detail: str = "auto",
        timeout_s: float = 45.0,
        max_retries: int = 3,
    ) -> None:
        self.model_name = model or os.environ.get("OPENAI_VISION_MODEL", "")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.detail = detail
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)
        if self.detail not in {"auto", "low", "high"}:
            raise ValueError("OpenAI image detail must be one of: auto, low, high.")
        if not self.model_name:
            raise ValueError("OpenAI vision scorer requires --openai-model or OPENAI_VISION_MODEL.")
        if not self.api_key:
            raise ValueError("OpenAI vision scorer requires OPENAI_API_KEY.")

    def score(
        self,
        *,
        objective: MissionObjective,
        frame_bgr: np.ndarray,
        crop_bgr: np.ndarray | None,
        detection: TargetDetection,
    ) -> SemanticVisionResult:
        image = crop_bgr if crop_bgr is not None else frame_bgr
        prompt = _semantic_prompt(objective, context="candidate crop")
        if detection.bbox:
            prompt += f"\nCandidate bounding box: {detection.bbox}."
        return self._score_image(prompt=prompt, image_bgr=image)

    def score_full_frame(
        self,
        *,
        objective: MissionObjective,
        frame_bgr: np.ndarray,
    ) -> SemanticVisionResult:
        return self._score_image(
            prompt=_semantic_prompt(objective, context="full drone frame"),
            image_bgr=frame_bgr,
        )

    def _score_image(self, *, prompt: str, image_bgr: np.ndarray) -> SemanticVisionResult:
        data_url = image_to_data_url(image_bgr)
        image_content = {"type": "input_image", "image_url": data_url}
        if self.detail != "auto":
            image_content["detail"] = self.detail
        body = {
            "model": self.model_name,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        image_content,
                    ],
                }
            ],
        }
        req = urlrequest.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        payload = self._open_with_retries(req)
        text = extract_response_text(payload)
        parsed = parse_semantic_json(text)
        return SemanticVisionResult(
            score=parsed["score"],
            decision=parsed["decision"],
            explanation=parsed["explanation"],
            model_name=self.model_name,
            tags=parsed["tags"],
            needs_human_review=parsed["needs_human_review"],
        )

    def _open_with_retries(self, req: urlrequest.Request) -> dict:
        for attempt in range(self.max_retries + 1):
            try:
                with urlrequest.urlopen(req, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urlerror.HTTPError as exc:
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise
                time.sleep(min(2.0 * (2 ** attempt), 12.0))
            except urlerror.URLError:
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(2.0 * (2 ** attempt), 12.0))
        raise RuntimeError("OpenAI request retry loop exited unexpectedly.")


def image_to_data_url(image_bgr: np.ndarray, *, max_width_px: int = 1024, quality: int = 82) -> str:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot encode empty image.")
    image = image_bgr
    height, width = image.shape[:2]
    if width > max_width_px:
        scale = max_width_px / float(width)
        image = cv2.resize(image, (max_width_px, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Could not JPEG-encode image.")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def extract_response_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    texts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)


def parse_semantic_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = {
            "score": 0.0,
            "decision": "NEEDS_REVIEW",
            "explanation": cleaned[:500] or "Model response was not valid JSON.",
            "tags": [],
            "needs_human_review": True,
        }
    score = max(0.0, min(1.0, float(data.get("score", 0.0))))
    raw_decision = str(data.get("decision", "NEEDS_REVIEW")).upper()
    try:
        decision = SemanticDecision(raw_decision)
    except ValueError:
        decision = SemanticDecision.NEEDS_REVIEW
    score, decision = normalize_score_decision(score, decision)
    tags = data.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    return {
        "score": round(score, 3),
        "decision": decision,
        "explanation": str(data.get("explanation", ""))[:800],
        "tags": [str(tag) for tag in tags][:20],
        "needs_human_review": bool(data.get("needs_human_review", True)),
    }


def normalize_score_decision(score: float, decision: SemanticDecision) -> tuple[float, SemanticDecision]:
    if decision == SemanticDecision.LIKELY_MATCH:
        score = max(score, 0.75)
    elif decision == SemanticDecision.POSSIBLE_MATCH:
        score = min(max(score, 0.55), 0.74)
    elif decision == SemanticDecision.NEEDS_REVIEW:
        score = min(max(score, 0.2), 0.54)
    else:
        score = min(score, 0.19)
    return score, decision


def _semantic_prompt(objective: MissionObjective, *, context: str) -> str:
    return (
        f"You are scoring a {context} from a search-and-rescue drone. "
        "Compare the image to the mission request. Return ONLY JSON with keys: "
        "score number 0.0-1.0, decision one of REJECT/POSSIBLE_MATCH/LIKELY_MATCH/NEEDS_REVIEW, "
        "explanation string, tags array, needs_human_review boolean. "
        f"{base_decision_rubric()} "
        f"{category_specific_guidance(objective)} "
        f"Mission request: {objective.raw_request}\n"
        f"Target description: {objective.target_description}\n"
        f"Extracted colors: {', '.join(objective.extracted_colors) or 'unknown'}\n"
        f"Extracted categories: {', '.join(objective.extracted_categories) or 'unknown'}"
    )


def base_decision_rubric() -> str:
    return (
        "Use this decision rubric strictly: "
        "LIKELY_MATCH means the requested target is clearly visible and matches the mission description. "
        "POSSIBLE_MATCH means there is visible target evidence, but details are incomplete or partly occluded. "
        "NEEDS_REVIEW means the image is ambiguous, suspicious, low-resolution, cluttered, or might contain the target but visible evidence is insufficient. "
        "REJECT means no plausible target evidence is visible. "
        "Favor recall for safety-critical missions by using NEEDS_REVIEW for ambiguous cases instead of REJECT, but do not inflate ambiguous background clutter to POSSIBLE_MATCH or LIKELY_MATCH. "
        "Use score bands consistently: LIKELY_MATCH 0.75-1.0, POSSIBLE_MATCH 0.55-0.74, NEEDS_REVIEW 0.20-0.54, REJECT 0.0-0.19. "
        "Do not claim certainty for exact identity if the image is unclear."
    )


def category_specific_guidance(objective: MissionObjective) -> str:
    categories = set(objective.extracted_categories)
    parts = []
    if "person" in categories:
        parts.append(
            "Person guidance: If a partially hidden person is visibly present, mark POSSIBLE_MATCH rather than NEEDS_REVIEW unless the visible evidence is only an indistinct blob. "
            "If multiple people, a person lying down, or a person partly concealed by grass is visible, use POSSIBLE_MATCH or LIKELY_MATCH depending on clarity. "
            "Do not mark vehicles, equipment, grass patches, shadows, or grey/green blobs as person matches unless a human body, limb, head, clothing on a person, or group of people is visibly present."
        )
    if "vehicle" in categories:
        parts.append(
            "Vehicle guidance: A visible car, truck, SUV, van, jeep, ATV, or similar vehicle should be POSSIBLE_MATCH or LIKELY_MATCH even if people are nearby. "
            "Do not reject a vehicle because it is partially occluded, parked near people, or only partly visible; use POSSIBLE_MATCH when the body, wheels, windshield, roofline, or cargo bed is visible. "
            "Do not mark campfires, people, grass, isolated shadows, or small non-vehicle objects as vehicle matches."
        )
    if "boat" in categories:
        parts.append(
            "Boat guidance: A visible boat, ship, hull, canoe, kayak, raft, docked vessel, or partially submerged vessel should be POSSIBLE_MATCH or LIKELY_MATCH. "
            "Use NEEDS_REVIEW for ambiguous floating debris or shoreline clutter unless boat structure is visible."
        )
    if "debris" in categories:
        parts.append(
            "Debris guidance: Scattered, damaged, irregular, or displaced material can be a match when it fits the mission context. "
            "Use NEEDS_REVIEW for ordinary repeated infrastructure or natural texture without visible disturbance."
        )
    if "signal" in categories:
        parts.append(
            "Signal guidance: Bright markers, smoke, fire, flares, high-contrast cloth, or deliberately arranged signals should be POSSIBLE_MATCH or LIKELY_MATCH. "
            "Use NEEDS_REVIEW for small bright clutter when intent is unclear."
        )
    if not parts:
        parts.append(
            "General guidance: Judge the visible evidence against the mission description and keep ambiguous but potentially relevant evidence in NEEDS_REVIEW."
        )
    return " ".join(parts) + "\n"


def crop_detection(frame_bgr: np.ndarray, detection: TargetDetection, padding_px: int = 24) -> np.ndarray | None:
    if frame_bgr is None or frame_bgr.size == 0 or detection.bbox is None:
        return None
    x, y, w, h = detection.bbox
    height, width = frame_bgr.shape[:2]
    x0 = max(0, x - padding_px)
    y0 = max(0, y - padding_px)
    x1 = min(width, x + w + padding_px)
    y1 = min(height, y + h + padding_px)
    if x1 <= x0 or y1 <= y0:
        return None
    return frame_bgr[y0:y1, x0:x1].copy()


def save_candidate_crop(crop_bgr: np.ndarray | None, path: str | Path) -> str | None:
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    Path(path).parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), crop_bgr)
    return str(path)


def _matching_color_tags(crop_bgr: np.ndarray, requested_colors: list[str]) -> list[str]:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    tags = []
    for color in requested_colors:
        ratio = _color_ratio(hsv, color)
        if ratio >= 0.08:
            tags.append(f"visual_color:{color}")
    return tags


def _color_ratio(hsv: np.ndarray, color: str) -> float:
    masks = {
        "red": [
            ((0, 80, 80), (10, 255, 255)),
            ((170, 80, 80), (180, 255, 255)),
        ],
        "orange": [((10, 80, 80), (24, 255, 255))],
        "yellow": [((24, 70, 70), (38, 255, 255))],
        "green": [((38, 50, 50), (85, 255, 255))],
        "blue": [((85, 50, 50), (130, 255, 255))],
        "purple": [((130, 45, 45), (160, 255, 255))],
        "black": [((0, 0, 0), (180, 255, 70))],
        "white": [((0, 0, 180), (180, 60, 255))],
        "gray": [((0, 0, 80), (180, 45, 200))],
        "grey": [((0, 0, 80), (180, 45, 200))],
    }
    ranges = masks.get(color)
    if not ranges:
        return 0.0
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
    return float(cv2.countNonZero(mask)) / float(mask.size)
