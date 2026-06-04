from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Protocol
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
    """Deterministic placeholder for the future vision-language model.

    This scorer does not claim make/model recognition. It uses objective text,
    candidate geometry, and simple visual cues to decide whether a candidate
    deserves human review while preserving the same interface a real VLM will use.
    """

    model_name = "local-semantic-placeholder-v1"

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
    ) -> None:
        self.model_name = model or os.environ.get("OPENAI_VISION_MODEL", "")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.detail = detail
        self.timeout_s = timeout_s
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
        with urlrequest.urlopen(req, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
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


def _semantic_prompt(objective: MissionObjective, *, context: str) -> str:
    return (
        f"You are scoring a {context} from a search-and-rescue drone. "
        "Compare the image to the mission request. Return ONLY JSON with keys: "
        "score number 0.0-1.0, decision one of REJECT/POSSIBLE_MATCH/LIKELY_MATCH/NEEDS_REVIEW, "
        "explanation string, tags array, needs_human_review boolean. "
        "Favor recall for SAR: uncertain plausible matches should be POSSIBLE_MATCH or NEEDS_REVIEW, "
        "not confidently rejected. Do not claim certainty for exact identity if the image is unclear.\n"
        f"Mission request: {objective.raw_request}\n"
        f"Target description: {objective.target_description}\n"
        f"Extracted colors: {', '.join(objective.extracted_colors) or 'unknown'}\n"
        f"Extracted categories: {', '.join(objective.extracted_categories) or 'unknown'}"
    )


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
