from __future__ import annotations

"""Mission-memory read-back: turn past analyst decisions into ranking priors.

Mission memory was previously write-only — it summarized past missions for
the dashboard but never influenced a subsequent mission. This module is the
influence layer. From prior reports and their analyst reviews it builds:

- term priors: tokens from confirmed candidates (image-name terms, proposal
  reasons, semantic tags) boost matching future candidates; tokens from
  repeatedly rejected candidates down-weight them. Learning from rejections
  is half the value: known clutter stops crowding the queue.
- location priors: in repeat missions over shared ground, candidates near a
  previously confirmed contact's (normalized) frame location get a bonus;
  candidates at repeatedly dismissed locations get a penalty.

The adjustment is deliberately bounded (±MAX_ADJUSTMENT on review priority)
and affects RANKING ONLY: no candidate is ever dropped from the queue by
memory. Memory nudges attention; the analyst still sees everything. The
effect of this module is measured by autonomy/memory_ablation.py — the same
missions with and without read-back.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


MAX_ADJUSTMENT = 0.15
TERM_BOOST_PER_HIT = 0.03
TERM_PENALTY_PER_HIT = 0.025
LOCATION_BOOST_MAX = 0.10
LOCATION_PENALTY_MAX = 0.08
DEFAULT_RADIUS_NORM = 0.10  # fraction of the frame diagonal

_IGNORED_TERMS = {
    "and", "with", "the", "for", "in", "on", "a", "an",
    "jpg", "png", "jpeg", "rf", "frame", "image", "img", "match",
    "proposal", "fallback", "full",
}


@dataclass
class MemoryPriors:
    confirmed_terms: Counter = field(default_factory=Counter)
    rejected_terms: Counter = field(default_factory=Counter)
    # (x_norm, y_norm, count) — normalized to frame width/height.
    confirmed_locations: list[tuple[float, float, int]] = field(default_factory=list)
    rejected_locations: list[tuple[float, float, int]] = field(default_factory=list)
    radius_norm: float = DEFAULT_RADIUS_NORM
    missions_observed: int = 0

    def is_empty(self) -> bool:
        return not (
            self.confirmed_terms
            or self.rejected_terms
            or self.confirmed_locations
            or self.rejected_locations
        )

    def as_dict(self) -> dict:
        return {
            "confirmed_terms": dict(self.confirmed_terms),
            "rejected_terms": dict(self.rejected_terms),
            "confirmed_locations": [list(item) for item in self.confirmed_locations],
            "rejected_locations": [list(item) for item in self.rejected_locations],
            "radius_norm": self.radius_norm,
            "missions_observed": self.missions_observed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryPriors":
        priors = cls(
            confirmed_terms=Counter(data.get("confirmed_terms") or {}),
            rejected_terms=Counter(data.get("rejected_terms") or {}),
            radius_norm=float(data.get("radius_norm") or DEFAULT_RADIUS_NORM),
            missions_observed=int(data.get("missions_observed") or 0),
        )
        priors.confirmed_locations = [tuple(item) for item in data.get("confirmed_locations") or []]
        priors.rejected_locations = [tuple(item) for item in data.get("rejected_locations") or []]
        return priors

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MemoryPriors":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# -- building priors from history ----------------------------------------------


def candidate_terms(result: dict) -> set[str]:
    """Stable tokens describing a candidate: name terms, reason, tags."""
    terms: set[str] = set()
    name = Path(str(result.get("image_path") or "")).stem.lower()
    name = re.sub(r"\.rf\.[a-f0-9]+", " ", name)
    for token in re.split(r"[^a-z0-9]+", name):
        if _is_term(token):
            terms.add(token)
    reason = str(result.get("proposal_reason") or "").lower()
    for token in re.split(r"[^a-z0-9]+", reason):
        if _is_term(token):
            terms.add(token)
    semantic = result.get("semantic") or {}
    for tag in semantic.get("tags") or []:
        tag = str(tag).lower()
        if tag.startswith("clip_top:"):
            for token in re.split(r"[^a-z0-9]+", tag.split(":", 1)[1]):
                if _is_term(token):
                    terms.add(token)
    return terms


def candidate_location_norm(result: dict, *, frame_size: tuple[int, int] | None = None) -> tuple[float, float] | None:
    """Candidate center normalized to [0,1]² when a bbox/frame size is known."""
    bbox = result.get("bbox")
    if bbox is None:
        return None
    x, y, w, h = bbox
    if frame_size:
        width, height = frame_size
    else:
        width = height = None
        size = result.get("frame_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            width, height = int(size[0]), int(size[1])
    if not width or not height:
        return None
    return ((x + w / 2.0) / width, (y + h / 2.0) / height)


def update_priors_from_mission(
    priors: MemoryPriors,
    *,
    results: list[dict],
    reviews: dict,
    frame_size: tuple[int, int] | None = None,
) -> MemoryPriors:
    """Fold one finished mission's analyst decisions into the priors.

    ``reviews`` maps candidate_id -> {"decision": approve|reject|investigate}
    (the dashboard's candidate_reviews.json shape). Investigations are
    neutral: uncertainty is not evidence in either direction.
    """
    by_id = {str(result.get("candidate_id")): result for result in results if result.get("candidate_id")}
    for candidate_id, review in (reviews or {}).items():
        result = by_id.get(str(candidate_id))
        if result is None:
            continue
        decision = str(review.get("decision") or review.get("status") or "").strip().lower()
        if decision in {"approve", "approved", "confirmed"}:
            priors.confirmed_terms.update(candidate_terms(result))
            location = candidate_location_norm(result, frame_size=frame_size)
            if location is not None:
                _merge_location(priors.confirmed_locations, location, priors.radius_norm)
        elif decision in {"reject", "rejected"}:
            priors.rejected_terms.update(candidate_terms(result))
            location = candidate_location_norm(result, frame_size=frame_size)
            if location is not None:
                _merge_location(priors.rejected_locations, location, priors.radius_norm)
    priors.missions_observed += 1
    return priors


def load_memory_priors(root: str | Path = ".") -> MemoryPriors:
    """Build priors from every reviewed vision report under logs/."""
    priors = MemoryPriors()
    for report_path in sorted(Path(root, "logs").glob("**/vision_report.json")):
        review_path = report_path.with_name("candidate_reviews.json")
        if not review_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            reviews = json.loads(review_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        update_priors_from_mission(priors, results=report.get("results") or [], reviews=reviews)
    return priors


# -- applying priors -------------------------------------------------------------


def memory_adjustment(
    priors: MemoryPriors | None,
    *,
    terms: set[str],
    location_norm: tuple[float, float] | None,
) -> tuple[float, list[str]]:
    """Bounded review-priority delta and human-readable reasons."""
    if priors is None or priors.is_empty():
        return 0.0, []
    delta = 0.0
    reasons: list[str] = []

    confirmed_hits = sorted(term for term in terms if priors.confirmed_terms.get(term))
    rejected_hits = sorted(term for term in terms if priors.rejected_terms.get(term))
    if confirmed_hits:
        boost = min(
            3 * TERM_BOOST_PER_HIT,
            sum(TERM_BOOST_PER_HIT * min(3, priors.confirmed_terms[t]) for t in confirmed_hits),
        )
        delta += boost
        reasons.append(f"memory: matches confirmed-contact cues ({', '.join(confirmed_hits[:3])})")
    if rejected_hits:
        penalty = min(
            3 * TERM_PENALTY_PER_HIT,
            sum(TERM_PENALTY_PER_HIT * min(3, priors.rejected_terms[t]) for t in rejected_hits),
        )
        delta -= penalty
        reasons.append(f"memory: matches dismissed-clutter cues ({', '.join(rejected_hits[:3])})")

    if location_norm is not None:
        confirmed_near = _nearest_count(priors.confirmed_locations, location_norm, priors.radius_norm)
        rejected_near = _nearest_count(priors.rejected_locations, location_norm, priors.radius_norm)
        if confirmed_near:
            delta += min(LOCATION_BOOST_MAX, 0.04 * confirmed_near)
            reasons.append(
                f"memory: near a previously confirmed contact location (seen {confirmed_near}x)"
            )
        if rejected_near:
            delta -= min(LOCATION_PENALTY_MAX, 0.03 * rejected_near)
            reasons.append(
                f"memory: location repeatedly dismissed by analysts ({rejected_near}x)"
            )

    delta = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, delta))
    return round(delta, 4), reasons


def _merge_location(
    locations: list[tuple[float, float, int]],
    point: tuple[float, float],
    radius_norm: float,
) -> None:
    for index, (x, y, count) in enumerate(locations):
        if _distance((x, y), point) <= radius_norm:
            new_count = count + 1
            locations[index] = (
                (x * count + point[0]) / new_count,
                (y * count + point[1]) / new_count,
                new_count,
            )
            return
    locations.append((point[0], point[1], 1))


def _nearest_count(
    locations: list[tuple[float, float, int]],
    point: tuple[float, float],
    radius_norm: float,
) -> int:
    best = 0
    for x, y, count in locations:
        if _distance((x, y), point) <= radius_norm:
            best = max(best, count)
    return best


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _is_term(token: str) -> bool:
    if len(token) <= 2 or token in _IGNORED_TERMS:
        return False
    if token.isdigit():
        return False
    if re.fullmatch(r"[a-f0-9]{8,}", token):
        return False
    return any(char.isalpha() for char in token)
