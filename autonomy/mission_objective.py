from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict

from autonomy.types import MissionObjective


COLORS = {
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "white",
    "black",
    "gray",
    "grey",
    "silver",
    "brown",
    "tan",
}

CATEGORY_HINTS = {
    "person": {"person", "people", "hiker", "child", "adult", "victim", "survivor"},
    "vehicle": {"vehicle", "car", "truck", "pickup", "suv", "van", "atv", "snowmobile"},
    "boat": {"boat", "ship", "vessel", "canoe", "kayak", "raft", "sailboat", "skiff", "t-top", "ttop", "center console"},
    "aircraft": {"aircraft", "plane", "helicopter", "drone"},
    "debris": {"debris", "wreckage", "crash", "crashed", "overturned"},
    "signal": {"flare", "smoke", "fire", "signal", "marker"},
}

URGENCY_HINTS = {
    "high": {"urgent", "emergency", "missing", "injured", "distress", "crashed", "sos"},
    "low": {"routine", "training", "practice", "survey"},
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "area",
    "for",
    "i",
    "in",
    "is",
    "it",
    "looking",
    "near",
    "of",
    "please",
    "search",
    "the",
    "this",
    "to",
}


def parse_mission_request(request: str) -> MissionObjective:
    cleaned = " ".join(request.strip().split())
    lowered = cleaned.lower()
    colors = sorted(color for color in COLORS if re.search(rf"\b{re.escape(color)}\b", lowered))
    categories = _extract_categories(lowered)
    urgency = _extract_urgency(lowered)
    search_area = _extract_search_area(cleaned)
    keywords = _extract_keywords(lowered)
    notes = [
        "Parsed locally with heuristic extraction.",
        "Use an open-vocabulary vision-language scorer for semantic target matching.",
    ]
    return MissionObjective(
        raw_request=cleaned,
        target_description=_extract_target_description(cleaned),
        search_area_description=search_area,
        urgency=urgency,
        confirmation_required=True,
        extracted_keywords=keywords,
        extracted_colors=colors,
        extracted_categories=categories,
        notes=notes,
    )


def objective_to_prompt(objective: MissionObjective) -> str:
    return (
        "Compare the candidate drone image to this search-and-rescue mission objective. "
        "Return a calibrated match score from 0.0 to 1.0, a short explanation, and whether "
        "human confirmation is recommended.\n\n"
        f"Mission request: {objective.raw_request}\n"
        f"Target description: {objective.target_description}\n"
        f"Extracted categories: {', '.join(objective.extracted_categories) or 'unknown'}\n"
        f"Extracted colors: {', '.join(objective.extracted_colors) or 'unknown'}\n"
        f"Search area: {objective.search_area_description or 'unspecified'}"
    )


def _has_term(text: str, term: str) -> bool:
    """Word-boundary match (with simple plural support) to avoid substring hits
    like 'car' in 'scar' or 'person' in 'personnel'."""
    return re.search(rf"\b{re.escape(term)}(?:s|es)?\b", text) is not None


def _extract_categories(text: str) -> list[str]:
    categories = []
    for category, hints in CATEGORY_HINTS.items():
        if any(_has_term(text, hint) for hint in hints):
            categories.append(category)
    return sorted(categories)


def _extract_urgency(text: str) -> str:
    if any(_has_term(text, hint) for hint in URGENCY_HINTS["high"]):
        return "high"
    if any(_has_term(text, hint) for hint in URGENCY_HINTS["low"]):
        return "low"
    return "normal"


def _extract_search_area(request: str) -> str | None:
    search_then_for = re.search(
        r"^\s*(?:search|scan|survey)\s+(?:the\s+)?(.+?)\s+for\s+.+$",
        request,
        flags=re.IGNORECASE,
    )
    if search_then_for:
        area = search_then_for.group(1).strip(" .")
        if area and area.lower() not in {"area", "this area"}:
            return area
    match = re.search(r"\b(?:near|around|within|along|by|at)\s+(.+)$", request, flags=re.IGNORECASE)
    return match.group(1).strip(" .") if match else None


def _extract_target_description(request: str) -> str:
    search_then_for = re.search(
        r"^\s*(?:search|scan|survey)\s+(?:the\s+)?(.+?)\s+for\s+(.+)$",
        request,
        flags=re.IGNORECASE,
    )
    if search_then_for and search_then_for.group(1).strip().lower() not in {"area", "this area"}:
        request = search_then_for.group(2)
    cleaned = re.sub(r"^\s*(?:search|look|find|locate)\s+(?:the\s+)?(?:area\s+)?(?:for\s+)?", "", request, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:near|around|within|along|by|at)\s+.+$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .") or request.strip()


def _extract_keywords(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9-]{2,}", text)
    return sorted({word for word in words if word not in STOPWORDS})[:24]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a natural-language SAR mission request")
    parser.add_argument("request", nargs="+")
    args = parser.parse_args()
    objective = parse_mission_request(" ".join(args.request))
    print(json.dumps(asdict(objective), indent=2))
    print()
    print(objective_to_prompt(objective))


if __name__ == "__main__":
    main()
