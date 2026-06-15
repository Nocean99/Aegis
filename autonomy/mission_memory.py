from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def build_mission_memory(root: str | Path = ".") -> dict:
    root_path = Path(root)
    report_paths = sorted((root_path / "logs").glob("**/vision_report.json"), reverse=True)
    acoustic_memory = build_acoustic_memory(root_path)
    mission_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    false_positive_terms: Counter[str] = Counter()
    false_negative_terms: Counter[str] = Counter()
    false_positive_causes: Counter[str] = Counter()
    confirmed_positive_indicators: Counter[str] = Counter()
    uncertainty_causes: Counter[str] = Counter()
    category_metrics: dict[str, list[dict]] = defaultdict(list)
    recent_reports = []

    for report_path in report_paths:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        objective = report.get("objective") or {}
        categories = objective.get("extracted_categories") or ["uncategorized"]
        mission_type = objective.get("mission_type") or "unknown"
        mission_counts[mission_type] += 1

        evaluation = report.get("evaluation") or {}
        analyst_capture = evaluation.get("analyst_capture") or {}
        for category in categories:
            category_metrics[str(category)].append(
                {
                    "precision": evaluation.get("precision"),
                    "recall": evaluation.get("recall"),
                    "f1": evaluation.get("f1"),
                    "capture_recall": analyst_capture.get("recall"),
                    "capture_f1": analyst_capture.get("f1"),
                }
            )

        for item in evaluation.get("false_positives") or []:
            false_positive_terms.update(_name_terms(item))
        for item in evaluation.get("false_negatives") or []:
            false_negative_terms.update(_name_terms(item))

        review_path = report_path.with_name("candidate_reviews.json")
        if review_path.exists():
            try:
                reviews = json.loads(review_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                reviews = {}
            for review in reviews.values():
                decision = normalize_review_decision(review)
                review_counts[decision] += 1
                reason_tag = normalize_reason_tag(review.get("reason_tag") or review.get("reason"))
                if reason_tag:
                    if decision == "reject":
                        false_positive_causes[reason_tag] += 1
                    elif decision == "approve":
                        confirmed_positive_indicators[reason_tag] += 1
                    else:
                        uncertainty_causes[reason_tag] += 1

        if len(recent_reports) < 10:
            summary = report.get("summary") or {}
            recent_reports.append(
                {
                    "path": _display_path(report_path, root_path),
                    "timestamp": report.get("timestamp"),
                    "mission_request": report.get("mission_request"),
                    "processed": summary.get("processed"),
                    "shortlist_count": summary.get("shortlist_count"),
                    "precision": evaluation.get("precision"),
                    "recall": evaluation.get("recall"),
                    "capture_recall": analyst_capture.get("recall"),
                }
            )

    return {
        "report_count": len(report_paths),
        "mission_types": dict(mission_counts.most_common()),
        "review_statuses": dict(review_counts.most_common()),
        "category_metrics": summarize_category_metrics(category_metrics),
        "common_false_positive_terms": dict(false_positive_terms.most_common(12)),
        "common_false_negative_terms": dict(false_negative_terms.most_common(12)),
        "mission_memory_v2": mission_memory_v2_snapshot(
            false_positive_causes=false_positive_causes,
            confirmed_positive_indicators=confirmed_positive_indicators,
            uncertainty_causes=uncertainty_causes,
            category_metrics=category_metrics,
        ),
        "mission_memory_v1": mission_memory_snapshot_from_patterns(
            false_positive_terms=false_positive_terms,
            false_negative_terms=false_negative_terms,
            weak_categories=_weak_categories(category_metrics),
        ),
        "recent_reports": recent_reports,
        "recommendations": memory_recommendations(false_positive_terms, false_negative_terms, category_metrics),
        "acoustic_memory": acoustic_memory,
    }


def build_acoustic_memory(root_path: Path) -> dict:
    recurring_false_positives: Counter[str] = Counter()
    recurring_uncertainty: Counter[str] = Counter()
    proposal_reasons: Counter[str] = Counter()
    report_count = 0
    candidate_count = 0
    for report_path in sorted((root_path / "logs").glob("**/acoustic_report.json"), reverse=True):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        report_count += 1
        candidates = report.get("candidates") or []
        candidate_count += len(candidates)
        proposal_reasons.update(str(item.get("proposal_reason") or "unknown") for item in candidates)
        evaluation = report.get("evaluation") or {}
        recurring_false_positives.update(evaluation.get("false_positive_causes") or {})
        recurring_uncertainty.update(evaluation.get("uncertainty_causes") or {})
        review_path = report_path.with_name("candidate_reviews.json")
        if review_path.exists():
            try:
                reviews = json.loads(review_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                reviews = {}
            for review in reviews.values():
                tag = normalize_reason_tag(review.get("reason_tag") or review.get("reason"))
                if not tag:
                    continue
                decision = normalize_review_decision(review)
                if decision == "reject":
                    recurring_false_positives[tag] += 1
                elif decision == "investigate":
                    recurring_uncertainty[tag] += 1
    return {
        "report_count": report_count,
        "candidate_count": candidate_count,
        "proposal_reasons": pretty_counts(proposal_reasons),
        "recurring_acoustic_false_positives": pretty_counts(recurring_false_positives),
        "recurring_acoustic_uncertainty": pretty_counts(recurring_uncertainty),
        "recommended_acoustic_data": recommended_acoustic_data(recurring_false_positives, recurring_uncertainty),
    }


def mission_memory_snapshot(report: dict, reviews: dict | None = None) -> dict:
    evaluation = report.get("evaluation") or {}
    objective = report.get("objective") or {}
    categories = [str(item) for item in objective.get("extracted_categories") or []]
    false_positive_terms: Counter[str] = Counter()
    false_negative_terms: Counter[str] = Counter()
    review_reasons: Counter[str] = Counter()
    false_positive_causes: Counter[str] = Counter()
    confirmed_positive_indicators: Counter[str] = Counter()
    uncertainty_causes: Counter[str] = Counter()
    for item in evaluation.get("false_positives") or []:
        false_positive_terms.update(_name_terms(item))
    for item in evaluation.get("false_negatives") or []:
        false_negative_terms.update(_name_terms(item))
    for review in (reviews or {}).values():
        reason = str(review.get("reason") or "").strip().lower()
        if reason:
            review_reasons[reason] += 1
        reason_tag = normalize_reason_tag(review.get("reason_tag") or reason)
        if reason_tag:
            decision = normalize_review_decision(review)
            if decision == "reject":
                false_positive_causes[reason_tag] += 1
            elif decision == "approve":
                confirmed_positive_indicators[reason_tag] += 1
            else:
                uncertainty_causes[reason_tag] += 1

    analyst_capture = evaluation.get("analyst_capture") or {}
    weak_categories = []
    if categories and (analyst_capture.get("recall") is not None) and float(analyst_capture.get("recall") or 0.0) < 0.9:
        weak_categories = categories

    snapshot = mission_memory_snapshot_from_patterns(
        false_positive_terms=false_positive_terms + review_reasons,
        false_negative_terms=false_negative_terms,
        weak_categories=weak_categories,
    )
    snapshot["analyst_decision_patterns"] = dict(review_reasons.most_common(8))
    snapshot["memory_v2"] = mission_memory_v2_snapshot(
        false_positive_causes=false_positive_causes,
        confirmed_positive_indicators=confirmed_positive_indicators,
        uncertainty_causes=uncertainty_causes,
        category_metrics={category: [{"capture_recall": analyst_capture.get("recall")}] for category in categories},
    )
    return snapshot


def mission_memory_v2_snapshot(
    *,
    false_positive_causes: Counter[str],
    confirmed_positive_indicators: Counter[str],
    uncertainty_causes: Counter[str],
    category_metrics: dict[str, list[dict]],
) -> dict:
    weak_categories = _weak_categories(category_metrics)
    modality_lessons = sensor_modality_lessons(category_metrics)
    return {
        "common_false_positive_causes": pretty_counts(false_positive_causes),
        "confirmed_positive_indicators": pretty_counts(confirmed_positive_indicators),
        "common_uncertainty_causes": pretty_counts(uncertainty_causes),
        "weak_categories": sorted(set(weak_categories)),
        "lessons": memory_v2_lessons(
            false_positive_causes=false_positive_causes,
            confirmed_positive_indicators=confirmed_positive_indicators,
            uncertainty_causes=uncertainty_causes,
            weak_categories=weak_categories,
        )
        + modality_lessons,
        "sensor_modality_lessons": modality_lessons,
    }


def memory_v2_lessons(
    *,
    false_positive_causes: Counter[str],
    confirmed_positive_indicators: Counter[str],
    uncertainty_causes: Counter[str],
    weak_categories: list[str],
) -> list[str]:
    lessons = []
    if false_positive_causes:
        causes = ", ".join(label for label, _ in false_positive_causes.most_common(3))
        lessons.append(f"Common false-positive causes: {causes}.")
    if confirmed_positive_indicators:
        indicators = ", ".join(label for label, _ in confirmed_positive_indicators.most_common(3))
        lessons.append(f"Confirmed target indicators: {indicators}.")
    if uncertainty_causes:
        causes = ", ".join(label for label, _ in uncertainty_causes.most_common(3))
        lessons.append(f"Common uncertainty causes: {causes}.")
    if weak_categories:
        lessons.append(f"Lower-capture categories need more data: {', '.join(sorted(set(weak_categories)))}.")
    if not lessons:
        lessons.append("No analyst reason-tag pattern is strong enough yet; keep tagging reviews.")
    return lessons


def sensor_modality_lessons(category_metrics: dict[str, list[dict]]) -> list[str]:
    categories = {str(category).lower() for category in category_metrics}
    if "vehicle" not in categories and "vehicles" not in categories:
        return []
    return [
        "RGB vehicle evidence benefits from selective API semantic review.",
        "Infrared vehicle evidence currently performs better with local hot-blob triage.",
        "API thermal review may need a stricter prompt or stricter NEEDS_REVIEW threshold.",
    ]


def pretty_counts(counter: Counter[str]) -> dict[str, int]:
    return {tag.replace("_", " "): count for tag, count in counter.most_common(8)}


def mission_memory_snapshot_from_patterns(
    *,
    false_positive_terms: Counter[str],
    false_negative_terms: Counter[str],
    weak_categories: list[str],
) -> dict:
    false_positive_patterns = [term for term, _ in false_positive_terms.most_common(8)]
    miss_patterns = [term for term, _ in false_negative_terms.most_common(8)]
    recommended_data = recommended_data_from_patterns(false_positive_patterns, miss_patterns, weak_categories)
    return {
        "false_positive_patterns": false_positive_patterns,
        "miss_patterns": miss_patterns,
        "weak_categories": sorted(set(weak_categories)),
        "recommended_data": recommended_data,
    }


def summarize_category_metrics(category_metrics: dict[str, list[dict]]) -> dict:
    return {
        category: {
            "runs": len(metrics),
            "avg_precision": _avg(item.get("precision") for item in metrics),
            "avg_recall": _avg(item.get("recall") for item in metrics),
            "avg_f1": _avg(item.get("f1") for item in metrics),
            "avg_capture_recall": _avg(item.get("capture_recall") for item in metrics),
            "avg_capture_f1": _avg(item.get("capture_f1") for item in metrics),
        }
        for category, metrics in sorted(category_metrics.items())
    }


def memory_recommendations(
    false_positive_terms: Counter[str],
    false_negative_terms: Counter[str],
    category_metrics: dict[str, list[dict]],
) -> list[str]:
    recommendations = []
    if false_positive_terms:
        terms = ", ".join(term for term, _ in false_positive_terms.most_common(3))
        recommendations.append(f"Review ranking should down-rank recurring false-positive cues: {terms}.")
    if false_negative_terms:
        terms = ", ".join(term for term, _ in false_negative_terms.most_common(3))
        recommendations.append(f"Keep full-frame fallback active for recurring misses involving: {terms}.")
    weak_categories = _weak_categories(category_metrics)
    if weak_categories:
        recommendations.append(f"Add more labeled examples for lower-capture categories: {', '.join(sorted(weak_categories))}.")
    if not recommendations:
        recommendations.append("Current reports show no obvious repeated failure pattern yet; add broader labeled missions next.")
    return recommendations


def recommended_data_from_patterns(
    false_positive_patterns: list[str],
    miss_patterns: list[str],
    weak_categories: list[str],
) -> list[str]:
    recommendations = []
    if weak_categories:
        recommendations.extend(f"more labeled {category} examples with hard negatives" for category in sorted(set(weak_categories)))
    if miss_patterns:
        recommendations.append(f"positive examples for recurring misses: {', '.join(miss_patterns[:3])}")
    if false_positive_patterns:
        recommendations.append(f"hard negatives for recurring false positives: {', '.join(false_positive_patterns[:3])}")
    if not recommendations:
        recommendations.append("broader labeled missions with positives, near misses, and hard negatives")
    return recommendations[:6]


def _weak_categories(category_metrics: dict[str, list[dict]]) -> list[str]:
    return [
        category
        for category, metrics in category_metrics.items()
        if (_avg(item.get("capture_recall") for item in metrics) or 0.0) < 0.9
    ]


def _name_terms(item) -> list[str]:
    if isinstance(item, dict):
        path = item.get("image_path") or ""
    else:
        path = str(item)
    name = Path(path).stem.lower()
    name = re.sub(r"\.rf\.[a-f0-9]+", " ", name)
    tokens = re.split(r"[^a-z0-9]+", name)
    ignored = {"and", "with", "the", "for", "in", "on", "a", "an", "jpg", "png", "jpeg", "rf"}
    return [part for part in tokens if is_memory_term(part, ignored)]


def is_memory_term(term: str, ignored: set[str]) -> bool:
    if len(term) <= 2 or term in ignored:
        return False
    if term.isdigit():
        return False
    if re.fullmatch(r"\d{4,}", term):
        return False
    if re.fullmatch(r"gss\d+", term):
        return False
    if re.fullmatch(r"[a-f0-9]{8,}", term):
        return False
    return any(char.isalpha() for char in term)


def normalize_review_decision(review: dict) -> str:
    decision = str(review.get("decision") or review.get("status") or "unknown").strip().lower()
    aliases = {
        "approved": "approve",
        "confirmed": "approve",
        "rejected": "reject",
        "needs_review": "investigate",
        "needs closer look": "investigate",
        "needs_closer_look": "investigate",
    }
    decision = aliases.get(decision, decision)
    if decision not in {"approve", "reject", "investigate"}:
        return "unknown"
    return decision


def normalize_reason_tag(value) -> str:
    tag = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    allowed = {
        "person_visible",
        "vehicle_visible",
        "too_small",
        "vegetation",
        "shadow",
        "debris",
        "rooftop",
        "road_marking",
        "building",
        "hot_object",
        "thermal_clutter",
        "wave_noise",
        "dock_machinery",
        "wind_noise",
        "low_snr",
        "overlapping_signals",
        "vessel_sound",
        "acoustic_clutter",
        "false_alarm",
        "uncertain_vehicle",
    }
    return tag if tag in allowed else ""


def recommended_acoustic_data(false_positive_causes: Counter[str], uncertainty_causes: Counter[str]) -> list[str]:
    recommendations = ["vessel passes", "quiet harbour negatives", "storm/wave negatives"]
    if false_positive_causes.get("dock_machinery"):
        recommendations.append("dock machinery negatives")
    if false_positive_causes.get("wind_noise"):
        recommendations.append("wind noise negatives")
    if uncertainty_causes.get("low_snr"):
        recommendations.append("low-SNR vessel clips")
    if uncertainty_causes.get("overlapping_signals"):
        recommendations.append("overlapping vessel and ambient sound clips")
    return list(dict.fromkeys(recommendations))


def _avg(values) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 4)


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
