from __future__ import annotations

"""Mission-memory ablation study.

Measures what memory read-back is actually worth by running the same
sequence of missions twice, identical in every way except one:

    Condition A (memory OFF): every mission runs cold and independent.
    Condition B (memory ON):  missions run in order; after each one, analyst
                              decisions are simulated from ground truth
                              (confirm true contacts, dismiss clutter) and
                              folded into priors that influence the NEXT
                              mission's candidate ranking.

Same detectors, same thresholds, same frames — the only difference is the
read-back, which is what makes this an ablation rather than a vibes
comparison.

Reported per condition, with repeat and novel targets SEPARATED — a memory
prior that helps re-find known contacts can plausibly hurt the genuinely new
one, and that trade-off must be visible, not averaged away:

- capture rate on repeat targets (seen in a previous mission) vs novel ones
- mean shortlist rank of true targets and precision@k
- analyst workload: shortlisted candidates per true target found

The harness ships with a synthetic mission-sequence generator (a persistent
moored vessel, persistent shoreline clutter, and a novel contact in the
final pass) so the study runs end-to-end offline:

    python3 -m autonomy.memory_ablation --generate-demo --output-dir logs/memory_ablation
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from autonomy.detection_metrics import iou, parse_gt_boxes
from autonomy.memory_priors import MemoryPriors, update_priors_from_mission
from autonomy.vision_lab import run_vision_lab


MATCH_IOU = 0.30
PRECISION_K = 3


# -- mission sequence spec -------------------------------------------------------


@dataclass(frozen=True)
class MissionSpec:
    name: str
    request: str
    images_dir: Path
    labels_csv: Path  # filename,label,gt_boxes,target_id

    @classmethod
    def from_dict(cls, data: dict, base: Path) -> "MissionSpec":
        return cls(
            name=str(data["name"]),
            request=str(data["request"]),
            images_dir=base / data["images_dir"],
            labels_csv=base / data["labels_csv"],
        )


def load_sequence(config_path: Path) -> list[MissionSpec]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    return [MissionSpec.from_dict(item, base) for item in data["missions"]]


def read_ground_truth(labels_csv: Path) -> dict[str, list[tuple[tuple[int, int, int, int], str]]]:
    """filename -> [(gt_box, target_id), ...]"""
    import csv

    truth: dict[str, list[tuple[tuple[int, int, int, int], str]]] = {}
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            boxes = parse_gt_boxes(row.get("gt_boxes"))
            target_ids = [part.strip() for part in str(row.get("target_id") or "").split(";") if part.strip()]
            entries = []
            for index, box in enumerate(boxes):
                target_id = target_ids[index] if index < len(target_ids) else f"{row['filename']}#{index}"
                entries.append((box, target_id))
            truth[row["filename"]] = entries
    return truth


# -- running one mission ----------------------------------------------------------


def run_mission(spec: MissionSpec, *, output_dir: Path, priors: MemoryPriors | None) -> dict:
    image_paths = sorted(
        path for path in spec.images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    report_path = run_vision_lab(
        mission_request=spec.request,
        image_paths=image_paths,
        output_dir=output_dir / spec.name,
        memory_priors=priors,
        save_only_detections=True,
    )
    return json.loads(report_path.read_text(encoding="utf-8"))


def simulate_analyst_reviews(report: dict, truth: dict) -> dict:
    """Ground-truth-driven reviews: confirm IoU matches, reject the rest.

    Mirrors what a real analyst does with the shortlist; using ground truth
    keeps the simulation deterministic and condition-neutral.
    """
    reviews: dict[str, dict] = {}
    for result in report.get("results", []):
        if not result.get("detected") or not result.get("candidate_id"):
            continue
        bbox = result.get("bbox")
        filename = Path(str(result.get("image_path") or "")).name
        gt_entries = truth.get(filename, [])
        matched = bbox is not None and any(
            iou(tuple(bbox), gt_box) >= MATCH_IOU for gt_box, _ in gt_entries
        )
        reviews[str(result["candidate_id"])] = {
            "decision": "approve" if matched else "reject",
            "reason_tag": "vehicle_visible" if matched else "false_alarm",
        }
    return reviews


# -- scoring a condition ----------------------------------------------------------


def evaluate_mission(report: dict, truth: dict, *, seen_target_ids: set[str]) -> dict:
    """Per-mission tallies, with repeat/novel targets separated."""
    shortlist = report.get("summary", {}).get("shortlist") or []
    ranked_ids = [entry.get("candidate_id") for entry in shortlist]
    results_by_id = {
        str(result.get("candidate_id")): result
        for result in report.get("results", [])
        if result.get("candidate_id")
    }

    target_hits: dict[str, int | None] = {}  # target_id -> best shortlist rank
    for filename, entries in truth.items():
        for gt_box, target_id in entries:
            best_rank = None
            for rank, candidate_id in enumerate(ranked_ids, start=1):
                result = results_by_id.get(str(candidate_id))
                if result is None or result.get("bbox") is None:
                    continue
                if Path(str(result.get("image_path") or "")).name != filename:
                    continue
                if iou(tuple(result["bbox"]), gt_box) >= MATCH_IOU:
                    best_rank = rank if best_rank is None else min(best_rank, rank)
            target_hits[target_id] = best_rank

    repeat = {tid: rank for tid, rank in target_hits.items() if tid in seen_target_ids}
    novel = {tid: rank for tid, rank in target_hits.items() if tid not in seen_target_ids}

    top_k = ranked_ids[:PRECISION_K]
    true_in_top_k = 0
    for candidate_id in top_k:
        result = results_by_id.get(str(candidate_id))
        if result is None or result.get("bbox") is None:
            continue
        filename = Path(str(result.get("image_path") or "")).name
        if any(iou(tuple(result["bbox"]), gt_box) >= MATCH_IOU for gt_box, _ in truth.get(filename, [])):
            true_in_top_k += 1

    found_ranks = [rank for rank in target_hits.values() if rank is not None]
    return {
        "targets_total": len(target_hits),
        "targets_found": len(found_ranks),
        "repeat_targets_total": len(repeat),
        "repeat_targets_found": sum(1 for rank in repeat.values() if rank is not None),
        "novel_targets_total": len(novel),
        "novel_targets_found": sum(1 for rank in novel.values() if rank is not None),
        "mean_true_target_rank": round(sum(found_ranks) / len(found_ranks), 2) if found_ranks else None,
        # Workload proxy: how deep the analyst must read the ranked queue to
        # have seen every real target — the metric memory reordering moves.
        "rank_of_last_true_target": max(found_ranks) if found_ranks else None,
        "precision_at_k": round(true_in_top_k / len(top_k), 4) if top_k else None,
        "shortlist_count": len(ranked_ids),
        "target_ids_present": sorted(target_hits.keys()),
    }


def aggregate_condition(mission_evaluations: list[dict]) -> dict:
    def total(key: str) -> int:
        return sum(item[key] for item in mission_evaluations)

    found = total("targets_found")
    ranks = [item["mean_true_target_rank"] for item in mission_evaluations if item["mean_true_target_rank"] is not None]
    last_ranks = [item["rank_of_last_true_target"] for item in mission_evaluations if item["rank_of_last_true_target"] is not None]
    precisions = [item["precision_at_k"] for item in mission_evaluations if item["precision_at_k"] is not None]
    shortlisted = total("shortlist_count")
    return {
        "mean_rank_of_last_true_target": round(sum(last_ranks) / len(last_ranks), 2) if last_ranks else None,
        "capture_rate": round(found / total("targets_total"), 4) if total("targets_total") else None,
        "repeat_capture_rate": (
            round(total("repeat_targets_found") / total("repeat_targets_total"), 4)
            if total("repeat_targets_total")
            else None
        ),
        "novel_capture_rate": (
            round(total("novel_targets_found") / total("novel_targets_total"), 4)
            if total("novel_targets_total")
            else None
        ),
        "mean_true_target_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
        "mean_precision_at_k": round(sum(precisions) / len(precisions), 4) if precisions else None,
        "shortlisted_per_target_found": round(shortlisted / found, 2) if found else None,
        "shortlisted_total": shortlisted,
        "targets_found": found,
        "targets_total": total("targets_total"),
    }


# -- the ablation -----------------------------------------------------------------


def run_ablation(missions: list[MissionSpec], *, output_dir: Path) -> dict:
    conditions: dict[str, dict] = {}
    for condition, memory_on in (("memory_off", False), ("memory_on", True)):
        priors = MemoryPriors() if memory_on else None
        seen_target_ids: set[str] = set()
        per_mission = []
        for spec in missions:
            truth = read_ground_truth(spec.labels_csv)
            report = run_mission(spec, output_dir=output_dir / condition, priors=priors)
            evaluation = evaluate_mission(report, truth, seen_target_ids=seen_target_ids)
            evaluation["mission"] = spec.name
            evaluation["memory_priors_active"] = bool(report.get("memory_priors_active"))
            per_mission.append(evaluation)
            if memory_on:
                reviews = simulate_analyst_reviews(report, truth)
                update_priors_from_mission(priors, results=report.get("results") or [], reviews=reviews)
            for entries in truth.values():
                seen_target_ids.update(target_id for _, target_id in entries)
        conditions[condition] = {
            "per_mission": per_mission,
            "aggregate": aggregate_condition(per_mission),
        }

    study = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "missions": [spec.name for spec in missions],
        "match_iou": MATCH_IOU,
        "precision_k": PRECISION_K,
        "conditions": conditions,
        "delta": _delta(conditions),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "memory_ablation.json").write_text(json.dumps(study, indent=2), encoding="utf-8")
    (output_dir / "memory_ablation.md").write_text(render_markdown(study), encoding="utf-8")
    return study


def _delta(conditions: dict) -> dict:
    off = conditions["memory_off"]["aggregate"]
    on = conditions["memory_on"]["aggregate"]
    delta = {}
    for key in (
        "capture_rate",
        "repeat_capture_rate",
        "novel_capture_rate",
        "mean_true_target_rank",
        "mean_precision_at_k",
        "shortlisted_per_target_found",
    ):
        if off.get(key) is not None and on.get(key) is not None:
            delta[key] = round(on[key] - off[key], 4)
    return delta


def render_markdown(study: dict) -> str:
    off = study["conditions"]["memory_off"]["aggregate"]
    on = study["conditions"]["memory_on"]["aggregate"]
    delta = study["delta"]

    def fmt(value) -> str:
        return "—" if value is None else f"{value}"

    lines = [
        "# Mission-Memory Ablation",
        "",
        f"Missions: {', '.join(study['missions'])} | match IoU ≥ {study['match_iou']} | k = {study['precision_k']}",
        "",
        "| Metric | Memory OFF | Memory ON | Δ (on − off) |",
        "|---|---|---|---|",
    ]
    rows = [
        ("Capture rate (all targets)", "capture_rate"),
        ("Capture rate (repeat targets)", "repeat_capture_rate"),
        ("Capture rate (novel targets)", "novel_capture_rate"),
        ("Mean true-target shortlist rank (lower is better)", "mean_true_target_rank"),
        (f"Precision@{study['precision_k']}", "mean_precision_at_k"),
        ("Shortlisted candidates per target found (workload)", "shortlisted_per_target_found"),
    ]
    for label, key in rows:
        lines.append(f"| {label} | {fmt(off.get(key))} | {fmt(on.get(key))} | {fmt(delta.get(key))} |")
    lines += [
        "",
        "Read novel-target capture before celebrating: a memory prior that",
        "boosts known contacts can rank the genuinely new contact lower. If",
        "novel capture dropped, that trade-off is the finding — report it.",
    ]
    return "\n".join(lines) + "\n"


# -- synthetic demo sequence -------------------------------------------------------


def generate_demo_sequence(base_dir: Path, *, missions: int = 4, frames_per_mission: int = 6) -> Path:
    """Sequential shoreline missions over shared ground, fully synthetic.

    Persistent across missions: a red vessel moored near the same spot
    (target_id=moored_vessel) and red-brown clutter on the shoreline that
    proposals fire on but analysts reject every time. The final mission adds
    a novel contact (target_id=novel_contact) at a fresh location.
    """
    rng = np.random.default_rng(7)
    width, height = 320, 240
    spec_entries = []
    for mission_index in range(missions):
        mission_name = f"pass_{mission_index + 1:02d}"
        images_dir = base_dir / mission_name / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for frame_index in range(frames_per_mission):
            frame = _shoreline_background(width, height, rng)
            gt_boxes: list[str] = []
            target_ids: list[str] = []
            # Persistent clutter: reddish rocks, fixed spot, every frame.
            _draw_blob(frame, center=(70, 185), size=12, color=(40, 45, 150), rng=rng)
            # The moored vessel appears in half the frames, near-fixed spot.
            if frame_index % 2 == 0:
                jitter = rng.integers(-4, 5, size=2)
                x, y = 230 + int(jitter[0]), 95 + int(jitter[1])
                _draw_vessel(frame, x=x, y=y, color=(30, 30, 200))
                gt_boxes.append(f"{x},{y},34,14")
                target_ids.append("moored_vessel")
            # Novel contact only in the final mission, different location.
            if mission_index == missions - 1 and frame_index in (1, 3):
                _draw_vessel(frame, x=60, y=60, color=(35, 35, 190))
                gt_boxes.append("60,60,34,14")
                target_ids.append("novel_contact")
            filename = f"frame_{frame_index:03d}.png"
            cv2.imwrite(str(images_dir / filename), frame)
            rows.append(
                {
                    "filename": filename,
                    "label": 1 if gt_boxes else 0,
                    "gt_boxes": ";".join(gt_boxes),
                    "target_id": ";".join(target_ids),
                }
            )
        labels_csv = base_dir / mission_name / "labels.csv"
        import csv

        with labels_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "label", "gt_boxes", "target_id"])
            writer.writeheader()
            writer.writerows(rows)
        spec_entries.append(
            {
                "name": mission_name,
                "request": "Monitor the protected shoreline for a red vessel",
                "images_dir": f"{mission_name}/images",
                "labels_csv": f"{mission_name}/labels.csv",
            }
        )
    config_path = base_dir / "sequence.json"
    config_path.write_text(json.dumps({"missions": spec_entries}, indent=2), encoding="utf-8")
    return config_path


def _shoreline_background(width: int, height: int, rng) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = 130  # water: blue-ish
    frame[:, :, 1] = 105
    frame[:, :, 2] = 70
    frame[170:, :, 0] = 90  # shoreline band
    frame[170:, :, 1] = 120
    frame[170:, :, 2] = 110
    noise = rng.integers(-12, 13, size=frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _draw_vessel(frame: np.ndarray, *, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (x, y), (x + 33, y + 13), color, thickness=-1)
    cv2.rectangle(frame, (x + 10, y - 5), (x + 22, y), (60, 60, 60), thickness=-1)


def _draw_blob(frame: np.ndarray, *, center: tuple[int, int], size: int, color: tuple[int, int, int], rng) -> None:
    cv2.circle(frame, center, size, color, thickness=-1)
    cv2.circle(frame, (center[0] + size // 2, center[1] + 3), size // 2, color, thickness=-1)


# -- CLI ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Mission-memory ablation study")
    parser.add_argument("--sequence", type=Path, default=None, help="sequence.json describing the mission sequence")
    parser.add_argument("--generate-demo", action="store_true", help="Generate the synthetic demo sequence first")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/memory_ablation"))
    args = parser.parse_args()

    sequence_path = args.sequence
    if args.generate_demo:
        sequence_path = generate_demo_sequence(args.output_dir / "demo_data")
        print(f"Demo sequence generated: {sequence_path}")
    if sequence_path is None:
        raise SystemExit("Provide --sequence or use --generate-demo.")
    study = run_ablation(load_sequence(sequence_path), output_dir=args.output_dir)
    print(json.dumps({"delta": study["delta"]}, indent=2))
    print(f"Study written to {args.output_dir}/memory_ablation.md")


if __name__ == "__main__":
    main()
