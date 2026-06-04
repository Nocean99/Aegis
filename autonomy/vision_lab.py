from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from autonomy.config_loader import load_search_mission_config
from autonomy.color_proposal_detector import MissionColorProposalDetector
from autonomy.mission_objective import parse_mission_request
from autonomy.mission_vision_plan import create_mission_vision_plan
from autonomy.objectness_proposal_detector import ObjectnessProposalDetector
from autonomy.red_block_detector import RedBlockDetector
from autonomy.semantic_vision import LocalSemanticVisionScorer, OpenAIVisionLanguageScorer, crop_detection, save_candidate_crop


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def run_vision_lab(
    *,
    mission_request: str,
    image_paths: list[Path],
    config_path: str | Path = "config/autonomy.yaml",
    output_dir: str | Path = "logs/vision_lab",
    save_only_detections: bool = False,
    proposal_mode: str = "mission-color",
    max_saved_candidates: int = 50,
    min_shortlist_score: float = 0.25,
    labels_csv: str | Path | None = None,
    eval_threshold: float = 0.25,
    semantic_vision: str = "local",
    openai_model: str | None = None,
    openai_detail: str = "auto",
    openai_timeout_s: float = 45.0,
    full_frame_semantic: str = "off",
) -> Path:
    config = load_search_mission_config(config_path)
    objective = parse_mission_request(mission_request)
    vision_plan = create_mission_vision_plan(objective)
    detector = RedBlockDetector(config.target)
    color_detector = MissionColorProposalDetector(vision_plan, min_area_px=max(25, int(config.target.min_area_px * 0.15)))
    scorer = LocalSemanticVisionScorer()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    frame_items = []
    for image_path in image_paths:
        frame_items.append({"source_path": image_path, "frame_index": None, "timestamp_s": None})
    return _run_vision_lab_on_frames(
        mission_request=mission_request,
        frame_items=frame_items,
        config_path=config_path,
        output_dir=output_dir,
        save_only_detections=save_only_detections,
        proposal_mode=proposal_mode,
        max_saved_candidates=max_saved_candidates,
        min_shortlist_score=min_shortlist_score,
        labels_csv=labels_csv,
        eval_threshold=eval_threshold,
        semantic_vision=semantic_vision,
        openai_model=openai_model,
        openai_detail=openai_detail,
        openai_timeout_s=openai_timeout_s,
        full_frame_semantic=full_frame_semantic,
        source_type="images",
    )


def run_video_vision_lab(
    *,
    mission_request: str,
    video_path: str | Path,
    config_path: str | Path = "config/autonomy.yaml",
    output_dir: str | Path = "logs/vision_lab",
    save_only_detections: bool = True,
    sample_every_s: float = 1.0,
    max_frames: int | None = None,
    proposal_mode: str = "mission-color",
    max_saved_candidates: int = 50,
    min_shortlist_score: float = 0.25,
    labels_csv: str | Path | None = None,
    eval_threshold: float = 0.25,
    semantic_vision: str = "local",
    openai_model: str | None = None,
    openai_detail: str = "auto",
    openai_timeout_s: float = 45.0,
    full_frame_semantic: str = "off",
) -> Path:
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * sample_every_s)))
    frame_items = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            frame_items.append(
                {
                    "source_path": video_path,
                    "frame_index": frame_index,
                    "timestamp_s": round(frame_index / fps, 3),
                    "frame": frame,
                }
            )
            if max_frames is not None and len(frame_items) >= max_frames:
                break
        frame_index += 1
    capture.release()
    return _run_vision_lab_on_frames(
        mission_request=mission_request,
        frame_items=frame_items,
        config_path=config_path,
        output_dir=output_dir,
        save_only_detections=save_only_detections,
        proposal_mode=proposal_mode,
        max_saved_candidates=max_saved_candidates,
        min_shortlist_score=min_shortlist_score,
        labels_csv=labels_csv,
        eval_threshold=eval_threshold,
        semantic_vision=semantic_vision,
        openai_model=openai_model,
        openai_detail=openai_detail,
        openai_timeout_s=openai_timeout_s,
        full_frame_semantic=full_frame_semantic,
        source_type="video",
    )


def _run_vision_lab_on_frames(
    *,
    mission_request: str,
    frame_items: list[dict],
    config_path: str | Path,
    output_dir: str | Path,
    save_only_detections: bool,
    proposal_mode: str,
    max_saved_candidates: int,
    min_shortlist_score: float,
    labels_csv: str | Path | None,
    eval_threshold: float,
    semantic_vision: str,
    openai_model: str | None,
    openai_detail: str,
    openai_timeout_s: float,
    full_frame_semantic: str,
    source_type: str,
) -> Path:
    config = load_search_mission_config(config_path)
    objective = parse_mission_request(mission_request)
    vision_plan = create_mission_vision_plan(objective)
    detector = RedBlockDetector(config.target)
    color_detector = MissionColorProposalDetector(vision_plan, min_area_px=max(25, int(config.target.min_area_px * 0.15)))
    objectness_detector = ObjectnessProposalDetector(min_area_px=max(30, int(config.target.min_area_px * 0.12)))
    scorer = build_semantic_scorer(
        semantic_vision,
        openai_model=openai_model,
        openai_detail=openai_detail,
        openai_timeout_s=openai_timeout_s,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(labels_csv) if labels_csv else {}

    pending_debug: list[tuple[int, np.ndarray, object, Path]] = []
    pending_crop: list[tuple[int, np.ndarray | None, Path]] = []
    results = []
    for index, item in enumerate(frame_items, start=1):
        source_path = Path(item["source_path"])
        frame = item.get("frame")
        if frame is None:
            frame = cv2.imread(str(source_path))
        if frame is None:
            results.append(
                {
                    "image_path": str(source_path),
                    "frame_index": item.get("frame_index"),
                    "timestamp_s": item.get("timestamp_s"),
                    "error": "Could not read image",
                    "detected": False,
                }
            )
            continue
        detection = detect_with_mode(detector, color_detector, objectness_detector, vision_plan, frame, proposal_mode)
        audit_mask = audit_mask_for_mode(detector, color_detector, objectness_detector, vision_plan, frame, proposal_mode)
        audit = red_region_audit(audit_mask)
        crop = crop_detection(frame, detection)
        semantic = scorer.score(
            objective=objective,
            frame_bgr=frame,
            crop_bgr=crop,
            detection=detection,
        )
        full_frame_result = None
        if should_run_full_frame_semantic(full_frame_semantic, detected=detection.detected):
            full_frame_result = scorer.score_full_frame(objective=objective, frame_bgr=frame)
        final_score = max(
            semantic.score,
            0.0 if full_frame_result is None else full_frame_result.score,
        )
        final_decision = semantic.decision if full_frame_result is None or semantic.score >= full_frame_result.score else full_frame_result.decision
        stem = _frame_stem(source_path, item.get("frame_index"))
        debug_path: Path | None = run_dir / f"{index:04d}_{stem}_debug.png"
        crop_path: Path | None = run_dir / f"{index:04d}_{stem}_crop.png"
        should_save_all = not save_only_detections
        if should_save_all:
            detector.save_snapshot(frame, debug_path, detection)
            saved_crop = save_candidate_crop(crop, crop_path)
        elif detection.detected:
            pending_debug.append((index, frame.copy(), detection, debug_path))
            pending_crop.append((index, crop, crop_path))
            saved_crop = str(crop_path)
        else:
            debug_path = None
            crop_path = None
            saved_crop = None
        results.append(
            {
                "image_path": str(source_path),
                "frame_index": item.get("frame_index"),
                "timestamp_s": item.get("timestamp_s"),
                "detected": detection.detected,
                "detector_confidence": detection.confidence,
                "bbox": detection.bbox,
                "red_audit": audit,
                "review_priority": detection.confidence,
                "crop_path": saved_crop,
                "debug_path": None if debug_path is None else str(debug_path),
                "semantic": asdict(semantic),
                "full_frame_semantic": None if full_frame_result is None else asdict(full_frame_result),
                "final_score": final_score,
                "final_decision": final_decision.value,
            }
        )

    for result in results:
        label = labels.get(_label_key(result.get("image_path"), result.get("frame_index")))
        if label is None:
            label = labels.get(_label_key(result.get("image_path"), None))
        if label is not None:
            result["label"] = label

    detected_count = sum(1 for result in results if result.get("detected"))
    shortlist_indexes = select_shortlist_indexes(
        results,
        max_items=max_saved_candidates,
        min_score=min_shortlist_score,
    )
    if save_only_detections:
        kept = set(shortlist_indexes)
        for index, frame, detection, path in pending_debug:
            if index in kept:
                detector.save_snapshot(frame, path, detection)
        for index, crop, path in pending_crop:
            if index in kept:
                save_candidate_crop(crop, path)
        for index, result in enumerate(results, start=1):
            if result.get("detected") and index not in kept:
                result["debug_path"] = None
                result["crop_path"] = None
    readable_results = [result for result in results if "error" not in result]
    possible_misses = [
        result for result in readable_results
        if not result.get("detected") and result.get("red_audit", {}).get("largest_red_area_px", 0) >= config.target.min_area_px * 0.35
    ]
    possible_misses = sorted(
        possible_misses,
        key=lambda result: result.get("red_audit", {}).get("largest_red_area_px", 0),
        reverse=True,
    )[:25]
    evaluation = evaluate_results(results, threshold=eval_threshold) if labels else None
    report = {
        "timestamp": stamp,
        "mission_request": mission_request,
        "objective": asdict(objective),
        "vision_plan": asdict(vision_plan),
        "scorer": scorer.model_name,
        "full_frame_semantic_mode": full_frame_semantic,
        "source_type": source_type,
        "proposal_mode": proposal_mode,
        "image_count": len(frame_items),
        "summary": {
            "processed": len(results),
            "detections": detected_count,
            "errors": sum(1 for result in results if "error" in result),
            "detection_rate": 0.0 if not results else round(detected_count / len(results), 4),
            "save_only_detections": save_only_detections,
            "max_saved_candidates": max_saved_candidates,
            "min_shortlist_score": min_shortlist_score,
            "shortlist_count": len(shortlist_indexes),
            "shortlist": [_shortlist_entry(results[index - 1]) for index in shortlist_indexes],
            "evaluation_threshold": eval_threshold,
            "possible_miss_count": len(possible_misses),
            "possible_misses": [
                {
                    "image_path": result["image_path"],
                    "largest_red_area_px": result["red_audit"]["largest_red_area_px"],
                    "red_pixel_ratio": result["red_audit"]["red_pixel_ratio"],
                }
                for result in possible_misses
            ],
        },
        "evaluation": evaluation,
        "results": results,
    }
    report_path = run_dir / "vision_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    shortlist_path = run_dir / "review_shortlist.json"
    shortlist_payload = {
        "timestamp": stamp,
        "mission_request": mission_request,
        "proposal_mode": proposal_mode,
        "shortlist": [_shortlist_entry(results[index - 1]) for index in shortlist_indexes],
    }
    shortlist_path.write_text(json.dumps(shortlist_payload, indent=2, default=str), encoding="utf-8")
    return report_path


def collect_image_paths(paths: list[str]) -> list[Path]:
    image_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            image_paths.extend(sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS))
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)
    return image_paths


def collect_video_paths(paths: list[str]) -> list[Path]:
    video_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            video_paths.extend(sorted(item for item in path.iterdir() if item.suffix.lower() in VIDEO_EXTENSIONS))
        elif path.suffix.lower() in VIDEO_EXTENSIONS:
            video_paths.append(path)
    return video_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vision-only candidate detection and semantic scoring")
    parser.add_argument("paths", nargs="+", help="Image files or folders of images")
    parser.add_argument("--mission-request", required=True)
    parser.add_argument("--config", default="config/autonomy.yaml")
    parser.add_argument("--output-dir", default="logs/vision_lab")
    parser.add_argument("--video", action="store_true", help="Treat the input path as a video file or folder of videos")
    parser.add_argument(
        "--proposal-mode",
        choices=["precise", "high-recall", "mission-color"],
        default="mission-color",
        help="mission-color adapts color proposals to the mission text; high-recall is broad red-focused; precise is stricter.",
    )
    parser.add_argument("--sample-every-s", type=float, default=1.0, help="For video mode, sample one frame every N seconds")
    parser.add_argument("--max-frames", type=int, default=None, help="For video mode, stop after N sampled frames")
    parser.add_argument("--max-saved-candidates", type=int, default=50, help="Maximum accepted detections to save as debug/crop files")
    parser.add_argument("--min-shortlist-score", type=float, default=0.25, help="Minimum semantic score for the review shortlist")
    parser.add_argument("--labels-csv", default=None, help="Optional CSV with image_path,expected_match,label columns for accuracy evaluation")
    parser.add_argument("--eval-threshold", type=float, default=0.25, help="Semantic score threshold for labeled precision/recall evaluation")
    parser.add_argument("--semantic-vision", choices=["local", "openai"], default="local", help="Semantic scorer backend")
    parser.add_argument("--openai-model", default=None, help="Required with --semantic-vision openai unless OPENAI_VISION_MODEL is set")
    parser.add_argument(
        "--openai-detail",
        choices=["auto", "low", "high"],
        default="auto",
        help="Image detail sent to the OpenAI vision model. low is cheaper/faster; high is better for small or distant targets.",
    )
    parser.add_argument("--openai-timeout-s", type=float, default=45.0, help="Timeout per OpenAI vision request")
    parser.add_argument(
        "--full-frame-semantic",
        choices=["off", "misses", "all"],
        default="off",
        help="Run semantic scoring on full frames, not just proposal crops.",
    )
    parser.add_argument(
        "--save-only-detections",
        action="store_true",
        help="Only save debug/crop images for detected candidates. The JSON report still includes every image.",
    )
    args = parser.parse_args()

    if args.video:
        video_paths = collect_video_paths(args.paths)
        if not video_paths:
            raise SystemExit("No video files found. Use .mp4, .mov, .avi, .mkv, .webm, or .m4v files.")
        if len(video_paths) > 1:
            raise SystemExit("Video mode currently accepts one video at a time.")
        report_path = run_video_vision_lab(
            mission_request=args.mission_request,
            video_path=video_paths[0],
            config_path=args.config,
            output_dir=args.output_dir,
            save_only_detections=args.save_only_detections,
            sample_every_s=args.sample_every_s,
            max_frames=args.max_frames,
            proposal_mode=args.proposal_mode,
            max_saved_candidates=args.max_saved_candidates,
            min_shortlist_score=args.min_shortlist_score,
            labels_csv=args.labels_csv,
            eval_threshold=args.eval_threshold,
            semantic_vision=args.semantic_vision,
            openai_model=args.openai_model,
            openai_detail=args.openai_detail,
            openai_timeout_s=args.openai_timeout_s,
            full_frame_semantic=args.full_frame_semantic,
        )
    else:
        image_paths = collect_image_paths(args.paths)
        if not image_paths:
            raise SystemExit("No image files found. Use .jpg, .jpeg, .png, .bmp, or .webp files.")
        report_path = run_vision_lab(
            mission_request=args.mission_request,
            image_paths=image_paths,
            config_path=args.config,
            output_dir=args.output_dir,
            save_only_detections=args.save_only_detections,
            proposal_mode=args.proposal_mode,
            max_saved_candidates=args.max_saved_candidates,
            min_shortlist_score=args.min_shortlist_score,
            labels_csv=args.labels_csv,
            eval_threshold=args.eval_threshold,
            semantic_vision=args.semantic_vision,
            openai_model=args.openai_model,
            openai_detail=args.openai_detail,
            openai_timeout_s=args.openai_timeout_s,
            full_frame_semantic=args.full_frame_semantic,
        )
    print(f"Vision report saved: {report_path}")


def detect_with_mode(
    detector: RedBlockDetector,
    color_detector: MissionColorProposalDetector,
    objectness_detector: ObjectnessProposalDetector,
    vision_plan,
    frame: np.ndarray,
    proposal_mode: str,
):
    if proposal_mode == "precise":
        return detector.detect(frame)
    if proposal_mode == "mission-color":
        if not vision_plan.important_colors and vision_plan.possible_categories:
            return objectness_detector.detect(frame)
        return color_detector.detect(frame)
    return detector.detect_high_recall(frame)


def audit_mask_for_mode(
    detector: RedBlockDetector,
    color_detector: MissionColorProposalDetector,
    objectness_detector: ObjectnessProposalDetector,
    vision_plan,
    frame: np.ndarray,
    proposal_mode: str,
) -> np.ndarray:
    if proposal_mode == "mission-color":
        if not vision_plan.important_colors and vision_plan.possible_categories:
            return objectness_detector.mask(frame)
        return color_detector.mask(frame)
    return detector.high_recall_mask(frame)


def build_semantic_scorer(
    name: str,
    *,
    openai_model: str | None = None,
    openai_detail: str = "auto",
    openai_timeout_s: float = 45.0,
):
    if name == "openai":
        return OpenAIVisionLanguageScorer(model=openai_model, detail=openai_detail, timeout_s=openai_timeout_s)
    return LocalSemanticVisionScorer()


def should_run_full_frame_semantic(mode: str, *, detected: bool) -> bool:
    if mode == "all":
        return True
    if mode == "misses":
        return not detected
    return False


def select_shortlist_indexes(results: list[dict], *, max_items: int, min_score: float) -> list[int]:
    scored: list[tuple[float, int]] = []
    for index, result in enumerate(results, start=1):
        if not result.get("detected"):
            continue
        score = float(result.get("final_score", result.get("semantic", {}).get("score", 0.0)))
        if score >= min_score:
            scored.append((score, index))
    scored.sort(reverse=True)
    return [index for _, index in scored[:max(0, max_items)]]


def load_labels(path: str | Path | None) -> dict[tuple[str, int | None], dict]:
    if path is None:
        return {}
    labels: dict[tuple[str, int | None], dict] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "expected_match"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Labels CSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            expected = _parse_bool(row["expected_match"])
            frame_index = row.get("frame_index", "").strip()
            labels[_label_key(row["image_path"], int(frame_index) if frame_index else None)] = {
                "expected_match": expected,
                "label": row.get("label", ""),
                "notes": row.get("notes", ""),
            }
    return labels


def evaluate_results(results: list[dict], *, threshold: float) -> dict:
    labeled = [result for result in results if "label" in result and "error" not in result]
    true_positive = false_positive = true_negative = false_negative = 0
    false_positives = []
    false_negatives = []
    for result in labeled:
        expected = bool(result["label"]["expected_match"])
        score = float(result.get("final_score", result.get("semantic", {}).get("score", 0.0)))
        predicted = is_predicted_match(result, threshold=threshold)
        if predicted and expected:
            true_positive += 1
        elif predicted and not expected:
            false_positive += 1
            false_positives.append(_shortlist_entry(result))
        elif not predicted and not expected:
            true_negative += 1
        else:
            false_negative += 1
            false_negatives.append(_shortlist_entry(result))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positive + true_negative) / len(labeled) if labeled else 0.0
    return {
        "labeled_count": len(labeled),
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "false_positives": false_positives[:25],
        "false_negatives": false_negatives[:25],
    }


def is_predicted_match(result: dict, *, threshold: float) -> bool:
    decision = str(result.get("final_decision", result.get("semantic", {}).get("decision", "")))
    score = float(result.get("final_score", result.get("semantic", {}).get("score", 0.0)))
    if decision == "LIKELY_MATCH":
        return score >= threshold
    if decision == "POSSIBLE_MATCH":
        return score >= threshold and bool(result.get("detected"))
    return False


def _shortlist_entry(result: dict) -> dict:
    semantic = result.get("semantic", {})
    red_audit = result.get("red_audit", {})
    return {
        "image_path": result.get("image_path"),
        "frame_index": result.get("frame_index"),
        "timestamp_s": result.get("timestamp_s"),
        "score": result.get("final_score", semantic.get("score")),
        "decision": result.get("final_decision", semantic.get("decision")),
        "detector_confidence": result.get("detector_confidence"),
        "review_priority": result.get("review_priority"),
        "bbox": result.get("bbox"),
        "crop_path": result.get("crop_path"),
        "debug_path": result.get("debug_path"),
        "largest_red_area_px": red_audit.get("largest_red_area_px"),
        "red_pixel_ratio": red_audit.get("red_pixel_ratio"),
        "label": result.get("label"),
    }


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "match", "positive"}


def _label_key(image_path: str | None, frame_index: int | None) -> tuple[str, int | None]:
    path = Path(image_path or "")
    return (path.name, frame_index)


def red_region_audit(mask: np.ndarray) -> dict[str, float | int]:
    if mask is None or mask.size == 0:
        return {"red_pixel_count": 0, "red_pixel_ratio": 0.0, "largest_red_area_px": 0}
    red_pixel_count = int(cv2.countNonZero(mask))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = int(max((cv2.contourArea(contour) for contour in contours), default=0))
    return {
        "red_pixel_count": red_pixel_count,
        "red_pixel_ratio": round(red_pixel_count / float(mask.size), 6),
        "largest_red_area_px": largest_area,
    }


def _frame_stem(source_path: Path, frame_index: int | None) -> str:
    if frame_index is None:
        return source_path.stem
    return f"{source_path.stem}_frame_{frame_index:06d}"


if __name__ == "__main__":
    main()
