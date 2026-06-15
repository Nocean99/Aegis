from __future__ import annotations

"""Fine-tune YOLO on a maritime dataset and benchmark it against the
pretrained baseline using the project's own two-view methodology.

Three subcommands:

    to-labels-csv   Convert a YOLO-format dataset split (Roboflow export)
                    into this project's labels CSV with ground-truth boxes,
                    so external datasets plug into the existing evaluation
                    harness (capture metrics + IoU localization metrics).

    train           Thin wrapper over ultralytics fine-tuning. Requires the
                    [ml] extras; everything else in this module runs without
                    them.

    benchmark       Run two or more weight files over the same labeled
                    images and report both metric views side by side:
                    capture (did evidence reach review) and localization
                    (was the box right). Writes JSON + a markdown table.

Typical flow (see docs/FINETUNE_YOLO.md for the full runbook):

    python3 -m autonomy.finetune_yolo to-labels-csv datasets/aerial_maritime/test \
        --output datasets/aerial_maritime/test_labels.csv
    python3 -m autonomy.finetune_yolo train datasets/aerial_maritime/data.yaml \
        --base-weights yolov8n.pt --epochs 50
    python3 -m autonomy.finetune_yolo benchmark \
        --weights yolov8n.pt runs/detect/train/weights/best.pt \
        --images datasets/aerial_maritime/test/images \
        --labels-csv datasets/aerial_maritime/test_labels.csv
"""

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from autonomy.detection_metrics import evaluate_localization, parse_gt_boxes

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# -- dataset conversion -------------------------------------------------------


def yolo_label_to_boxes(label_text: str, *, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Convert YOLO txt rows (class cx cy w h, normalized) to pixel x,y,w,h."""
    boxes = []
    for line in label_text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = (float(value) for value in parts[:5])
        box_w = max(1, round(w * width))
        box_h = max(1, round(h * height))
        x = round(cx * width - box_w / 2)
        y = round(cy * height - box_h / 2)
        boxes.append((x, y, box_w, box_h))
    return boxes


def yolo_split_to_labels_csv(split_dir: Path, output_csv: Path) -> int:
    """Convert a YOLO-format split (images/ + labels/) to the project labels CSV.

    Emits: filename, label (1 if any ground-truth box), gt_boxes ("x,y,w,h;...").
    Returns the number of images written.
    """
    import cv2

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No images/ directory under {split_dir}")
    rows = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        label_path = labels_dir / (image_path.stem + ".txt")
        boxes = (
            yolo_label_to_boxes(label_path.read_text(encoding="utf-8"), width=width, height=height)
            if label_path.exists()
            else []
        )
        rows.append(
            {
                "filename": image_path.name,
                "label": 1 if boxes else 0,
                "gt_boxes": ";".join(f"{x},{y},{w},{h}" for x, y, w, h in boxes),
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "label", "gt_boxes"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_labels_csv(labels_csv: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    """filename -> ground-truth boxes (may be empty for negative frames)."""
    labels: dict[str, list[tuple[int, int, int, int]]] = {}
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[row["filename"]] = parse_gt_boxes(row.get("gt_boxes"))
    return labels


# -- benchmark ----------------------------------------------------------------


@dataclass
class BenchmarkResult:
    name: str
    capture: dict
    localization: dict

    def as_dict(self) -> dict:
        return {"name": self.name, "capture": self.capture, "localization": self.localization}


def benchmark_detector(
    name: str,
    detector,
    *,
    images_dir: Path,
    labels: dict[str, list[tuple[int, int, int, int]]],
    iou_threshold: float = 0.5,
) -> BenchmarkResult:
    """Two-view evaluation of one detector over a labeled image folder.

    Capture view: frame-level — did the detector surface anything on frames
    that contain a target (recall) and how often it fired on empty frames
    (precision). Localization view: box-level IoU metrics. The same split,
    both stories.
    """
    import cv2

    items = []
    frame_tp = frame_fp = frame_fn = frame_tn = 0
    for filename, gt_boxes in sorted(labels.items()):
        image_path = images_dir / filename
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        detections = detector.detect_all(image)
        pred_boxes = [
            (detection.bbox, float(detection.confidence))
            for detection in detections
            if detection.bbox is not None
        ]
        items.append({"gt_boxes": gt_boxes, "pred_boxes": pred_boxes})
        fired = bool(pred_boxes)
        has_target = bool(gt_boxes)
        if has_target and fired:
            frame_tp += 1
        elif has_target:
            frame_fn += 1
        elif fired:
            frame_fp += 1
        else:
            frame_tn += 1

    capture_precision = frame_tp / (frame_tp + frame_fp) if frame_tp + frame_fp else 0.0
    capture_recall = frame_tp / (frame_tp + frame_fn) if frame_tp + frame_fn else 0.0
    capture_f1 = (
        2 * capture_precision * capture_recall / (capture_precision + capture_recall)
        if capture_precision + capture_recall
        else 0.0
    )
    capture = {
        "metric_mode": "frame_capture",
        "frames": len(items),
        "true_positive": frame_tp,
        "false_positive": frame_fp,
        "false_negative": frame_fn,
        "true_negative": frame_tn,
        "precision": round(capture_precision, 4),
        "recall": round(capture_recall, 4),
        "f1": round(capture_f1, 4),
    }
    localization = evaluate_localization(items, iou_threshold=iou_threshold).as_dict()
    return BenchmarkResult(name=name, capture=capture, localization=localization)


def comparison_markdown(results: list[BenchmarkResult]) -> str:
    lines = [
        "| Model | Capture P | Capture R | Capture F1 | Loc P | Loc R | Loc F1 | Mean IoU | AP@0.5 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        capture, loc = result.capture, result.localization
        lines.append(
            f"| {result.name} | {capture['precision']:.3f} | {capture['recall']:.3f} "
            f"| {capture['f1']:.3f} | {loc['precision']:.3f} | {loc['recall']:.3f} "
            f"| {loc['f1']:.3f} | {loc['mean_iou']:.3f} | {loc['average_precision']:.3f} |"
        )
    return "\n".join(lines)


def run_benchmark(
    weights_list: list[str],
    *,
    images_dir: Path,
    labels_csv: Path,
    output_dir: Path,
    confidence_threshold: float = 0.2,
    iou_threshold: float = 0.5,
    detector_factory=None,
) -> dict:
    """Benchmark each weights file; write JSON + markdown. Returns the report.

    ``detector_factory(weights)`` may be injected for tests; the default
    builds a YoloProposalDetector (requires the [ml] extras).
    """
    if detector_factory is None:
        from autonomy.yolo_proposal_detector import YoloProposalDetector

        def detector_factory(weights: str):
            return YoloProposalDetector(
                weights=weights, confidence_threshold=confidence_threshold
            )

    labels = read_labels_csv(labels_csv)
    results = [
        benchmark_detector(weights, detector_factory(weights), images_dir=images_dir, labels=labels, iou_threshold=iou_threshold)
        for weights in weights_list
    ]
    report = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "images_dir": str(images_dir),
        "labels_csv": str(labels_csv),
        "labeled_frames": len(labels),
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "results": [result.as_dict() for result in results],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "finetune_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "finetune_benchmark.md").write_text(
        "# Fine-tune benchmark\n\n" + comparison_markdown(results) + "\n", encoding="utf-8"
    )
    return report


# -- training -----------------------------------------------------------------


def train(
    data_yaml: Path,
    *,
    base_weights: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    project: str = "logs/finetune_yolo",
) -> str:
    """Fine-tune. Returns the path to the best weights."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Training requires ultralytics. Install with: pip install '.[ml]'"
        ) from exc
    model = YOLO(base_weights)
    results = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, project=project)
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best weights: {best}")
    return str(best)


# -- CLI ------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("to-labels-csv", help="YOLO split -> project labels CSV")
    convert.add_argument("split_dir", type=Path)
    convert.add_argument("--output", type=Path, required=True)

    fit = sub.add_parser("train", help="Fine-tune YOLO (requires [ml] extras)")
    fit.add_argument("data_yaml", type=Path)
    fit.add_argument("--base-weights", default="yolov8n.pt")
    fit.add_argument("--epochs", type=int, default=50)
    fit.add_argument("--imgsz", type=int, default=640)
    fit.add_argument("--project", default="logs/finetune_yolo")

    compare = sub.add_parser("benchmark", help="Compare weights with two-view metrics")
    compare.add_argument("--weights", nargs="+", required=True)
    compare.add_argument("--images", type=Path, required=True)
    compare.add_argument("--labels-csv", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, default=Path("logs/finetune_yolo/benchmark"))
    compare.add_argument("--confidence-threshold", type=float, default=0.2)
    compare.add_argument("--iou-threshold", type=float, default=0.5)

    args = parser.parse_args()
    if args.command == "to-labels-csv":
        count = yolo_split_to_labels_csv(args.split_dir, args.output)
        print(f"Wrote {count} rows to {args.output}")
    elif args.command == "train":
        train(
            args.data_yaml,
            base_weights=args.base_weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            project=args.project,
        )
    else:
        report = run_benchmark(
            args.weights,
            images_dir=args.images,
            labels_csv=args.labels_csv,
            output_dir=args.output_dir,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
        )
        print(json.dumps(report["results"], indent=2))
        print(f"\nReport written to {args.output_dir}")


if __name__ == "__main__":
    main()
