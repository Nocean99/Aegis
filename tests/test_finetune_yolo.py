from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.finetune_yolo import (
    benchmark_detector,
    comparison_markdown,
    read_labels_csv,
    run_benchmark,
    yolo_label_to_boxes,
    yolo_split_to_labels_csv,
)
from autonomy.types import TargetDetection


class FakeDetector:
    """Returns preset boxes per filename-independent frame marker pixel."""

    def __init__(self, boxes_for_marked: list[tuple[int, int, int, int]], confidence: float = 0.9):
        self.boxes_for_marked = boxes_for_marked
        self.confidence = confidence

    def detect_all(self, frame):
        # Frames with a white marker pixel at (0, 0) are "target" frames.
        if frame[0, 0].sum() > 600:
            return [
                TargetDetection(True, confidence=self.confidence, bbox=box)
                for box in self.boxes_for_marked
            ]
        return []


def write_image(path: Path, *, marked: bool, size: tuple[int, int] = (100, 100)) -> None:
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if marked:
        image[0, 0] = (255, 255, 255)
    cv2.imwrite(str(path), image)


def make_dataset(root: Path) -> Path:
    """Two positives with known boxes, one negative."""
    images = root / "split" / "images"
    labels = root / "split" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    write_image(images / "pos1.png", marked=True)
    write_image(images / "pos2.png", marked=True)
    write_image(images / "neg1.png", marked=False)
    # YOLO normalized: class cx cy w h on 100x100 -> box (40,40,20,20)
    labels.joinpath("pos1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    labels.joinpath("pos2.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return root / "split"


def test_yolo_label_to_boxes_denormalizes() -> None:
    boxes = yolo_label_to_boxes("0 0.5 0.5 0.2 0.2", width=100, height=100)
    assert boxes == [(40, 40, 20, 20)]
    assert yolo_label_to_boxes("", width=100, height=100) == []
    assert yolo_label_to_boxes("bad line", width=100, height=100) == []


def test_split_conversion_writes_project_labels_csv() -> None:
    with TemporaryDirectory() as tmp:
        split = make_dataset(Path(tmp))
        output = Path(tmp) / "labels.csv"
        count = yolo_split_to_labels_csv(split, output)
        assert count == 3
        with output.open(newline="", encoding="utf-8") as handle:
            rows = {row["filename"]: row for row in csv.DictReader(handle)}
        assert rows["pos1.png"]["label"] == "1"
        assert rows["pos1.png"]["gt_boxes"] == "40,40,20,20"
        assert rows["neg1.png"]["label"] == "0"
        assert rows["neg1.png"]["gt_boxes"] == ""
        labels = read_labels_csv(output)
        assert labels["pos2.png"] == [(40, 40, 20, 20)]
        assert labels["neg1.png"] == []


def test_benchmark_detector_two_views() -> None:
    with TemporaryDirectory() as tmp:
        split = make_dataset(Path(tmp))
        labels_csv = Path(tmp) / "labels.csv"
        yolo_split_to_labels_csv(split, labels_csv)
        labels = read_labels_csv(labels_csv)

        perfect = benchmark_detector(
            "perfect",
            FakeDetector([(40, 40, 20, 20)]),
            images_dir=split / "images",
            labels=labels,
        )
        assert perfect.capture["recall"] == 1.0
        assert perfect.capture["precision"] == 1.0
        assert perfect.localization["recall"] == 1.0
        assert perfect.localization["mean_iou"] == 1.0

        # Fires on the right frames but with a poorly placed box:
        # perfect capture, failed localization — the two-view split in action.
        misplaced = benchmark_detector(
            "misplaced",
            FakeDetector([(0, 0, 20, 20)]),
            images_dir=split / "images",
            labels=labels,
        )
        assert misplaced.capture["recall"] == 1.0
        assert misplaced.localization["recall"] == 0.0


def test_run_benchmark_writes_reports_with_injected_factory() -> None:
    with TemporaryDirectory() as tmp:
        split = make_dataset(Path(tmp))
        labels_csv = Path(tmp) / "labels.csv"
        yolo_split_to_labels_csv(split, labels_csv)
        output_dir = Path(tmp) / "bench"

        detectors = {
            "baseline.pt": FakeDetector([(0, 0, 20, 20)]),
            "finetuned.pt": FakeDetector([(40, 40, 20, 20)]),
        }
        report = run_benchmark(
            ["baseline.pt", "finetuned.pt"],
            images_dir=split / "images",
            labels_csv=labels_csv,
            output_dir=output_dir,
            detector_factory=lambda weights: detectors[weights],
        )
        assert len(report["results"]) == 2
        baseline, finetuned = report["results"]
        assert finetuned["localization"]["f1"] > baseline["localization"]["f1"]
        saved = json.loads((output_dir / "finetune_benchmark.json").read_text(encoding="utf-8"))
        assert saved["labeled_frames"] == 3
        markdown = (output_dir / "finetune_benchmark.md").read_text(encoding="utf-8")
        assert "finetuned.pt" in markdown and "baseline.pt" in markdown


def test_comparison_markdown_renders_one_row_per_model() -> None:
    with TemporaryDirectory() as tmp:
        split = make_dataset(Path(tmp))
        labels_csv = Path(tmp) / "labels.csv"
        yolo_split_to_labels_csv(split, labels_csv)
        labels = read_labels_csv(labels_csv)
        result = benchmark_detector(
            "model-a", FakeDetector([(40, 40, 20, 20)]), images_dir=split / "images", labels=labels
        )
        text = comparison_markdown([result])
        assert text.count("model-a") == 1
        assert "Mean IoU" in text


if __name__ == "__main__":
    tests = [
        test_yolo_label_to_boxes_denormalizes,
        test_split_conversion_writes_project_labels_csv,
        test_benchmark_detector_two_views,
        test_run_benchmark_writes_reports_with_injected_factory,
        test_comparison_markdown_renders_one_row_per_model,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
