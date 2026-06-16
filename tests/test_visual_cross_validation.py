from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.visual_cross_validation import create_visual_evaluation_protocol
from autonomy.visual_cross_validation import read_visual_labels
from autonomy.visual_cross_validation import run_visual_cross_validation


def make_visual_fixture(root: Path) -> tuple[Path, Path]:
    dataset = root / "dataset"
    dataset.mkdir()
    labels_csv = root / "labels.csv"
    rows = []
    for index in range(10):
        path = f"rgb/positive_{index}.jpg"
        (dataset / "rgb").mkdir(exist_ok=True)
        (dataset / path).write_bytes(b"positive")
        rows.append(
            {
                "image_path": path,
                "expected_match": "true",
                "label": "positive",
                "modality": "rgb",
                "source_dataset": "fixture",
            }
        )
    for index in range(10):
        path = f"rgb/negative_{index}.jpg"
        (dataset / path).write_bytes(b"negative")
        rows.append(
            {
                "image_path": path,
                "expected_match": "false",
                "label": "negative",
                "modality": "rgb",
                "source_dataset": "fixture",
            }
        )
    with labels_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "expected_match", "label", "modality", "source_dataset"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return dataset, labels_csv


def test_visual_protocol_keeps_lockbox_out_of_development() -> None:
    with TemporaryDirectory() as tmp:
        dataset, labels_csv = make_visual_fixture(Path(tmp))
        rows = read_visual_labels(labels_csv)
        protocol = create_visual_evaluation_protocol(
            rows,
            folds=4,
            seed=5,
            lockbox_fraction=0.2,
        )
        lockbox_paths = {row["image_path"] for row in protocol["lockbox"]}
        development_paths = {row["image_path"] for row in protocol["development"]}
        fold_paths = {row["image_path"] for fold in protocol["development_folds"] for row in fold}

        assert len(lockbox_paths) == 4
        assert len(development_paths) == 16
        assert not lockbox_paths & development_paths
        assert fold_paths == development_paths
        assert all(len(fold) == 4 for fold in protocol["development_folds"])
        assert dataset.exists()


def test_visual_cross_validation_split_only_writes_lockbox_and_fold_manifests() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset, labels_csv = make_visual_fixture(root)
        report_path = run_visual_cross_validation(
            dataset_root=dataset,
            labels_csv=labels_csv,
            output_dir=root / "logs",
            mission_request="Search aerial RGB imagery for vehicles.",
            modality="rgb",
            folds=2,
            seed=5,
            lockbox_fraction=0.2,
            run_evaluation=False,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["benchmark"] == "Aegis RGB Stratified Cross-Validation v1"
        assert report["dataset_counts"] == {"positive": 10, "negative": 10, "total": 20}
        assert report["development_counts"] == {"positive": 8, "negative": 8, "total": 16}
        assert report["lockbox_counts"] == {"positive": 2, "negative": 2, "total": 4}
        assert len(report["fold_reports"]) == 2
        assert all(item["leakage_check"]["overlap_count"] == 0 for item in report["fold_reports"])
        assert (root / "logs" / "development_manifest.csv").exists()
        assert (root / "logs" / "final_test_lockbox.csv").exists()
        assert (root / "logs" / "fold_01" / "tuning_manifest.csv").exists()
        fold_labels = root / "logs" / "fold_01" / "evaluation_sample" / "labels.csv"
        assert fold_labels.exists()
        fold_rows = list(csv.DictReader(fold_labels.open(newline="", encoding="utf-8")))
        assert any(row["expected_match"] == "false" and row["label"] == "negative" for row in fold_rows)
        assert report_path.with_suffix(".html").exists()


def test_visual_protocol_can_cap_development_without_touching_lockbox() -> None:
    with TemporaryDirectory() as tmp:
        _, labels_csv = make_visual_fixture(Path(tmp))
        rows = read_visual_labels(labels_csv)
        protocol = create_visual_evaluation_protocol(
            rows,
            folds=2,
            seed=5,
            lockbox_fraction=0.2,
            max_development_rows=6,
        )
        assert len(protocol["development"]) == 6
        assert len(protocol["lockbox"]) == 4
        assert not ({row["image_path"] for row in protocol["development"]} & {row["image_path"] for row in protocol["lockbox"]})


if __name__ == "__main__":
    tests = [
        test_visual_protocol_keeps_lockbox_out_of_development,
        test_visual_cross_validation_split_only_writes_lockbox_and_fold_manifests,
        test_visual_protocol_can_cap_development_without_touching_lockbox,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
