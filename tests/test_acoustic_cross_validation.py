from __future__ import annotations

import json
import math
import struct
import sys
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.acoustic_cross_validation import run_acoustic_cross_validation
from autonomy.acoustic_cross_validation import create_evaluation_protocol
from autonomy.acoustic_cross_validation import stratified_folds


def write_wav(path: Path, *, amplitude: float = 0.1, sample_rate: int = 8000) -> None:
    samples = []
    for index in range(sample_rate):
        t = index / sample_rate
        value = int(amplitude * math.sin(2 * math.pi * 220 * t) * 32767)
        samples.append(value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def make_dataset(root: Path) -> Path:
    dataset = root / "dataset"
    for label in ("anthropogenic", "animal", "sonar"):
        folder = dataset / label
        folder.mkdir(parents=True)
        for index in range(4):
            write_wav(folder / f"{label}_{index}.wav", amplitude=0.2 if label == "anthropogenic" else 0.02)
    return dataset


def test_stratified_folds_cover_every_clip_once() -> None:
    with TemporaryDirectory() as tmp:
        dataset = make_dataset(Path(tmp))
        folds = stratified_folds(dataset, folds=2, seed=3)
        flattened = [item["path"] for fold in folds for item in fold]
        assert len(flattened) == 12
        assert len(set(flattened)) == 12
        assert all(counts == {"anthropogenic": 2, "animal": 2, "sonar": 2} for counts in [count_labels(fold) for fold in folds])


def test_acoustic_cross_validation_writes_report() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset = make_dataset(root)
        report_path = run_acoustic_cross_validation(
            dataset_root=dataset,
            output_dir=root / "logs",
            folds=2,
            seed=3,
            lockbox_fraction=0.25,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["dataset_counts"] == {"anthropogenic": 4, "animal": 4, "sonar": 4}
        assert report["folds"] == 2
        assert report["lockbox_counts"] == {"anthropogenic": 1, "animal": 1, "sonar": 1}
        assert report["development_counts"] == {"anthropogenic": 3, "animal": 3, "sonar": 3}
        assert len(report["fold_reports"]) == 2
        assert all(item["leakage_check"]["overlap_count"] == 0 for item in report["fold_reports"])
        assert set(report["mean_std_metrics"]) == {"capture_precision", "capture_recall", "capture_f1"}
        assert report["aggregate_metrics"]["labeled_count"] == 9
        assert (root / "logs" / "development_manifest.csv").exists()
        assert (root / "logs" / "final_test_lockbox.csv").exists()
        assert report_path.with_suffix(".html").exists()


def test_protocol_keeps_lockbox_out_of_development_folds() -> None:
    with TemporaryDirectory() as tmp:
        dataset = make_dataset(Path(tmp))
        protocol = create_evaluation_protocol(dataset, folds=2, seed=3, lockbox_fraction=0.25)
        lockbox_paths = {item["path"] for item in protocol["lockbox"]}
        development_paths = {item["path"] for item in protocol["development"]}
        fold_paths = {item["path"] for fold in protocol["development_folds"] for item in fold}
        assert lockbox_paths
        assert not lockbox_paths & development_paths
        assert fold_paths == development_paths


def count_labels(fold: list[dict]) -> dict[str, int]:
    counts = {"anthropogenic": 0, "animal": 0, "sonar": 0}
    for item in fold:
        counts[item["label"]] += 1
    return counts


if __name__ == "__main__":
    tests = [
        test_stratified_folds_cover_every_clip_once,
        test_acoustic_cross_validation_writes_report,
        test_protocol_keeps_lockbox_out_of_development_folds,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
