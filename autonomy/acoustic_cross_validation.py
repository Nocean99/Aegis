from __future__ import annotations

import argparse
import csv
import html
import json
import random
import statistics
from pathlib import Path

from autonomy.acoustic_benchmark import CLASSES, POSITIVE_CLASS, inspect_dataset
from autonomy.acoustic_intelligence import analyze_acoustic_evidence


DEFAULT_LOCKBOX_FRACTION = 0.2


def run_acoustic_cross_validation(
    *,
    dataset_root: str | Path,
    output_dir: str | Path = "logs/acoustic_cross_validation_v1",
    folds: int = 5,
    seed: int = 7,
    lockbox_fraction: float = DEFAULT_LOCKBOX_FRACTION,
) -> Path:
    dataset = Path(dataset_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    split_protocol = create_evaluation_protocol(
        dataset,
        folds=folds,
        seed=seed,
        lockbox_fraction=lockbox_fraction,
    )
    split = split_protocol["development_folds"]
    fold_reports = []
    totals = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0, "proposal_count": 0, "labeled_count": 0}
    for index, fold in enumerate(split, start=1):
        fold_dir = output / f"fold_{index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        tuning_set = [item for fold_index, tuning_fold in enumerate(split, start=1) if fold_index != index for item in tuning_fold]
        assert_disjoint(tuning_set, fold)
        write_fold_labels(fold_dir / "tuning_manifest.csv", tuning_set)
        labels_csv = write_fold_labels(fold_dir / "labels.csv", fold)
        report_path = analyze_acoustic_evidence(
            [item["path"] for item in fold],
            mission_request="Identify anthropogenic underwater noise such as vessel or machinery activity.",
            output_dir=fold_dir / "evidence",
            labels_csv=labels_csv,
        )
        acoustic_report = json.loads(report_path.read_text(encoding="utf-8"))
        evaluation = acoustic_report.get("evaluation") or {}
        metrics = fold_metrics(evaluation)
        for key in totals:
            totals[key] += int(evaluation.get(key) or 0)
        fold_reports.append(
            {
                "fold": index,
                "labels_csv": str(labels_csv),
                "tuning_manifest": str(fold_dir / "tuning_manifest.csv"),
                "acoustic_report": str(report_path),
                "tuning_counts": count_labels(tuning_set),
                "sample_counts": count_labels(fold),
                "leakage_check": {
                    "tuning_count": len(tuning_set),
                    "evaluation_count": len(fold),
                    "overlap_count": overlap_count(tuning_set, fold),
                },
                "confusion_matrix": {
                    "true_positive": evaluation.get("true_positive", 0),
                    "false_positive": evaluation.get("false_positive", 0),
                    "true_negative": evaluation.get("true_negative", 0),
                    "false_negative": evaluation.get("false_negative", 0),
                },
                "metrics": metrics,
            }
        )

    aggregate = aggregate_metrics(totals)
    fold_summary = summarize_fold_metrics(fold_reports)
    report = {
        "benchmark": "Aegis Acoustic Stratified Cross-Validation v1",
        "dataset_root": str(dataset),
        "dataset_counts": inspect_dataset(dataset),
        "folds": folds,
        "seed": seed,
        "lockbox_fraction": lockbox_fraction,
        "positive_class": POSITIVE_CLASS,
        "negative_classes": [label for label in CLASSES if label != POSITIVE_CLASS],
        "method_note": (
            "This evaluates the currently frozen acoustic heuristic on development folds only. "
            "Each fold writes a tuning manifest containing the other folds and an evaluation labels file "
            "containing only the held-out fold. The final lockbox manifest is written but not evaluated."
        ),
        "leakage_policy": (
            "No clip may appear in both the tuning manifest and evaluation labels for the same fold. "
            "The lockbox split must not be used for tuning; evaluate it exactly once only after all tuning is frozen."
        ),
        "lockbox_manifest": str(output / "final_test_lockbox.csv"),
        "lockbox_counts": count_labels(split_protocol["lockbox"]),
        "development_counts": count_labels(split_protocol["development"]),
        "fold_reports": fold_reports,
        "mean_std_metrics": fold_summary,
        "aggregate_confusion_matrix": {
            "true_positive": totals["true_positive"],
            "false_positive": totals["false_positive"],
            "true_negative": totals["true_negative"],
            "false_negative": totals["false_negative"],
        },
        "aggregate_metrics": aggregate,
    }
    write_fold_labels(output / "development_manifest.csv", split_protocol["development"])
    write_fold_labels(output / "final_test_lockbox.csv", split_protocol["lockbox"])
    report_path = output / "acoustic_cross_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path.with_suffix(".html").write_text(render_html(report), encoding="utf-8")
    return report_path


def create_evaluation_protocol(
    dataset_root: Path,
    *,
    folds: int,
    seed: int,
    lockbox_fraction: float,
) -> dict:
    if not 0.0 <= lockbox_fraction < 0.5:
        raise ValueError("lockbox_fraction must be >= 0.0 and < 0.5")
    development: list[dict] = []
    lockbox: list[dict] = []
    rng = random.Random(seed)
    for label in CLASSES:
        files = sorted((dataset_root / label).glob("*.wav"))
        rng.shuffle(files)
        lockbox_count = int(round(len(files) * lockbox_fraction))
        if lockbox_fraction and files:
            lockbox_count = max(1, lockbox_count)
        label_lockbox = files[:lockbox_count]
        label_development = files[lockbox_count:]
        for path in label_development:
            development.append(label_item(path, label))
        for path in label_lockbox:
            lockbox.append(label_item(path, label))
    development_folds = stratified_folds_from_items(development, folds=folds, seed=seed + 1)
    assert_disjoint(development, lockbox)
    return {
        "development": sorted(development, key=lambda item: (item["label"], item["path"])),
        "lockbox": sorted(lockbox, key=lambda item: (item["label"], item["path"])),
        "development_folds": development_folds,
    }


def stratified_folds(dataset_root: Path, *, folds: int, seed: int) -> list[list[dict]]:
    items = []
    for label in CLASSES:
        items.extend(label_item(path, label) for path in sorted((dataset_root / label).glob("*.wav")))
    return stratified_folds_from_items(items, folds=folds, seed=seed)


def stratified_folds_from_items(items: list[dict], *, folds: int, seed: int) -> list[list[dict]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    result: list[list[dict]] = [[] for _ in range(folds)]
    rng = random.Random(seed)
    for label in CLASSES:
        label_items = [item for item in items if item["label"] == label]
        rng.shuffle(label_items)
        for index, item in enumerate(label_items):
            result[index % folds].append(item)
    for fold in result:
        fold.sort(key=lambda item: (item["label"], item["path"]))
    return result


def label_item(path: Path, label: str) -> dict:
    return {
        "path": str(path),
        "expected_match": label == POSITIVE_CLASS,
        "label": label,
    }


def write_fold_labels(path: Path, fold: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "expected_match", "label"])
        writer.writeheader()
        for item in fold:
            writer.writerow(
                {
                    "audio_path": item["path"],
                    "expected_match": str(item["expected_match"]).lower(),
                    "label": item["label"],
                }
            )
    return path


def count_labels(fold: list[dict]) -> dict[str, int]:
    counts = {label: 0 for label in CLASSES}
    for item in fold:
        counts[item["label"]] = counts.get(item["label"], 0) + 1
    return counts


def assert_disjoint(left: list[dict], right: list[dict]) -> None:
    overlap = overlap_count(left, right)
    if overlap:
        raise ValueError(f"Evaluation leakage detected: {overlap} clips appear in both sets.")


def overlap_count(left: list[dict], right: list[dict]) -> int:
    left_paths = {item["path"] for item in left}
    right_paths = {item["path"] for item in right}
    return len(left_paths & right_paths)


def fold_metrics(evaluation: dict) -> dict[str, float | int]:
    return {
        "capture_precision": evaluation.get("capture_precision", 0.0),
        "capture_recall": evaluation.get("capture_recall", 0.0),
        "capture_f1": evaluation.get("capture_f1", 0.0),
        "proposal_count": evaluation.get("proposal_count", 0),
        "labeled_count": evaluation.get("labeled_count", 0),
    }


def aggregate_metrics(totals: dict[str, int]) -> dict[str, float | int]:
    tp = totals["true_positive"]
    fp = totals["false_positive"]
    tn = totals["true_negative"]
    fn = totals["false_negative"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "capture_precision": round(precision, 4),
        "capture_recall": round(recall, 4),
        "capture_f1": round(f1, 4),
        "proposal_count": totals["proposal_count"],
        "labeled_count": totals["labeled_count"],
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def summarize_fold_metrics(fold_reports: list[dict]) -> dict[str, dict[str, float]]:
    summary = {}
    for metric in ("capture_precision", "capture_recall", "capture_f1"):
        values = [float(report["metrics"].get(metric) or 0.0) for report in fold_reports]
        summary[metric] = {
            "mean": round(statistics.fmean(values), 4) if values else 0.0,
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        }
    return summary


def render_html(report: dict) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{esc(item['fold'])}</td>"
        f"<td>{esc(json.dumps(item['sample_counts']))}</td>"
        f"<td>{percent(item['metrics']['capture_precision'])}</td>"
        f"<td>{percent(item['metrics']['capture_recall'])}</td>"
        f"<td>{percent(item['metrics']['capture_f1'])}</td>"
        f"<td>{esc(item['metrics']['proposal_count'])}</td>"
        "</tr>"
        for item in report.get("fold_reports", [])
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Aegis Acoustic Cross-Validation v1</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border-bottom: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    pre {{ background: #f3f4f6; padding: 16px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Aegis Acoustic Cross-Validation v1</h1>
  <p>{esc(report.get("method_note"))}</p>
  <h2>Dataset Counts</h2>
  <pre>{esc(json.dumps(report.get("dataset_counts", {}), indent=2))}</pre>
  <h2>Aggregate Metrics</h2>
  <pre>{esc(json.dumps(report.get("aggregate_metrics", {}), indent=2))}</pre>
  <h2>Mean / Standard Deviation Across Folds</h2>
  <pre>{esc(json.dumps(report.get("mean_std_metrics", {}), indent=2))}</pre>
  <h2>Lockbox</h2>
  <p>{esc(report.get("leakage_policy"))}</p>
  <pre>{esc(json.dumps({"manifest": report.get("lockbox_manifest"), "counts": report.get("lockbox_counts")}, indent=2))}</pre>
  <h2>Folds</h2>
  <table>
    <thead><tr><th>Fold</th><th>Sample Counts</th><th>Precision</th><th>Recall</th><th>F1</th><th>Proposals</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def percent(value) -> str:
    return f"{float(value or 0.0) * 100:.1f}%"


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stratified k-fold acoustic evaluation.")
    parser.add_argument("dataset_root")
    parser.add_argument("--output-dir", default="logs/acoustic_cross_validation_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lockbox-fraction", type=float, default=DEFAULT_LOCKBOX_FRACTION)
    args = parser.parse_args()
    report_path = run_acoustic_cross_validation(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        folds=args.folds,
        seed=args.seed,
        lockbox_fraction=args.lockbox_fraction,
    )
    print(f"Acoustic cross-validation report saved: {report_path}")
    print(f"HTML report saved: {report_path.with_suffix('.html')}")


if __name__ == "__main__":
    main()
