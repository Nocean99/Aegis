from __future__ import annotations

import argparse
import csv
import html
import json
import random
import statistics
from pathlib import Path

from autonomy.benchmark_sample import link_or_copy
from autonomy.mission_evaluation import run_mission_evaluation


DEFAULT_LOCKBOX_FRACTION = 0.2


def run_visual_cross_validation(
    *,
    dataset_root: str | Path,
    labels_csv: str | Path,
    output_dir: str | Path,
    mission_request: str,
    modality: str,
    folds: int = 5,
    seed: int = 7,
    lockbox_fraction: float = DEFAULT_LOCKBOX_FRACTION,
    max_development_rows: int | None = None,
    run_evaluation: bool = True,
) -> Path:
    dataset = Path(dataset_root)
    labels_path = Path(labels_csv)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = read_visual_labels(labels_path)
    protocol = create_visual_evaluation_protocol(
        rows,
        folds=folds,
        seed=seed,
        lockbox_fraction=lockbox_fraction,
        max_development_rows=max_development_rows,
    )
    write_rows(output / "development_manifest.csv", protocol["development"])
    write_rows(output / "final_test_lockbox.csv", protocol["lockbox"])

    fold_reports = []
    totals = zero_totals()
    for index, fold in enumerate(protocol["development_folds"], start=1):
        fold_dir = output / f"fold_{index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        tuning_set = [
            item
            for fold_index, tuning_fold in enumerate(protocol["development_folds"], start=1)
            if fold_index != index
            for item in tuning_fold
        ]
        assert_disjoint(tuning_set, fold)
        write_rows(fold_dir / "tuning_manifest.csv", tuning_set)
        labels_for_eval = materialize_visual_fold(
            dataset_root=dataset,
            rows=fold,
            output_dir=fold_dir / "evaluation_sample",
        )
        fold_record = {
            "fold": index,
            "tuning_manifest": str(fold_dir / "tuning_manifest.csv"),
            "labels_csv": str(labels_for_eval),
            "tuning_counts": count_labels(tuning_set),
            "sample_counts": count_labels(fold),
            "leakage_check": {
                "tuning_count": len(tuning_set),
                "evaluation_count": len(fold),
                "overlap_count": overlap_count(tuning_set, fold),
            },
        }
        if run_evaluation:
            report_path = run_mission_evaluation(
                mission_request=mission_request,
                paths=[str(labels_for_eval.parent / "images")],
                output_dir=fold_dir / "mission_evaluation",
                proposal_mode="vehicle",
                save_only_detections=False,
                max_saved_candidates=max(50, len(fold)),
                labels_csv=labels_for_eval,
                semantic_vision="local",
                full_frame_semantic="misses",
            )
            mission_report = json.loads(report_path.read_text(encoding="utf-8"))
            vision_report = mission_report.get("vision_report") or {}
            summary = vision_report.get("summary") or {}
            evaluation = vision_report.get("evaluation") or {}
            analyst_capture = evaluation.get("analyst_capture") or {}
            add_totals(totals, evaluation, summary)
            fold_record.update(
                {
                    "mission_report": str(report_path),
                    "vision_report_path": mission_report.get("vision_report_path"),
                    "proposal_count": summary.get("detections", 0),
                    "shortlist_count": summary.get("shortlist_count", 0),
                    "confusion_matrix": confusion_matrix(analyst_capture),
                    "metrics": fold_metrics(evaluation),
                }
            )
        else:
            fold_record.update({"confusion_matrix": {}, "metrics": {}})
        fold_reports.append(fold_record)

    report = {
        "benchmark": f"Aegis {modality.upper()} Stratified Cross-Validation v1",
        "dataset_root": str(dataset),
        "labels_csv": str(labels_path),
        "modality": modality,
        "folds": folds,
        "seed": seed,
        "lockbox_fraction": lockbox_fraction,
        "max_development_rows": max_development_rows,
        "run_evaluation": run_evaluation,
        "dataset_counts": count_labels(rows),
        "development_counts": count_labels(protocol["development"]),
        "lockbox_counts": count_labels(protocol["lockbox"]),
        "lockbox_manifest": str(output / "final_test_lockbox.csv"),
        "leakage_policy": (
            "No image may appear in both a fold's tuning manifest and evaluation labels. "
            "The final lockbox manifest is written but must not be evaluated until tuning is frozen."
        ),
        "fold_reports": fold_reports,
        "mean_std_metrics": summarize_fold_metrics(fold_reports) if run_evaluation else {},
        "aggregate_confusion_matrix": confusion_matrix(totals),
        "aggregate_metrics": aggregate_metrics(totals) if run_evaluation else {},
    }
    report_path = output / "visual_cross_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path.with_suffix(".html").write_text(render_html(report), encoding="utf-8")
    return report_path


def read_visual_labels(labels_csv: Path) -> list[dict]:
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "expected_match", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Visual labels CSV missing columns: {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            rows.append(
                {
                    "image_path": row["image_path"],
                    "expected_match": parse_bool(row["expected_match"]),
                    "label": row.get("label") or ("positive" if str(row["expected_match"]).strip().lower() == "true" else "negative"),
                    "modality": row.get("modality") or "",
                    "source_dataset": row.get("source_dataset") or "",
                }
            )
        return rows


def create_visual_evaluation_protocol(
    rows: list[dict],
    *,
    folds: int,
    seed: int,
    lockbox_fraction: float,
    max_development_rows: int | None = None,
) -> dict:
    if not 0.0 <= lockbox_fraction < 0.5:
        raise ValueError("lockbox_fraction must be >= 0.0 and < 0.5")
    rng = random.Random(seed)
    positives = [row for row in rows if parse_bool(row["expected_match"])]
    negatives = [row for row in rows if not parse_bool(row["expected_match"])]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    lockbox = select_lockbox(positives, lockbox_fraction) + select_lockbox(negatives, lockbox_fraction)
    lockbox_paths = {row["image_path"] for row in lockbox}
    development_pool = [row for row in positives + negatives if row["image_path"] not in lockbox_paths]
    development = cap_development_rows(development_pool, max_rows=max_development_rows, rng=rng)
    folds_out = stratified_folds_from_rows(development, folds=folds, seed=seed + 1)
    assert_disjoint(development, lockbox)
    return {
        "development": sorted(development, key=row_key),
        "lockbox": sorted(lockbox, key=row_key),
        "development_folds": folds_out,
    }


def select_lockbox(rows: list[dict], fraction: float) -> list[dict]:
    count = int(round(len(rows) * fraction))
    if fraction and rows:
        count = max(1, count)
    return rows[:count]


def cap_development_rows(rows: list[dict], *, max_rows: int | None, rng: random.Random) -> list[dict]:
    if max_rows is None or len(rows) <= max_rows:
        return rows
    positives = [row for row in rows if parse_bool(row["expected_match"])]
    negatives = [row for row in rows if not parse_bool(row["expected_match"])]
    # Keep rare negatives when possible, then fill remaining slots with positives.
    rng.shuffle(positives)
    rng.shuffle(negatives)
    negative_count = min(len(negatives), max_rows // 2 if len(negatives) >= max_rows // 2 else len(negatives))
    positive_count = max_rows - negative_count
    selected = positives[:positive_count] + negatives[:negative_count]
    rng.shuffle(selected)
    return selected


def stratified_folds_from_rows(rows: list[dict], *, folds: int, seed: int) -> list[list[dict]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    rng = random.Random(seed)
    result: list[list[dict]] = [[] for _ in range(folds)]
    for expected in (True, False):
        class_rows = [row for row in rows if parse_bool(row["expected_match"]) is expected]
        rng.shuffle(class_rows)
        for index, row in enumerate(class_rows):
            result[index % folds].append(row)
    for fold in result:
        fold.sort(key=row_key)
    return result


def materialize_visual_fold(*, dataset_root: Path, rows: list[dict], output_dir: Path) -> Path:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    used_names = set()
    for index, row in enumerate(rows, start=1):
        source = dataset_root / row["image_path"]
        if not source.exists():
            continue
        name = unique_name(f"{index:05d}_{source.name}", used_names)
        target = image_dir / name
        link_or_copy(source, target)
        output_rows.append(
            {
                "image_path": f"images/{name}",
                "expected_match": parse_bool(row["expected_match"]),
                "label": row["label"],
                "modality": row.get("modality", ""),
                "source_dataset": row.get("source_dataset", ""),
            }
        )
    labels_path = output_dir / "labels.csv"
    write_rows(labels_path, output_rows)
    return labels_path


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "expected_match", "label", "modality", "source_dataset"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "expected_match": str(parse_bool(row["expected_match"])).lower(),
                    "label": row.get("label") or ("positive" if parse_bool(row["expected_match"]) else "negative"),
                    "modality": row.get("modality", ""),
                    "source_dataset": row.get("source_dataset", ""),
                }
            )
    return path


def count_labels(rows: list[dict]) -> dict[str, int]:
    positives = sum(1 for row in rows if parse_bool(row["expected_match"]))
    negatives = sum(1 for row in rows if not parse_bool(row["expected_match"]))
    return {"positive": positives, "negative": negatives, "total": len(rows)}


def assert_disjoint(left: list[dict], right: list[dict]) -> None:
    overlap = overlap_count(left, right)
    if overlap:
        raise ValueError(f"Evaluation leakage detected: {overlap} images appear in both sets.")


def overlap_count(left: list[dict], right: list[dict]) -> int:
    return len({row["image_path"] for row in left} & {row["image_path"] for row in right})


def zero_totals() -> dict[str, int]:
    return {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0, "proposal_count": 0, "labeled_count": 0}


def add_totals(totals: dict[str, int], evaluation: dict, summary: dict | None = None) -> None:
    analyst_capture = evaluation.get("analyst_capture") or {}
    totals["true_positive"] += int(analyst_capture.get("true_positive") or 0)
    totals["false_positive"] += int(analyst_capture.get("false_positive") or 0)
    totals["true_negative"] += int(analyst_capture.get("true_negative") or 0)
    totals["false_negative"] += int(analyst_capture.get("false_negative") or 0)
    totals["proposal_count"] += int((summary or {}).get("detections") or 0)
    totals["labeled_count"] += int(evaluation.get("labeled_count") or 0)


def confusion_matrix(evaluation_or_totals: dict) -> dict[str, int]:
    return {
        "true_positive": int(evaluation_or_totals.get("true_positive") or 0),
        "false_positive": int(evaluation_or_totals.get("false_positive") or 0),
        "true_negative": int(evaluation_or_totals.get("true_negative") or 0),
        "false_negative": int(evaluation_or_totals.get("false_negative") or 0),
    }


def fold_metrics(evaluation: dict) -> dict[str, float | int]:
    analyst_capture = evaluation.get("analyst_capture") or {}
    return {
        "capture_precision": analyst_capture.get("precision", 0.0),
        "capture_recall": analyst_capture.get("recall", 0.0),
        "capture_f1": analyst_capture.get("f1", 0.0),
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
        "labeled_count": totals["labeled_count"],
        **confusion_matrix(totals),
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


def unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    index = 2
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1


def row_key(row: dict) -> tuple:
    return (not parse_bool(row["expected_match"]), row["image_path"])


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "match", "positive"}


def render_html(report: dict) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{esc(item['fold'])}</td>"
        f"<td>{esc(json.dumps(item['sample_counts']))}</td>"
        f"<td>{percent(item.get('metrics', {}).get('capture_precision'))}</td>"
        f"<td>{percent(item.get('metrics', {}).get('capture_recall'))}</td>"
        f"<td>{percent(item.get('metrics', {}).get('capture_f1'))}</td>"
        f"<td>{esc(item['leakage_check']['overlap_count'])}</td>"
        "</tr>"
        for item in report.get("fold_reports", [])
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{esc(report.get("benchmark"))}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border-bottom: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    pre {{ background: #f3f4f6; padding: 16px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{esc(report.get("benchmark"))}</h1>
  <p>{esc(report.get("leakage_policy"))}</p>
  <h2>Counts</h2>
  <pre>{esc(json.dumps({"dataset": report.get("dataset_counts"), "development": report.get("development_counts"), "lockbox": report.get("lockbox_counts")}, indent=2))}</pre>
  <h2>Mean / Standard Deviation</h2>
  <pre>{esc(json.dumps(report.get("mean_std_metrics", {}), indent=2))}</pre>
  <h2>Aggregate Metrics</h2>
  <pre>{esc(json.dumps(report.get("aggregate_metrics", {}), indent=2))}</pre>
  <h2>Folds</h2>
  <table>
    <thead><tr><th>Fold</th><th>Sample Counts</th><th>Precision</th><th>Recall</th><th>F1</th><th>Leakage Overlap</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def percent(value) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value or 0.0) * 100:.1f}%"


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-controlled visual benchmark cross-validation.")
    parser.add_argument("dataset_root")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mission-request", required=True)
    parser.add_argument("--modality", choices=["rgb", "infrared"], required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lockbox-fraction", type=float, default=DEFAULT_LOCKBOX_FRACTION)
    parser.add_argument("--max-development-rows", type=int, default=None)
    parser.add_argument("--split-only", action="store_true")
    args = parser.parse_args()
    report_path = run_visual_cross_validation(
        dataset_root=args.dataset_root,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        mission_request=args.mission_request,
        modality=args.modality,
        folds=args.folds,
        seed=args.seed,
        lockbox_fraction=args.lockbox_fraction,
        max_development_rows=args.max_development_rows,
        run_evaluation=not args.split_only,
    )
    print(f"Visual cross-validation report saved: {report_path}")
    print(f"HTML report saved: {report_path.with_suffix('.html')}")


if __name__ == "__main__":
    main()
