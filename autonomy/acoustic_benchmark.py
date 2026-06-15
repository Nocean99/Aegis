from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path

from autonomy.acoustic_intelligence import analyze_acoustic_evidence


CLASSES = ("anthropogenic", "animal", "sonar")
POSITIVE_CLASS = "anthropogenic"


def run_acoustic_benchmark(
    *,
    dataset_root: str | Path,
    benchmark_root: str | Path = "benchmark_data/acoustic_v1",
    output_dir: str | Path = "logs/acoustic_benchmark_v1",
    sample_limit: int = 20,
    docs_snippet_path: str | Path = "docs/ACOUSTIC_BENCHMARK_V1_SNIPPET.md",
) -> Path:
    dataset = Path(dataset_root)
    benchmark = Path(benchmark_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_counts = inspect_dataset(dataset)
    sampled = create_benchmark_dataset(dataset, benchmark, sample_limit=sample_limit)
    csv_path = write_benchmark_csv(benchmark, sampled)

    evidence_dir = output / "evidence"
    acoustic_report_path = analyze_acoustic_evidence(
        [benchmark],
        mission_request="Identify anthropogenic underwater noise such as vessel or machinery activity.",
        output_dir=evidence_dir,
        labels_csv=csv_path,
    )
    acoustic_report = json.loads(acoustic_report_path.read_text(encoding="utf-8"))
    evaluation = acoustic_report.get("evaluation") or {}
    items = evaluation.get("items") or []
    candidate_counts = {
        Path(item.get("audio_path", "")).name: int(item.get("proposal_count") or 0)
        for item in items
    }

    report = {
        "benchmark": "Aegis Acoustic Benchmark v1",
        "dataset_root": str(dataset),
        "benchmark_root": str(benchmark),
        "benchmark_csv": str(csv_path),
        "acoustic_report": str(acoustic_report_path),
        "dataset_counts": source_counts,
        "sample_counts": {label: len(paths) for label, paths in sampled.items()},
        "positive_class": POSITIVE_CLASS,
        "negative_classes": [label for label in CLASSES if label != POSITIVE_CLASS],
        "prediction_rule": "candidate generated = positive; no candidate generated = negative",
        "confusion_matrix": {
            "true_positive": evaluation.get("true_positive", 0),
            "false_positive": evaluation.get("false_positive", 0),
            "true_negative": evaluation.get("true_negative", 0),
            "false_negative": evaluation.get("false_negative", 0),
        },
        "metrics": {
            "capture_precision": evaluation.get("capture_precision", 0.0),
            "capture_recall": evaluation.get("capture_recall", 0.0),
            "capture_f1": evaluation.get("capture_f1", 0.0),
            "proposal_count": evaluation.get("proposal_count", 0),
            "labeled_count": evaluation.get("labeled_count", 0),
        },
        "failure_analysis": failure_analysis(items, candidate_counts),
    }
    report["benchmark_observations"] = benchmark_observations(report)
    report["readme_snippet"] = readme_snippet(report)

    report_path = output / "acoustic_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "acoustic_benchmark_report.html").write_text(render_html(report), encoding="utf-8")

    docs_path = Path(docs_snippet_path)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(report["readme_snippet"] + "\n", encoding="utf-8")
    return report_path


def inspect_dataset(dataset_root: Path) -> dict[str, int]:
    counts = {}
    for label in CLASSES:
        folder = dataset_root / label
        counts[label] = len(sorted(folder.glob("*.wav"))) if folder.exists() else 0
    return counts


def create_benchmark_dataset(dataset_root: Path, benchmark_root: Path, *, sample_limit: int) -> dict[str, list[Path]]:
    benchmark_root.mkdir(parents=True, exist_ok=True)
    sampled: dict[str, list[Path]] = {}
    for label in CLASSES:
        source_folder = dataset_root / label
        target_folder = benchmark_root / label
        target_folder.mkdir(parents=True, exist_ok=True)
        source_files = sorted(source_folder.glob("*.wav"))[:sample_limit] if source_folder.exists() else []
        copied = []
        for source in source_files:
            target = target_folder / source.name
            shutil.copy2(source, target)
            copied.append(target)
        sampled[label] = copied
    return sampled


def write_benchmark_csv(benchmark_root: Path, sampled: dict[str, list[Path]]) -> Path:
    csv_path = benchmark_root / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "expected_match", "label"])
        writer.writeheader()
        for label in CLASSES:
            for path in sampled.get(label, []):
                writer.writerow(
                    {
                        "audio_path": str(path),
                        "expected_match": str(label == POSITIVE_CLASS).lower(),
                        "label": label,
                    }
                )
    return csv_path


def failure_analysis(items: list[dict], candidate_counts: dict[str, int]) -> dict:
    misses = []
    animal_false_positives = []
    sonar_false_positives = []
    for item in items:
        label = item.get("label")
        expected = bool(item.get("expected_match"))
        predicted = bool(item.get("predicted_match"))
        row = {
            "audio_path": item.get("audio_path"),
            "label": label,
            "proposal_count": candidate_counts.get(Path(item.get("audio_path", "")).name, 0),
        }
        if label == POSITIVE_CLASS and expected and not predicted:
            misses.append(row)
        elif label == "animal" and not expected and predicted:
            animal_false_positives.append(row)
        elif label == "sonar" and not expected and predicted:
            sonar_false_positives.append(row)
    return {
        "top_anthropogenic_misses": misses[:10],
        "top_animal_false_positives": sorted(animal_false_positives, key=lambda row: row["proposal_count"], reverse=True)[:10],
        "top_sonar_false_positives": sorted(sonar_false_positives, key=lambda row: row["proposal_count"], reverse=True)[:10],
    }


def benchmark_observations(report: dict) -> list[str]:
    metrics = report.get("metrics") or {}
    precision = float(metrics.get("capture_precision") or 0.0)
    recall = float(metrics.get("capture_recall") or 0.0)
    observations = [
        "This benchmark uses the current configured acoustic proposal pipeline.",
        "Anthropogenic clips are treated as positive; animal and sonar clips are treated as negative.",
    ]
    if recall >= 0.8 and precision < 0.6:
        observations.append("The current pipeline preserves positive acoustic evidence but over-proposes on negative classes.")
    elif precision >= 0.8 and recall < 0.6:
        observations.append("The current pipeline is conservative: generated candidates are often useful, but it misses too many positive clips.")
    elif precision >= 0.7 and recall >= 0.7:
        observations.append("The current proposal layer is a workable acoustic triage baseline before classifier training.")
    else:
        observations.append("The current proposal layer needs tuning before it can separate anthropogenic clips from negative acoustic classes.")
    return observations


def readme_snippet(report: dict) -> str:
    counts = report.get("dataset_counts") or {}
    metrics = report.get("metrics") or {}
    precision = percent(metrics.get("capture_precision"))
    recall = percent(metrics.get("capture_recall"))
    f1 = percent(metrics.get("capture_f1"))
    finding = (report.get("benchmark_observations") or ["Benchmark complete."])[-1]
    return f"""## Aegis Acoustic Benchmark v1

Dataset:

- Anthropogenic: {counts.get("anthropogenic", 0)}
- Animal: {counts.get("animal", 0)}
- Sonar: {counts.get("sonar", 0)}

Results:

- Precision: {precision}
- Recall: {recall}
- F1: {f1}

Key Finding: {finding}"""


def render_html(report: dict) -> str:
    failure = report.get("failure_analysis") or {}
    observations = "".join(f"<li>{esc(item)}</li>" for item in report.get("benchmark_observations", []))
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Aegis Acoustic Benchmark v1</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    pre {{ background: #f3f4f6; padding: 16px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border-bottom: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Aegis Acoustic Benchmark v1</h1>
  <p>Anthropogenic underwater noise is positive. Animal and sonar clips are negative.</p>
  <h2>Dataset Counts</h2>
  <pre>{esc(json.dumps(report.get("dataset_counts", {}), indent=2))}</pre>
  <h2>Sample Counts</h2>
  <pre>{esc(json.dumps(report.get("sample_counts", {}), indent=2))}</pre>
  <h2>Confusion Matrix</h2>
  <pre>{esc(json.dumps(report.get("confusion_matrix", {}), indent=2))}</pre>
  <h2>Metrics</h2>
  <pre>{esc(json.dumps(report.get("metrics", {}), indent=2))}</pre>
  <h2>Failure Analysis</h2>
  <pre>{esc(json.dumps(failure, indent=2))}</pre>
  <h2>Benchmark Observations</h2>
  <ul>{observations}</ul>
  <h2>README Snippet</h2>
  <pre>{esc(report.get("readme_snippet", ""))}</pre>
</body>
</html>
"""


def percent(value) -> str:
    return f"{float(value or 0.0) * 100:.1f}%"


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and run Aegis Acoustic Benchmark v1.")
    parser.add_argument("dataset_root")
    parser.add_argument("--benchmark-root", default="benchmark_data/acoustic_v1")
    parser.add_argument("--output-dir", default="logs/acoustic_benchmark_v1")
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()
    report_path = run_acoustic_benchmark(
        dataset_root=args.dataset_root,
        benchmark_root=args.benchmark_root,
        output_dir=args.output_dir,
        sample_limit=args.sample_limit,
    )
    print(f"Acoustic benchmark report saved: {report_path}")
    print(f"HTML report saved: {report_path.with_name('acoustic_benchmark_report.html')}")
    print("README snippet saved: docs/ACOUSTIC_BENCHMARK_V1_SNIPPET.md")


if __name__ == "__main__":
    main()
