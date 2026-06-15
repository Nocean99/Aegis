from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

from autonomy.mission_evaluation import run_mission_evaluation


def run_benchmark_suite(
    *,
    suite_path: str | Path = "config/mission_benchmark_suite.json",
    output_dir: str | Path = "logs/mission_benchmark_suites",
    semantic_vision: str | None = None,
    openai_detail: str | None = None,
    only: list[str] | None = None,
    include_disabled: bool = False,
) -> Path:
    suite_path = Path(suite_path)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = set(only or [])
    results = []
    for benchmark in suite.get("benchmarks", []):
        benchmark_id = benchmark.get("id", "unnamed")
        if selected_ids and benchmark_id not in selected_ids:
            continue
        if not include_disabled and not benchmark.get("enabled", True):
            results.append({"id": benchmark_id, "status": "skipped", "reason": "disabled"})
            continue
        result = run_one_benchmark(
            benchmark,
            run_dir=run_dir,
            semantic_vision=semantic_vision,
            openai_detail=openai_detail,
        )
        results.append(result)
    payload = {
        "timestamp": stamp,
        "suite_name": suite.get("suite_name", suite_path.stem),
        "suite_path": str(suite_path),
        "summary": summarize_suite(results),
        "benchmarks": results,
    }
    report_path = run_dir / "mission_benchmark_suite_report.json"
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    html_path = run_dir / "mission_benchmark_suite_report.html"
    html_path.write_text(render_html(payload, report_path.parent), encoding="utf-8")
    return report_path


def run_one_benchmark(
    benchmark: dict,
    *,
    run_dir: Path,
    semantic_vision: str | None,
    openai_detail: str | None,
) -> dict:
    benchmark_id = benchmark.get("id", "unnamed")
    benchmark_dir = run_dir / benchmark_id
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    try:
        report_path = run_mission_evaluation(
            mission_request=benchmark["mission_request"],
            paths=list(benchmark.get("paths", [])),
            operating_mode=benchmark.get("operating_mode", "connected-supervised"),
            output_dir=benchmark_dir,
            video=bool(benchmark.get("video", False)),
            sample_every_s=float(benchmark.get("sample_every_s", 1.0)),
            max_frames=benchmark.get("max_frames"),
            proposal_mode=benchmark.get("proposal_mode", "mission-color"),
            save_only_detections=bool(benchmark.get("save_only_detections", True)),
            max_saved_candidates=int(benchmark.get("max_saved_candidates", 50)),
            min_shortlist_score=float(benchmark.get("min_shortlist_score", 0.25)),
            labels_csv=benchmark.get("labels_csv"),
            eval_threshold=float(benchmark.get("eval_threshold", 0.25)),
            semantic_vision=semantic_vision or benchmark.get("semantic_vision", "local"),
            openai_model=benchmark.get("openai_model"),
            openai_detail=openai_detail or benchmark.get("openai_detail", "auto"),
            openai_timeout_s=float(benchmark.get("openai_timeout_s", 45.0)),
            full_frame_semantic=benchmark.get("full_frame_semantic", "off"),
        )
    except Exception as exc:
        return {
            "id": benchmark_id,
            "mission_type": benchmark.get("mission_type"),
            "status": "error",
            "error": str(exc),
        }
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "id": benchmark_id,
        "mission_type": benchmark.get("mission_type"),
        "status": "ok",
        "mission_request": benchmark.get("mission_request"),
        "report_path": str(report_path),
        "html_report_path": str(report_path.with_name("mission_evaluation_report.html")),
        "stage_summary": data.get("stage_summary"),
        "vision_summary": data.get("vision_summary"),
    }


def summarize_suite(results: list[dict]) -> dict:
    completed = [result for result in results if result.get("status") == "ok"]
    metrics = [result.get("vision_summary") or {} for result in completed]
    return {
        "configured": len(results),
        "completed": len(completed),
        "skipped": sum(1 for result in results if result.get("status") == "skipped"),
        "errors": sum(1 for result in results if result.get("status") == "error"),
        "avg_confirmed_precision": average(metric.get("precision") for metric in metrics),
        "avg_confirmed_recall": average(metric.get("recall") for metric in metrics),
        "avg_confirmed_f1": average(metric.get("f1") for metric in metrics),
        "avg_capture_precision": average(metric.get("capture_precision") for metric in metrics),
        "avg_capture_recall": average(metric.get("capture_recall") for metric in metrics),
        "avg_capture_f1": average(metric.get("capture_f1") for metric in metrics),
        "semantic_errors": sum(int(metric.get("semantic_errors") or 0) for metric in metrics),
    }


def average(values) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 4)


def render_html(data: dict, base_dir: Path) -> str:
    rows = "\n".join(result_row(result, base_dir) for result in data.get("benchmarks", []))
    summary = data.get("summary") or {}
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Mission Benchmark Suite</title>
    <style>{CSS}</style>
  </head>
  <body>
    <main class="shell">
      <header>
        <p class="eyebrow">Mission Intelligence Benchmark Suite</p>
        <h1>{esc(data.get("suite_name"))}</h1>
      </header>
      <section class="metrics">
        {metric("Completed", summary.get("completed"))}
        {metric("Errors", summary.get("errors"))}
        {metric("Confirmed F1", summary.get("avg_confirmed_f1"))}
        {metric("Capture F1", summary.get("avg_capture_f1"))}
        {metric("Capture Recall", summary.get("avg_capture_recall"))}
        {metric("Semantic Errors", summary.get("semantic_errors"))}
      </section>
      <section class="panel">
        <h2>Benchmarks</h2>
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Type</th>
              <th>Status</th>
              <th>Confirmed P/R/F1</th>
              <th>Capture P/R/F1</th>
              <th>Report</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </section>
    </main>
  </body>
</html>
"""


def result_row(result: dict, base_dir: Path) -> str:
    summary = result.get("vision_summary") or {}
    report_link = ""
    if result.get("html_report_path"):
        path = Path(result["html_report_path"])
        try:
            path = path.relative_to(base_dir)
        except ValueError:
            pass
        report_link = f'<a href="{esc(str(path))}">HTML</a>'
    return (
        "<tr>"
        f"<td>{esc(result.get('id'))}</td>"
        f"<td>{esc(result.get('mission_type'))}</td>"
        f"<td>{esc(result.get('status'))}</td>"
        f"<td>{triple(summary.get('precision'), summary.get('recall'), summary.get('f1'))}</td>"
        f"<td>{triple(summary.get('capture_precision'), summary.get('capture_recall'), summary.get('capture_f1'))}</td>"
        f"<td>{report_link}</td>"
        "</tr>"
    )


def metric(label: str, value) -> str:
    return f'<div class="metric"><span>{esc(label)}</span><strong>{esc("n/a" if value is None else value)}</strong></div>'


def triple(a, b, c) -> str:
    return f"{esc(a)} / {esc(b)} / {esc(c)}"


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


CSS = """
:root { color-scheme: dark; --bg: #101417; --panel: #182024; --panel2: #202a2f; --line: #324147; --text: #eef5f2; --muted: #a9bbb4; --accent: #63cdda; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.shell { width: min(1500px, calc(100% - 28px)); margin: 0 auto; padding: 22px 0 32px; }
.eyebrow { margin: 0 0 6px; color: var(--accent); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
h1, h2 { margin: 0; }
h1 { margin-bottom: 16px; font-size: clamp(1.45rem, 3vw, 2.55rem); }
h2 { margin-bottom: 12px; font-size: 1rem; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 12px; }
.metric, .panel { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
.metric { padding: 12px; }
.metric span { display: block; color: var(--muted); font-size: .78rem; }
.metric strong { display: block; margin-top: 4px; font-size: 1.25rem; }
.panel { padding: 14px; }
table { width: 100%; border-collapse: collapse; }
th, td { border-top: 1px solid var(--line); padding: 10px 8px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
a { color: var(--accent); }
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configured mission benchmark evaluations")
    parser.add_argument("--suite", default="config/mission_benchmark_suite.json")
    parser.add_argument("--output-dir", default="logs/mission_benchmark_suites")
    parser.add_argument("--semantic-vision", choices=["local", "clip", "openai"], default=None)
    parser.add_argument("--openai-detail", choices=["auto", "low", "high"], default=None)
    parser.add_argument("--only", action="append", default=None, help="Run only this benchmark id. May be repeated.")
    parser.add_argument("--include-disabled", action="store_true")
    args = parser.parse_args()
    report_path = run_benchmark_suite(
        suite_path=args.suite,
        output_dir=args.output_dir,
        semantic_vision=args.semantic_vision,
        openai_detail=args.openai_detail,
        only=args.only,
        include_disabled=args.include_disabled,
    )
    print(f"Mission benchmark suite saved: {report_path}")
    print(f"HTML report saved: {report_path.with_name('mission_benchmark_suite_report.html')}")


if __name__ == "__main__":
    main()
