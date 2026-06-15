from __future__ import annotations

import argparse
import html
import json
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from autonomy.contextual_search_plan import create_contextual_search_plan
from autonomy.mission_command import create_mission_command
from autonomy.mission_memory import mission_memory_snapshot
from autonomy.mission_vision_plan import create_mission_vision_plan
from autonomy.vision_lab import collect_image_paths, collect_video_paths, run_video_vision_lab, run_vision_lab


def run_mission_evaluation(
    *,
    mission_request: str,
    paths: list[str],
    operating_mode: str = "connected-supervised",
    output_dir: str | Path = "logs/mission_evaluations",
    config_path: str | Path = "config/autonomy.yaml",
    video: bool = False,
    sample_every_s: float = 1.0,
    max_frames: int | None = None,
    proposal_mode: str = "mission-color",
    save_only_detections: bool = True,
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
    full_frame_semantic = effective_full_frame_semantic_mode(semantic_vision, full_frame_semantic)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    stages: list[dict] = []
    payload: dict = {
        "timestamp": stamp,
        "mission_request": mission_request,
        "operating_mode": operating_mode,
        "source_paths": paths,
        "source_type": "video" if video else "images",
        "resilience_notes": [
            "Stages are isolated so a failed component records an error without erasing earlier outputs.",
            "Navigation, collection, perception, semantic scoring, and analyst review should remain decoupled in future live systems.",
        ],
    }

    command = _run_stage(stages, "mission_command", lambda: create_mission_command(mission_request, operating_mode=operating_mode))
    if command is not None:
        payload["command"] = asdict(command)
    vision_plan = _run_stage(stages, "vision_plan", lambda: create_mission_vision_plan(command.objective if command else mission_request))
    if vision_plan is not None:
        payload["vision_plan"] = asdict(vision_plan)
    contextual_plan = _run_stage(stages, "contextual_search_plan", lambda: create_contextual_search_plan(command.objective if command else mission_request))
    if contextual_plan is not None:
        payload["contextual_search_plan"] = asdict(contextual_plan)

    evidence_paths = _run_stage(
        stages,
        "evidence_collection",
        lambda: collect_video_paths(paths) if video else collect_image_paths(paths),
    )
    if evidence_paths:
        payload["evidence_count"] = len(evidence_paths)
        vision_report_path = _run_stage(
            stages,
            "vision_benchmark",
            lambda: _run_vision_stage(
                mission_request=mission_request,
                evidence_paths=evidence_paths,
                config_path=config_path,
                output_dir=run_dir / "vision_lab",
                video=video,
                sample_every_s=sample_every_s,
                max_frames=max_frames,
                proposal_mode=proposal_mode,
                save_only_detections=save_only_detections,
                max_saved_candidates=max_saved_candidates,
                min_shortlist_score=min_shortlist_score,
                labels_csv=labels_csv,
                eval_threshold=eval_threshold,
                semantic_vision=semantic_vision,
                openai_model=openai_model,
                openai_detail=openai_detail,
                openai_timeout_s=openai_timeout_s,
                full_frame_semantic=full_frame_semantic,
            ),
        )
        if vision_report_path is not None:
            payload["vision_report_path"] = str(vision_report_path)
            vision_report = _safe_load_json(vision_report_path)
            payload["vision_summary"] = _summarize_vision_report(vision_report)
            payload["vision_report"] = vision_report
            payload["mission_memory"] = mission_memory_snapshot(vision_report)
    else:
        stages.append(
            {
                "name": "vision_benchmark",
                "status": "skipped",
                "error": "No compatible evidence files found.",
            }
        )
        payload["evidence_count"] = 0

    payload["stages"] = stages
    payload["stage_summary"] = _stage_summary(stages)
    report_path = run_dir / "mission_evaluation_report.json"
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    html_path = run_dir / "mission_evaluation_report.html"
    html_path.write_text(render_html(payload, report_path.parent), encoding="utf-8")
    return report_path


def effective_full_frame_semantic_mode(semantic_vision: str, full_frame_semantic: str) -> str:
    if semantic_vision == "openai" and full_frame_semantic == "off":
        return "misses"
    return full_frame_semantic


def _run_vision_stage(**kwargs) -> Path:
    evidence_paths: list[Path] = kwargs.pop("evidence_paths")
    video: bool = kwargs.pop("video")
    sample_every_s = kwargs.pop("sample_every_s")
    max_frames = kwargs.pop("max_frames")
    if video:
        if len(evidence_paths) != 1:
            raise ValueError("Video evaluation currently accepts exactly one video file.")
        return run_video_vision_lab(
            video_path=evidence_paths[0],
            sample_every_s=sample_every_s,
            max_frames=max_frames,
            **kwargs,
        )
    return run_vision_lab(image_paths=evidence_paths, **kwargs)


def _run_stage(stages: list[dict], name: str, fn):
    try:
        result = fn()
    except Exception as exc:  # Keep the mission report alive even when one subsystem breaks.
        stages.append(
            {
                "name": name,
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(limit=6),
            }
        )
        return None
    stages.append({"name": name, "status": "ok"})
    return result


def _safe_load_json(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": f"Could not load vision report: {exc}"}


def _summarize_vision_report(report: dict) -> dict:
    summary = report.get("summary") or {}
    evaluation = report.get("evaluation") or {}
    analyst_capture = evaluation.get("analyst_capture") or {}
    return {
        "processed": summary.get("processed"),
        "detections": summary.get("detections"),
        "shortlist_count": summary.get("shortlist_count"),
        "semantic_errors": summary.get("semantic_errors"),
        "precision": evaluation.get("precision"),
        "recall": evaluation.get("recall"),
        "f1": evaluation.get("f1"),
        "capture_precision": analyst_capture.get("precision"),
        "capture_recall": analyst_capture.get("recall"),
        "capture_f1": analyst_capture.get("f1"),
        "possible_miss_count": summary.get("possible_miss_count"),
    }


def _stage_summary(stages: list[dict]) -> dict:
    return {
        "ok": sum(1 for stage in stages if stage.get("status") == "ok"),
        "error": sum(1 for stage in stages if stage.get("status") == "error"),
        "skipped": sum(1 for stage in stages if stage.get("status") == "skipped"),
    }


def render_html(data: dict, base_dir: Path) -> str:
    command = data.get("command") or {}
    objective = command.get("objective") or {}
    vision = data.get("vision_plan") or {}
    context = data.get("contextual_search_plan") or {}
    vision_summary = data.get("vision_summary") or {}
    vision_report = data.get("vision_report") or {}
    mission_memory = data.get("mission_memory") or {}
    review_summary = analyst_decision_summary(data)
    stages = data.get("stages") or []
    likely = context.get("likely_locations") or []
    deprioritized = context.get("deprioritized_locations") or []
    shortlist = ((vision_report.get("summary") or {}).get("shortlist") or [])[:8]
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Mission Evaluation Report</title>
    <style>{CSS}</style>
  </head>
  <body>
    <main class="shell">
      <header class="top">
        <div>
          <p class="eyebrow">Mission Intelligence Evaluation</p>
          <h1>{esc(data.get("mission_request"))}</h1>
        </div>
        <div class="meta">
          <span>{esc(data.get("timestamp"))}</span>
          <span>{esc(data.get("operating_mode"))}</span>
          <span>{esc(data.get("source_type"))}</span>
        </div>
      </header>

      <section class="metrics">
        {metric("Evidence", data.get("evidence_count"))}
        {metric("Processed", vision_summary.get("processed"))}
        {metric("Detections", vision_summary.get("detections"))}
        {metric("Shortlist", vision_summary.get("shortlist_count"))}
        {metric("Semantic Errors", vision_summary.get("semantic_errors"))}
        {metric("Precision", vision_summary.get("precision"))}
        {metric("Recall", vision_summary.get("recall"))}
        {metric("Capture Recall", vision_summary.get("capture_recall"))}
        {metric("Stage Errors", (data.get("stage_summary") or {}).get("error"))}
      </section>

      <section class="grid">
        <article class="panel">
          <h2>Mission Objective</h2>
          {row("Target", objective.get("target_description"))}
          {row("Search Area", objective.get("search_area_description") or "unspecified")}
          {row("Urgency", objective.get("urgency"))}
          {row("Categories", join_values(objective.get("extracted_categories")))}
        </article>
        <article class="panel">
          <h2>Vision Strategy</h2>
          {row("Colors", join_values(vision.get("important_colors")))}
          {row("Proposal Modes", join_values(vision.get("proposal_modes")))}
          {row("Review Threshold", vision.get("review_threshold"))}
        </article>
      </section>

      <section class="grid">
        <article class="panel">
          <h2>Evidence Collected</h2>
          {row("Source Type", data.get("source_type"))}
          {row("Evidence Files", data.get("evidence_count"))}
          {row("Processed Frames", vision_summary.get("processed"))}
          {row("Semantic Errors", vision_summary.get("semantic_errors"))}
        </article>
        <article class="panel">
          <h2>Performance Metrics</h2>
          {row("Confirmed Precision", vision_summary.get("precision"))}
          {row("Confirmed Recall", vision_summary.get("recall"))}
          {row("Confirmed F1", vision_summary.get("f1"))}
          {row("Capture Recall", vision_summary.get("capture_recall"))}
        </article>
      </section>

      <section class="panel">
        <h2>Candidates Found</h2>
        {candidate_table(shortlist)}
      </section>

      <section class="panel">
        <h2>Analyst Decisions</h2>
        {decision_summary(review_summary)}
      </section>

      <section class="panel">
        <h2>Search Priorities</h2>
        {priority_list(likely) or '<p class="muted">No priorities inferred.</p>'}
      </section>

      <section class="panel">
        <h2>Deprioritized Areas</h2>
        {priority_list(deprioritized) or '<p class="muted">None listed.</p>'}
      </section>

      <section class="panel">
        <h2>Stage Health</h2>
        {stage_table(stages)}
      </section>

      <section class="grid">
        <article class="panel">
          <h2>Mission Memory</h2>
          {memory_list("False Positive Patterns", mission_memory.get("false_positive_patterns"))}
          {memory_list("Miss Patterns", mission_memory.get("miss_patterns"))}
          {memory_list("Weak Categories", mission_memory.get("weak_categories"))}
        </article>
        <article class="panel">
          <h2>Recommendations</h2>
          {memory_list("Recommended Data", mission_memory.get("recommended_data"))}
        </article>
      </section>

      <section class="panel">
        <h2>Artifacts</h2>
        {artifact_link("Mission JSON", "mission_evaluation_report.json")}
        {artifact_link("Vision JSON", data.get("vision_report_path"), base_dir)}
      </section>
    </main>
  </body>
</html>
"""


def metric(label: str, value) -> str:
    return f'<div class="metric"><span>{esc(label)}</span><strong>{esc("n/a" if value is None else value)}</strong></div>'


def row(label: str, value) -> str:
    return f'<div class="row"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'


def priority_list(items: list[dict]) -> str:
    rendered = []
    for item in items:
        rendered.append(
            f"""
            <article class="priority">
              <strong>{esc(item.get("priority"))} · {esc(item.get("name"))}</strong>
              <p>{esc(item.get("rationale"))}</p>
              <span>{esc(join_values(item.get("cues")))}</span>
            </article>
            """
        )
    return "\n".join(rendered)


def stage_table(stages: list[dict]) -> str:
    rows = []
    for stage in stages:
        rows.append(
            f"<tr><td>{esc(stage.get('name'))}</td><td>{esc(stage.get('status'))}</td><td>{esc(stage.get('error', ''))}</td></tr>"
        )
    return f"<table><thead><tr><th>Stage</th><th>Status</th><th>Notes</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"


def candidate_table(candidates: list[dict]) -> str:
    if not candidates:
        return '<p class="muted">No candidates found.</p>'
    rows = []
    for item in candidates:
        rows.append(
            "<tr>"
            f"<td>{esc(item.get('candidate_id'))}</td>"
            f"<td>{esc(Path(str(item.get('image_path') or '')).name)}</td>"
            f"<td>{esc(item.get('decision'))}</td>"
            f"<td>{esc(item.get('review_priority'))}</td>"
            f"<td>{esc(join_values(item.get('review_reasons')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>ID</th><th>Evidence</th><th>Decision</th><th>Priority</th><th>Why</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def analyst_decision_summary(data: dict) -> dict:
    report_path = data.get("vision_report_path")
    if not report_path:
        return {"reviewed": 0, "decisions": {}, "reviews": []}
    review_path = Path(report_path).with_name("candidate_reviews.json")
    if not review_path.exists():
        return {"reviewed": 0, "decisions": {}, "reviews": []}
    try:
        reviews = json.loads(review_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"reviewed": 0, "decisions": {}, "reviews": []}
    decisions: dict[str, int] = {}
    for review in reviews.values():
        decision = str(review.get("decision") or review.get("status") or "unknown")
        decisions[decision] = decisions.get(decision, 0) + 1
    return {"reviewed": len(reviews), "decisions": decisions, "reviews": list(reviews.values())[:12]}


def decision_summary(summary: dict) -> str:
    if not summary.get("reviewed"):
        return '<p class="muted">No analyst decisions saved yet.</p>'
    rows = "".join(
        f"<tr><td>{esc(review.get('candidate_id'))}</td><td>{esc(review.get('decision') or review.get('status'))}</td><td>{esc(review.get('reason'))}</td><td>{esc(review.get('notes'))}</td></tr>"
        for review in summary.get("reviews", [])
    )
    counts = ", ".join(f"{key}: {value}" for key, value in (summary.get("decisions") or {}).items())
    return f"<p class=\"muted\">Reviewed {esc(summary.get('reviewed'))}. {esc(counts)}</p><table><thead><tr><th>Candidate</th><th>Decision</th><th>Reason</th><th>Notes</th></tr></thead><tbody>{rows}</tbody></table>"


def memory_list(label: str, values) -> str:
    items = [str(value) for value in values or []]
    if not items:
        return f'<div class="row"><span>{esc(label)}</span><strong>none yet</strong></div>'
    return f'<div class="row"><span>{esc(label)}</span><strong>{esc(", ".join(items))}</strong></div>'


def artifact_link(label: str, path_value, base_dir: Path | None = None) -> str:
    if not path_value:
        return f'<p class="muted">{esc(label)} unavailable.</p>'
    path = Path(path_value)
    if base_dir is not None and path.is_absolute():
        try:
            path = path.relative_to(base_dir)
        except ValueError:
            pass
    return f'<p><a href="{quote(str(path))}">{esc(label)}</a></p>'


def join_values(values) -> str:
    return ", ".join(str(value) for value in values or []) or "none"


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


CSS = """
:root { color-scheme: dark; --bg: #101417; --panel: #182024; --panel2: #202a2f; --line: #324147; --text: #eef5f2; --muted: #a9bbb4; --accent: #63cdda; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.shell { width: min(1500px, calc(100% - 28px)); margin: 0 auto; padding: 22px 0 32px; }
.top { display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; margin-bottom: 16px; }
.eyebrow { margin: 0 0 6px; color: var(--accent); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
h1, h2, p { margin: 0; }
h1 { max-width: 960px; font-size: clamp(1.45rem, 3vw, 2.55rem); line-height: 1.08; }
h2 { margin-bottom: 12px; font-size: 1rem; }
.meta { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
.meta span { display: inline-flex; min-height: 30px; align-items: center; border: 1px solid var(--line); border-radius: 999px; background: var(--panel2); padding: 0 10px; color: var(--muted); font-size: .8rem; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(115px, 1fr)); gap: 10px; margin-bottom: 12px; }
.metric, .panel { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
.metric { padding: 12px; }
.metric span, .row span { display: block; color: var(--muted); font-size: .78rem; }
.metric strong { display: block; margin-top: 4px; font-size: 1.25rem; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.panel { padding: 14px; margin-bottom: 12px; }
.row { display: grid; gap: 4px; margin-top: 10px; }
.row strong { overflow-wrap: anywhere; }
.priority { border-top: 1px solid var(--line); padding: 12px 0; }
.priority:first-of-type { border-top: 0; padding-top: 0; }
.priority p { margin-top: 4px; color: var(--text); }
.priority span, .muted { color: var(--muted); }
table { width: 100%; border-collapse: collapse; }
th, td { border-top: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
a { color: var(--accent); }
@media (max-width: 980px) { .top, .grid { display: block; } .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full mission intelligence evaluation over images or one video")
    parser.add_argument("paths", nargs="+", help="Image/video files or folders")
    parser.add_argument("--mission-request", required=True)
    parser.add_argument("--mode", default="connected-supervised", help="connected-supervised or autonomous-return-report")
    parser.add_argument("--output-dir", default="logs/mission_evaluations")
    parser.add_argument("--config", default="config/autonomy.yaml")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--sample-every-s", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--proposal-mode", choices=["precise", "high-recall", "mission-color", "vehicle", "yolo"], default="mission-color")
    parser.add_argument("--max-saved-candidates", type=int, default=50)
    parser.add_argument("--min-shortlist-score", type=float, default=0.25)
    parser.add_argument("--labels-csv", default=None)
    parser.add_argument("--eval-threshold", type=float, default=0.25)
    parser.add_argument("--semantic-vision", choices=["local", "clip", "openai"], default="local")
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--openai-detail", choices=["auto", "low", "high"], default="auto")
    parser.add_argument("--openai-timeout-s", type=float, default=45.0)
    parser.add_argument("--full-frame-semantic", choices=["off", "misses", "all"], default="off")
    parser.add_argument("--save-all-debug-images", action="store_true", help="Save debug/crop files for all processed frames.")
    args = parser.parse_args()

    report_path = run_mission_evaluation(
        mission_request=args.mission_request,
        paths=args.paths,
        operating_mode=args.mode,
        output_dir=args.output_dir,
        config_path=args.config,
        video=args.video,
        sample_every_s=args.sample_every_s,
        max_frames=args.max_frames,
        proposal_mode=args.proposal_mode,
        save_only_detections=not args.save_all_debug_images,
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
    print(f"Mission evaluation saved: {report_path}")
    print(f"HTML report saved: {report_path.with_name('mission_evaluation_report.html')}")


if __name__ == "__main__":
    main()
