from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2

from autonomy.semantic_vision import extract_response_text, image_to_data_url


DEFAULT_TRIAL_COUNT = 5


@dataclass(frozen=True)
class CascadeTrial:
    target_id: str
    target_image_path: str
    target_description: str


@dataclass(frozen=True)
class CascadeClaim:
    claimed_find: bool
    confidence: float
    explanation: str


class CascadeReviewer(Protocol):
    model_name: str

    def review(self, *, target_description: str, image_path: str, candidate_id: str, rank: int) -> CascadeClaim:
        ...


class OpenAICascadeReviewer:
    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        detail: str = "auto",
        timeout_s: float = 45.0,
        max_retries: int = 3,
    ) -> None:
        self.model_name = model or os.environ.get("OPENAI_VISION_MODEL", "")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.detail = detail
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)
        if self.detail not in {"auto", "low", "high"}:
            raise ValueError("OpenAI image detail must be one of: auto, low, high.")
        if not self.model_name:
            raise ValueError("OpenAI cascade reviewer requires --openai-model or OPENAI_VISION_MODEL.")
        if not self.api_key:
            raise ValueError("OpenAI cascade reviewer requires OPENAI_API_KEY.")

    def review(self, *, target_description: str, image_path: str, candidate_id: str, rank: int) -> CascadeClaim:
        frame = cv2.imread(str(image_path))
        if frame is None:
            return CascadeClaim(False, 0.0, f"Could not read candidate image: {image_path}")
        image_content = {"type": "input_image", "image_url": image_to_data_url(frame)}
        if self.detail != "auto":
            image_content["detail"] = self.detail
        prompt = cascade_prompt(target_description=target_description, candidate_id=candidate_id, rank=rank)
        body = {
            "model": self.model_name,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        image_content,
                    ],
                }
            ],
        }
        req = urlrequest.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        payload = self._open_with_retries(req)
        return parse_cascade_claim(extract_response_text(payload))

    def _open_with_retries(self, req: urlrequest.Request) -> dict:
        for attempt in range(self.max_retries + 1):
            try:
                with urlrequest.urlopen(req, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urlerror.HTTPError as exc:
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise
                time.sleep(min(2.0 * (2 ** attempt), 12.0))
            except urlerror.URLError:
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(2.0 * (2 ** attempt), 12.0))
        raise RuntimeError("OpenAI request retry loop exited unexpectedly.")


def cascade_prompt(*, target_description: str, candidate_id: str, rank: int) -> str:
    return (
        "You are the semantic confirmation stage in a cascade ranking benchmark. "
        "The local layer has already ranked all images. You are walking that ranked list in order. "
        "Your task is only to decide whether THIS candidate image matches the planted target description. "
        "Return ONLY JSON with keys: claimed_find boolean, confidence number 0.0-1.0, explanation string. "
        "Set claimed_find=true only when the target is clearly present. "
        "If the image merely contains a generic vehicle but does not match the target description, return false. "
        "Do not use rank as evidence; rank is provided only for audit logging. "
        f"Candidate id: {candidate_id}. Rank: {rank}. "
        f"Target description: {target_description}"
    )


def parse_cascade_claim(text: str) -> CascadeClaim:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return CascadeClaim(False, 0.0, cleaned[:500] or "Model response was not valid JSON.")
    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    return CascadeClaim(
        claimed_find=bool(data.get("claimed_find", False)),
        confidence=round(confidence, 3),
        explanation=str(data.get("explanation", ""))[:800],
    )


def load_ranked_candidates(report_path: str | Path) -> list[dict]:
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report.get("results", [])
    readable = [result for result in results if not result.get("error") and result.get("image_path")]
    return sorted(
        readable,
        key=lambda result: (
            float(result.get("review_priority", 0.0)),
            float(result.get("final_score", 0.0)),
            str(result.get("candidate_id", "")),
        ),
        reverse=True,
    )


def default_trials_from_report(report_path: str | Path, *, limit: int = DEFAULT_TRIAL_COUNT) -> list[CascadeTrial]:
    ranked = load_ranked_candidates(report_path)
    positives = [item for item in ranked if bool((item.get("label") or {}).get("expected_match"))]
    descriptions = [
        "a clearly visible road vehicle in an overhead aerial traffic scene",
        "a parked vehicle or small cluster of parked vehicles in an aerial lot scene",
        "a vehicle visible in a dark or night-like aerial scene",
        "a small vehicle on a road or lane viewed from above",
        "a vehicle in a dense parking-lot style aerial image",
    ]
    trials: list[CascadeTrial] = []
    for index, item in enumerate(positives[:limit]):
        trials.append(
            CascadeTrial(
                target_id=str(item.get("candidate_id") or Path(item["image_path"]).stem),
                target_image_path=str(item["image_path"]),
                target_description=descriptions[index % len(descriptions)],
            )
        )
    return trials


def load_trials(path: str | Path | None, report_path: str | Path, *, limit: int = DEFAULT_TRIAL_COUNT) -> list[CascadeTrial]:
    if path is None:
        return default_trials_from_report(report_path, limit=limit)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw.get("trials", raw if isinstance(raw, list) else [])
    trials = [
        CascadeTrial(
            target_id=str(item["target_id"]),
            target_image_path=str(item["target_image_path"]),
            target_description=str(item["target_description"]),
        )
        for item in items
    ]
    return trials[:limit]


def run_cascade_benchmark(
    *,
    report_path: str | Path,
    trials: list[CascadeTrial],
    reviewer: CascadeReviewer,
    output_dir: str | Path,
) -> Path:
    ranked = load_ranked_candidates(report_path)
    if not ranked:
        raise ValueError("No ranked candidates found in report.")
    ranked_by_path = {normalize_path(item["image_path"]): index for index, item in enumerate(ranked, start=1)}
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    trial_results = []
    for trial in trials:
        target_key = normalize_path(trial.target_image_path)
        true_target_rank = ranked_by_path.get(target_key)
        checks = []
        claimed = None
        api_errors = []
        for rank, candidate in enumerate(ranked, start=1):
            candidate_path = str(candidate["image_path"])
            try:
                claim = reviewer.review(
                    target_description=trial.target_description,
                    image_path=candidate_path,
                    candidate_id=str(candidate.get("candidate_id", "")),
                    rank=rank,
                )
            except Exception as exc:
                api_errors.append({"rank": rank, "candidate_id": candidate.get("candidate_id"), "error": f"{type(exc).__name__}: {exc}"})
                claim = CascadeClaim(False, 0.0, f"Reviewer error: {type(exc).__name__}: {exc}")
            check = {
                "rank": rank,
                "candidate_id": candidate.get("candidate_id"),
                "image_path": candidate_path,
                "review_priority": candidate.get("review_priority"),
                "expected_match": bool((candidate.get("label") or {}).get("expected_match")),
                "claimed_find": claim.claimed_find,
                "confidence": claim.confidence,
                "explanation": claim.explanation,
            }
            checks.append(check)
            if claim.claimed_find:
                claimed = check
                break
        claimed_rank = None if claimed is None else int(claimed["rank"])
        claimed_path = None if claimed is None else str(claimed["image_path"])
        claim_correct = claimed_path is not None and normalize_path(claimed_path) == target_key
        trial_results.append(
            {
                "target_id": trial.target_id,
                "target_image_path": trial.target_image_path,
                "target_description": trial.target_description,
                "true_target_rank": true_target_rank,
                "claimed_rank": claimed_rank,
                "claimed_candidate_id": None if claimed is None else claimed.get("candidate_id"),
                "claimed_image_path": claimed_path,
                "claim_correct": claim_correct,
                "found_true_target": claim_correct,
                "stopped_on_false_positive": claimed is not None and not claim_correct,
                "api_calls_made": len(checks),
                "api_errors": api_errors,
                "checked": checks,
            }
        )

    report = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_type": "cascade_ranking_efficiency_test",
        "note": (
            "This is a ranking-quality and efficiency test. Stopping at first claimed find is a test construct, "
            "not normal mission behavior; real missions keep a review queue and preserve uncertain evidence."
        ),
        "source_report": str(report_path),
        "reviewer": getattr(reviewer, "model_name", "unknown"),
        "candidate_count": len(ranked),
        "trial_count": len(trial_results),
        "summary": summarize_trials(trial_results),
        "ranked_order": [
            {
                "rank": index,
                "candidate_id": item.get("candidate_id"),
                "image_path": item.get("image_path"),
                "review_priority": item.get("review_priority"),
                "expected_match": bool((item.get("label") or {}).get("expected_match")),
            }
            for index, item in enumerate(ranked, start=1)
        ],
        "trials": trial_results,
    }
    json_path = run_dir / "cascade_ranking_report.json"
    html_path = run_dir / "cascade_ranking_report.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return json_path


def summarize_trials(trials: list[dict]) -> dict:
    correct = sum(1 for trial in trials if trial["claim_correct"])
    false_stops = sum(1 for trial in trials if trial["stopped_on_false_positive"])
    misses = sum(1 for trial in trials if trial["claimed_rank"] is None)
    api_calls = [int(trial["api_calls_made"]) for trial in trials]
    true_ranks = [int(trial["true_target_rank"]) for trial in trials if trial["true_target_rank"] is not None]
    top5 = sum(1 for trial in trials if trial["claim_correct"] and (trial["claimed_rank"] or 9999) <= 5)
    return {
        "correct_finds": correct,
        "false_positive_stops": false_stops,
        "misses_no_claim": misses,
        "correct_find_rate": round(correct / len(trials), 4) if trials else 0.0,
        "correct_top5_count": top5,
        "mean_api_calls": round(statistics.mean(api_calls), 2) if api_calls else 0.0,
        "median_api_calls": round(statistics.median(api_calls), 2) if api_calls else 0.0,
        "mean_true_target_rank": round(statistics.mean(true_ranks), 2) if true_ranks else None,
        "median_true_target_rank": round(statistics.median(true_ranks), 2) if true_ranks else None,
    }


def normalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def render_html(report: dict) -> str:
    rows = []
    for trial in report["trials"]:
        status = "correct" if trial["claim_correct"] else ("false stop" if trial["stopped_on_false_positive"] else "miss")
        rows.append(
            "<tr>"
            f"<td>{escape(trial['target_id'])}</td>"
            f"<td>{escape(status)}</td>"
            f"<td>{trial['claimed_rank'] if trial['claimed_rank'] is not None else 'none'}</td>"
            f"<td>{trial['true_target_rank'] if trial['true_target_rank'] is not None else 'not ranked'}</td>"
            f"<td>{trial['api_calls_made']}</td>"
            f"<td>{escape(trial['target_description'])}</td>"
            "</tr>"
        )
    summary = report["summary"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Aegis Cascade Ranking Efficiency Test</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#0c1418; color:#edf5f7; margin:32px; }}
    .panel {{ border:1px solid #29414a; border-radius:8px; padding:20px; background:#142025; margin-bottom:18px; }}
    table {{ border-collapse:collapse; width:100%; }}
    th, td {{ border-bottom:1px solid #29414a; padding:10px; text-align:left; vertical-align:top; }}
    th {{ color:#9db4bc; }}
    .note {{ color:#b8c8ce; }}
  </style>
</head>
<body>
  <h1>Aegis Cascade Ranking Efficiency Test</h1>
  <div class="panel">
    <p class="note">{escape(report['note'])}</p>
    <p><strong>Reviewer:</strong> {escape(report['reviewer'])}</p>
    <p><strong>Candidates:</strong> {report['candidate_count']} | <strong>Trials:</strong> {report['trial_count']}</p>
  </div>
  <div class="panel">
    <h2>Summary</h2>
    <p><strong>Correct finds:</strong> {summary['correct_finds']} | <strong>False stops:</strong> {summary['false_positive_stops']} | <strong>Misses:</strong> {summary['misses_no_claim']}</p>
    <p><strong>Correct find rate:</strong> {summary['correct_find_rate']:.1%} | <strong>Mean API calls:</strong> {summary['mean_api_calls']} | <strong>Median true target rank:</strong> {summary['median_true_target_rank']}</p>
  </div>
  <div class="panel">
    <h2>Per-Trial Results</h2>
    <table>
      <thead><tr><th>Target</th><th>Status</th><th>Claimed Rank</th><th>True Rank</th><th>API Calls</th><th>Target Description</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</body>
</html>
"""


def escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aegis cascade ranking/efficiency benchmark")
    parser.add_argument("--report", required=True, help="Existing vision_report.json with ranked candidates")
    parser.add_argument("--trials-json", default=None, help="Optional JSON manifest with planted target trials")
    parser.add_argument("--trial-count", type=int, default=DEFAULT_TRIAL_COUNT)
    parser.add_argument("--output-dir", default="logs/cascade_ranking_tests/latest")
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--openai-detail", choices=["auto", "low", "high"], default=os.environ.get("OPENAI_IMAGE_DETAIL", "auto"))
    parser.add_argument("--openai-timeout-s", type=float, default=45.0)
    args = parser.parse_args()

    trials = load_trials(args.trials_json, args.report, limit=args.trial_count)
    if not trials:
        raise SystemExit("No cascade trials available. Provide --trials-json or a report with positive labels.")
    reviewer = OpenAICascadeReviewer(
        model=args.openai_model,
        detail=args.openai_detail,
        timeout_s=args.openai_timeout_s,
    )
    path = run_cascade_benchmark(
        report_path=args.report,
        trials=trials,
        reviewer=reviewer,
        output_dir=args.output_dir,
    )
    print(f"Cascade ranking report saved: {path}")
    print(f"HTML report saved: {path.with_suffix('.html')}")


if __name__ == "__main__":
    main()
