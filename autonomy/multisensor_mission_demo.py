from __future__ import annotations

import argparse
import html
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from autonomy.acoustic_intelligence import analyze_acoustic_evidence
from autonomy.mission_memory import build_mission_memory
from autonomy.vision_lab import collect_image_paths, run_vision_lab


def run_multisensor_demo(
    *,
    mission_request: str,
    rgb_images: str | Path,
    ir_images: str | Path,
    acoustic: str | Path,
    output_dir: str | Path = "logs/multisensor_missions",
    max_saved_candidates: int = 50,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    stages: list[dict] = []

    rgb_report_path = run_stage(
        stages,
        "rgb_vision",
        lambda: run_vision_lab(
            image_paths=collect_image_paths([str(rgb_images)]),
            mission_request=mission_request,
            output_dir=run_dir / "rgb",
            proposal_mode="vehicle",
            semantic_vision="local",
            max_saved_candidates=max_saved_candidates,
        ),
    )
    ir_report_path = run_stage(
        stages,
        "infrared_vision",
        lambda: run_vision_lab(
            image_paths=collect_image_paths([str(ir_images)]),
            mission_request=mission_request,
            output_dir=run_dir / "infrared",
            proposal_mode="vehicle",
            semantic_vision="local",
            max_saved_candidates=max_saved_candidates,
        ),
    )
    acoustic_report_path = run_stage(
        stages,
        "acoustic_evidence",
        lambda: analyze_acoustic_evidence(
            [acoustic],
            mission_request=mission_request,
            output_dir=run_dir / "acoustic",
        ),
    )

    reports = {
        "rgb": safe_load_json(rgb_report_path),
        "infrared": safe_load_json(ir_report_path),
        "acoustic": safe_load_json(acoustic_report_path),
    }
    unified_candidates = unified_candidate_list(reports)
    contacts = maritime_contact_summary(unified_candidates)
    payload = {
        "timestamp": stamp,
        "mission_request": mission_request,
        "source_paths": {
            "rgb_images": str(rgb_images),
            "ir_images": str(ir_images),
            "acoustic": str(acoustic),
        },
        "reports": {
            "rgb": str(rgb_report_path) if rgb_report_path else None,
            "infrared": str(ir_report_path) if ir_report_path else None,
            "acoustic": str(acoustic_report_path) if acoustic_report_path else None,
        },
        "stage_summary": stage_summary(stages),
        "stages": stages,
        "unified_candidates": unified_candidates,
        "contacts": contacts,
        "summary": {
            "candidate_count": len(unified_candidates),
            "contact_count": len(contacts),
            "high_priority_contacts": sum(1 for item in contacts if float(item.get("priority") or 0.0) >= 0.75),
            "multi_sensor_confirmation": any(item.get("multi_sensor_confirmation") for item in contacts),
            "rgb_candidates": sum(1 for item in unified_candidates if item["sensor_modality"] == "rgb"),
            "ir_candidates": sum(1 for item in unified_candidates if item["sensor_modality"] == "infrared"),
            "acoustic_candidates": sum(1 for item in unified_candidates if item["sensor_modality"] == "acoustic"),
        },
        "mission_memory_summary": build_mission_memory("."),
        "product_note": "Aegis Multi-Sensor Mission Demo v1 unifies RGB, infrared, and acoustic evidence into one reviewable mission record.",
    }
    report_path = run_dir / "multisensor_mission_report.json"
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (run_dir / "multisensor_mission_report.html").write_text(render_html(payload), encoding="utf-8")
    return report_path


def run_stage(stages: list[dict], name: str, fn):
    try:
        result = fn()
    except Exception as exc:
        stages.append({"name": name, "status": "error", "error": str(exc), "traceback": traceback.format_exc(limit=6)})
        return None
    stages.append({"name": name, "status": "ok", "report_path": str(result)})
    return result


def safe_load_json(path: str | Path | None) -> dict:
    if path is None:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def unified_candidate_list(reports: dict[str, dict]) -> list[dict]:
    candidates: list[dict] = []
    for modality in ("rgb", "infrared"):
        report = reports.get(modality) or {}
        for item in (report.get("summary") or {}).get("shortlist") or []:
            candidates.append(
                {
                    "candidate_id": item.get("candidate_id"),
                    "sensor_modality": item.get("sensor_modality") or modality,
                    "source_path": item.get("image_path"),
                    "artifact_path": item.get("debug_path") or item.get("crop_path") or item.get("image_path"),
                    "proposal_score": item.get("proposal_score"),
                    "semantic_score": item.get("semantic_score"),
                    "uncertainty_score": item.get("uncertainty_score"),
                    "review_priority": item.get("review_priority"),
                    "decision": item.get("decision"),
                    "reason": item.get("proposal_reason"),
                    "time_range": None,
                }
            )
    acoustic = reports.get("acoustic") or {}
    for item in acoustic.get("candidates") or []:
        candidates.append(
            {
                "candidate_id": item.get("candidate_id"),
                "sensor_modality": "acoustic",
                "source_path": item.get("audio_path"),
                "artifact_path": item.get("spectrogram_path"),
                "proposal_score": item.get("proposal_score"),
                "semantic_score": None,
                "uncertainty_score": item.get("uncertainty_score"),
                "review_priority": item.get("review_priority"),
                "decision": None,
                "reason": item.get("proposal_reason"),
                "time_range": {
                    "start_s": item.get("start_s"),
                    "end_s": item.get("end_s"),
                    "duration_s": item.get("duration_s"),
                },
            }
        )
    return sorted(candidates, key=lambda item: float(item.get("review_priority") or item.get("proposal_score") or 0.0), reverse=True)


def stage_summary(stages: list[dict]) -> dict:
    return {
        "ok": sum(1 for stage in stages if stage.get("status") == "ok"),
        "error": sum(1 for stage in stages if stage.get("status") == "error"),
    }


def maritime_contact_summary(candidates: list[dict]) -> list[dict]:
    confirming_candidates = [item for item in candidates if confirms_contact_evidence(item)]
    by_modality = {
        "rgb": [item for item in confirming_candidates if item.get("sensor_modality") == "rgb"],
        "infrared": [item for item in confirming_candidates if item.get("sensor_modality") == "infrared"],
        "acoustic": [item for item in confirming_candidates if item.get("sensor_modality") == "acoustic"],
    }
    strongest = {key: values[0] for key, values in by_modality.items() if values}
    contacts = []
    if len(strongest) >= 2:
        priority = min(0.99, 0.55 + 0.14 * len(strongest) + 0.12 * average_priority(strongest.values()))
        contacts.append(
            {
                "contact_id": "contact-1",
                "assessment": "possible vessel activity",
                "priority": round(priority, 3),
                "multi_sensor_confirmation": True,
                "evidence": contact_evidence(strongest),
                "candidate_ids": [item.get("candidate_id") for item in strongest.values()],
            }
        )
    elif strongest:
        item = next(iter(strongest.values()))
        modality = item.get("sensor_modality")
        assessment = "possible acoustic contact" if modality == "acoustic" else "possible single-sensor contact"
        evidence = contact_evidence({modality: item})
        if modality == "infrared":
            assessment = "possible false alarm"
            evidence = ["Thermal hotspot only"]
        priority = float(item.get("review_priority") or item.get("proposal_score") or 0.0)
        if modality == "acoustic":
            priority = min(priority, 0.62)
        contacts.append(
            {
                "contact_id": "contact-1",
                "assessment": assessment,
                "priority": round(priority, 3),
                "multi_sensor_confirmation": False,
                "evidence": evidence,
                "candidate_ids": [item.get("candidate_id")],
            }
        )
    thermal_only = [item for item in by_modality.get("infrared", [])[1:] if float(item.get("review_priority") or 0.0) < 0.65]
    if thermal_only:
        item = thermal_only[0]
        contacts.append(
            {
                "contact_id": f"contact-{len(contacts) + 1}",
                "assessment": "possible false alarm",
                "priority": round(float(item.get("review_priority") or item.get("proposal_score") or 0.0), 3),
                "multi_sensor_confirmation": False,
                "evidence": ["Thermal hotspot only"],
                "candidate_ids": [item.get("candidate_id")],
            }
        )
    return contacts


def confirms_contact_evidence(candidate: dict) -> bool:
    reason = str(candidate.get("reason") or "").lower()
    modality = candidate.get("sensor_modality")
    proposal_score = float(candidate.get("proposal_score") or 0.0)
    if "full-frame fallback" in reason:
        return False
    if modality == "acoustic":
        return proposal_score >= 0.45
    if modality in {"rgb", "infrared"}:
        return proposal_score >= 0.55
    return False


def average_priority(items) -> float:
    values = [float(item.get("review_priority") or item.get("proposal_score") or 0.0) for item in items]
    return sum(values) / len(values) if values else 0.0


def contact_evidence(strongest: dict) -> list[str]:
    evidence = []
    if strongest.get("rgb"):
        evidence.append("RGB vessel silhouette or visual proposal")
    if strongest.get("infrared"):
        evidence.append("Thermal hotspot")
    if strongest.get("acoustic"):
        evidence.append("Engine-like acoustic segment")
    return evidence


def render_html(report: dict) -> str:
    contact_rows = "\n".join(
        "<tr>"
        f"<td>{esc(item.get('contact_id'))}</td>"
        f"<td>{esc(item.get('assessment'))}</td>"
        f"<td>{esc(item.get('priority'))}</td>"
        f"<td>{esc(', '.join(item.get('evidence') or []))}</td>"
        f"<td>{esc(item.get('multi_sensor_confirmation'))}</td>"
        "</tr>"
        for item in report.get("contacts", [])
    )
    rows = "\n".join(
        "<tr>"
        f"<td>{esc(item.get('sensor_modality'))}</td>"
        f"<td>{esc(item.get('candidate_id'))}</td>"
        f"<td>{esc(item.get('review_priority'))}</td>"
        f"<td>{esc(item.get('proposal_score'))}</td>"
        f"<td>{esc(item.get('reason'))}</td>"
        f"<td>{esc(item.get('source_path'))}</td>"
        "</tr>"
        for item in report.get("unified_candidates", [])
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Aegis Multi-Sensor Mission Demo</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    .muted {{ color: #6b7280; }}
  </style>
</head>
<body>
  <h1>Aegis Multi-Sensor Mission Demo</h1>
  <p><strong>Mission:</strong> {esc(report.get("mission_request"))}</p>
  <p class="muted">{esc(report.get("product_note"))}</p>
  <h2>Summary</h2>
  <pre>{esc(json.dumps(report.get("summary", {}), indent=2))}</pre>
  <h2>Stage Health</h2>
  <pre>{esc(json.dumps(report.get("stage_summary", {}), indent=2))}</pre>
  <h2>Contact Summary</h2>
  <table>
    <thead><tr><th>Contact</th><th>Assessment</th><th>Priority</th><th>Evidence</th><th>Multi-sensor</th></tr></thead>
    <tbody>{contact_rows}</tbody>
  </table>
  <h2>Unified Candidate List</h2>
  <table>
    <thead><tr><th>Modality</th><th>Candidate</th><th>Priority</th><th>Proposal</th><th>Reason</th><th>Source</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aegis Multi-Sensor Mission Demo v1.")
    parser.add_argument("--mission-request", required=True)
    parser.add_argument("--rgb-images", required=True)
    parser.add_argument("--ir-images", required=True)
    parser.add_argument("--acoustic", required=True)
    parser.add_argument("--output-dir", default="logs/multisensor_missions")
    parser.add_argument("--max-saved-candidates", type=int, default=50)
    args = parser.parse_args()
    report_path = run_multisensor_demo(
        mission_request=args.mission_request,
        rgb_images=args.rgb_images,
        ir_images=args.ir_images,
        acoustic=args.acoustic,
        output_dir=args.output_dir,
        max_saved_candidates=args.max_saved_candidates,
    )
    print(f"Multi-sensor mission report saved: {report_path}")
    print(f"HTML report saved: {report_path.with_name('multisensor_mission_report.html')}")


if __name__ == "__main__":
    main()
