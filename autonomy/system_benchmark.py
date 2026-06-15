from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from autonomy.multisensor_mission_demo import run_multisensor_demo


def run_system_benchmark(
    *,
    missions_root: str | Path = "benchmark_data/missions",
    output_dir: str | Path = "logs/system_benchmark_v1",
) -> Path:
    missions_path = Path(missions_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mission_files = sorted(missions_path.glob("mission_*/mission.json"))
    results = []
    for mission_file in mission_files:
        mission = json.loads(mission_file.read_text(encoding="utf-8"))
        mission_dir = mission_file.parent
        run_root = output_path / mission["mission_id"]
        report_path = run_multisensor_demo(
            mission_request=mission["mission_request"],
            rgb_images=resolve_mission_path(mission_dir, mission["rgb_images"]),
            ir_images=resolve_mission_path(mission_dir, mission["ir_images"]),
            acoustic=resolve_mission_path(mission_dir, mission["acoustic"]),
            output_dir=run_root,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        results.append(score_mission(mission, report, report_path))

    metrics = system_metrics(results)
    report = {
        "benchmark": "Aegis System Benchmark v1",
        "missions_root": str(missions_path),
        "mission_count": len(results),
        "metrics": metrics,
        "results": results,
        "interpretation": benchmark_interpretation(metrics),
    }
    report_path = output_path / "system_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_path / "system_benchmark_report.html").write_text(render_html(report), encoding="utf-8")
    return report_path


def resolve_mission_path(mission_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (mission_dir / path).resolve()


def score_mission(mission: dict, report: dict, report_path: Path) -> dict:
    summary = report.get("summary") or {}
    contacts = report.get("contacts") or []
    expected_contact = bool(mission.get("expected_contact"))
    expected_priority = str(mission.get("expected_priority") or "").lower()
    positive_contacts = [contact for contact in contacts if is_positive_contact(contact)]
    predicted_contact = bool(positive_contacts)
    predicted_high_priority = any(float(contact.get("priority") or 0.0) >= 0.75 for contact in positive_contacts)
    predicted_medium_priority = any(0.35 <= float(contact.get("priority") or 0.0) < 0.75 for contact in positive_contacts)
    candidate_modalities = {
        item.get("sensor_modality")
        for item in report.get("unified_candidates", [])
        if item.get("sensor_modality")
    }
    contact_modalities = contact_evidence_modalities(positive_contacts, report.get("unified_candidates", []))
    expected_evidence = set(mission.get("expected_evidence") or [])
    evidence_met = expected_evidence <= contact_modalities
    priority_met = (
        not expected_contact
        or expected_priority not in {"high", "medium", "low"}
        or (expected_priority == "high" and predicted_high_priority)
        or (expected_priority == "medium" and (predicted_high_priority or predicted_medium_priority))
        or (expected_priority == "low" and predicted_contact)
    )
    mission_success = expected_contact == predicted_contact and priority_met and evidence_met
    return {
        "mission_id": mission.get("mission_id"),
        "name": mission.get("name"),
        "report_path": str(report_path),
        "expected_contact": expected_contact,
        "predicted_contact": predicted_contact,
        "expected_priority": expected_priority,
        "predicted_high_priority": predicted_high_priority,
        "positive_contact_count": len(positive_contacts),
        "expected_evidence": sorted(expected_evidence),
        "observed_evidence": sorted(contact_modalities),
        "observed_candidate_modalities": sorted(candidate_modalities),
        "evidence_met": evidence_met,
        "mission_success": mission_success,
        "contact_count": summary.get("contact_count", 0),
        "candidate_count": summary.get("candidate_count", 0),
        "contacts": contacts,
    }


def is_positive_contact(contact: dict) -> bool:
    assessment = str(contact.get("assessment") or "").lower()
    if "false alarm" in assessment:
        return False
    return "vessel" in assessment or "contact" in assessment


def contact_evidence_modalities(contacts: list[dict], candidates: list[dict]) -> set[str]:
    candidate_modalities = {
        item.get("candidate_id"): item.get("sensor_modality")
        for item in candidates
        if item.get("candidate_id") and item.get("sensor_modality")
    }
    modalities = set()
    for contact in contacts:
        for candidate_id in contact.get("candidate_ids") or []:
            modality = candidate_modalities.get(candidate_id)
            if modality:
                modalities.add(modality)
    return modalities


def system_metrics(results: list[dict]) -> dict:
    tp = sum(1 for item in results if item["expected_contact"] and item["predicted_contact"])
    fp = sum(1 for item in results if not item["expected_contact"] and item["predicted_contact"])
    tn = sum(1 for item in results if not item["expected_contact"] and not item["predicted_contact"])
    fn = sum(1 for item in results if item["expected_contact"] and not item["predicted_contact"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    success_rate = sum(1 for item in results if item["mission_success"]) / len(results) if results else 0.0
    return {
        "mission_success_rate": round(success_rate, 4),
        "contact_precision": round(precision, 4),
        "contact_recall": round(recall, 4),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def benchmark_interpretation(metrics: dict) -> str:
    success = float(metrics.get("mission_success_rate") or 0.0)
    if success >= 0.9:
        return "The current mission workflow succeeds on the configured system benchmark cases."
    if success >= 0.6:
        return "The current mission workflow is partially successful and needs more mission cases before claiming robustness."
    return "The current mission workflow needs tuning before it can be treated as a reliable system benchmark."


def render_html(report: dict) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{esc(item.get('mission_id'))}</td>"
        f"<td>{esc(item.get('name'))}</td>"
        f"<td>{esc(item.get('mission_success'))}</td>"
        f"<td>{esc(item.get('contact_count'))}</td>"
        f"<td>{esc(', '.join(item.get('observed_evidence') or []))}</td>"
        f"<td>{esc(item.get('report_path'))}</td>"
        "</tr>"
        for item in report.get("results", [])
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Aegis System Benchmark v1</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    pre {{ background: #f3f4f6; padding: 16px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Aegis System Benchmark v1</h1>
  <p>{esc(report.get("interpretation"))}</p>
  <h2>Metrics</h2>
  <pre>{esc(json.dumps(report.get("metrics", {}), indent=2))}</pre>
  <h2>Missions</h2>
  <table>
    <thead><tr><th>Mission</th><th>Name</th><th>Success</th><th>Contacts</th><th>Evidence</th><th>Report</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aegis System Benchmark v1.")
    parser.add_argument("--missions-root", default="benchmark_data/missions")
    parser.add_argument("--output-dir", default="logs/system_benchmark_v1")
    args = parser.parse_args()
    report_path = run_system_benchmark(missions_root=args.missions_root, output_dir=args.output_dir)
    print(f"System benchmark report saved: {report_path}")
    print(f"HTML report saved: {report_path.with_name('system_benchmark_report.html')}")


if __name__ == "__main__":
    main()
