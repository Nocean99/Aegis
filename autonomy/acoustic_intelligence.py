from __future__ import annotations

import argparse
import csv
import html
import json
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


AUDIO_EXTENSIONS = {".wav"}


@dataclass(frozen=True)
class AcousticMetadata:
    audio_path: str
    sample_rate_hz: int
    channels: int
    sample_count: int
    duration_s: float
    peak_amplitude: float
    rms_amplitude: float


@dataclass(frozen=True)
class AcousticCandidate:
    candidate_id: str
    audio_path: str
    start_s: float
    end_s: float
    duration_s: float
    proposal_score: float
    uncertainty_score: float
    review_priority: float
    sensor_modality: str
    proposal_reason: str
    peak_amplitude: float
    rms_amplitude: float
    spectrogram_path: str | None = None


def analyze_acoustic_evidence(
    paths: list[str | Path],
    *,
    mission_request: str,
    output_dir: str | Path = "logs/acoustic_evaluations",
    frame_ms: int = 1000,
    hop_ms: int = 500,
    labels_csv: str | Path | None = None,
) -> Path:
    audio_paths = discover_audio_paths(paths)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    spectrogram_dir = run_dir / "spectrograms"
    spectrogram_dir.mkdir(exist_ok=True)

    metadata: list[dict] = []
    candidates: list[dict] = []
    errors: list[dict] = []
    for audio_path in audio_paths:
        try:
            samples, sample_rate, channels = read_wav_mono(audio_path)
            item = acoustic_metadata(audio_path, samples, sample_rate, channels)
            metadata.append(asdict(item))
            spectrogram_path = spectrogram_dir / f"{audio_path.stem}_spectrogram.png"
            write_spectrogram_png(samples, sample_rate, spectrogram_path)
            metadata[-1]["spectrogram_path"] = str(spectrogram_path)
            audio_candidates = propose_acoustic_segments(
                audio_path=audio_path,
                samples=samples,
                sample_rate=sample_rate,
                frame_ms=frame_ms,
                hop_ms=hop_ms,
                spectrogram_path=spectrogram_path,
                target_profile=target_profile_for_mission(mission_request),
            )
            candidates.extend(asdict(candidate) for candidate in audio_candidates)
        except Exception as exc:  # Keep acoustic reports resilient.
            errors.append({"audio_path": str(audio_path), "error": str(exc)})

    labels = load_acoustic_labels(labels_csv) if labels_csv else {}
    evaluation = evaluate_acoustic_candidates(metadata=metadata, candidates=candidates, labels=labels)
    report = {
        "mission_request": mission_request,
        "source_paths": [str(path) for path in paths],
        "sensor_modality": "acoustic",
        "metadata": metadata,
        "candidates": candidates,
        "evaluation": evaluation,
        "summary": {
            "audio_files": len(audio_paths),
            "processed": len(metadata),
            "errors": len(errors),
            "candidate_count": len(candidates),
            "spectrogram_count": sum(1 for item in metadata if item.get("spectrogram_path")),
            "labeled_count": evaluation.get("labeled_count"),
        },
        "errors": errors,
        "next_step": "Use analyst review and labeled acoustic data before training a classifier.",
    }
    report_path = run_dir / "acoustic_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "acoustic_report.html").write_text(render_acoustic_html(report), encoding="utf-8")
    return report_path


def discover_audio_paths(paths: list[str | Path]) -> list[Path]:
    discovered: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            discovered.extend(sorted(item for item in path.rglob("*") if item.suffix.lower() in AUDIO_EXTENSIONS))
        elif path.suffix.lower() in AUDIO_EXTENSIONS:
            discovered.append(path)
    return discovered


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int, int]:
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate, channels


def acoustic_metadata(path: Path, samples: np.ndarray, sample_rate: int, channels: int) -> AcousticMetadata:
    duration = len(samples) / float(sample_rate) if sample_rate else 0.0
    return AcousticMetadata(
        audio_path=str(path),
        sample_rate_hz=sample_rate,
        channels=channels,
        sample_count=int(len(samples)),
        duration_s=round(duration, 3),
        peak_amplitude=round(float(np.max(np.abs(samples))) if samples.size else 0.0, 4),
        rms_amplitude=round(float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0, 4),
    )


def write_spectrogram_png(samples: np.ndarray, sample_rate: int, output_path: str | Path) -> Path:
    spec = compute_spectrogram(samples, sample_rate)
    image = spectrogram_to_image(spec)
    output = Path(output_path)
    ok = cv2.imwrite(str(output), image)
    if not ok:
        raise ValueError(f"Could not write spectrogram: {output}")
    return output


def compute_spectrogram(samples: np.ndarray, sample_rate: int, *, window_ms: int = 40, hop_ms: int = 20) -> np.ndarray:
    if samples.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    window_size = max(64, int(sample_rate * window_ms / 1000))
    hop_size = max(1, int(sample_rate * hop_ms / 1000))
    if len(samples) < window_size:
        samples = np.pad(samples, (0, window_size - len(samples)))
    window = np.hanning(window_size).astype(np.float32)
    columns = []
    for start in range(0, len(samples) - window_size + 1, hop_size):
        frame = samples[start : start + window_size] * window
        spectrum = np.abs(np.fft.rfft(frame))
        columns.append(spectrum)
    if not columns:
        columns = [np.abs(np.fft.rfft(samples[:window_size] * window))]
    spec = np.stack(columns, axis=1)
    return np.log1p(spec).astype(np.float32)


def spectrogram_to_image(spec: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(spec, None, 0, 255, cv2.NORM_MINMAX)
    image = normalized.astype(np.uint8)
    image = np.flipud(image)
    image = cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)
    return image


def propose_acoustic_segments(
    *,
    audio_path: Path,
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: int = 1000,
    hop_ms: int = 500,
    spectrogram_path: str | Path | None = None,
    target_profile: str = "activity",
) -> list[AcousticCandidate]:
    if samples.size == 0 or sample_rate <= 0:
        return []
    profile = acoustic_spectral_profile(samples, sample_rate)
    anthropogenic_mode = target_profile == "anthropogenic"
    if anthropogenic_mode and not is_anthropogenic_like(profile):
        return []
    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    hop_size = max(1, int(sample_rate * hop_ms / 1000))
    if len(samples) < frame_size:
        padded = np.pad(samples, (0, frame_size - len(samples)))
    else:
        padded = samples
    frame_stats = []
    for start in range(0, len(padded) - frame_size + 1, hop_size):
        frame = padded[start : start + frame_size]
        rms = float(np.sqrt(np.mean(np.square(frame))))
        peak = float(np.max(np.abs(frame)))
        frame_stats.append((start, rms, peak))
    if not frame_stats:
        return []
    rms_values = np.array([item[1] for item in frame_stats], dtype=np.float32)
    if anthropogenic_mode:
        threshold = max(float(np.percentile(rms_values, 70)), float(np.mean(rms_values) + np.std(rms_values) * 0.25), 0.004)
    else:
        threshold = max(float(np.percentile(rms_values, 75)), float(np.mean(rms_values) + np.std(rms_values) * 0.5), 0.03)
    candidates = []
    for index, (start, rms, peak) in enumerate(frame_stats, start=1):
        peak_floor = 0.04 if anthropogenic_mode else 0.35
        if rms < threshold and peak < peak_floor:
            continue
        start_s = start / float(sample_rate)
        end_s = min(len(samples), start + frame_size) / float(sample_rate)
        score = min(1.0, max(rms / max(threshold, 1e-6), peak) * 0.7)
        uncertainty = 1.0 - min(0.85, abs(rms - threshold) / max(threshold, 1e-6))
        review_priority = min(1.0, score * 0.72 + uncertainty * 0.28)
        candidates.append(
            AcousticCandidate(
                candidate_id=f"{audio_path.stem}_acoustic_{index:04d}",
                audio_path=str(audio_path),
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                duration_s=round(max(0.0, end_s - start_s), 3),
                proposal_score=round(float(score), 3),
                uncertainty_score=round(float(max(0.0, min(1.0, uncertainty))), 3),
                review_priority=round(float(review_priority), 3),
                sensor_modality="acoustic",
                proposal_reason="broadband anthropogenic acoustic segment" if anthropogenic_mode else "high-energy acoustic segment",
                peak_amplitude=round(float(peak), 4),
                rms_amplitude=round(float(rms), 4),
                spectrogram_path=str(spectrogram_path) if spectrogram_path else None,
            )
        )
    if anthropogenic_mode and not candidates:
        rms = float(profile.get("rms") or 0.0)
        peak = float(profile.get("peak") or 0.0)
        score = min(1.0, 0.45 + float(profile.get("entropy") or 0.0) * 0.35 + float(profile.get("flatness") or 0.0) * 0.2)
        uncertainty = max(0.1, 1.0 - score)
        candidates.append(
            AcousticCandidate(
                candidate_id=f"{audio_path.stem}_acoustic_profile_0001",
                audio_path=str(audio_path),
                start_s=0.0,
                end_s=round(len(samples) / float(sample_rate), 3),
                duration_s=round(len(samples) / float(sample_rate), 3),
                proposal_score=round(float(score), 3),
                uncertainty_score=round(float(uncertainty), 3),
                review_priority=round(float(min(1.0, score * 0.74 + uncertainty * 0.26)), 3),
                sensor_modality="acoustic",
                proposal_reason="broadband anthropogenic acoustic profile",
                peak_amplitude=round(float(peak), 4),
                rms_amplitude=round(float(rms), 4),
                spectrogram_path=str(spectrogram_path) if spectrogram_path else None,
            )
        )
    return candidates


def target_profile_for_mission(mission_request: str) -> str:
    text = mission_request.lower()
    anthropogenic_terms = (
        "anthropogenic",
        "vessel",
        "ship",
        "boat",
        "engine",
        "machinery",
        "human-generated",
        "human generated",
    )
    if any(term in text for term in anthropogenic_terms):
        return "anthropogenic"
    return "activity"


def acoustic_spectral_profile(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    if samples.size == 0 or sample_rate <= 0:
        return {}
    windowed = samples[: min(len(samples), sample_rate * 5)]
    if len(windowed) < 64:
        return {
            "rms": float(np.sqrt(np.mean(np.square(windowed)))) if windowed.size else 0.0,
            "peak": float(np.max(np.abs(windowed))) if windowed.size else 0.0,
        }
    spectrum = np.abs(np.fft.rfft(windowed * np.hanning(len(windowed)))) + 1e-12
    frequencies = np.fft.rfftfreq(len(windowed), 1 / sample_rate)
    total = float(np.sum(spectrum))
    probabilities = spectrum / total
    return {
        "rms": float(np.sqrt(np.mean(np.square(windowed)))),
        "peak": float(np.max(np.abs(windowed))),
        "centroid": float(np.sum(frequencies * spectrum) / total),
        "low_ratio": float(np.sum(spectrum[frequencies < 300]) / total),
        "mid_ratio": float(np.sum(spectrum[(frequencies >= 300) & (frequencies < 2000)]) / total),
        "high_ratio": float(np.sum(spectrum[frequencies >= 2000]) / total),
        "entropy": float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(probabilities))),
        "flatness": float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum)),
        "zero_crossing_rate": float(np.mean(np.abs(np.diff(np.signbit(windowed))))),
    }


def is_anthropogenic_like(profile: dict[str, float]) -> bool:
    entropy = float(profile.get("entropy") or 0.0)
    flatness = float(profile.get("flatness") or 0.0)
    centroid = float(profile.get("centroid") or 0.0)
    zcr = float(profile.get("zero_crossing_rate") or 0.0)
    rms = float(profile.get("rms") or 0.0)
    peak = float(profile.get("peak") or 0.0)
    low_ratio = float(profile.get("low_ratio") or 0.0)
    high_ratio = float(profile.get("high_ratio") or 0.0)
    energy_in_vessel_bands = high_ratio >= 0.45 or low_ratio >= 0.25
    broad_continuous = (
        rms >= 0.006
        and peak >= 0.035
        and entropy >= 0.94
        and flatness >= 0.45
        and centroid >= 1500
        and zcr < 0.45
    )
    energetic_broadband = (
        rms >= 0.03
        and entropy >= 0.88
        and centroid >= 1400
        and zcr < 0.45
        and energy_in_vessel_bands
    )
    strong_machinery_event = (
        rms >= 0.08
        and flatness >= 0.32
        and centroid >= 1200
        and zcr < 0.45
        and entropy >= 0.84
        and energy_in_vessel_bands
    )
    steady_low_frequency_vessel = (
        rms >= 0.024
        and peak >= 0.12
        and 1450 <= centroid <= 2200
        and low_ratio >= 0.34
        and zcr < 0.07
        and entropy >= 0.86
    )
    tonal_antifouling = (
        rms >= 0.03
        and peak >= 0.15
        and 1700 <= centroid <= 2200
        and high_ratio >= 0.55
        and flatness <= 0.05
        and 0.82 <= entropy <= 0.89
        and 0.15 <= zcr <= 0.25
    )
    return (
        broad_continuous
        or energetic_broadband
        or strong_machinery_event
        or steady_low_frequency_vessel
        or tonal_antifouling
    )


def load_acoustic_labels(labels_csv: str | Path | None) -> dict[str, dict]:
    if not labels_csv:
        return {}
    path = Path(labels_csv)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"audio_path", "expected_match", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Acoustic labels CSV missing columns: {', '.join(sorted(missing))}")
        labels = {}
        for row in reader:
            key = Path(row["audio_path"]).name
            labels[key] = {
                "expected_match": str(row["expected_match"]).strip().lower() == "true",
                "label": row.get("label") or "",
                "false_positive_cause": row.get("false_positive_cause") or "",
                "uncertainty_cause": row.get("uncertainty_cause") or "",
            }
        return labels


def evaluate_acoustic_candidates(*, metadata: list[dict], candidates: list[dict], labels: dict[str, dict]) -> dict:
    if not labels:
        return {}
    candidate_counts: dict[str, int] = {}
    for candidate in candidates:
        key = Path(candidate.get("audio_path", "")).name
        candidate_counts[key] = candidate_counts.get(key, 0) + 1
    rows = []
    for item in metadata:
        key = Path(item.get("audio_path", "")).name
        if key not in labels:
            continue
        label = labels[key]
        predicted = candidate_counts.get(key, 0) > 0
        rows.append(
            {
                "audio_path": item.get("audio_path"),
                "expected_match": label["expected_match"],
                "predicted_match": predicted,
                "proposal_count": candidate_counts.get(key, 0),
                "label": label.get("label"),
                "false_positive_cause": label.get("false_positive_cause"),
                "uncertainty_cause": label.get("uncertainty_cause"),
            }
        )
    tp = sum(1 for row in rows if row["expected_match"] and row["predicted_match"])
    fp = sum(1 for row in rows if not row["expected_match"] and row["predicted_match"])
    tn = sum(1 for row in rows if not row["expected_match"] and not row["predicted_match"])
    fn = sum(1 for row in rows if row["expected_match"] and not row["predicted_match"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    false_positive_causes = {}
    uncertainty_causes = {}
    for row in rows:
        if not row["expected_match"] and row["predicted_match"] and row.get("false_positive_cause"):
            cause = str(row["false_positive_cause"])
            false_positive_causes[cause] = false_positive_causes.get(cause, 0) + 1
        if row["predicted_match"] and row.get("uncertainty_cause"):
            cause = str(row["uncertainty_cause"])
            uncertainty_causes[cause] = uncertainty_causes.get(cause, 0) + 1
    return {
        "labeled_count": len(rows),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "capture_precision": round(precision, 4),
        "capture_recall": round(recall, 4),
        "capture_f1": round(f1, 4),
        "proposal_count": len(candidates),
        "false_positive_causes": false_positive_causes,
        "uncertainty_causes": uncertainty_causes,
        "items": rows,
    }


def render_acoustic_html(report: dict) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{esc(item.get('candidate_id'))}</td>"
        f"<td>{esc(item.get('start_s'))}-{esc(item.get('end_s'))}</td>"
        f"<td>{esc(item.get('proposal_score'))}</td>"
        f"<td>{esc(item.get('uncertainty_score'))}</td>"
        f"<td>{esc(item.get('proposal_reason'))}</td>"
        "</tr>"
        for item in report.get("candidates", [])
    )
    spectrograms = "\n".join(
        f"<li>{esc(Path(item.get('audio_path', '')).name)}: {esc(item.get('spectrogram_path'))}</li>"
        for item in report.get("metadata", [])
        if item.get("spectrogram_path")
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Aegis Acoustic Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .muted {{ color: #6b7280; }}
  </style>
</head>
<body>
  <h1>Aegis Acoustic Report</h1>
  <p><strong>Mission:</strong> {esc(report.get("mission_request"))}</p>
  <p class="muted">This is a Phase 1/2 acoustic evidence report, not a trained classifier.</p>
  <h2>Summary</h2>
  <pre>{esc(json.dumps(report.get("summary", {}), indent=2))}</pre>
  <h2>Evaluation</h2>
  <pre>{esc(json.dumps(report.get("evaluation", {}), indent=2))}</pre>
  <h2>Spectrograms</h2>
  <ul>{spectrograms}</ul>
  <h2>Candidate Segments</h2>
  <table>
    <thead><tr><th>Candidate</th><th>Time</th><th>Proposal</th><th>Uncertainty</th><th>Reason</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate acoustic evidence metadata, spectrograms, candidates, and a simple report.")
    parser.add_argument("paths", nargs="+", help="WAV files or folders containing WAV files")
    parser.add_argument("--mission-request", required=True)
    parser.add_argument("--output-dir", default="logs/acoustic_evaluations/latest")
    parser.add_argument("--frame-ms", type=int, default=1000)
    parser.add_argument("--hop-ms", type=int, default=500)
    parser.add_argument("--labels-csv", default=None)
    args = parser.parse_args()
    report_path = analyze_acoustic_evidence(
        args.paths,
        mission_request=args.mission_request,
        output_dir=args.output_dir,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        labels_csv=args.labels_csv,
    )
    print(f"Acoustic report saved: {report_path}")
    print(f"HTML report saved: {report_path.with_name('acoustic_report.html')}")


if __name__ == "__main__":
    main()
