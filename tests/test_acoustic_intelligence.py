from __future__ import annotations

import json
import math
import struct
import sys
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.acoustic_intelligence import analyze_acoustic_evidence
from autonomy.acoustic_intelligence import is_anthropogenic_like
from autonomy.acoustic_intelligence import propose_acoustic_segments
from autonomy.acoustic_intelligence import read_wav_mono


def write_test_wav(path: Path, sample_rate: int = 8000) -> None:
    samples = []
    for index in range(sample_rate * 2):
        t = index / sample_rate
        amplitude = 0.08
        if 0.75 <= t <= 1.1:
            amplitude = 0.75
        signal = (
            math.sin(2 * math.pi * 140 * t) * 0.45
            + math.sin(2 * math.pi * 310 * t) * 0.25
            + math.sin(2 * math.pi * 760 * t) * 0.15
            + math.sin(2 * math.pi * 1530 * t) * 0.10
            + math.sin(2 * math.pi * 2470 * t) * 0.05
        )
        if 0.75 <= t <= 1.1:
            noise = ((((index * 1103515245 + 12345) >> 16) & 0x7FFF) / 16384.0) - 1.0
            signal = signal * 0.55 + noise * 0.45
        value = int(max(-1.0, min(1.0, amplitude * signal)) * 32767)
        samples.append(value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def test_acoustic_pipeline_writes_metadata_spectrogram_candidates_and_report() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        wav_path = root / "test_signal.wav"
        write_test_wav(wav_path)
        report_path = analyze_acoustic_evidence(
            [wav_path],
            mission_request="Listen for engine or distress sounds near the search area",
            output_dir=root / "report",
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["sensor_modality"] == "acoustic"
        assert report["summary"]["processed"] == 1
        assert report["summary"]["spectrogram_count"] == 1
        assert report["summary"]["candidate_count"] >= 1
        assert report["evaluation"] == {}
        assert Path(report["metadata"][0]["spectrogram_path"]).exists()
        assert (report_path.parent / "acoustic_report.html").exists()
        candidate = report["candidates"][0]
        assert candidate["sensor_modality"] == "acoustic"
        assert candidate["proposal_reason"] in {
            "high-energy acoustic segment",
            "broadband anthropogenic acoustic segment",
            "broadband anthropogenic acoustic profile",
        }
        assert set(candidate) >= {
            "candidate_id",
            "audio_path",
            "start_s",
            "end_s",
            "proposal_score",
            "uncertainty_score",
            "review_priority",
            "spectrogram_path",
        }


def test_acoustic_segment_proposals_find_high_energy_region() -> None:
    with TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "burst.wav"
        write_test_wav(wav_path)
        samples, sample_rate, _channels = read_wav_mono(wav_path)
        candidates = propose_acoustic_segments(
            audio_path=wav_path,
            samples=samples,
            sample_rate=sample_rate,
            frame_ms=500,
            hop_ms=250,
        )
        assert candidates
        assert any(candidate.start_s <= 1.0 <= candidate.end_s for candidate in candidates)


def test_acoustic_labeled_benchmark_reports_capture_metrics() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        positive = root / "vessel.wav"
        negative = root / "ambient.wav"
        write_test_wav(positive)
        write_quiet_wav(negative)
        labels = root / "labels.csv"
        labels.write_text(
            "\n".join(
                [
                    "audio_path,expected_match,label,false_positive_cause,uncertainty_cause",
                    "vessel.wav,true,positive,,",
                    "ambient.wav,false,negative,wave_noise,low_snr",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        report_path = analyze_acoustic_evidence(
            [root],
            mission_request="Listen for vessel activity",
            output_dir=root / "benchmark",
            labels_csv=labels,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        evaluation = report["evaluation"]
        assert evaluation["labeled_count"] == 2
        assert set(evaluation) >= {
            "capture_precision",
            "capture_recall",
            "capture_f1",
            "proposal_count",
            "false_positive_causes",
        }


def test_anthropogenic_gate_rejects_quiet_animal_like_broadband() -> None:
    assert not is_anthropogenic_like(
        {
            "rms": 0.001,
            "peak": 0.012,
            "centroid": 3900,
            "low_ratio": 0.05,
            "mid_ratio": 0.22,
            "high_ratio": 0.73,
            "entropy": 0.97,
            "flatness": 0.72,
            "zero_crossing_rate": 0.33,
        }
    )


def test_anthropogenic_gate_accepts_steady_vessel_and_antifouling_profiles() -> None:
    assert is_anthropogenic_like(
        {
            "rms": 0.028,
            "peak": 0.16,
            "centroid": 1710,
            "low_ratio": 0.40,
            "mid_ratio": 0.29,
            "high_ratio": 0.31,
            "entropy": 0.88,
            "flatness": 0.42,
            "zero_crossing_rate": 0.04,
        }
    )
    assert is_anthropogenic_like(
        {
            "rms": 0.034,
            "peak": 0.18,
            "centroid": 1880,
            "low_ratio": 0.07,
            "mid_ratio": 0.35,
            "high_ratio": 0.57,
            "entropy": 0.86,
            "flatness": 0.012,
            "zero_crossing_rate": 0.20,
        }
    )


def write_quiet_wav(path: Path, sample_rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(struct.pack("<h", 0) for _ in range(sample_rate * 2)))


if __name__ == "__main__":
    tests = [
        test_acoustic_pipeline_writes_metadata_spectrogram_candidates_and_report,
        test_acoustic_segment_proposals_find_high_energy_region,
        test_acoustic_labeled_benchmark_reports_capture_metrics,
        test_anthropogenic_gate_rejects_quiet_animal_like_broadband,
        test_anthropogenic_gate_accepts_steady_vessel_and_antifouling_profiles,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
