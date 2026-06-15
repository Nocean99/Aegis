from __future__ import annotations

import json
import math
import struct
import sys
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.system_benchmark import run_system_benchmark


def write_rgb_vehicle_image(path: Path) -> None:
    image = np.full((160, 220, 3), 35, dtype=np.uint8)
    cv2.rectangle(image, (80, 70), (135, 98), (220, 220, 220), -1)
    cv2.imwrite(str(path), image)


def write_ir_vehicle_image(path: Path) -> None:
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    cv2.rectangle(image, (90, 72), (135, 100), (240, 240, 240), -1)
    cv2.imwrite(str(path), image)


def write_engine_wav(path: Path, sample_rate: int = 8000) -> None:
    samples = []
    for index in range(sample_rate * 2):
        t = index / sample_rate
        amplitude = 0.08
        signal = math.sin(2 * math.pi * 140 * t) * 0.45 + math.sin(2 * math.pi * 310 * t) * 0.25
        if 0.75 <= t <= 1.1:
            amplitude = 0.75
            noise = ((((index * 1103515245 + 12345) >> 16) & 0x7FFF) / 16384.0) - 1.0
            signal = signal * 0.55 + noise * 0.45
        samples.append(int(max(-1.0, min(1.0, amplitude * signal)) * 32767))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def test_system_benchmark_scores_mission_success() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        mission_dir = root / "benchmark_data" / "missions" / "mission_001"
        rgb_dir = root / "rgb"
        ir_dir = root / "ir"
        audio_dir = root / "audio"
        for folder in (mission_dir, rgb_dir, ir_dir, audio_dir):
            folder.mkdir(parents=True)
        write_rgb_vehicle_image(rgb_dir / "rgb_contact_001.jpg")
        write_ir_vehicle_image(ir_dir / "ir_contact_001.jpg")
        write_engine_wav(audio_dir / "hydrophone_contact_001.wav")
        (mission_dir / "mission.json").write_text(
            json.dumps(
                {
                    "mission_id": "mission_001",
                    "name": "Synthetic shoreline mission",
                    "mission_request": "Monitor a protected shoreline for possible vessel activity",
                    "rgb_images": str(rgb_dir),
                    "ir_images": str(ir_dir),
                    "acoustic": str(audio_dir / "hydrophone_contact_001.wav"),
                    "expected_contact": True,
                    "expected_priority": "high",
                    "expected_evidence": ["rgb", "infrared", "acoustic"],
                }
            ),
            encoding="utf-8",
        )
        report_path = run_system_benchmark(
            missions_root=root / "benchmark_data" / "missions",
            output_dir=root / "logs" / "system_benchmark_v1",
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["mission_count"] == 1
        assert report["metrics"]["mission_success_rate"] == 1.0
        assert report["metrics"]["contact_precision"] == 1.0
        assert report["metrics"]["contact_recall"] == 1.0
        assert report["results"][0]["mission_success"] is True
        assert report_path.with_name("system_benchmark_report.html").exists()


if __name__ == "__main__":
    tests = [test_system_benchmark_scores_mission_success]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
