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

from autonomy.multisensor_mission_demo import run_multisensor_demo


def write_rgb_vehicle_image(path: Path) -> None:
    image = np.full((160, 220, 3), 35, dtype=np.uint8)
    cv2.rectangle(image, (80, 70), (135, 98), (220, 220, 220), -1)
    cv2.rectangle(image, (86, 75), (129, 93), (40, 40, 40), 2)
    cv2.imwrite(str(path), image)


def write_ir_vehicle_image(path: Path) -> None:
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    cv2.rectangle(image, (90, 72), (135, 100), (240, 240, 240), -1)
    cv2.imwrite(str(path), image)


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


def test_multisensor_demo_writes_unified_candidates_and_report() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        rgb_dir = root / "rgb"
        ir_dir = root / "ir"
        rgb_dir.mkdir()
        ir_dir.mkdir()
        write_rgb_vehicle_image(rgb_dir / "rgb_vehicle.jpg")
        write_ir_vehicle_image(ir_dir / "ir_vehicle.jpg")
        wav_path = root / "hydrophone.wav"
        write_test_wav(wav_path)
        report_path = run_multisensor_demo(
            mission_request="Monitor a shoreline for possible vessel activity",
            rgb_images=rgb_dir,
            ir_images=ir_dir,
            acoustic=wav_path,
            output_dir=root / "multisensor",
            max_saved_candidates=5,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["stage_summary"]["error"] == 0
        assert report["summary"]["candidate_count"] >= 3
        assert report["summary"]["contact_count"] >= 1
        assert report["summary"]["high_priority_contacts"] >= 1
        assert report["summary"]["multi_sensor_confirmation"] is True
        assert report["contacts"][0]["assessment"] == "possible vessel activity"
        assert set(report["contacts"][0]["evidence"]) == {
            "RGB vessel silhouette or visual proposal",
            "Thermal hotspot",
            "Engine-like acoustic segment",
        }
        modalities = {item["sensor_modality"] for item in report["unified_candidates"]}
        assert {"rgb", "infrared", "acoustic"} <= modalities
        assert (report_path.parent / "multisensor_mission_report.html").exists()


if __name__ == "__main__":
    tests = [test_multisensor_demo_writes_unified_candidates_and_report]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
