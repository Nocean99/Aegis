from __future__ import annotations

import json
import math
import struct
import sys
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.acoustic_benchmark import run_acoustic_benchmark


def write_wav(path: Path, *, amplitude: float, frequency: float = 220.0, sample_rate: int = 8000) -> None:
    samples = []
    for index in range(sample_rate):
        t = index / sample_rate
        value = int(amplitude * math.sin(2 * math.pi * frequency * t) * 32767)
        samples.append(value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def test_acoustic_benchmark_creates_sample_csv_and_report() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset = root / "dataset"
        for label in ("anthropogenic", "animal", "sonar"):
            (dataset / label).mkdir(parents=True)
        write_wav(dataset / "anthropogenic" / "vessel.wav", amplitude=0.8)
        write_wav(dataset / "animal" / "whale.wav", amplitude=0.02)
        write_wav(dataset / "sonar" / "ping.wav", amplitude=0.02)

        report_path = run_acoustic_benchmark(
            dataset_root=dataset,
            benchmark_root=root / "benchmark_data" / "acoustic_v1",
            output_dir=root / "logs" / "acoustic_benchmark_v1",
            sample_limit=2,
            docs_snippet_path=root / "docs" / "ACOUSTIC_BENCHMARK_V1_SNIPPET.md",
        )

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["dataset_counts"] == {"anthropogenic": 1, "animal": 1, "sonar": 1}
        assert report["sample_counts"] == {"anthropogenic": 1, "animal": 1, "sonar": 1}
        assert report["metrics"]["labeled_count"] == 3
        assert (root / "benchmark_data" / "acoustic_v1" / "benchmark.csv").exists()
        assert report_path.with_name("acoustic_benchmark_report.html").exists()
        assert "Aegis Acoustic Benchmark v1" in report["readme_snippet"]


if __name__ == "__main__":
    tests = [test_acoustic_benchmark_creates_sample_csv_and_report]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
