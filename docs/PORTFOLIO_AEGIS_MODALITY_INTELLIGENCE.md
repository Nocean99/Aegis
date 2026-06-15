# Aegis Modality Intelligence Portfolio Note

## Summary

Aegis Mission Intelligence now supports modality-aware benchmark interpretation for vehicle search. The benchmark results show that RGB and infrared imagery should not use the same review policy.

## Vehicle Benchmark Results

| Modality | Strategy | Images | Capture Precision | Capture Recall | Capture F1 |
|---|---|---:|---:|---:|---:|
| RGB | local vehicle proposals | 500 | 50.0% | 100.0% | 66.7% |
| RGB | API semantic review | 100 | 73.2% | 95.3% | 82.8% |
| Infrared | local hot-blob triage | 500 | 89.4% | 100.0% | 94.4% |
| Infrared | API semantic review | 100 | 61.1% | 100.0% | 75.9% |

## Interpretation

RGB vehicle evidence benefits from selective API semantic review. The local RGB proposal layer is useful because it preserves likely evidence, but it creates a noisy analyst queue. API review improves capture precision while keeping recall high.

Infrared vehicle evidence currently performs better with local hot-blob triage. The API preserved recall, but it treated too many ambiguous thermal negatives as reviewable evidence. Thermal API review needs stricter prompting or stricter NEEDS_REVIEW thresholds before it improves the IR workflow.

## Product Lesson

The benchmark result supports a key Aegis design principle: mission review policy should depend on sensor modality.

A single mission can combine multiple intelligence layers:

- Aegis Vision Intelligence for RGB imagery
- Aegis Infrared Intelligence for thermal imagery
- Aegis Acoustic Intelligence for sonar, audio, and other signal-based sensing

The next logical expansion is an acoustic/sonar module that can evaluate non-visual evidence with the same mission-memory and benchmark discipline.

## Acoustic Intelligence MVP

Aegis Acoustic Intelligence starts with evidence workflow, not classifier training.

Phase 1:

- ingest `.wav` evidence
- generate spectrograms
- extract sample rate, duration, channel count, RMS, and peak metadata
- write JSON and HTML reports

Phase 2:

- propose high-energy acoustic segments
- emit candidate JSON with timestamps, proposal score, uncertainty, and proposal reason
- preserve ambiguous audio evidence for analyst review

Only after this workflow is benchmarked should classifier training become the focus.
