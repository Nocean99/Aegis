# Aegis Acoustic Benchmark Tuning Report

## Objective

Measure the acoustic proposal layer first, then tune the proposal logic without changing the benchmark runner.

Positive class:

- anthropogenic underwater noise

Negative classes:

- animal
- sonar

## Dataset

| Class | Full Dataset Count | Benchmark Sample |
|---|---:|---:|
| Anthropogenic | 77 | 20 |
| Animal | 265 | 20 |
| Sonar | 26 | 20 |

## Before Tuning

The first benchmark measured the original high-energy acoustic proposal layer.

| Metric | Result |
|---|---:|
| True positives | 9 |
| False positives | 29 |
| True negatives | 11 |
| False negatives | 11 |
| Capture precision | 23.7% |
| Capture recall | 45.0% |
| Capture F1 | 31.0% |

Finding: the original acoustic layer behaved like a rough activity detector. It fired on many animal and sonar clips while missing low-amplitude vessel or machinery clips.

## Tuning Change

The proposal layer now uses an anthropogenic acoustic profile when the mission mentions vessel, ship, boat, engine, machinery, or anthropogenic noise.

The new proposal gate looks for:

- broad continuous acoustic energy
- energetic broadband noise
- strong machinery-like acoustic events

Generic acoustic activity detection still preserves loud events for non-anthropogenic missions.

## After Tuning

| Metric | Result |
|---|---:|
| True positives | 19 |
| False positives | 8 |
| True negatives | 32 |
| False negatives | 1 |
| Capture precision | 70.4% |
| Capture recall | 95.0% |
| Capture F1 | 80.8% |

## Interpretation

The acoustic module is now a useful triage layer for anthropogenic underwater noise. It is not a trained classifier yet, but it can preserve likely vessel or machinery evidence while reducing animal and sonar false positives.

The next acoustic improvement should be a small classifier or stricter scorer trained/evaluated against this same benchmark, not a replacement for the benchmark workflow.
