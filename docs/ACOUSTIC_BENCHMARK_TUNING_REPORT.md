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

The first tuned proposal gate looked for:

- broad continuous acoustic energy
- energetic broadband noise
- strong machinery-like acoustic events

Generic acoustic activity detection still preserves loud events for non-anthropogenic missions.

## First Tuned Result

| Metric | Result |
|---|---:|
| True positives | 19 |
| False positives | 8 |
| True negatives | 32 |
| False negatives | 1 |
| Capture precision | 70.4% |
| Capture recall | 95.0% |
| Capture F1 | 80.8% |

## Holdout Retest

A second sample was created from later files in `/Users/noah/Downloads/dataset_final` to avoid reusing the first sorted benchmark sample where possible.

Because the sonar class only contains 26 files total, the holdout contains the 6 remaining sonar files instead of a full 20.

| Class | Holdout Sample |
|---|---:|
| Anthropogenic | 20 |
| Animal | 20 |
| Sonar | 6 |

The first tuned gate dropped on this holdout:

| Metric | Result |
|---|---:|
| True positives | 11 |
| False positives | 15 |
| True negatives | 11 |
| False negatives | 9 |
| Capture precision | 42.3% |
| Capture recall | 55.0% |
| Capture F1 | 47.8% |

Failure pattern:

- missed steady cruise-ship and high-frequency antifouling clips
- over-triggered on dolphin, sea-lion, and humpback clips
- over-triggered on one sonar clip

## Vessel-Aware Gate Update

The current gate adds two vessel-specific profiles and tightens animal/sonar false positives:

- rejects very quiet broadband clips that looked like dolphin noise
- keeps steady low-frequency vessel hum
- keeps high-frequency antifouling-style vessel noise
- requires vessel-band energy for broad machinery events

Original 60-clip sample after the update:

| Metric | Result |
|---|---:|
| True positives | 16 |
| False positives | 2 |
| True negatives | 38 |
| False negatives | 4 |
| Capture precision | 88.9% |
| Capture recall | 80.0% |
| Capture F1 | 84.2% |

Holdout sample after the update:

| Metric | Result |
|---|---:|
| True positives | 20 |
| False positives | 2 |
| True negatives | 24 |
| False negatives | 0 |
| Capture precision | 90.9% |
| Capture recall | 100.0% |
| Capture F1 | 95.2% |

## Interpretation

The acoustic module is now a stronger heuristic triage layer for anthropogenic underwater noise. It is not a trained classifier yet, but it can preserve likely vessel or machinery evidence while reducing animal and sonar false positives.

The current tuning is a better overall detector shape: the original sample loses some recall but becomes much cleaner, while the holdout sample improves substantially. The next acoustic step should be a larger fixed train/validation/test split so tuning decisions are measured against a stable validation set instead of hand-picked samples.

## Leakage-Controlled Evaluation Protocol

After the two sample-based debugging runs, Aegis now uses a leakage-controlled protocol:

- set aside a final lockbox split
- do not evaluate the lockbox during tuning
- run stratified 5-fold cross-validation on the development split only
- ensure each fold's tuning manifest and evaluation manifest have zero overlapping clips
- report mean and standard deviation across folds

Important caveat: because earlier heuristic work had already looked at parts of this dataset, the development cross-validation result is not a pristine final test-set claim. The lockbox exists to prevent further leakage from this point forward.

| Dataset Class | Clips |
|---|---:|
| Anthropogenic | 77 |
| Animal | 265 |
| Sonar | 26 |

Aggregate cross-validation result:

| Metric | Result |
|---|---:|
| True positives | 57 |
| False positives | 99 |
| True negatives | 192 |
| False negatives | 20 |
| Development clips | 295 |
| Lockbox clips, unevaluated | 73 |
| Lockbox anthropogenic / animal / sonar | 15 / 53 / 5 |

Aggregate development cross-validation result:

| Metric | Result |
|---|---:|
| True positives | 44 |
| False positives | 85 |
| True negatives | 148 |
| False negatives | 18 |
| Capture precision | 34.1% |
| Capture recall | 71.0% |
| Capture F1 | 46.1% |

Mean and standard deviation across folds:

| Metric | Mean | Std |
|---|---:|---:|
| Capture precision | 34.2% | 8.9% |
| Capture recall | 70.6% | 16.7% |
| Capture F1 | 46.0% | 11.5% |

Fold results:

| Fold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 1 | 35.7% | 76.9% | 48.8% |
| 2 | 42.3% | 84.6% | 56.4% |
| 3 | 40.9% | 75.0% | 52.9% |
| 4 | 20.0% | 41.7% | 27.0% |
| 5 | 32.1% | 75.0% | 45.0% |

Interpretation: the acoustic heuristic is recall-oriented but noisy. It is good enough to demonstrate acoustic evidence entering the Aegis mission workflow, but not strong enough to claim robust acoustic classification. Fold 4 is a warning sign: performance varies meaningfully by split, so any single-sample improvement should be treated skeptically.

## Physics-Grounded Feature Improvements

The next acoustic improvements should be based on general signal properties that distinguish vessel or machinery noise from biological/natural sounds, not on individual clips that failed.

Candidate features:

- Harmonic stack stability: engines often create stable fundamental frequencies and harmonics over time; animal calls often sweep, pulse, or vary more strongly.
- Temporal stationarity: machinery tends to be continuous or slowly varying; marine mammal vocalizations are often transient, pulsed, tonal sweeps, clicks, or songs.
- Modulation rate: propeller/engine activity can create regular modulation bands; animal vocalizations may have different call/click repetition structures.
- Spectral slope and band energy ratios: vessel noise often has persistent low-frequency energy plus machinery harmonics; dolphin/whale signals may concentrate in narrower or higher-frequency structures.
- Tonal persistence: antifouling and machinery tones can remain narrowband and persistent over multiple windows.
- Event density: sonar and animal clicks can create sparse high-energy events; engine noise often fills more of the time-frequency plane.

Overfitting risk: adding these features is lower risk than hand-tuning thresholds to named clips, but any threshold or classifier using them must still be selected on development folds only. The lockbox must stay untouched until the model and thresholds are frozen.

The next acoustic improvement should be a small classifier or stricter scorer trained/evaluated against this same benchmark, not a replacement for the benchmark workflow.
