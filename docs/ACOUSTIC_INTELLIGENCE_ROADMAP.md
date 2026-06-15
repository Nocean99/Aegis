# Aegis Acoustic Intelligence Roadmap

## Why Acoustic Evidence Belongs In Aegis

Aegis is becoming a multi-sensor mission intelligence platform. Vision and infrared evidence already flow through the same loop:

```text
Mission request
-> sensor evidence
-> proposal generation
-> candidate ranking
-> analyst review
-> mission memory
-> mission report
```

Acoustic evidence should enter that same workflow. The first goal is not a deep model. The first goal is making audio and sonar-like observations reviewable, measurable, and reportable.

This expands the platform from:

```text
Aegis Vision Intelligence
+ Aegis Infrared Intelligence
```

to:

```text
Aegis Vision Intelligence
+ Aegis Infrared Intelligence
+ Aegis Acoustic Intelligence
```

## Mission Examples

Acoustic evidence can support missions such as:

- locating engine sounds near a search area
- detecting distress calls, whistles, knocks, or impact sounds
- monitoring machinery, motors, pumps, or generators
- identifying underwater or shoreline acoustic anomalies
- combining audio with RGB/IR evidence for stronger mission reports

## MVP Definition

The acoustic MVP should prove that non-visual evidence can use the Aegis mission workflow.

The MVP is complete when Aegis can:

1. Ingest `.wav` evidence.
2. Generate spectrogram artifacts.
3. Extract basic audio metadata.
4. Propose reviewable acoustic segments.
5. Emit candidate JSON using the Aegis candidate style.
6. Produce a simple acoustic report.
7. Preserve uncertainty for analyst review.
8. Show acoustic candidates in the analyst dashboard.
9. Produce a basic labeled acoustic benchmark.

Classifier training comes later.

## Data Sources

Initial data sources:

- `.wav` files from microphones
- exported hydrophone or sonar-like audio clips
- simulated engine, alarm, whistle, or impact sounds
- field recordings from phones or external recorders
- future live audio streams from robots, drones, buoys, boats, or fixed sensors

Future benchmark folders should follow the same pattern as vision benchmarks:

```text
datasets/benchmarks/acoustic/
  engines/
    positives/
    near_misses/
    hard_negatives/
  distress_calls/
    positives/
    near_misses/
    hard_negatives/
  underwater_anomalies/
    positives/
    near_misses/
    hard_negatives/
```

## Candidate Format

Acoustic candidates should look like other Aegis candidates, but with time ranges instead of image bounding boxes.

```json
{
  "candidate_id": "engine_clip_acoustic_0004",
  "audio_path": "datasets/acoustic/engine_clip.wav",
  "start_s": 3.5,
  "end_s": 5.0,
  "duration_s": 1.5,
  "proposal_score": 0.82,
  "uncertainty_score": 0.37,
  "sensor_modality": "acoustic",
  "proposal_reason": "high-energy acoustic segment",
  "peak_amplitude": 0.91,
  "rms_amplitude": 0.42
}
```

Later semantic or classifier fields can be added:

```json
{
  "semantic_score": 0.76,
  "mission_relevance_score": 0.81,
  "review_priority": 0.79,
  "tags": ["engine", "repeating pulse", "low frequency"]
}
```

## Phase 1: Acoustic Evidence Ingestion

Implemented first:

- `.wav` ingestion
- mono conversion
- sample rate, channel count, duration, RMS, and peak amplitude metadata
- spectrogram PNG generation
- JSON and HTML acoustic report shell

Command:

```bash
./scripts/run_acoustic_evidence.sh "/path/to/wav_or_folder" \
  --mission-request "Listen for engine or distress sounds near the search area" \
  --output-dir logs/acoustic_evaluations/demo
```

Outputs:

```text
logs/acoustic_evaluations/demo/acoustic_report.json
logs/acoustic_evaluations/demo/acoustic_report.html
logs/acoustic_evaluations/demo/spectrograms/
```

## Phase 2: Acoustic Segment Proposals

Implemented second:

- sliding-window energy analysis
- high-energy acoustic segment proposals
- candidate JSON with timestamps, proposal score, uncertainty, and proposal reason
- simple report table for analyst review
- dashboard review cards with spectrogram image, time range, proposal score, reason, review priority, and analyst decisions

This is intentionally simple. The point is to preserve acoustic evidence and make it reviewable before training any classifier.

## Phase 3: Benchmarking

Before training, use a small labeled evaluation:

```text
20 vessel-like clips
20 non-vessel or ambient clips
```

Measure:

- capture precision
- capture recall
- proposal count
- false positive causes
- uncertainty causes

The first benchmark headline should be:

```text
Aegis now evaluates acoustic evidence using the same benchmark and reporting workflow as visual and infrared evidence.
```

Then expand:

- collect labeled clips
- define positives, near misses, and hard negatives
- measure capture precision and capture recall
- track false positives such as wind, handling noise, machinery hum, speech, and water noise
- add mission-memory lessons for acoustic evidence

## Phase 4: Classifier Or Semantic Review

Only after the evidence workflow works:

- train small acoustic classifiers
- test audio embeddings
- compare local classifiers against API or larger model review
- tune thresholds by mission type
- integrate review priority with visual and infrared candidates

## Product Lesson

The move is not "add a model."

The move is:

```text
Make acoustic evidence fit the Aegis mission workflow.
```

That is what turns Aegis from vision intelligence into true multi-sensor mission intelligence.

## Multi-Sensor Demo Scenario

Mission:

```text
Monitor a shoreline for possible vessel activity.
```

Inputs:

- RGB image folder
- IR image folder
- hydrophone `.wav` file

Aegis outputs:

- visual candidates
- thermal candidates
- acoustic contact candidates
- analyst review queue
- mission report
- mission-memory lessons by modality
