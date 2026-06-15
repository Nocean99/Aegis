# Multi-Sensor Shoreline Demo

## Scenario

Protected coastal zone.

## Mission

```text
Identify possible vessel activity.
```

## Inputs

| Sensor Layer | Evidence |
|---|---|
| Aegis Vision Intelligence | RGB shoreline camera image folder |
| Aegis Infrared Intelligence | thermal shoreline camera image folder |
| Aegis Acoustic Intelligence | hydrophone `.wav` recording |

## Expected Aegis Output

- RGB visual candidates
- IR thermal candidates
- acoustic contact candidates
- spectrogram artifacts
- unified candidate list
- analyst review queue
- approve / reject / investigate decisions
- mission report
- mission-memory lessons by modality

## Example Contact Output

| Contact | Assessment | Evidence | Priority |
|---|---|---|---|
| Candidate 1 | Possible vessel activity | RGB vessel silhouette, thermal hotspot, engine-like acoustic segment | 0.95 |
| Candidate 2 | Possible false alarm | Thermal hotspot only | 0.41 |

Example mission report summary:

```json
{
  "candidate_count": 2,
  "high_priority_contacts": 1,
  "multi_sensor_confirmation": true
}
```

This is the first demo where Aegis looks less like a drone simulator and more like a maritime intelligence platform: it does not just detect objects, it explains why a contact matters across sensors.

## Why This Demo Matters

This demo makes Aegis easy to understand as a multi-sensor mission intelligence platform.

The point is not that one model solves everything. The point is that different sensors produce different evidence, and Aegis can preserve, rank, review, and learn from each evidence stream.

## Review Policy

| Modality | Current Best Policy |
|---|---|
| RGB | local proposals followed by selective API cleanup |
| Infrared | local hot-blob triage first |
| Acoustic | local segment proposals first |

## Acoustic MVP Role

The acoustic layer does not need classifier training to be useful in this demo.

It only needs to:

1. Ingest `.wav` evidence.
2. Generate a spectrogram.
3. Propose acoustic segments.
4. Show candidate time ranges in the dashboard.
5. Save analyst decisions.
6. Add acoustic patterns to mission memory.

That is enough to show the architecture moving from vision intelligence to multi-sensor mission intelligence.

## Demo Command

```bash
./scripts/run_multisensor_demo.sh \
  --mission-request "Identify possible vessel activity in a protected coastal zone" \
  --rgb-images "/path/to/rgb_images" \
  --ir-images "/path/to/ir_images" \
  --acoustic "/path/to/hydrophone.wav" \
  --output-dir logs/multisensor_missions
```

Outputs:

```text
logs/multisensor_missions/<timestamp>/multisensor_mission_report.json
logs/multisensor_missions/<timestamp>/multisensor_mission_report.html
```

The report links the three evidence stages:

- RGB vision report
- infrared vision report
- acoustic report

It then normalizes their candidates into one review-priority ordered list.
