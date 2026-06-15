# Aegis Mission Intelligence v1

## Definition

Aegis Mission Intelligence v1 is the first complete MVP of the project.

It is not just a drone simulator and not just a sensor demo. It is a mission intelligence workflow that turns multi-sensor evidence into prioritized contacts, analyst decisions, mission memory, and structured reports.

## Supported Modalities

| Modality | Current Role |
|---|---|
| RGB | visual proposal detection and semantic review |
| Infrared | thermal proposal detection and local triage |
| Acoustic | hydrophone `.wav` ingestion, spectrograms, and anthropogenic acoustic triage |

## Supported Outputs

- Candidates
- Analyst review queue
- Mission memory
- Mission report

## Benchmarked Areas

| Benchmark | Status |
|---|---|
| SAR people | measured |
| RGB vehicles | measured |
| IR vehicles | measured |
| Acoustic underwater noise | measured and tuned |
| System benchmark | measured across five mission cases |

## Multi-Sensor Demo

Current demo:

```text
Mission: Monitor protected shoreline for possible vessel activity
Inputs: RGB image, IR image, hydrophone recording
Output: 1 high-priority contact with multi-sensor confirmation
```

Evidence:

- RGB vessel silhouette
- IR hotspot
- engine-like acoustic signature

## System Benchmark

Current mission cases:

| Mission | Scenario |
|---|---|
| mission_001 | vessel present, all sensors agree |
| mission_002 | acoustic only, medium priority |
| mission_003 | thermal false positive, no vessel |
| mission_004 | RGB ambiguous, acoustic confirms |
| mission_005 | no contact |

Current result:

```text
Mission success rate: 100.0%
Contact precision: 100.0%
Contact recall: 100.0%
```

## MVP Boundary

This is the v1 boundary:

```text
Mission Request
  -> RGB / IR / Acoustic Evidence
  -> Candidate Generation
  -> Contact Summary
  -> Analyst Review
  -> Mission Memory
  -> Mission Report
```

Future work should improve the demo experience, system benchmark coverage, and dashboard polish before adding more modalities.
