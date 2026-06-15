# Aegis System Benchmark v1

## Purpose

This benchmark evaluates Aegis as a mission intelligence workflow, not as separate sensor models.

Each mission includes:

- RGB evidence
- infrared evidence
- acoustic evidence
- expected contact outcome
- expected priority
- expected supporting evidence

## Mission Cases

| Mission | Scenario | Expected Result |
|---|---|---|
| mission_001 | vessel present, all sensors agree | high-priority contact |
| mission_002 | acoustic only | medium-priority contact |
| mission_003 | thermal false positive | no positive vessel contact |
| mission_004 | RGB ambiguous, acoustic confirms | high-priority contact |
| mission_005 | no contact | no positive vessel contact |

## Results

| Metric | Result |
|---|---:|
| Mission success rate | 100.0% |
| Contact precision | 100.0% |
| Contact recall | 100.0% |
| True positives | 3 |
| False positives | 0 |
| True negatives | 2 |
| False negatives | 0 |

## Interpretation

Aegis now has a first mission-level benchmark. The system preserves uncertain candidates for review, but only counts contact-confirming evidence toward mission success.

This matters because a thermal-only hotspot should stay reviewable without becoming a confirmed vessel contact. Acoustic-only evidence can raise a medium-priority contact, while multi-sensor evidence can raise a high-priority contact.

## Rerun Command

```bash
./scripts/run_system_benchmark_v1.sh
```

Outputs:

```text
logs/system_benchmark_v1/system_benchmark_report.json
logs/system_benchmark_v1/system_benchmark_report.html
```
