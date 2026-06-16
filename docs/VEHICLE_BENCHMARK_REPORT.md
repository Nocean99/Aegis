# Vehicle Benchmark Results

Milestone summary:

Aegis now has a second measured mission category beyond people search. The first vehicle benchmark uses the curated aerial image set from the drone vision test folder and evaluates whether the system can preserve vehicle evidence for analyst review.

## Dataset

The vehicle benchmark contains 43 labeled images:

- vehicle positives: cars, trucks, and jeeps
- hard negatives: people, grass-only images, ambiguous grey objects, and campfire/field scenes

Labels:

```text
config/vision_labels_vehicles_drone_vision_test.csv
```

## Local Vehicle Baseline

Report:

```text
logs/mission_benchmark_suites/20260606T171304Z/vehicles_aerial_search/20260606T171304Z/mission_evaluation_report.html
```

Configuration:

- images processed: 43
- scorer: `local-semantic-placeholder-v1`
- API usage: none
- local detections: 26
- shortlist size: 43
- semantic errors: 0

Confirmed-match metrics:

| Metric | Result |
|---|---:|
| Precision | 0.7143 |
| Recall | 0.4545 |
| F1 | 0.5556 |

Analyst-capture metrics:

| Metric | Result |
|---|---:|
| Capture precision | 0.3462 |
| Capture recall | 0.8182 |
| Capture F1 | 0.4865 |

## Interpretation

The vehicle baseline proves that Aegis can run the same mission-intelligence loop on a second category: mission request, evidence processing, candidate generation, scoring, ranking, report generation, and benchmark metrics.

The local vehicle layer preserved most vehicle evidence for review, with 81.82% capture recall. The current weakness is noise: capture precision was 34.62%, so the review queue still includes too many non-vehicle items.

That is a useful result. It shows the platform is category-general, while also identifying the next improvement: vehicle-specific ranking and stronger semantic review should reduce false positives from grass, people, shadows, and ambiguous grey objects.

## DroneVehicle RGB/Infrared Benchmark Readiness

A larger DroneVehicle RGB/infrared dataset has been imported for the next vehicle benchmark phase.

Analysis:

```text
docs/DRONEVEHICLE_BENCHMARK_ANALYSIS.md
```

Generated labels:

```text
datasets/benchmarks/vehicles/dronevehicle_rgb_labels.csv
datasets/benchmarks/vehicles/dronevehicle_ir_labels.csv
```

Stats:

```text
datasets/benchmarks/vehicles/dronevehicle_stats.json
```

Dataset summary:

| Metric | Result |
|---|---:|
| RGB images | 28,439 |
| Infrared images | 28,439 |
| RGB/IR pairs | 28,439 |
| Total annotations | 953,164 |
| Positive image cases | 56,040 |
| Negative image cases | 838 |

Class distribution:

| Class | Annotations |
|---|---:|
| car | 817,926 |
| truck | 48,086 |
| bus | 31,924 |
| freight car | 30,583 |
| van | 24,643 |

This dataset is stronger than the earlier YOLO OBB export because it includes negative image cases. That means future local RGB and infrared runs can measure both capture recall and capture precision directly.

No OpenAI/API evaluation has been run on the full DroneVehicle dataset. The next step should be a local-only benchmark pass first, then a 100-250 image API review sample if needed.

## DroneVehicle Local Subset Baselines

Aegis ran local-only baselines on 500-image RGB and IR subsets, then added a lightweight vehicle-specific proposal layer for aerial RGB and infrared imagery.

Reports:

```text
docs/DRONEVEHICLE_RGB_BENCHMARK_REPORT.md
docs/DRONEVEHICLE_IR_BENCHMARK_REPORT.md
```

| Benchmark | Images | Positives | Negatives | Local detections | Confirmed F1 | Capture F1 |
|---|---:|---:|---:|---:|---:|---:|
| DroneVehicle RGB local subset | 500 | 250 | 250 | 500 | 0.7622 | 0.6667 |
| DroneVehicle IR local subset | 500 | 447 | 53 | 500 | 0.9460 | 0.9440 |

Interpretation:

The vehicle proposal layer changed the DroneVehicle result from a zero-detection failure into a measurable local benchmark. RGB capture recall reached 100.00%, but capture precision is still 50.00% because the fallback policy preserves uncertain frames for review. IR performed much better immediately, reaching 100.00% capture recall and 89.40% capture precision through hot-blob proposals.

This is the intended resilience behavior: uncertain evidence degrades into analyst review instead of becoming a silent miss.

## Next Vehicle Benchmark Step

Review-priority API samples have been prepared for RGB and IR, limited to 100 images each:

```text
docs/DRONEVEHICLE_RGB_API_BENCHMARK_REPORT.md
docs/DRONEVEHICLE_IR_API_BENCHMARK_REPORT.md
```

The RGB API review benchmark has now completed successfully. The first attempt used broken relative symlinks and produced 100 image-read errors; that invalid report is ignored. After fixing the sampler and recreating readable samples, the completed RGB run processed 100 images with 1 semantic/API error.

## Modality-Aware Strategy Comparison

Vehicle benchmark results now show that the best review strategy depends on sensor modality.

| Modality | Strategy | Images | Capture Precision | Capture Recall | Capture F1 | Recommendation |
|---|---|---:|---:|---:|---:|---|
| RGB | local vehicle proposals | 500 | 50.0% | 100.0% | 66.7% | use as broad first pass |
| RGB | API semantic review | 100 | 73.2% | 95.3% | 82.8% | best RGB review strategy |
| Infrared | local hot-blob triage | 500 | 89.4% | 100.0% | 94.4% | best IR review strategy |
| Infrared | API semantic review | 100 | 61.1% | 100.0% | 75.9% | not recommended yet |

Interpretation:

The API semantic layer cleaned the RGB analyst queue meaningfully. Capture precision improved from 50.0% to 73.2%, while capture recall stayed high at 95.3%. This matches the intended layered design: local RGB proposals preserve evidence, then semantic review reduces noise.

IR behaved differently. Local hot-blob triage was already strong, with 89.4% capture precision and 100.0% capture recall. The IR API sample preserved recall but over-kept ambiguous thermal negatives as `NEEDS_REVIEW`, reducing capture precision to 61.1%.

Mission-memory lessons:

- RGB vehicle evidence benefits from selective API semantic review.
- IR vehicle evidence currently performs better with local hot-blob triage.
- API thermal review may need a stricter prompt or stricter `NEEDS_REVIEW` threshold.

Next step:

Keep selective API review in the RGB vehicle workflow. For IR, prefer local hot-blob triage until the thermal API prompt and review threshold are tuned against hard negatives.

## Leakage-Controlled Local RGB/IR Development Evaluation

Aegis now has the same anti-leakage benchmark protocol for RGB and infrared vehicle layers that it uses for acoustic evaluation.

Protocol:

- Set aside a final 20% lockbox manifest for RGB and IR. Do not evaluate it during tuning.
- Run stratified 5-fold cross-validation on capped 500-image development sets.
- For each fold, write a tuning manifest and a separate held-out evaluation labels file.
- Confirm zero overlap between tuning and evaluation examples for every fold.
- Use local vehicle proposals only. No OpenAI/API calls are made.

Results:

| Modality | Development Images | Lockbox Images | Capture Precision | Capture Recall | Capture F1 |
|---|---:|---:|---:|---:|---:|
| RGB local dev CV | 500 | 5,688 | 50.0% +/- 0.0% | 100.0% +/- 0.0% | 66.7% +/- 0.0% |
| IR local dev CV | 500 | 5,688 | 91.6% +/- 0.5% | 100.0% +/- 0.0% | 95.6% +/- 0.3% |

Interpretation:

The corrected cross-validation results support the same modality lesson as the earlier subset/API experiments. RGB local triage is high-recall but noisy, so RGB benefits from selective semantic cleanup. IR local triage is already strong for DroneVehicle-style thermal imagery, so IR should trust local hot-blob triage first and use API review more selectively.

Full report:

```text
docs/DRONEVEHICLE_VISUAL_CROSS_VALIDATION_REPORT.md
```
