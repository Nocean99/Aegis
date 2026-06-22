# Aegis — Mission Intelligence Layer

Aegis is a mission intelligence platform that turns sensor overload into ranked, analyst-confirmed findings for robotic and sensor systems.

Aegis helps operators convert large volumes of image, infrared, acoustic, and mission data into prioritized findings, analyst decisions, mission memory, and structured mission reports.

The platform combines mission planning, contextual search priorities, proposal detection, semantic analysis, candidate ranking, analyst review, and mission memory into a single workflow. Current validation uses drone simulation, image/video benchmarks, and PX4/Gazebo integration paths, but the architecture is designed to support multiple sensor sources including drones, fixed cameras, robotics platforms, acoustic sensors, telemetry feeds, and recorded datasets.

This repository is for simulation, autonomy workflow development, and perception evaluation. It is not flight-control firmware and should not be connected directly to real motors.

## Aegis Mission Intelligence v1

Current MVP boundary:

| Area | v1 Status |
|---|---|
| Modalities | RGB, infrared, acoustic |
| Outputs | candidates, analyst review, mission memory, mission reports |
| Benchmarks | SAR, RGB vehicles, IR vehicles, acoustic underwater noise |
| Multi-sensor demo | shoreline monitoring workflow |
| System benchmark | five mission-level contact cases in development |

The v1 product story is simple: Aegis takes RGB imagery, thermal imagery, and hydrophone audio from protected shoreline missions, then produces ranked contacts with supporting evidence across sensors.

## Why This Exists

Modern missions can generate thousands of images, video frames, detections, and sensor observations.

Human operators often have to review that evidence manually. Important findings can be missed while analysts spend time sorting through irrelevant frames, ambiguous detections, or low-quality sensor outputs.

Aegis focuses on deciding which observations deserve attention while preserving uncertain evidence for review. The goal is not to remove the operator. The goal is to help the operator move faster, miss less, and leave behind a useful mission record.

## System Architecture

```mermaid
flowchart TD
    A[Mission Request] --> B[Mission Planning]
    B --> C[Contextual Search Priorities]
    C --> D[Sensor Collection]
    D --> E[Proposal Detection]
    E --> F[Semantic Scoring]
    F --> G[Candidate Ranking]
    G --> H[Analyst Review]
    H --> I[Mission Memory]
    I --> J[Mission Report]

    D -.-> D1[Image Folder]
    D -.-> D2[Video]
    D -.-> D3[Drone Camera]
    D -.-> D4[Future Sensor Sources]

    F -.-> F1[Local Semantic Scorer]
    F -.-> F2[Optional Vision-Language Scorer]

    G -.-> G1[Proposal Score]
    G -.-> G2[Semantic Score]
    G -.-> G3[Uncertainty Score]
    G -.-> G4[Mission Relevance Score]
```

PX4, Gazebo, dashboards, cameras, videos, and future sensor feeds are integration points. The mission intelligence layer is the product.

Low-level vehicle control stays separated from mission reasoning. Perception scoring does not directly control a vehicle; it produces reviewable evidence and mission reports.

## Benchmark Snapshot

![Aegis benchmark snapshot](docs/assets/aegis_benchmark_snapshot.png)

Current leakage-controlled development estimates:

| Benchmark | Strategy | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| IR Vehicles | local triage, cross-validated | 91.6% ± 0.5% capture | 100.0% ± 0.0% capture | 95.6% ± 0.3% capture |
| RGB Vehicles | local proposals, cross-validated | 50.0% ± 0.0% capture | 100.0% ± 0.0% capture | 66.7% ± 0.0% capture |
| Acoustic | development cross-validation | 34.2% ± 8.9% capture | 70.6% ± 16.7% capture | 46.0% ± 11.5% capture |
| Maritime SAR | SeaDronesSee local recall-only | n/a | 78.0% ± 11.0% confirmed recall | n/a |

The headline lesson: review policy should depend on modality. Infrared vehicle evidence currently performs best with local hot-blob triage, RGB vehicle evidence benefits from semantic cleanup after local proposals, and acoustic evidence is useful but still noisy.

Cross-surface consistency check: this table is the canonical public benchmark snapshot and matches the current benchmark graphic. Older single-sample and API-review numbers remain traceable in reports, but they are not the current headline generalization estimates.

## Evaluation Methodology

Aegis is evaluated with leakage-controlled splits. For acoustic and DroneVehicle RGB/IR development runs, the pipeline creates a protected final lockbox and leaves it untouched while development uses stratified cross-validation. No clip or image appears in both the tuning and evaluation side of the same fold.

Aegis reports two metric families. Confirmed metrics measure what the system identifies as a match. Capture metrics measure whether relevant evidence is preserved for analyst review, even when the system is uncertain. Capture recall is intentionally recall-biased because full-frame fallback can preserve a low-confidence candidate instead of silently dropping a target.

SeaDronesSee is currently reported as recall-only because the local train/validation labels contain no negative maritime cases under the current target rule. Precision and F1 are intentionally omitted until no-target maritime imagery is added.

Note: earlier single-sample benchmark figures were superseded by leakage-controlled cross-validation after held-out testing showed they overstated generalization. The cross-validated numbers above are the current honest development estimates.

Fixed benchmark issue: a label-parsing bug that affected earlier benchmark scores was corrected, and a regression test now verifies that negative labels stay negative.

## Multi-Sensor Shoreline Demo

Aegis includes a small maritime monitoring demo that runs visual, infrared, and acoustic stages through one mission workflow.

Scenario:

```text
Protected coastal zone.
Mission: monitor for possible vessel activity.
Inputs: RGB shoreline image, thermal shoreline image, hydrophone recording.
Output: candidates by modality, analyst review queue, mission memory, mission report.
```

The demo is a workflow demonstration, not a benchmark score. It shows how RGB, infrared, and hydrophone evidence move through the same mission reporting and review structure. Current recording assets demonstrate RGB/IR review and a labeled acoustic clip separately; they should not be presented as a scored three-sensor fusion result.

The acoustic recording path is active: `.wav` files produce spectrograms, acoustic segment candidates, candidate JSON, and report entries. The current labeled acoustic recording demo is a single positive clip showing acoustic evidence entering the workflow, while the broader acoustic benchmark is reported separately in the benchmark section.

Run it with:

```bash
./scripts/run_multisensor_demo.sh \
  --mission-request "Monitor a protected shoreline for possible vessel activity" \
  --rgb-images demo_data/shoreline_v1/rgb \
  --ir-images demo_data/shoreline_v1/ir \
  --acoustic demo_data/shoreline_v1/acoustic/hydrophone_contact_001.wav \
  --output-dir logs/multisensor_missions
```

## Core Concepts

**Mission Planning**
Converts plain-English objectives into structured mission commands, operating modes, target descriptions, confirmation policies, and link-loss behavior.

**Contextual Search Priorities**
Infers likely places to search first based on mission context. A vehicle mission, person search, vessel search, debris search, and signal search should not all be treated as the same generic grid problem.

**Proposal Detection**
Uses lightweight local detection to find candidate regions before heavier semantic review. Current proposal layers include color-based detection, high-recall detection, and objectness-style proposals.

**Semantic Scoring**
Scores candidate crops and full frames against the mission request. The system supports a local semantic scorer and an optional OpenAI-backed vision scorer for stronger open-vocabulary review.

**Candidate Ranking**
Separates confirmed matches from review preservation. A likely match ranks high, uncertain evidence stays reviewable, scorer failures remain visible, and rejected crops can trigger full-frame review.

Each candidate receives an explicit ranking object:

```json
{
  "proposal_score": 0.81,
  "semantic_score": 0.67,
  "uncertainty_score": 0.43,
  "mission_relevance_score": 0.74,
  "review_priority": 0.72
}
```

The analyst queue is sorted by `review_priority`.

**Analyst Review**
Provides a human decision layer for approving, rejecting, or investigating candidates. Review decisions are saved beside mission reports.

**Mission Memory**
Summarizes previous reports and analyst decisions to expose recurring false positives, recurring misses, weak categories, and recommended benchmark data to collect next.

**Mission Reports**
Generates structured JSON and HTML reports covering mission understanding, contextual planning, vision strategy, candidate results, metrics, and stage health.

## System Principles

**Human In The Loop**
The platform assists operators. It does not remove oversight from high-stakes search, rescue, inspection, or security workflows.

**Resilient By Design**
No single component should erase the mission.

- Detector fails -> preserve the frame and continue reporting.
- Crop is wrong -> run full-frame semantic review.
- Semantic scorer times out -> keep the candidate for review.
- Dashboard fails -> reports and review files remain on disk.
- Benchmark fails on one mission -> the suite records the error and continues.

The preferred failure mode is degraded confidence and more review, not silent misses.

**Layered Evaluation**
Local and API evaluation are kept as separate signals rather than merged into one verdict. They fail on different cases, so preserving both signals and surfacing disagreements for human review prevents any single layer's blind spot from silently losing a target.

**Mission Intelligence Before Drone Simulation**
The drone simulator is a validation platform. The broader system is a reusable mission intelligence layer for robotic and sensor workflows.

## Example Workflow

```text
Mission: Search for a missing person near a shoreline
  -> Mission planner extracts target, urgency, context, and operating mode
  -> Context planner prioritizes likely locations
  -> Sensor source provides imagery or video
  -> Proposal detector generates candidate observations
  -> Semantic scorer reviews crops and full frames
  -> Candidate ranking sorts the analyst queue
  -> Analyst approves, rejects, or investigates findings
  -> Mission report is generated
  -> Mission memory records patterns and weaknesses
```

## Current Capabilities

- Plain-English mission objective parsing
- Mission command generation with operating modes
- Contextual search priority planning
- Search mission state machines for simulated robotic workflows
- PX4/Gazebo helper scripts for drone-based validation
- Fast dashboard simulation for command, telemetry, alerts, and logs
- Image and video mission evaluation
- Vision benchmark suite across mission types
- Color, high-recall, objectness, vehicle, and optional YOLO proposal detection
- Local semantic scoring interface
- Optional local CLIP open-vocabulary semantic scoring (offline, `pip install '.[ml]'`, see `docs/LEARNED_MODELS.md`)
- Optional OpenAI vision-language scoring backend
- Acoustic `.wav` ingestion, spectrogram generation, and anthropogenic proposal triage
- Multi-sensor shoreline demo combining RGB, infrared, and acoustic evidence
- Full-frame fallback review for detector misses and rejected crops
- IoU-based localization metrics (precision/recall/F1, mean IoU, AP) alongside capture metrics via a `gt_boxes` labels column
- Multi-frame contact tracking for video missions (one contact per track, not per frame)
- Pixel-to-ground georeferencing for camera detections (NED + lat/lon)
- Candidate ranking with review-priority explanations
- Analyst dashboard for reviewing candidates, metrics, reports, and mission memory
- Mission memory summaries from past reports and analyst decisions
- JSON and HTML mission reports
- Structured logs, debug images, candidate crops, and review files
- Safety-oriented mission concepts: return-home, geofence, abort, manual override, and link-loss policy

## Analyst Dashboard

Run:

```bash
./scripts/run_analyst_dashboard.sh
```

Open:

```text
http://localhost:8010
```

The dashboard provides:

- saved mission and vision reports
- mission planning from a plain-English request
- candidate queue and shortlist review
- confidence scores and review-priority reasons
- precision, recall, F1, and capture-recall metrics
- approve, reject, and investigate review states
- mission memory from previous reports and analyst decisions

Review decisions are saved beside each report:

```text
candidate_reviews.json
```

Example decision record:

```json
{
  "candidate_id": "0042_shoreline_frame",
  "decision": "reject",
  "reason": "shoreline debris",
  "notes": "Bright clutter, no vessel structure visible",
  "updated_at": "2026-06-05T12:00:00Z"
}
```

## Benchmarking

The benchmark suite evaluates the system across mission contexts and sensor modalities, not just one detector task. The current headline numbers are the leakage-controlled estimates in the Benchmark Snapshot near the top of this README.

Historical single-sample and API-review reports remain in `docs/` for traceability, but they are not the current generalization claim. Use the cross-validated development estimates for the public benchmark story until the protected lockboxes are evaluated.

Current benchmark lessons:

- full-frame fallback significantly improved target capture
- RGB vehicle evidence is captured reliably by local proposals; API review can be useful as selective cleanup, but API sample runs are not the headline benchmark estimate
- IR vehicle evidence currently performs best with local hot-blob triage; thermal API review did not improve confirmed accuracy in a small 30-image probe
- thermal hard negatives and a trained thermal detector are the likely long-term fix for IR false positives
- SeaDronesSee maritime SAR is recall-only until no-target maritime negatives are added

### Mixed-Set Demo Probes

For demo recording, Aegis was also run on balanced 40-image mixed vehicle sets with 20 positives and 20 negatives. These are not cross-validation headline numbers; they are small, interpretable viewing probes where precision is meaningful because negatives are present.

| Probe | Confirmed Precision | Confirmed Recall | Confirmed F1 | Capture Precision | Capture Recall |
|---|---:|---:|---:|---:|---:|
| RGB mixed 40 | 64.5% | 100.0% | 78.4% | 50.0% | 100.0% |
| IR mixed 40 | 52.6% | 100.0% | 69.0% | 50.0% | 100.0% |

The RGB demo run shows cleaner ranking behavior: strong positives tend to rank high and many negatives sink into lower-priority `NEEDS_REVIEW` fallback. The IR demo run preserves target evidence but is weaker at confirmed discrimination because warm non-vehicle thermal structures can resemble vehicle-like hot blobs.

### Cascade Ranking Test

A separate cascade ranking/efficiency test uses the mixed RGB set to test ranking quality rather than mission behavior. The local layer ranks all 40 images, then the API confirmation layer walks the ranked list in order for a planted target and stops at the first claimed find. Correctness is checked against the hidden ground-truth target image, not against API confidence.

Across five planted targets:

- 3 were correctly found in the top few ranks
- median API calls was 2 instead of reviewing all 40 images
- 1 trial stopped early on a similar-looking false positive
- 1 trial missed the target even though the true target was ranked 4th

That result is useful because it shows both the efficiency upside and the failure modes. The API can make wrong early stops, and the local ranker can surface a target that the API still fails to confirm. That is why Aegis keeps local ranking, API review, and human analyst decisions as separate layers.

### IR Recalibration Note

IR confirmed-match thresholds were tightened after diagnosing thermal false positives, then validated through the held-out cross-validation harness. The change modestly improved confirmed precision from 91.8% to 92.5% while confirmed recall moved from 100.0% to 99.6%. This is a scoring recalibration, not a solved thermal detector; trained thermal detection and broader hard negatives remain future work.

Detailed benchmark reports and commands:

```text
docs/RUNNING_BENCHMARKS.md
docs/AEGIS_MISSION_INTELLIGENCE_V1.md
docs/ACOUSTIC_INTELLIGENCE_ROADMAP.md
docs/ACOUSTIC_BENCHMARK_V1_SNIPPET.md
docs/ACOUSTIC_BENCHMARK_TUNING_REPORT.md
docs/MULTISENSOR_SHORELINE_DEMO.md
docs/SYSTEM_BENCHMARK_V1_REPORT.md
docs/SARD_BENCHMARK_REPORT.md
docs/VEHICLE_BENCHMARK_REPORT.md
docs/DRONEVEHICLE_BENCHMARK_ANALYSIS.md
docs/DRONEVEHICLE_VISUAL_CROSS_VALIDATION_REPORT.md
docs/DRONEVEHICLE_RGB_BENCHMARK_REPORT.md
docs/DRONEVEHICLE_IR_BENCHMARK_REPORT.md
docs/DRONEVEHICLE_RGB_API_BENCHMARK_REPORT.md
docs/DRONEVEHICLE_IR_API_BENCHMARK_REPORT.md
docs/IR_LAYER_RECALIBRATION_REPORT.md
docs/LINKEDIN_POST_AEGIS_VEHICLE_MODALITY_BENCHMARK.md
docs/PORTFOLIO_AEGIS_MODALITY_INTELLIGENCE.md
```

Current multi-sensor boundary:

```text
Aegis Vision Intelligence
+ Aegis Infrared Intelligence
+ Aegis Acoustic Intelligence
```

Aegis now includes acoustic evidence as a first-class non-visual modality, not just a future roadmap item.

Phase 1/2 acoustic evidence support includes `.wav` ingestion, spectrogram generation, high-energy and anthropogenic acoustic proposals, candidate JSON, and a simple acoustic report.

The analyst dashboard can review acoustic candidates with spectrograms, time ranges, proposal scores, reasons, review priority, and approve/reject/investigate decisions.

The first multi-sensor demo runner combines one RGB image set, one IR image set, and one acoustic recording into a unified candidate list and mission report.

## Mission Memory

Mission memory reads past reports and analyst review decisions to summarize what the platform is learning operationally.

Example shape:

```json
{
  "recurring_false_positives": ["grass", "grey objects", "white vehicles"],
  "recurring_misses": ["partially hidden person", "small distant vehicle"],
  "common_false_positive_causes": ["vegetation", "shadow", "debris"],
  "common_uncertainty_causes": ["too small"],
  "sensor_modality_lessons": [
    "RGB vehicle evidence benefits from selective API semantic review.",
    "Infrared vehicle evidence currently performs better with local hot-blob triage."
  ],
  "weak_categories": ["boats", "smoke", "signals"],
  "recommended_data": ["shoreline vessel imagery", "hard-negative smoke/fog examples"]
}
```

This is not model training yet. It is operational learning: the system records where it is weak, what it tends to over-prioritize, which analyst reason tags keep appearing, and what benchmark data should be collected next.

## Mission Evaluation

Run the full mission-intelligence loop over image or video evidence:

```bash
./scripts/run_mission_evaluation.sh "/path/to/images" \
  --mission-request "Search the shoreline for a missing person wearing an orange life vest" \
  --labels-csv "/path/to/labels.csv"
```

The report combines mission command parsing, contextual search priorities, vision planning, candidate detection, semantic scoring, evaluation metrics, and stage health.

Each mission report includes:

- mission objective
- search area
- evidence collected
- candidates found
- analyst decisions
- performance metrics
- mission memory
- recommendations

Detailed local/API benchmark commands are in [docs/RUNNING_BENCHMARKS.md](docs/RUNNING_BENCHMARKS.md).

## Optional Vision-Language Scoring

The local semantic scorer is intentionally conservative. It ranks candidates but does not claim exact arbitrary object recognition. For stronger open-vocabulary testing, the project supports an optional OpenAI-backed vision scorer. Setup and benchmark commands are documented in [docs/RUNNING_BENCHMARKS.md](docs/RUNNING_BENCHMARKS.md).

## Simulation And Drone Validation

The lightweight dashboard simulation:

```bash
python3 server.py
```

Open:

```text
http://localhost:8000
```

PX4 remains responsible for low-level stabilization and flight control. Mission logic should call controller interfaces, not publish raw flight-control messages directly.

PX4/Gazebo setup details live in [docs/PX4_GAZEBO_SETUP.md](docs/PX4_GAZEBO_SETUP.md).

## Technology Stack

- Python
- OpenCV
- HTML/CSS/JavaScript dashboard
- JSON/CSV mission logs and reports
- PX4/Gazebo validation path
- ROS 2 integration path
- Optional OpenAI vision-language scoring

Core modules live in `autonomy/`:

- `mission_command.py`: mission command and operating-mode policy
- `mission_objective.py`: plain-English objective parsing
- `contextual_search_plan.py`: contextual priority planning
- `mission_vision_plan.py`: mission-specific perception plan
- `vision_lab.py`: image/video benchmark runner
- `semantic_vision.py`: semantic scoring interface
- `mission_evaluation.py`: full mission evaluation pipeline
- `mission_benchmark_suite.py`: benchmark suite runner
- `mission_memory.py`: report and analyst-review memory
- `acoustic_intelligence.py`: WAV ingestion, spectrograms, acoustic proposals, and reports
- `multisensor_mission_demo.py`: RGB, IR, and acoustic mission demo report
- `world_model.py`: local grid map of searched cells, candidates, and confidence
- `px4_controller_interface.py`: ROS 2/PX4 Offboard wrapper

## Tests

The repository currently has 40+ Python test files and a GitHub Actions CI workflow. Run the main mission-intelligence checks:

```bash
python3 tests/test_mission_memory.py
python3 tests/test_mission_evaluation.py
python3 tests/test_vehicle_proposal_layer.py
python3 tests/test_multisensor_mission_demo.py
```

## Roadmap

Recently completed:

- IoU localization metrics (`autonomy/detection_metrics.py`) reported alongside capture metrics
- multi-frame contact tracking for video missions (`autonomy/contact_tracker.py`)
- local CLIP semantic scorer and YOLO proposal detector as optional `[ml]` extras (`docs/LEARNED_MODELS.md`)
- pixel-to-ground georeferencing (`autonomy/georeference.py`)
- closed-loop PX4/Gazebo runbook and chained launcher (`docs/CLOSED_LOOP_DEMO_RUNBOOK.md`, `scripts/run_closed_loop_demo.sh`)

Near-term:

- record and link the demo video (`docs/DEMO_VIDEO_PLAN.md`) and add a dashboard screenshot to this README
- collect broader labeled datasets for boats, debris, signals, fire/smoke, structure damage, and animals
- improve candidate ranking to reduce noisy review items while preserving capture recall
- improve analyst review workflow and report browsing, including a collapsed low-priority review group
- keep expanding mission memory into practical recommendations
- wire georeferenced contact locations into live search missions and the dashboard
- add SeaDronesSee no-target maritime negatives so maritime SAR precision can be measured
- collect thermal hard negatives and train or fine-tune a thermal vehicle detector
- rerun small, held-out IR API probes only after thermal-specific prompts or thresholds are frozen

Medium-term:

- keep the canonical project/repository name as `Aegis`
- add cleaner sensor abstraction for folders, videos, live cameras, drone cameras, hydrophones, and telemetry feeds
- support disconnected/edge collection with host-side semantic review after reconnect or return
- add richer report export for portfolio/demo use

Architecture details and longer-term dataset priorities are tracked in:

```text
docs/MISSION_INTELLIGENCE_ROADMAP.md
```

## Safety

This project is simulation-first. For any future hardware work, start with bench tests, props-off tests, tethered hover, manual flight, assisted waypoint flight, and only then controlled autonomous tests in a legal area with permission.

Keep human override, logging, geofencing, and return-home behavior in the loop.
