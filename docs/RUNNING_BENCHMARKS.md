# Running Benchmarks

This page keeps the operational benchmark commands out of the main README.

## Benchmark Concepts

Aegis reports two kinds of benchmark metrics:

- **Confirmed-match metrics:** how often the system confidently identifies the target.
- **Analyst-capture metrics:** how often the system preserves the right evidence for review.

For mission workflows, missing a possible target is usually worse than showing an analyst a few extra uncertain images.

## Mission Evaluation

Run the full mission-intelligence loop over image or video evidence:

```bash
./scripts/run_mission_evaluation.sh "/path/to/images" \
  --mission-request "Search the shoreline for a missing person wearing an orange life vest" \
  --labels-csv "/path/to/labels.csv"
```

Run with OpenAI semantic vision:

```bash
./scripts/run_mission_evaluation.sh "/path/to/images" \
  --mission-request "Search the shoreline for a missing person wearing an orange life vest" \
  --labels-csv "/path/to/labels.csv" \
  --semantic-vision openai \
  --openai-detail high \
  --full-frame-semantic misses
```

Reports are written to:

```text
logs/mission_evaluations/<timestamp>/mission_evaluation_report.json
logs/mission_evaluations/<timestamp>/mission_evaluation_report.html
```

## Benchmark Suite

Run the configured suite:

```bash
./scripts/run_mission_benchmark_suite.sh
```

Suite reports are written to:

```text
logs/mission_benchmark_suites/<timestamp>/mission_benchmark_suite_report.json
logs/mission_benchmark_suites/<timestamp>/mission_benchmark_suite_report.html
```

## SAR People Dataset

Convert a YOLOv8 person dataset into Aegis benchmark labels:

```bash
./scripts/import_yolo_person_benchmark.sh "/path/to/yolo_dataset"
```

Output:

```text
datasets/benchmarks/people/sard_labels.csv
```

## DroneVehicle RGB/Infrared Dataset

Analyze and import the DroneVehicle RGB/infrared dataset:

```bash
./scripts/analyze_dronevehicle_benchmark.sh "/path/to/VisDrone-DroneVehicle"
./scripts/import_dronevehicle_vehicle_benchmark.sh "/path/to/VisDrone-DroneVehicle"
```

Outputs:

```text
datasets/benchmarks/vehicles/dronevehicle_rgb_labels.csv
datasets/benchmarks/vehicles/dronevehicle_ir_labels.csv
datasets/benchmarks/vehicles/dronevehicle_stats.json
docs/DRONEVEHICLE_BENCHMARK_ANALYSIS.md
```

## DroneVehicle Local RGB/IR Baselines

Create 500-image local subsets:

```bash
./scripts/create_benchmark_sample.sh "/path/to/VisDrone-DroneVehicle" \
  --labels-csv datasets/benchmarks/vehicles/dronevehicle_rgb_labels.csv \
  --output-dir logs/benchmark_samples/dronevehicle_rgb_local_500 \
  --max-images 500 \
  --seed 21

./scripts/create_benchmark_sample.sh "/path/to/VisDrone-DroneVehicle" \
  --labels-csv datasets/benchmarks/vehicles/dronevehicle_ir_labels.csv \
  --output-dir logs/benchmark_samples/dronevehicle_ir_local_500 \
  --max-images 500 \
  --seed 22
```

Run local-only RGB and IR evaluations:

```bash
./scripts/run_mission_evaluation.sh \
  logs/benchmark_samples/dronevehicle_rgb_local_500/images \
  --mission-request "Search aerial RGB imagery for vehicles relevant to incident response" \
  --labels-csv logs/benchmark_samples/dronevehicle_rgb_local_500/labels.csv \
  --semantic-vision local \
  --proposal-mode vehicle \
  --full-frame-semantic misses \
  --max-saved-candidates 500

./scripts/run_mission_evaluation.sh \
  logs/benchmark_samples/dronevehicle_ir_local_500/images \
  --mission-request "Search infrared aerial imagery for vehicles relevant to incident response" \
  --labels-csv logs/benchmark_samples/dronevehicle_ir_local_500/labels.csv \
  --semantic-vision local \
  --proposal-mode vehicle \
  --full-frame-semantic misses \
  --max-saved-candidates 500
```

## DroneVehicle API Review Samples

Do not run OpenAI/API review on the full DroneVehicle dataset. Use small review-priority samples.

RGB API review:

```bash
./scripts/run_mission_evaluation.sh \
  logs/benchmark_samples/dronevehicle_rgb_api_review_sample_100/images \
  --mission-request "Search these aerial RGB images for vehicles including cars, trucks, vans, buses, and freight vehicles." \
  --labels-csv logs/benchmark_samples/dronevehicle_rgb_api_review_sample_100/labels.csv \
  --semantic-vision openai \
  --openai-detail high \
  --proposal-mode vehicle \
  --full-frame-semantic misses \
  --max-saved-candidates 100 \
  --output-dir logs/mission_evaluations/dronevehicle_rgb_api_review_sample_100
```

IR API review:

```bash
./scripts/run_mission_evaluation.sh \
  logs/benchmark_samples/dronevehicle_ir_api_review_sample_100/images \
  --mission-request "Search these infrared aerial images for vehicles including cars, trucks, vans, buses, and freight vehicles." \
  --labels-csv logs/benchmark_samples/dronevehicle_ir_api_review_sample_100/labels.csv \
  --semantic-vision openai \
  --openai-detail auto \
  --proposal-mode vehicle \
  --full-frame-semantic misses \
  --max-saved-candidates 100 \
  --output-dir logs/mission_evaluations/dronevehicle_ir_api_review_sample_100_auto
```

## Acoustic Benchmark v1

Create a 60-clip underwater-noise benchmark and evaluate the acoustic proposal layer:

```bash
./scripts/run_acoustic_benchmark_v1.sh "/path/to/dataset_final"
```

Expected source folders:

```text
dataset_final/
  anthropogenic/
  animal/
  sonar/
```

Outputs:

```text
benchmark_data/acoustic_v1/benchmark.csv
logs/acoustic_benchmark_v1/acoustic_benchmark_report.json
logs/acoustic_benchmark_v1/acoustic_benchmark_report.html
docs/ACOUSTIC_BENCHMARK_V1_SNIPPET.md
docs/ACOUSTIC_BENCHMARK_TUNING_REPORT.md
```

## System Benchmark v1

Run the first mission-level benchmark:

```bash
./scripts/run_system_benchmark_v1.sh
```

Mission cases live in:

```text
benchmark_data/missions/
```

Each mission contains a `mission.json` file with RGB input, IR input, acoustic input, and expected contact outcome.

Outputs:

```text
logs/system_benchmark_v1/system_benchmark_report.json
logs/system_benchmark_v1/system_benchmark_report.html
```

System-level metrics:

- mission success rate
- contact precision
- contact recall

## OpenAI Vision Setup

The `.env` file is ignored by Git and should not be committed:

```bash
cp .env.example .env
```

Then set:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_VISION_MODEL=gpt-4o
OPENAI_IMAGE_DETAIL=auto
```

Check the environment:

```bash
./scripts/check_openai_vision_env.sh
```
