#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-/Users/noah/Downloads/VisDrone-DroneVehicle}"
MAX_DEVELOPMENT_ROWS="${2:-500}"
OUTPUT_DIR="${3:-logs/visual_cross_validation/dronevehicle_rgb}"

python3 -m autonomy.visual_cross_validation "$DATASET_ROOT" \
  --labels-csv datasets/benchmarks/vehicles/dronevehicle_rgb_labels.csv \
  --output-dir "$OUTPUT_DIR" \
  --mission-request "Search aerial RGB imagery for vehicles including cars, trucks, vans, buses, and freight vehicles." \
  --modality rgb \
  --folds 5 \
  --seed 31 \
  --lockbox-fraction 0.2 \
  --max-development-rows "$MAX_DEVELOPMENT_ROWS"
