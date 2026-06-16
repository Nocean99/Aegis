#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-/Users/noah/Downloads/VisDrone-DroneVehicle}"
MAX_DEVELOPMENT_ROWS="${2:-500}"
OUTPUT_DIR="${3:-logs/visual_cross_validation/dronevehicle_ir}"

python3 -m autonomy.visual_cross_validation "$DATASET_ROOT" \
  --labels-csv datasets/benchmarks/vehicles/dronevehicle_ir_labels.csv \
  --output-dir "$OUTPUT_DIR" \
  --mission-request "Search infrared aerial imagery for vehicles including cars, trucks, vans, buses, and freight vehicles." \
  --modality infrared \
  --folds 5 \
  --seed 32 \
  --lockbox-fraction 0.2 \
  --max-development-rows "$MAX_DEVELOPMENT_ROWS"
