#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-/Users/noah/Downloads/dataset_final}"

python3 -m autonomy.acoustic_cross_validation \
  "$DATASET_ROOT" \
  --output-dir logs/acoustic_cross_validation_v1 \
  --folds 5 \
  --seed 7 \
  --lockbox-fraction 0.2
