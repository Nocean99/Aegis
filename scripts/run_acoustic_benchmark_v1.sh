#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-/Users/noah/Downloads/dataset_final}"

python3 -m autonomy.acoustic_benchmark \
  "$DATASET_ROOT" \
  --benchmark-root benchmark_data/acoustic_v1 \
  --output-dir logs/acoustic_benchmark_v1 \
  --sample-limit 20
