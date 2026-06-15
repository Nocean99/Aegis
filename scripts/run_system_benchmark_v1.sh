#!/usr/bin/env bash
set -euo pipefail

python3 -m autonomy.system_benchmark \
  --missions-root benchmark_data/missions \
  --output-dir logs/system_benchmark_v1
