#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/load_project_env.sh

python3 -m autonomy.cascade_ranking_benchmark "$@"
