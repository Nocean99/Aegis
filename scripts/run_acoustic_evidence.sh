#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m autonomy.acoustic_intelligence "$@"
