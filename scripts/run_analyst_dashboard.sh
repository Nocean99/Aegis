#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 analyst_server.py "$@"
