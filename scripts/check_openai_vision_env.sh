#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "OpenAI vision encoder environment check"
echo

missing=0

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo "[ok] OPENAI_API_KEY is set"
else
  echo "[missing] OPENAI_API_KEY is not set"
  missing=1
fi

if [[ -n "${OPENAI_VISION_MODEL:-}" ]]; then
  echo "[ok] OPENAI_VISION_MODEL=${OPENAI_VISION_MODEL}"
else
  echo "[missing] OPENAI_VISION_MODEL is not set"
  missing=1
fi

if [[ "$missing" -eq 0 ]]; then
  python3 - <<'PY'
import os
import sys

try:
    from autonomy.semantic_vision import OpenAIVisionLanguageScorer
except Exception as exc:
    print(f"[missing] Could not import OpenAIVisionLanguageScorer: {exc}")
    sys.exit(1)

try:
    OpenAIVisionLanguageScorer(detail=os.environ.get("OPENAI_IMAGE_DETAIL", "auto"))
except Exception as exc:
    print(f"[missing] Could not initialize OpenAI vision scorer: {exc}")
    sys.exit(1)

print("[ok] OpenAI vision scorer can initialize")
PY
else
  python3 - <<'PY'
import sys

try:
    from autonomy.semantic_vision import OpenAIVisionLanguageScorer  # noqa: F401
except Exception as exc:
    print(f"[missing] Could not import OpenAIVisionLanguageScorer: {exc}")
    sys.exit(1)

print("[ok] OpenAI vision scorer import works")
PY
fi

echo
if [[ "$missing" -ne 0 ]]; then
  echo "Set these before running the AI vision benchmark:"
  echo '  export OPENAI_API_KEY="..."'
  echo '  export OPENAI_VISION_MODEL="your-vision-capable-model"'
  exit 1
fi

echo "Ready for AI vision benchmark runs."
