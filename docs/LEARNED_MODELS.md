# Learned Models: CLIP Scoring and YOLO Proposals

The platform runs end-to-end with zero ML dependencies — heuristic proposal
layers plus the local semantic scorer. This document covers the optional
learned upgrades: a CLIP open-vocabulary semantic scorer and a YOLO proposal
detector. Both run fully offline (CPU or GPU) once weights are downloaded,
which matters for edge and disconnected missions. No cloud API is in the loop.

## Install

```bash
pip install '.[ml]'
# or directly:
pip install torch open-clip-torch pillow ultralytics
```

Everything else in the project continues to work without these packages. The
imports are lazy (the same optional pattern as the PX4 interface), so simply
having `--semantic-vision local` selected never touches torch.

## CLIP semantic scorer (`--semantic-vision clip`)

`autonomy/clip_semantic_scorer.py` implements the `SemanticVisionScorer`
protocol with open-vocabulary scoring:

1. The mission objective is expanded into positive prompts (target
   description, category phrasings, color phrasings).
2. The candidate crop is ranked against positive prompts plus a fixed set of
   aerial background prompts.
3. The softmax probability mass on positive prompts becomes the score, mapped
   onto the standard decision bands (LIKELY_MATCH >= 0.75, POSSIBLE_MATCH >=
   0.55, NEEDS_REVIEW >= 0.20, REJECT below).

Every result keeps `needs_human_review=True`: the scorer prioritizes, the
analyst decides.

Default model is `ViT-B-32` with `laion2b_s34b_b79k` weights (~600 MB
download on first use, then cached).

```bash
python3 -m autonomy.vision_lab \
  --request "find a red boat near the shoreline" \
  --source demo_data/shoreline_rgb \
  --semantic-vision clip
```

## YOLO proposal detector (`--proposal-mode yolo`)

`autonomy/yolo_proposal_detector.py` is a drop-in alternative to the
heuristic color/objectness/vehicle proposal layers. Detections are filtered
to COCO classes relevant to the mission's extracted categories (person,
vehicle, boat, aircraft); categories COCO cannot express (debris, signal)
accept any class and let semantic scoring sort it out. When nothing clears
the confidence threshold it emits the same full-frame fallback as the other
layers, so recall-first review behavior is preserved.

Default weights are `yolov8n.pt` (~6 MB, auto-downloaded on first use).

```bash
python3 -m autonomy.vision_lab \
  --request "find people near the waterline" \
  --source demo_data/shoreline_rgb \
  --proposal-mode yolo --semantic-vision clip
```

## Benchmarks with learned models

Both flags are wired through mission evaluation and the benchmark suite, so
you can compare heuristic and learned configurations on the same labeled
data:

```bash
python3 -m autonomy.mission_evaluation \
  --request "find vehicles" \
  --source benchmark_data/dronevehicle_rgb \
  --labels benchmark_data/dronevehicle_rgb/labels.csv \
  --proposal-mode yolo --semantic-vision clip

python3 -m autonomy.mission_benchmark_suite \
  --config benchmark_data/missions/system_benchmark_v1.json \
  --proposal-mode yolo --semantic-vision clip
```

For box-level quality (not just capture), add a `gt_boxes` column to the
labels CSV (`x,y,w,h` boxes separated by `;`). The report then includes an
`iou_localization` block: precision/recall/F1 at IoU 0.5, mean IoU, and
average precision. See `autonomy/detection_metrics.py` for the metric
definitions and why capture and localization views are reported separately.

## Testing without the ML stack

The unit tests inject fake encoders/models, so
`tests/test_clip_semantic_scorer.py` and
`tests/test_yolo_proposal_detector.py` pass without torch or ultralytics
installed — CI stays light while the integration paths stay covered.
