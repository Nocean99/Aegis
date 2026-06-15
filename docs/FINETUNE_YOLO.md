# Fine-Tuning YOLO on a Maritime Dataset

Runbook: fine-tune YOLOv8n for aerial boat detection and benchmark it
against the pretrained baseline using this project's two-view methodology
(frame capture + IoU localization). Produces the resume sentence honestly:
*"fine-tuned and benchmarked a detector against a pretrained baseline using
my own two-view evaluation harness."*

Everything ML-heavy is behind the `[ml]` extras; the dataset conversion and
benchmark report code run (and are unit-tested) without them.

## 1. Get the dataset

Roboflow's **Aerial Maritime Drone** dataset: small (74 images), aerial,
boats/docks/lifts — purpose-sized for a CPU/M-series fine-tune.

1. Create a free account at https://universe.roboflow.com
2. Search "Aerial Maritime Drone Dataset", open the project
3. Download → Format: **YOLOv8** → unzip to `datasets/aerial_maritime/`

Expected layout (Roboflow standard):

```text
datasets/aerial_maritime/
  data.yaml
  train/images  train/labels
  valid/images  valid/labels
  test/images   test/labels
```

`datasets/` is gitignored — check the dataset's license page before any
redistribution, same policy as the other benchmark data.

## 2. Install the ML extras

```bash
pip install '.[ml]'
```

## 3. Convert the test split into the project's evaluation format

```bash
python3 -m autonomy.finetune_yolo to-labels-csv datasets/aerial_maritime/test \
  --output datasets/aerial_maritime/test_labels.csv
```

This writes the standard labels CSV (filename, label, `gt_boxes`) so the
external dataset flows through the same `detection_metrics` harness as every
other benchmark in the repo.

## 4. Baseline before training

```bash
python3 -m autonomy.finetune_yolo benchmark \
  --weights yolov8n.pt \
  --images datasets/aerial_maritime/test/images \
  --labels-csv datasets/aerial_maritime/test_labels.csv \
  --output-dir logs/finetune_yolo/baseline
```

Expect mediocre numbers — COCO's "boat" class from satellite-ish nadir
views is exactly the domain gap fine-tuning exists to close. Save this
report; it's the "before" picture.

## 5. Fine-tune

```bash
python3 -m autonomy.finetune_yolo train datasets/aerial_maritime/data.yaml \
  --base-weights yolov8n.pt --epochs 50 --imgsz 640
```

~50 epochs on 74 images takes minutes on an M-series Mac (ultralytics uses
the `mps` device automatically), longer but fine on CPU. Best weights land
in `logs/finetune_yolo/train*/weights/best.pt`.

## 6. The head-to-head

```bash
python3 -m autonomy.finetune_yolo benchmark \
  --weights yolov8n.pt logs/finetune_yolo/train/weights/best.pt \
  --images datasets/aerial_maritime/test/images \
  --labels-csv datasets/aerial_maritime/test_labels.csv \
  --output-dir logs/finetune_yolo/head_to_head
```

Output: `finetune_benchmark.json` + `finetune_benchmark.md` with one row per
model across both views:

| Model | Capture P | Capture R | Capture F1 | Loc P | Loc R | Loc F1 | Mean IoU | AP@0.5 |
|---|---|---|---|---|---|---|---|---|
| yolov8n.pt | … | … | … | … | … | … | … | … |
| best.pt | … | … | … | … | … | … | … | … |

## 7. Reading the result honestly

- The interesting comparison is the *gap between views*: pretrained COCO
  weights often capture frames (something fires somewhere) while failing
  localization (wrong box, wrong object). Fine-tuning should close the
  localization gap most.
- 74 images is a small test set — report the raw counts (TP/FP/FN), not
  just ratios, and don't claim a half-point of F1 as signal.
- Watch for the failure mode too: a small fine-tune can overfit to docks
  and lose generic-boat recall. If localization improves but capture recall
  drops, say so — that trade-off is the finding.
- The fine-tuned weights drop straight into the pipeline:
  `--proposal-mode yolo` with `YoloProposalDetector(weights="...best.pt")`,
  or pass the weights path through your own runner.
