from __future__ import annotations

"""Standard IoU-based detection metrics.

The capture metrics elsewhere in this project measure "did the target reach
analyst review," which is intentionally biased toward recall. This module adds
the conventional computer-vision view: box-level precision/recall/F1 at an IoU
threshold, mean IoU over matched boxes, and average precision (AP) from
score-ranked detections. Use both views together: capture metrics describe the
review workflow, localization metrics describe detector quality.

Ground-truth boxes are provided per image via an optional ``gt_boxes`` column
in the labels CSV, formatted as ``x,y,w,h`` with multiple boxes separated by
``;`` (e.g. ``"10,20,30,40;100,120,25,25"``). An empty value means the image
contains no target.
"""

from dataclasses import dataclass


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class LocalizationMetrics:
    labeled_images: int
    ground_truth_boxes: int
    predicted_boxes: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    mean_iou: float
    average_precision: float
    iou_threshold: float

    def as_dict(self) -> dict:
        return {
            "metric_mode": "iou_localization",
            "iou_threshold": self.iou_threshold,
            "labeled_images": self.labeled_images,
            "ground_truth_boxes": self.ground_truth_boxes,
            "predicted_boxes": self.predicted_boxes,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mean_iou": round(self.mean_iou, 4),
            "average_precision": round(self.average_precision, 4),
        }


def iou(box_a: Box, box_b: Box) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    intersection = float((right - left) * (bottom - top))
    union = float(aw * ah + bw * bh) - intersection
    return intersection / union if union > 0 else 0.0


def parse_gt_boxes(value: str | None) -> list[Box]:
    """Parse 'x,y,w,h;x,y,w,h' (spaces tolerated) into a list of boxes."""
    if not value or not str(value).strip():
        return []
    boxes: list[Box] = []
    for chunk in str(value).replace("|", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part for part in chunk.replace(",", " ").split() if part]
        if len(parts) != 4:
            raise ValueError(f"Ground-truth box must have 4 values (x,y,w,h), got: {chunk!r}")
        x, y, w, h = (int(round(float(part))) for part in parts)
        if w <= 0 or h <= 0:
            raise ValueError(f"Ground-truth box must have positive size, got: {chunk!r}")
        boxes.append((x, y, w, h))
    return boxes


def evaluate_localization(
    items: list[dict],
    *,
    iou_threshold: float = 0.5,
) -> LocalizationMetrics:
    """Evaluate predicted boxes against ground truth.

    Each item is a dict with:
      - ``gt_boxes``: list of (x, y, w, h) ground-truth boxes (may be empty)
      - ``pred_boxes``: list of (box, score) predictions (may be empty)

    Matching is greedy per image: predictions are sorted by score, each
    prediction consumes the best remaining ground-truth box if IoU clears the
    threshold. AP is computed from the global score-ranked prediction list
    (area under the precision-recall curve, all-point interpolation).
    """
    total_gt = 0
    total_pred = 0
    true_positive = 0
    matched_ious: list[float] = []
    ranked: list[tuple[float, bool]] = []  # (score, is_true_positive)

    for item in items:
        gt_boxes = list(item.get("gt_boxes") or [])
        pred_boxes = sorted(item.get("pred_boxes") or [], key=lambda pair: -float(pair[1]))
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        unmatched = list(range(len(gt_boxes)))
        for box, score in pred_boxes:
            best_index = -1
            best_iou = 0.0
            for gt_index in unmatched:
                value = iou(tuple(box), tuple(gt_boxes[gt_index]))
                if value > best_iou:
                    best_iou = value
                    best_index = gt_index
            if best_index >= 0 and best_iou >= iou_threshold:
                unmatched.remove(best_index)
                true_positive += 1
                matched_ious.append(best_iou)
                ranked.append((float(score), True))
            else:
                ranked.append((float(score), False))

    false_positive = total_pred - true_positive
    false_negative = total_gt - true_positive
    precision = true_positive / total_pred if total_pred else 0.0
    recall = true_positive / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    average_precision = _average_precision(ranked, total_gt)
    return LocalizationMetrics(
        labeled_images=len(items),
        ground_truth_boxes=total_gt,
        predicted_boxes=total_pred,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=mean_iou,
        average_precision=average_precision,
        iou_threshold=iou_threshold,
    )


def localization_items_from_results(results: list[dict]) -> list[dict]:
    """Build evaluation items from vision-lab result dicts.

    Only results whose label includes ``gt_boxes`` participate. The pipeline
    currently emits at most one predicted box per frame; full-frame fallbacks
    (bbox is None) contribute no predicted box, so they count against recall
    here even though they are preserved for analyst review. That asymmetry is
    deliberate: this metric measures localization, not review capture.
    """
    items: list[dict] = []
    for result in results:
        label = result.get("label")
        if not isinstance(label, dict) or "gt_boxes" not in label:
            continue
        pred_boxes: list[tuple[Box, float]] = []
        bbox = result.get("bbox")
        if result.get("detected") and bbox is not None:
            score = float(
                result.get("final_score")
                or result.get("detector_confidence")
                or 0.0
            )
            pred_boxes.append((tuple(bbox), score))
        items.append({"gt_boxes": label["gt_boxes"], "pred_boxes": pred_boxes})
    return items


def _average_precision(ranked: list[tuple[float, bool]], total_gt: int) -> float:
    if total_gt <= 0 or not ranked:
        return 0.0
    ranked = sorted(ranked, key=lambda pair: -pair[0])
    cumulative_tp = 0
    points: list[tuple[float, float]] = []  # (recall, precision)
    for index, (_, is_tp) in enumerate(ranked, start=1):
        if is_tp:
            cumulative_tp += 1
        points.append((cumulative_tp / total_gt, cumulative_tp / index))
    # All-point interpolation: precision envelope from the right.
    best_precision = 0.0
    envelope: list[tuple[float, float]] = []
    for recall_value, precision_value in reversed(points):
        best_precision = max(best_precision, precision_value)
        envelope.append((recall_value, best_precision))
    envelope.reverse()
    area = 0.0
    previous_recall = 0.0
    for recall_value, precision_value in envelope:
        area += (recall_value - previous_recall) * precision_value
        previous_recall = recall_value
    return max(0.0, min(1.0, area))
