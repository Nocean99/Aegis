from __future__ import annotations

"""Learned proposal layer backed by a YOLO detector (ultralytics).

Drop-in alternative to the heuristic color/objectness/vehicle proposal layers,
selected with ``--proposal-mode yolo``. It runs fully offline once weights are
downloaded, and filters detections to COCO classes relevant to the mission's
extracted categories so a vessel mission is not flooded with traffic-light
boxes.

The ultralytics dependency is imported lazily (same optional pattern as the
PX4 interface and CLIP scorer). Install with:

    pip install '.[ml]'        # or: pip install ultralytics

A ``model`` object may be injected for tests; it must be callable like an
ultralytics model: ``model(frame, verbose=False)`` returning results with
``.boxes`` (xywh, conf, cls) and ``.names``.
"""

import numpy as np

from autonomy.types import TargetDetection


# Mission category -> COCO class names accepted as proposals.
CATEGORY_TO_COCO = {
    "person": {"person"},
    "vehicle": {"car", "truck", "bus", "motorcycle", "bicycle"},
    "boat": {"boat"},
    "aircraft": {"airplane"},
    # debris/signal have no COCO equivalent; fall through to "any class".
}


class YoloProposalDetector:
    def __init__(
        self,
        *,
        weights: str = "yolov8n.pt",
        confidence_threshold: float = 0.2,
        categories: list[str] | None = None,
        model=None,
    ) -> None:
        self.weights = weights
        self.confidence_threshold = confidence_threshold
        self.categories = list(categories or [])
        self._model = model

    def detect(self, frame_bgr: np.ndarray, *, modality: str = "rgb", allow_fallback: bool = True) -> TargetDetection:
        if frame_bgr is None or frame_bgr.size == 0:
            return TargetDetection(False, sensor_modality=modality)
        detections = self.detect_all(frame_bgr, modality=modality)
        if detections:
            return max(detections, key=lambda item: item.confidence)
        if allow_fallback:
            height, width = frame_bgr.shape[:2]
            return TargetDetection(
                True,
                confidence=0.2,
                bbox=None,
                center_px=(width // 2, height // 2),
                area_px=float(width * height),
                area_ratio=1.0,
                sensor_modality=modality,
                proposal_reason="yolo full-frame fallback (no class above threshold)",
            )
        return TargetDetection(False, sensor_modality=modality)

    def detect_all(self, frame_bgr: np.ndarray, *, modality: str = "rgb") -> list[TargetDetection]:
        """All class-filtered detections above threshold, for multi-box workflows."""
        model = self._load_model()
        results = model(frame_bgr, verbose=False)
        height, width = frame_bgr.shape[:2]
        frame_area = float(max(1, height * width))
        allowed = self._allowed_class_names()
        detections: list[TargetDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", {}) or {}
            if boxes is None:
                continue
            for xywh, confidence, class_id in zip(
                _to_numpy(boxes.xywh), _to_numpy(boxes.conf), _to_numpy(boxes.cls)
            ):
                confidence = float(confidence)
                if confidence < self.confidence_threshold:
                    continue
                class_name = str(names.get(int(class_id), int(class_id)))
                if allowed is not None and class_name not in allowed:
                    continue
                center_x, center_y, box_w, box_h = (float(v) for v in xywh)
                x = int(round(center_x - box_w / 2))
                y = int(round(center_y - box_h / 2))
                w = max(1, int(round(box_w)))
                h = max(1, int(round(box_h)))
                detections.append(
                    TargetDetection(
                        True,
                        confidence=confidence,
                        bbox=(x, y, w, h),
                        center_px=(int(center_x), int(center_y)),
                        area_px=float(w * h),
                        area_ratio=float(w * h) / frame_area,
                        sensor_modality=modality,
                        proposal_reason=f"yolo:{class_name} {confidence:.2f}",
                    )
                )
        return detections

    def mask(self, frame_bgr: np.ndarray, *, modality: str = "rgb") -> np.ndarray:
        """Filled detection boxes as an audit mask, matching other detectors."""
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        for detection in self.detect_all(frame_bgr, modality=modality):
            if detection.bbox is None:
                continue
            x, y, w, h = detection.bbox
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(mask.shape[1], x + w)
            y1 = min(mask.shape[0], y + h)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 255
        return mask

    def _allowed_class_names(self) -> set[str] | None:
        allowed: set[str] = set()
        unmapped_category = False
        for category in self.categories:
            names = CATEGORY_TO_COCO.get(category)
            if names is None:
                unmapped_category = True
            else:
                allowed.update(names)
        if not allowed or unmapped_category:
            # No mapped categories (or a category COCO can't express):
            # accept any class and let semantic scoring sort it out.
            return None
        return allowed

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO proposal mode requires ultralytics. Install with: pip install '.[ml]'"
            ) from exc
        self._model = YOLO(self.weights)
        return self._model


def _to_numpy(value) -> np.ndarray:
    if value is None:
        return np.zeros((0,))
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)
