from __future__ import annotations

import cv2
import numpy as np

from autonomy.types import TargetDetection


class ObjectnessProposalDetector:
    """High-recall generic object proposal detector for category-only missions.

    This is not object recognition. It looks for compact non-background regions
    that are worth sending to a heavier semantic model.
    """

    def __init__(self, *, min_area_px: int = 90) -> None:
        self.min_area_px = min_area_px

    def detect(self, bgr_image: np.ndarray) -> TargetDetection:
        proposals = self.detect_all(bgr_image, max_regions=1)
        return proposals[0] if proposals else TargetDetection(False)

    def detect_all(self, bgr_image: np.ndarray, *, max_regions: int = 5) -> list[TargetDetection]:
        if bgr_image is None or bgr_image.size == 0:
            return []
        mask = self.mask(bgr_image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(bgr_image.shape[0] * bgr_image.shape[1])
        proposals: list[TargetDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            rect_area = float(w * h)
            area_ratio = rect_area / image_area
            if area_ratio > 0.55:
                continue
            compactness = area / rect_area
            size_score = min(area / max(self.min_area_px * 12, 1), 1.0)
            confidence = min(0.74, 0.18 + size_score * 0.28 + compactness * 0.22 + min(area_ratio / 0.12, 1.0) * 0.12)
            proposals.append(
                TargetDetection(
                    detected=True,
                    confidence=round(confidence, 3),
                    bbox=(x, y, w, h),
                    center_px=(int(x + w / 2), int(y + h / 2)),
                    area_px=area,
                    area_ratio=area_ratio,
                )
            )
        proposals.sort(key=lambda item: item.confidence, reverse=True)
        return proposals[:max_regions]

    def mask(self, bgr_image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        vegetation = ((hue >= 35) & (hue <= 95) & (sat >= 35) & (val >= 35)).astype(np.uint8) * 255
        non_vegetation = cv2.bitwise_not(vegetation)

        edges = cv2.Canny(gray, 65, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

        bright_or_dark = (((val > 165) | (val < 45)) & (sat < 210)).astype(np.uint8) * 255
        mask = cv2.bitwise_or(cv2.bitwise_and(non_vegetation, bright_or_dark), cv2.bitwise_and(non_vegetation, edges))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask
