"""Folded page detector — finds folded corners and edges using triangle anomaly detection."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import cv2

from .base import BaseDetector
from ..models.schemas import PageIssue, IssueType, Severity, BoundingBox
from config.settings import settings

# Corner names and the search region offsets (row_start, row_end, col_start, col_end as ratios)
CORNERS = {
    "top-left":     (0.0,  None, 0.0,  None),
    "top-right":    (0.0,  None, None, 1.0),
    "bottom-left":  (None, 1.0,  0.0,  None),
    "bottom-right": (None, 1.0,  None, 1.0),
}


class FoldedPageDetector(BaseDetector):
    """Detects folded page corners by looking for triangular bright/dark anomalies."""

    def __init__(self) -> None:
        super().__init__("folded_page")
        self.corner_ratio = settings.fold_corner_search_ratio
        self.min_triangle_area = settings.fold_triangle_area_min

    def _extract_corner(self, image: np.ndarray, corner: str) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        ch = int(h * self.corner_ratio)
        cw = int(w * self.corner_ratio)

        rects = {
            "top-left":     (0,      ch,     0,      cw),
            "top-right":    (0,      ch,     w - cw, w),
            "bottom-left":  (h - ch, h,      0,      cw),
            "bottom-right": (h - ch, h,      w - cw, w),
        }
        r1, r2, c1, c2 = rects[corner]
        return image[r1:r2, c1:c2], (r1, r2, c1, c2)

    def _detect_fold_in_corner(self, corner_img: np.ndarray) -> float:
        """Return confidence (0-1) that a fold is present. Uses contour triangle analysis."""
        gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY) if corner_img.ndim == 3 else corner_img
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_triangle_area:
                continue

            # Approximate contour and check for triangular shape (3 vertices)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) == 3:
                # Score by how well it fills the corner region
                region_area = corner_img.shape[0] * corner_img.shape[1]
                confidence = min(1.0, area / (region_area * 0.5))
                return confidence

        return 0.0

    def detect(self, page_image: np.ndarray, page_number: int, **kwargs) -> List[PageIssue]:
        issues: List[PageIssue] = []

        for corner_name in CORNERS:
            corner_img, (r1, r2, c1, c2) = self._extract_corner(page_image, corner_name)
            confidence = self._detect_fold_in_corner(corner_img)

            if confidence < 0.3:
                continue

            severity = Severity.critical if confidence > 0.7 else Severity.warning
            issues.append(PageIssue(
                type=IssueType.folded,
                severity=severity,
                description=f"Folded corner detected at {corner_name} (confidence {confidence:.0%}).",
                location=corner_name,
                confidence=confidence,
                recommended_action="Unfold the page and rescan. Content may be obscured under the fold.",
                bounding_box=BoundingBox(x1=c1, y1=r1, x2=c2, y2=r2),
            ))

        return issues
