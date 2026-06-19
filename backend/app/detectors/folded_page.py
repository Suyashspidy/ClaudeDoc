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

    # the page's outer corner, expressed as (row, col) within each corner crop
    _OUTER_PIXEL = {
        "top-left":     lambda h, w: (0,     0),
        "top-right":    lambda h, w: (0,     w - 1),
        "bottom-left":  lambda h, w: (h - 1, 0),
        "bottom-right": lambda h, w: (h - 1, w - 1),
    }

    def __init__(self) -> None:
        super().__init__("folded_page")
        self.corner_ratio = settings.fold_corner_search_ratio
        self.min_triangle_area = settings.fold_triangle_area_min
        self.deviation = settings.fold_deviation_threshold

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

    def _detect_fold_in_corner(self, corner_img: np.ndarray, page_bg: float, corner_name: str) -> float:
        """Return confidence (0-1) that a fold is present in this corner.

        A folded corner leaves a triangular region — the revealed page back or its
        cast shadow — anchored at the page's outer corner, whose intensity differs
        markedly from the page background. We threshold on that *deviation* (so dark
        folds, which `RETR_EXTERNAL` on a bright-pixel mask would miss as holes, are
        now the foreground), then keep contours that (a) are large enough, (b) touch
        the outer corner, and (c) are triangular (fill ~half their bounding box).
        """
        gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY) if corner_img.ndim == 3 else corner_img
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return 0.0

        # Pixels that deviate strongly from the page background, in either direction.
        diff = np.abs(gray.astype(np.int16) - int(round(page_bg))).astype(np.uint8)
        mask = (diff > self.deviation).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        oy, ox = self._OUTER_PIXEL[corner_name](h, w)
        region_area = h * w
        anchor_tol = 0.20 * max(h, w)
        best = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_triangle_area:
                continue

            # Must be anchored at the page's outer corner (a real corner fold is).
            pts = contour.reshape(-1, 2)  # (x, y)
            if np.min(np.hypot(pts[:, 0] - ox, pts[:, 1] - oy)) > anchor_tol:
                continue

            # Triangular, not a solid block (text line / full rectangle ~1.0) and
            # not a thin streak: a corner-fold triangle fills roughly half its bbox.
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw == 0 or bh == 0:
                continue
            rect_fill = area / float(bw * bh)
            if not (0.20 <= rect_fill <= 0.80):
                continue

            best = max(best, float(min(1.0, area / (region_area * 0.5))))

        return best

    def detect(self, page_image: np.ndarray, page_number: int, **kwargs) -> List[PageIssue]:
        issues: List[PageIssue] = []

        gray_full = (
            cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY) if page_image.ndim == 3 else page_image
        )
        page_bg = float(np.median(gray_full))  # dominant background intensity

        for corner_name in CORNERS:
            corner_img, (r1, r2, c1, c2) = self._extract_corner(page_image, corner_name)
            confidence = self._detect_fold_in_corner(corner_img, page_bg, corner_name)

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
