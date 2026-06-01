"""Annotates page images with bounding boxes and severity overlays for the QA report."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2

from ..models.schemas import PageIssue, Severity

SEVERITY_COLORS: dict = {
    Severity.critical: (0, 0, 220),    # Red (BGR)
    Severity.warning:  (0, 165, 255),  # Orange
    Severity.info:     (255, 200, 0),  # Cyan-ish
}


class PageAnnotator:
    """Draws issue bounding boxes and labels onto page images."""

    def annotate(
        self,
        page_image: np.ndarray,
        issues: List[PageIssue],
        page_number: int,
    ) -> np.ndarray:
        annotated = page_image.copy()
        h, w = annotated.shape[:2]

        for issue in issues:
            color = SEVERITY_COLORS.get(issue.severity, (128, 128, 128))

            if issue.bounding_box is not None:
                bb = issue.bounding_box
                x1, y1 = int(bb.x1), int(bb.y1)
                x2, y2 = int(bb.x2), int(bb.y2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{issue.type.value} [{issue.severity.value}]"
                self._draw_label(annotated, label, x1, y1, color)
            else:
                # Full-page overlay strip at top
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (w, 30), color, -1)
                cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
                label = f"P{page_number}: {issue.type.value} [{issue.severity.value}]"
                cv2.putText(annotated, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def _draw_label(
        self, image: np.ndarray, text: str, x: int, y: int, color: tuple
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.5, 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        # Background rectangle for readability
        cv2.rectangle(image, (x, max(0, y - th - 4)), (x + tw + 4, y), color, -1)
        cv2.putText(image, text, (x + 2, y - 2), font, scale, (255, 255, 255), thickness)

    def save_thumbnail(
        self,
        annotated: np.ndarray,
        output_path: Path,
        width: int = 400,
    ) -> Path:
        h, w = annotated.shape[:2]
        scale = width / w
        new_h = int(h * scale)
        thumbnail = cv2.resize(annotated, (width, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output_path), thumbnail)
        return output_path
