"""Skew and rotation detector — flags pages scanned at an angle."""
from __future__ import annotations
from typing import List
import numpy as np
import cv2

from .base import BaseDetector
from ..models.schemas import PageIssue, IssueType, Severity
from config.settings import settings


class SkewDetector(BaseDetector):
    """Calculates page skew angle using Hough line analysis on binarised text."""

    def __init__(self) -> None:
        super().__init__("skew")
        self.threshold_deg = settings.skew_threshold_degrees
        self.max_angle_deg = settings.skew_max_angle_degrees

    def _calculate_skew_angle(self, gray: np.ndarray) -> float:
        """Return skew angle in degrees (positive = clockwise tilt)."""
        # Otsu threshold to isolate dark content
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological closing to connect text into lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        coords = np.column_stack(np.where(connected > 0))
        if coords.shape[0] < 100:
            return 0.0

        angle = cv2.minAreaRect(coords)[-1]
        # minAreaRect returns angle in [-90, 0); normalise to [-45, 45)
        if angle < -45:
            angle += 90
        return -angle  # positive = clockwise

    def detect(self, page_image: np.ndarray, page_number: int, **kwargs) -> List[PageIssue]:
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        angle = self._calculate_skew_angle(gray)

        if abs(angle) <= self.threshold_deg:
            return []

        severity = Severity.critical if abs(angle) > 10.0 else Severity.warning
        direction = "clockwise" if angle > 0 else "counter-clockwise"

        return [PageIssue(
            type=IssueType.skew,
            severity=severity,
            description=f"Page is skewed {abs(angle):.2f}° {direction}.",
            location="full_page",
            confidence=min(1.0, abs(angle) / 15.0),
            recommended_action=(
                "Deskew page before OCR or archiving. "
                f"Rotate {abs(angle):.1f}° {'counter-clockwise' if angle > 0 else 'clockwise'} to correct."
            ),
        )]
