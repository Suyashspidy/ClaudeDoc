"""Blank page detector — identifies pages with no meaningful content."""
from __future__ import annotations
from typing import List
import numpy as np
import cv2

from .base import BaseDetector
from ..models.schemas import PageIssue, IssueType, Severity
from config.settings import settings


class BlankPageDetector(BaseDetector):
    """Detects blank or near-blank pages by analysing pixel density and std deviation."""

    def __init__(self) -> None:
        super().__init__("blank_page")
        self.density_threshold = settings.blank_page_pixel_density_threshold
        self.std_threshold = settings.blank_page_std_threshold

    def detect(self, page_image: np.ndarray, page_number: int, **kwargs) -> List[PageIssue]:
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)

        # Pixel density: ratio of dark pixels (< 200 intensity) to total pixels
        dark_pixels = np.sum(gray < 200)
        total_pixels = gray.size
        density = dark_pixels / total_pixels

        # Standard deviation of pixel intensities
        std_dev = float(np.std(gray))

        is_blank = density < self.density_threshold and std_dev < self.std_threshold

        if not is_blank:
            return []

        # Distinguish near-blank (might be intentional) from truly empty
        severity = Severity.warning if density > (self.density_threshold / 2) else Severity.info
        description = (
            f"Page appears blank (dark pixel density: {density:.2%}, "
            f"intensity std: {std_dev:.1f})"
        )

        return [PageIssue(
            type=IssueType.blank,
            severity=severity,
            description=description,
            location="full_page",
            confidence=min(1.0, (1.0 - density / self.density_threshold) * 0.9 + 0.1),
            recommended_action="Verify whether this page was intentionally blank; rescan if content expected.",
        )]
