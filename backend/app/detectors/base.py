"""Abstract base class for all document quality detectors."""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from ..models.schemas import PageIssue

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """All detectors implement this interface so the pipeline can run them uniformly."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(f"claudedoc.{name}")

    @abstractmethod
    def detect(self, page_image: np.ndarray, page_number: int, **kwargs) -> List[PageIssue]:
        """Analyse a single page image and return any detected issues.

        Args:
            page_image: BGR numpy array from OpenCV.
            page_number: 1-based page index.
            **kwargs: Detector-specific extra context.

        Returns:
            List of PageIssue objects (empty list = no issues found).
        """

    def _safe_detect(self, page_image: np.ndarray, page_number: int, **kwargs) -> List[PageIssue]:
        """Wraps detect() with error isolation so one detector can't crash the pipeline."""
        try:
            return self.detect(page_image, page_number, **kwargs)
        except Exception as exc:
            self.logger.error("Detector %s failed on page %d: %s", self.name, page_number, exc, exc_info=True)
            return []
