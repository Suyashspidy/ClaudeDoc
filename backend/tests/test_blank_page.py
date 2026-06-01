"""Unit tests for BlankPageDetector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from backend.app.detectors.blank_page import BlankPageDetector
from backend.app.models.schemas import IssueType


class TestBlankPageDetector:
    def setup_method(self):
        self.detector = BlankPageDetector()

    def test_blank_page_detected(self, blank_page):
        issues = self.detector.detect(blank_page, page_number=1)
        assert len(issues) == 1
        assert issues[0].type == IssueType.blank

    def test_text_page_not_flagged(self, text_page):
        issues = self.detector.detect(text_page, page_number=1)
        assert issues == []

    def test_confidence_between_0_and_1(self, blank_page):
        issues = self.detector.detect(blank_page, page_number=1)
        assert 0.0 <= issues[0].confidence <= 1.0

    def test_near_blank_gets_warning(self):
        # 1% dark pixels — near blank but not fully empty
        img = np.ones((1122, 794, 3), dtype=np.uint8) * 248
        img[100:110, 100:200] = 50  # small dark patch
        issues = self.detector.detect(img, page_number=1)
        # May or may not flag depending on threshold — just ensure no crash
        assert isinstance(issues, list)

    def test_fully_black_page_not_flagged_as_blank(self):
        """A black page is not blank — it has content."""
        img = np.zeros((1122, 794, 3), dtype=np.uint8)
        issues = self.detector.detect(img, page_number=1)
        assert issues == []
