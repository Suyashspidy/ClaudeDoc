"""Unit tests for SkewDetector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from backend.app.detectors.skew import SkewDetector
from backend.app.models.schemas import IssueType


class TestSkewDetector:
    def setup_method(self):
        self.detector = SkewDetector()

    def test_clean_page_not_flagged(self, text_page):
        issues = self.detector.detect(text_page, page_number=1)
        assert issues == []

    def test_skewed_page_detected(self, skewed_page):
        issues = self.detector.detect(skewed_page, page_number=1)
        assert len(issues) >= 1
        assert issues[0].type == IssueType.skew

    def test_skew_angle_in_description(self, skewed_page):
        issues = self.detector.detect(skewed_page, page_number=1)
        if issues:
            assert "°" in issues[0].description

    def test_confidence_range(self, skewed_page):
        issues = self.detector.detect(skewed_page, page_number=1)
        for issue in issues:
            assert 0.0 <= issue.confidence <= 1.0
