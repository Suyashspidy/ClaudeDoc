"""Unit tests for MissingPageDetector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from backend.app.detectors.missing_page import MissingPageDetector
from backend.app.models.schemas import IssueType, Severity


class TestMissingPageDetector:
    def setup_method(self):
        self.detector = MissingPageDetector()

    def test_metadata_mismatch_detected(self):
        issues = self.detector.detect_from_pdf_metadata(
            metadata_page_count=10, actual_page_count=8
        )
        assert len(issues) == 1
        assert issues[0].type == IssueType.missing
        assert issues[0].severity == Severity.critical

    def test_no_issue_when_counts_match(self):
        issues = self.detector.detect_from_pdf_metadata(
            metadata_page_count=5, actual_page_count=5
        )
        assert issues == []

    def test_no_issue_when_metadata_none(self):
        issues = self.detector.detect_from_pdf_metadata(
            metadata_page_count=None, actual_page_count=5
        )
        assert issues == []

    def test_sequence_gap_detected(self):
        # Physical pages 1-4 with detected numbers 1,2,5,6 (pages 3,4 missing)
        page_numbers = {1: 1, 2: 2, 3: 5, 4: 6}
        issues = self.detector.detect_sequence_gaps(page_numbers)
        assert len(issues) == 1
        assert "3" in issues[0].description or "4" in issues[0].description

    def test_no_gap_in_complete_sequence(self):
        page_numbers = {1: 1, 2: 2, 3: 3, 4: 4}
        issues = self.detector.detect_sequence_gaps(page_numbers)
        assert issues == []

    def test_missing_pages_5_and_6(self):
        """Matches the spec sample test case: PDF with missing pages 5 and 6."""
        page_numbers = {1: 1, 2: 2, 3: 3, 4: 4, 5: 7, 6: 8}
        issues = self.detector.detect_sequence_gaps(page_numbers)
        assert len(issues) == 1
        description = issues[0].description
        assert "5" in description and "6" in description
