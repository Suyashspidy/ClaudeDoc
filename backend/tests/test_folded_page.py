"""Unit tests for FoldedPageDetector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from backend.app.detectors.folded_page import FoldedPageDetector
from backend.app.models.schemas import IssueType


class TestFoldedPageDetector:
    def setup_method(self):
        self.detector = FoldedPageDetector()

    def test_folded_corner_detected(self, folded_page):
        issues = self.detector.detect(folded_page, page_number=3)
        # The bottom-right triangle fold should trigger detection
        fold_issues = [i for i in issues if i.type == IssueType.folded]
        assert len(fold_issues) >= 1

    def test_clean_page_no_fold(self, text_page):
        issues = self.detector.detect(text_page, page_number=1)
        assert issues == []

    def test_fold_location_reported(self, folded_page):
        issues = self.detector.detect(folded_page, page_number=3)
        for issue in issues:
            assert issue.location in {"top-left", "top-right", "bottom-left", "bottom-right"}

    def test_bounding_box_present(self, folded_page):
        issues = self.detector.detect(folded_page, page_number=3)
        for issue in issues:
            assert issue.bounding_box is not None
