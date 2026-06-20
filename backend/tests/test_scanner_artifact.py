"""Unit tests for ScannerArtifactDetector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from backend.app.detectors.scanner_artifact import ScannerArtifactDetector


class TestScannerArtifactDetector:
    def setup_method(self):
        self.detector = ScannerArtifactDetector()

    def test_artifact_detected_across_pages(self, artifact_pages):
        # Feed pages through single-page detect first
        for pn, img in artifact_pages.items():
            self.detector.detect(img, pn)

        issues, health = self.detector.detect_batch(artifact_pages)
        # With same-position artifact on 3+ pages, expect at least one artifact flagged
        assert health["artifact_detected"] is True
        assert len(health["affected_pages"]) >= 1

    def test_clean_pages_produce_no_artifacts(self, clean_pages):
        self.detector = ScannerArtifactDetector()  # fresh instance
        for pn, img in clean_pages.items():
            self.detector.detect(img, pn)

        issues, health = self.detector.detect_batch(clean_pages)
        assert health["artifact_detected"] is False

    def test_cleaning_recommended_when_many_pages(self, artifact_pages):
        # Add more pages with the same artifact
        import numpy as np
        import cv2
        for i in range(4, 8):
            img = np.ones((1122, 794, 3), dtype=np.uint8) * 245
            cv2.line(img, (300, 0), (300, 1122), (80, 80, 80), 1)
            artifact_pages[i] = img

        self.detector = ScannerArtifactDetector()
        for pn, img in artifact_pages.items():
            self.detector.detect(img, pn)

        _, health = self.detector.detect_batch(artifact_pages)
        assert health["cleaning_recommended"] is True
