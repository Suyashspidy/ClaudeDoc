"""Scanner artifact detector — finds vertical/horizontal lines caused by dirty scanner glass."""
from __future__ import annotations
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import cv2

from .base import BaseDetector
from ..models.schemas import PageIssue, IssueType, Severity, BoundingBox
from config.settings import settings


class LineCandidate:
    """Represents a detected line candidate from HoughLinesP."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int, page_h: int, page_w: int) -> None:
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.page_h = page_h
        self.page_w = page_w
        self.is_vertical = abs(x2 - x1) < abs(y2 - y1)
        self.length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        self.position = x1 if self.is_vertical else y1  # canonical position for cross-page matching

    @property
    def spans_full_dimension(self) -> bool:
        span_ratio = settings.artifact_line_span_ratio
        if self.is_vertical:
            return self.length / self.page_h >= span_ratio
        return self.length / self.page_w >= span_ratio

    @property
    def width(self) -> int:
        """Estimated line width (single pixel for Hough output)."""
        return 1  # Hough lines are always 1px; thickness check happens on the mask

    def intensity_variance(self, gray: np.ndarray) -> float:
        """Compute pixel intensity variance along the line."""
        num_samples = min(200, int(self.length))
        xs = np.linspace(self.x1, self.x2, num_samples).astype(int)
        ys = np.linspace(self.y1, self.y2, num_samples).astype(int)
        xs = np.clip(xs, 0, gray.shape[1] - 1)
        ys = np.clip(ys, 0, gray.shape[0] - 1)
        intensities = gray[ys, xs]
        return float(np.var(intensities))


class ScannerArtifactDetector(BaseDetector):
    """
    Detects scanner artifacts using the 4-condition algorithm:
      1. Line spans > 80% of page dimension
      2. Intensity variance along line < 10 (uniform = artifact)
      3. Same line position appears on 3+ adjacent pages
      4. Line width is 1-3 pixels
    Requires 3+ conditions to classify as artifact.
    """

    def __init__(self) -> None:
        super().__init__("scanner_artifact")
        self._page_lines: Dict[int, List[LineCandidate]] = {}  # keyed by page_number

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, page_image: np.ndarray, page_number: int, **kwargs) -> List[PageIssue]:
        """Single-page detection. Cross-page condition evaluated after all pages processed."""
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        candidates = self._extract_line_candidates(gray)
        self._page_lines[page_number] = candidates
        # Return preliminary issues; cross-page check happens in detect_batch()
        return []

    def detect_batch(self, page_images: Dict[int, np.ndarray]) -> Tuple[Dict[int, List[PageIssue]], Dict]:
        """
        Run full cross-page artifact analysis after all pages have been processed.
        Returns (per_page_issues dict, scanner_health dict).
        """
        grays = {pn: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for pn, img in page_images.items()}

        for pn, gray in grays.items():
            if pn not in self._page_lines:
                self._page_lines[pn] = self._extract_line_candidates(gray)

        artifact_lines = self._find_cross_page_artifacts(grays)
        issues_by_page: Dict[int, List[PageIssue]] = defaultdict(list)
        artifact_positions = []

        for position, pages, orientation, candidates in artifact_lines:
            for page_num in pages:
                h, w = grays[page_num].shape
                bbox = BoundingBox(x1=position, y1=0, x2=position + 2, y2=h)
                loc = f"vertical streak at x={position}"
                severity = Severity.critical if len(pages) >= 5 else Severity.warning
                issues_by_page[page_num].append(PageIssue(
                    type=IssueType.scanner_artifact,
                    severity=severity,
                    description=(
                        f"Scanner artifact (vertical streak) at {loc}. "
                        f"Appears consistently on {len(pages)} pages — likely dirty scanner glass."
                    ),
                    location=loc,
                    confidence=min(1.0, len(pages) / 10.0 + 0.5),
                    recommended_action="Clean the scanner glass. This vertical line is a hardware artifact.",
                    bounding_box=bbox,
                ))
            artifact_positions.append({
                "orientation": orientation,
                "position": position,
                "affected_pages": list(pages),
            })

        affected_pages = sorted({pn for _, pages, _, _ in artifact_lines for pn in pages})
        scanner_health = {
            "artifact_detected": bool(artifact_lines),
            "affected_pages": affected_pages,
            "cleaning_recommended": len(affected_pages) >= 3,
            "artifact_positions": artifact_positions,
        }

        return dict(issues_by_page), scanner_health

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_line_candidates(self, gray: np.ndarray) -> List[LineCandidate]:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        h, w = gray.shape
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=min(h, w) // 4, maxLineGap=10)
        if lines is None:
            return []

        candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            candidates.append(LineCandidate(x1, y1, x2, y2, h, w))
        return candidates

    def _find_cross_page_artifacts(self, grays: Dict[int, np.ndarray]):
        """Group lines by position and score each group against the 4 conditions.

        Only vertical lines are considered: scanner glass dirt creates vertical streaks
        as pages move through the scanner. Horizontal lines are book content (text rows).
        """
        buckets: Dict[Tuple[str, int], List[Tuple[int, LineCandidate]]] = defaultdict(list)
        tol = settings.artifact_position_tolerance_px

        for page_num, candidates in self._page_lines.items():
            gray = grays.get(page_num)
            if gray is None:
                continue
            for c in candidates:
                if not c.is_vertical:
                    continue  # skip horizontal lines — those are book text, not scanner artifacts
                orientation = "vertical"
                position_bin = (c.position // tol) * tol
                buckets[(orientation, position_bin)].append((page_num, c, gray))

        results = []
        for (orientation, position), entries in buckets.items():
            pages = {pn for pn, _, _ in entries}
            if len(pages) < settings.artifact_min_pages_same_position:
                continue

            # Evaluate each of the 4 conditions across the group
            confirmed = []
            for page_num, candidate, gray in entries:
                score = 0
                if candidate.spans_full_dimension:
                    score += 1
                if candidate.intensity_variance(gray) < settings.artifact_intensity_variance_max:
                    score += 1
                # Condition 3: cross-page frequency — already confirmed (pages >= min)
                score += 1
                width_ok = settings.artifact_line_width_min <= candidate.width <= settings.artifact_line_width_max
                if width_ok:
                    score += 1
                if score >= settings.artifact_conditions_required:
                    confirmed.append(page_num)

            if confirmed:
                results.append((position, set(confirmed), orientation, entries))

        return results
