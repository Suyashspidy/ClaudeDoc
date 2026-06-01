"""Missing page detector — extracts page numbers from content and identifies sequence gaps."""
from __future__ import annotations
import re
from typing import List, Optional, Dict, Tuple
import numpy as np
import cv2

from .base import BaseDetector
from ..models.schemas import PageIssue, IssueType, Severity
from config.settings import settings


_PAGE_NUMBER_PATTERN = re.compile(
    r'\b(?:page|pg|p\.?\s*)?\s*(\d{1,4})\b',
    re.IGNORECASE,
)


class MissingPageDetector(BaseDetector):
    """
    Combines two signals:
    1. PDF metadata page count vs actual extracted pages.
    2. Detected page numbers in headers/footers to find sequence gaps.
    """

    def __init__(self) -> None:
        super().__init__("missing_page")

    def detect(self, page_image: np.ndarray, page_number: int, **kwargs) -> List[PageIssue]:
        # Single-page variant does nothing; gaps only become visible after batch analysis.
        return []

    def detect_from_pdf_metadata(
        self, metadata_page_count: Optional[int], actual_page_count: int
    ) -> List[PageIssue]:
        """Compare PDF metadata page count with actual extracted pages."""
        if metadata_page_count is None or metadata_page_count <= actual_page_count:
            return []

        missing = metadata_page_count - actual_page_count
        return [PageIssue(
            type=IssueType.missing,
            severity=Severity.critical,
            description=(
                f"PDF metadata reports {metadata_page_count} pages but only "
                f"{actual_page_count} pages were extracted. "
                f"Approximately {missing} page(s) may be missing."
            ),
            location="document_level",
            confidence=0.95,
            recommended_action="Re-export or rescan the document; verify source file integrity.",
        )]

    def detect_sequence_gaps(
        self, page_numbers: Dict[int, Optional[int]]
    ) -> List[Tuple[int, PageIssue]]:
        """
        Given a mapping of {physical_page -> detected_page_number},
        find gaps in the detected sequence.

        Returns a list of (physical_page_after_gap, PageIssue) so the pipeline
        can assign each issue to the correct page in the report.

        Strategy: every page in the main numbered body of a book shares roughly
        the same offset = physical_page - printed_number (e.g. if printed page 1
        is at physical page 15, offset = 14 throughout). False OCR reads from
        chapter headings, copyright years, etc. have very different offsets and
        are filtered out via the median. Roman numeral front matter returns None
        and is already excluded.

        A real missing page means physical pages are consecutive in the PDF but
        the printed numbers skip — because the deleted page is simply gone.
        If the physical gap is as large as the printed gap, it is explained by
        OCR failures on existing pages, not missing scans.
        """
        pairs = sorted(
            [(phys, num) for phys, num in page_numbers.items() if num is not None],
            key=lambda x: x[0],
        )
        if len(pairs) < 5:
            return []

        # Filter to the main numbered section using the median offset
        offsets = [phys - num for phys, num in pairs]
        median_offset = sorted(offsets)[len(offsets) // 2]
        OFFSET_TOLERANCE = 5
        main_pairs = [
            (phys, num) for phys, num in pairs
            if abs((phys - num) - median_offset) <= OFFSET_TOLERANCE
        ]
        if len(main_pairs) < 5:
            return []

        results: List[Tuple[int, PageIssue]] = []
        for i in range(len(main_pairs) - 1):
            curr_phys, curr_num = main_pairs[i]
            next_phys, next_num = main_pairs[i + 1]
            num_gap = next_num - curr_num - 1     # printed page numbers skipped
            phys_gap = next_phys - curr_phys - 1  # physical pages not in main_pairs

            if num_gap <= 0:
                continue

            # If phys_gap >= num_gap the "missing" numbers are explained by OCR
            # failures on pages that physically exist — not a real missing scan.
            # A genuine missing page has phys_gap < num_gap (typically phys_gap == 0).
            if phys_gap >= num_gap:
                continue

            missing_nums = list(range(curr_num + 1, next_num))
            issue = PageIssue(
                type=IssueType.missing,
                severity=Severity.critical,
                description=(
                    f"Missing pages detected: page numbers {missing_nums} are absent. "
                    f"Sequence jumps from {curr_num} to {next_num}."
                ),
                location=f"after physical page {curr_phys}",
                confidence=0.9,
                recommended_action=(
                    f"Locate and rescan missing page(s) {missing_nums}. "
                    "Check the original document for completeness."
                ),
            )
            results.append((curr_phys, issue))

        return results

    @staticmethod
    def extract_page_numbers_from_pdf(file_path) -> Dict[int, Optional[int]]:
        """
        Extract printed page numbers from a PDF's embedded text layer.
        Looks for isolated numbers (a number on its own line) in the bottom
        30% of each page, then falls back to the top 15% (header).
        Far more reliable than OCR on page images.
        Returns {physical_page_number: detected_printed_number}.
        """
        try:
            import fitz
        except ImportError:
            return {}

        isolated = re.compile(r'^\s*(\d{1,4})\s*$')
        results: Dict[int, Optional[int]] = {}

        try:
            doc = fitz.open(str(file_path))
            for i in range(len(doc)):
                page = doc[i]
                h = page.rect.height
                footer_candidates: List[int] = []
                header_candidates: List[int] = []

                for x0, y0, x1, y1, text, _bno, _btype in page.get_text("blocks"):
                    for line in text.split("\n"):
                        m = isolated.match(line.strip())
                        if not m:
                            continue
                        num = int(m.group(1))
                        if not (1 <= num <= 9999):
                            continue
                        if y0 > h * 0.70:
                            footer_candidates.append(num)
                        elif y1 < h * 0.15:
                            header_candidates.append(num)

                if footer_candidates:
                    results[i + 1] = footer_candidates[0]
                elif header_candidates:
                    results[i + 1] = header_candidates[0]
                else:
                    results[i + 1] = None
            doc.close()
        except Exception:
            pass

        return results

    @staticmethod
    def extract_page_number_from_image(
        page_image: np.ndarray, search_footer: bool = True, search_header: bool = True
    ) -> Optional[int]:
        """Tesseract OCR fallback for image-only PDFs (no embedded text layer)."""
        try:
            import pytesseract
        except ImportError:
            return None

        h, w = page_image.shape[:2]
        strip_height = int(h * 0.08)
        candidates: List[int] = []

        regions = []
        if search_header:
            regions.append(page_image[:strip_height, :])
        if search_footer:
            regions.append(page_image[h - strip_height:, :])

        for region in regions:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(binary, config="--psm 7 -c tessedit_char_whitelist=0123456789 ")
            for match in _PAGE_NUMBER_PATTERN.finditer(text):
                num = int(match.group(1))
                if 1 <= num <= 9999:
                    candidates.append(num)

        if not candidates:
            return None
        candidates.sort()
        return candidates[len(candidates) // 2]
