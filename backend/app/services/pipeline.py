"""Main document QA pipeline — orchestrates all detectors and produces a QAReport."""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ..detectors import (
    BlankPageDetector,
    SkewDetector,
    FoldedPageDetector,
    MissingPageDetector,
)
from ..models.schemas import (
    IssuesSummary, PageReport, QAReport, ScannerHealth, Severity,
)
from ..services.pdf_processor import PDFProcessor, ImageProcessor
from ..services.quality_scorer import score_page, score_document
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentQAPipeline:
    """Runs the full QA pipeline on a PDF or image file."""

    def __init__(self, job_id: str, output_dir: Path) -> None:
        self.job_id = job_id
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._blank = BlankPageDetector()
        self._skew = SkewDetector()
        self._fold = FoldedPageDetector()
        self._missing = MissingPageDetector()

    def run(
        self,
        file_path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> QAReport:
        start = time.time()

        def _progress(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)
            logger.info("[%s] %.0f%% %s", self.job_id, pct, msg)

        _progress(2, "Loading document…")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            processor = PDFProcessor(dpi=settings.report_dpi)
            metadata = processor.get_metadata(file_path)
            pages: Dict[int, np.ndarray] = processor.extract_all_pages(file_path)
            metadata_page_count = metadata.get("page_count")
        else:
            processor = ImageProcessor()
            metadata = processor.get_metadata(file_path)
            pages = processor.extract_all_pages(file_path)
            metadata_page_count = None

        total_pages = len(pages)
        _progress(10, f"Extracted {total_pages} pages.")

        # ---- Per-page detectors ----
        page_issues: Dict[int, list] = {pn: [] for pn in pages}

        for idx, (page_num, img) in enumerate(sorted(pages.items())):
            pct = 10 + (idx / total_pages) * 75
            _progress(pct, f"Analysing page {int(page_num)}/{total_pages}…")

            page_issues[page_num] += self._blank._safe_detect(img, page_num)
            page_issues[page_num] += self._skew._safe_detect(img, page_num)
            page_issues[page_num] += self._fold._safe_detect(img, page_num)

        # ---- Missing page detection ----
        _progress(90, "Checking for missing pages…")

        # Metadata check: assign to page 1 if metadata says pages are missing
        for issue in self._missing.detect_from_pdf_metadata(metadata_page_count, total_pages):
            page_issues[1].append(issue)

        # For PDFs: use embedded text layer (reliable). For images: fall back to OCR.
        if suffix == ".pdf":
            detected_page_numbers = MissingPageDetector.extract_page_numbers_from_pdf(file_path)
        else:
            detected_page_numbers = {
                pn: MissingPageDetector.extract_page_number_from_image(img)
                for pn, img in pages.items()
            }

        for physical_page, issue in self._missing.detect_sequence_gaps(detected_page_numbers):
            if physical_page in page_issues:
                page_issues[physical_page].append(issue)

        # ---- Build page reports ----
        page_reports: List[PageReport] = []
        for page_num in sorted(pages.keys()):
            issues = page_issues[page_num]
            page_reports.append(PageReport(
                page_number=int(page_num),
                issues=issues,
                quality_score=score_page(issues),
            ))

        # ---- Aggregate scores & summary ----
        page_scores = [p.quality_score for p in page_reports]
        overall_score = score_document(page_scores)

        all_issues = [i for p in page_reports for i in p.issues]
        summary = IssuesSummary(
            critical=sum(1 for i in all_issues if i.severity == Severity.critical),
            warnings=sum(1 for i in all_issues if i.severity == Severity.warning),
            info=sum(1 for i in all_issues if i.severity == Severity.info),
        )

        scanner_health = ScannerHealth(
            artifact_detected=False,
            affected_pages=[],
            cleaning_recommended=False,
            artifact_positions=[],
        )

        report = QAReport(
            job_id=self.job_id,
            document_name=file_path.name,
            total_pages=total_pages,
            processing_time_seconds=time.time() - start,
            overall_quality_score=overall_score,
            issues_summary=summary,
            pages=page_reports,
            scanner_health=scanner_health,
        )

        _progress(100, "Done.")
        return report
