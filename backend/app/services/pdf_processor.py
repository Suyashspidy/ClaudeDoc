"""PDF processing service — extracts pages as images using PyMuPDF."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Iterator, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Wraps PyMuPDF (fitz) to extract pages as numpy BGR arrays."""

    def __init__(self, dpi: int = 150) -> None:
        self.dpi = dpi
        self._zoom = dpi / 72.0  # PyMuPDF default is 72 DPI

    def get_metadata(self, pdf_path: Path) -> dict:
        import fitz
        doc = fitz.open(str(pdf_path))
        meta = doc.metadata or {}
        meta["page_count"] = doc.page_count
        doc.close()
        return meta

    def iter_pages(self, pdf_path: Path) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (1-based page number, BGR image array) for each page."""
        import fitz
        doc = fitz.open(str(pdf_path))
        matrix = fitz.Matrix(self._zoom, self._zoom)

        try:
            for idx in range(doc.page_count):
                page = doc.load_page(idx)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                import cv2
                bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                yield idx + 1, bgr
        finally:
            doc.close()

    def extract_all_pages(self, pdf_path: Path) -> Dict[int, np.ndarray]:
        """Return dict of {page_number: BGR image}."""
        return dict(self.iter_pages(pdf_path))

    def get_page_count(self, pdf_path: Path) -> int:
        import fitz
        doc = fitz.open(str(pdf_path))
        count = doc.page_count
        doc.close()
        return count


class ImageProcessor:
    """Handles single image files (PNG, JPEG, TIFF) as single-page documents."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

    def extract_all_pages(self, image_path: Path) -> Dict[int, np.ndarray]:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return {1: img}

    def get_metadata(self, image_path: Path) -> dict:
        return {"page_count": 1, "format": image_path.suffix.lower()}
