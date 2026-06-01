"""
Generates synthetic test PDFs covering each QA scenario.
Run: python sample_docs/generate_samples.py
Requires: reportlab, Pillow, numpy, opencv-python
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OUTPUT = Path(__file__).parent


def _numpy_to_pil(arr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _make_text_page(page_num: int, label: str = "", h: int = 1122, w: int = 794) -> np.ndarray:
    img = np.ones((h, w, 3), dtype=np.uint8) * 245
    for y in range(120, h - 120, 35):
        cv2.line(img, (80, y), (w - 80, y), (40, 40, 40), 1)
    # Page number in footer
    cv2.putText(img, str(page_num), (w // 2 - 10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
    if label:
        cv2.putText(img, label, (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 50, 50), 2)
    return img


def _save_pdf(pages: list[np.ndarray], name: str) -> None:
    try:
        import fitz
        doc = fitz.open()
        for arr in pages:
            pil = _numpy_to_pil(arr)
            buf = io.BytesIO()
            pil.save(buf, format="PDF")
            page_pdf = fitz.open("pdf", buf.getvalue())
            doc.insert_pdf(page_pdf)
        doc.save(str(OUTPUT / name))
        doc.close()
        print(f"  Saved {name}")
    except ImportError:
        print("  PyMuPDF not available; saving as JPEG instead")
        for idx, arr in enumerate(pages):
            cv2.imwrite(str(OUTPUT / f"{name.replace('.pdf', '')}_page{idx+1}.jpg"), arr)


def generate_missing_pages() -> None:
    """PDF with page numbers 1-4, 7-10 (pages 5-6 missing)."""
    pages = [_make_text_page(i) for i in [1, 2, 3, 4, 7, 8, 9, 10]]
    _save_pdf(pages, "test_missing_pages_5_6.pdf")


def generate_folded_corner() -> None:
    """Document with folded bottom-right corner on page 3."""
    pages = []
    for i in range(1, 6):
        page = _make_text_page(i)
        if i == 3:
            h, w = page.shape[:2]
            size = int(min(h, w) * 0.12)
            pts = np.array([[w, h], [w - size, h], [w, h - size]])
            cv2.fillPoly(page, [pts], (255, 255, 255))
            cv2.putText(page, "FOLD", (w - size + 10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 100), 1)
        pages.append(page)
    _save_pdf(pages, "test_folded_corner_page3.pdf")


def generate_scanner_artifact() -> None:
    """Pages 10-15 have a vertical scanner artifact line at x=350."""
    pages = []
    for i in range(1, 16):
        page = _make_text_page(i)
        if 10 <= i <= 15:
            cv2.line(page, (350, 0), (350, page.shape[0]), (90, 90, 90), 1)
        pages.append(page)
    _save_pdf(pages, "test_scanner_artifact_pages10_15.pdf")


def generate_foreign_object() -> None:
    """Page 8 has a dark rectangle blocking 20% of content (simulated bookmark)."""
    pages = []
    for i in range(1, 10):
        page = _make_text_page(i)
        if i == 8:
            h, w = page.shape[:2]
            # Dark bookmark-shaped rectangle
            cv2.rectangle(page, (w // 2 - 30, 0), (w // 2 + 30, int(h * 0.4)),
                          (30, 20, 10), -1)
            cv2.putText(page, "BOOKMARK", (w // 2 - 45, int(h * 0.42)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 100), 1)
        pages.append(page)
    _save_pdf(pages, "test_foreign_object_page8.pdf")


def generate_perfect_document() -> None:
    """Clean document — should score 100."""
    pages = [_make_text_page(i) for i in range(1, 6)]
    _save_pdf(pages, "test_perfect_document.pdf")


if __name__ == "__main__":
    print("Generating sample test documents…")
    generate_missing_pages()
    generate_folded_corner()
    generate_scanner_artifact()
    generate_foreign_object()
    generate_perfect_document()
    print("Done. Files saved to sample_docs/")
