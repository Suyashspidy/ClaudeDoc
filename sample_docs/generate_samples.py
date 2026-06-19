"""
Generates synthetic test PDFs covering each QA scenario.
Run: python sample_docs/generate_samples.py
Requires: PyMuPDF, Pillow, numpy, opencv-python

Notes on realism (so the detectors actually fire):
  * Printed page numbers are written as a real PDF **text layer** (not baked-in
    pixels), because MissingPageDetector reads the embedded text layer.
  * The folded corner is a **dark shadow triangle** (a real fold casts a shadow /
    reveals the darker page back), not a white-on-white wedge.
"""
from __future__ import annotations
import io
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

OUTPUT = Path(__file__).parent


def _numpy_to_pil(arr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _make_text_page(label: str = "", h: int = 1122, w: int = 794) -> np.ndarray:
    """A page of ruled 'text' lines. The page number is added later as real PDF text."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 245
    for y in range(120, h - 120, 35):
        cv2.line(img, (80, y), (w - 80, y), (40, 40, 40), 1)
    if label:
        cv2.putText(img, label, (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 50, 50), 2)
    return img


def _save_pdf(pages: List[np.ndarray], name: str, numbers: Optional[List[Optional[int]]] = None) -> None:
    """Save pages as a PDF, embedding each page number as a real text-layer footer."""
    import fitz

    doc = fitz.open()
    for idx, arr in enumerate(pages):
        h, w = arr.shape[:2]
        pil = _numpy_to_pil(arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")

        page = doc.new_page(width=w, height=h)  # 1 PDF point == 1 image pixel
        page.insert_image(fitz.Rect(0, 0, w, h), stream=buf.getvalue())

        if numbers is not None and numbers[idx] is not None:
            # Real text layer: an isolated number centred in the footer.
            page.insert_text(
                fitz.Point(w / 2 - 8, h - 45),
                str(numbers[idx]),
                fontsize=20,
                color=(0.3, 0.3, 0.3),
            )

    doc.save(str(OUTPUT / name))
    doc.close()
    print(f"  Saved {name}")


def generate_missing_pages() -> None:
    """PDF with printed page numbers 1-4, 7-10 (printed pages 5-6 missing)."""
    nums = [1, 2, 3, 4, 7, 8, 9, 10]
    pages = [_make_text_page() for _ in nums]
    _save_pdf(pages, "test_missing_pages_5_6.pdf", numbers=nums)


def generate_folded_corner() -> None:
    """Document with a folded (shadowed) bottom-right corner on page 3."""
    nums = list(range(1, 6))
    pages = []
    for i in nums:
        page = _make_text_page()
        if i == 3:
            h, w = page.shape[:2]
            size = int(min(h, w) * 0.13)
            # Dark triangular shadow anchored at the outer (bottom-right) corner.
            pts = np.array([[w, h], [w - size, h], [w, h - size]])
            cv2.fillPoly(page, [pts], (70, 70, 70))
            # A slightly darker diagonal crease along the fold line.
            cv2.line(page, (w - size, h), (w, h - size), (45, 45, 45), 3)
        pages.append(page)
    _save_pdf(pages, "test_folded_corner_page3.pdf", numbers=nums)


def generate_scanner_artifact() -> None:
    """Pages 10-15 have a vertical scanner artifact line at x=350."""
    nums = list(range(1, 16))
    pages = []
    for i in nums:
        page = _make_text_page()
        if 10 <= i <= 15:
            cv2.line(page, (350, 0), (350, page.shape[0]), (90, 90, 90), 1)
        pages.append(page)
    _save_pdf(pages, "test_scanner_artifact_pages10_15.pdf", numbers=nums)


def generate_foreign_object() -> None:
    """Page 8 has a dark rectangle blocking ~20% of content (simulated bookmark)."""
    nums = list(range(1, 10))
    pages = []
    for i in nums:
        page = _make_text_page()
        if i == 8:
            h, w = page.shape[:2]
            cv2.rectangle(page, (w // 2 - 30, 0), (w // 2 + 30, int(h * 0.4)),
                          (30, 20, 10), -1)
            cv2.putText(page, "BOOKMARK", (w // 2 - 45, int(h * 0.42)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 100), 1)
        pages.append(page)
    _save_pdf(pages, "test_foreign_object_page8.pdf", numbers=nums)


def generate_perfect_document() -> None:
    """Clean document — should score 100."""
    nums = list(range(1, 6))
    pages = [_make_text_page() for _ in nums]
    _save_pdf(pages, "test_perfect_document.pdf", numbers=nums)


if __name__ == "__main__":
    print("Generating sample test documents...")
    generate_missing_pages()
    generate_folded_corner()
    generate_scanner_artifact()
    generate_foreign_object()
    generate_perfect_document()
    print("Done. Files saved to sample_docs/")
