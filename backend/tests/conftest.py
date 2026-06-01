"""Shared test fixtures for ClaudeDoc detection algorithms."""
from __future__ import annotations
import numpy as np
import pytest
import cv2


def _make_blank_page(h: int = 1122, w: int = 794) -> np.ndarray:
    """White page with almost no content."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 250
    return img


def _make_page_with_text(h: int = 1122, w: int = 794) -> np.ndarray:
    """Page with simulated text lines."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 245
    for y in range(100, h - 100, 30):
        cv2.line(img, (80, y), (w - 80, y), (30, 30, 30), 2)
    return img


def _make_skewed_page(angle: float = 5.0) -> np.ndarray:
    """Page with text rotated by `angle` degrees."""
    base = _make_page_with_text()
    h, w = base.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(base, M, (w, h), borderValue=(245, 245, 245))


def _make_page_with_vertical_artifact(
    h: int = 1122, w: int = 794, x_pos: int = 300
) -> np.ndarray:
    """Page with a thin, uniform-intensity vertical line simulating a scanner artifact."""
    img = _make_page_with_text(h, w)
    cv2.line(img, (x_pos, 0), (x_pos, h), (80, 80, 80), 1)
    return img


def _make_page_with_folded_corner(corner: str = "bottom-right") -> np.ndarray:
    """Page with a white triangle in one corner simulating a fold."""
    img = _make_page_with_text()
    h, w = img.shape[:2]
    corner_size = int(min(h, w) * 0.12)

    corners_pts = {
        "top-left":     np.array([[0, 0], [corner_size, 0], [0, corner_size]]),
        "top-right":    np.array([[w, 0], [w - corner_size, 0], [w, corner_size]]),
        "bottom-left":  np.array([[0, h], [corner_size, h], [0, h - corner_size]]),
        "bottom-right": np.array([[w, h], [w - corner_size, h], [w, h - corner_size]]),
    }
    pts = corners_pts[corner]
    cv2.fillPoly(img, [pts], (255, 255, 255))
    return img


@pytest.fixture
def blank_page() -> np.ndarray:
    return _make_blank_page()


@pytest.fixture
def text_page() -> np.ndarray:
    return _make_page_with_text()


@pytest.fixture
def skewed_page() -> np.ndarray:
    return _make_skewed_page(angle=5.0)


@pytest.fixture
def artifact_pages() -> dict:
    """3 pages with the same vertical artifact at x=300."""
    return {i: _make_page_with_vertical_artifact(x_pos=300) for i in range(1, 4)}


@pytest.fixture
def clean_pages() -> dict:
    return {i: _make_page_with_text() for i in range(1, 4)}


@pytest.fixture
def folded_page() -> np.ndarray:
    return _make_page_with_folded_corner("bottom-right")
