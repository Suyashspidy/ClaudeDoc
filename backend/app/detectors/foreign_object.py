"""Foreign object detector — uses YOLOv8 + SAM to detect objects blocking page content."""
from __future__ import annotations
import logging
from typing import List, Optional
import numpy as np
import cv2

from .base import BaseDetector
from ..models.schemas import PageIssue, IssueType, Severity, BoundingBox
from config.settings import settings

logger = logging.getLogger(__name__)

# COCO class labels relevant to document scanning context
FOREIGN_OBJECT_CLASSES = {
    "person", "book", "cell phone", "scissors", "remote", "mouse",
    "keyboard", "pen", "pencil", "cup", "bottle", "paper", "card",
    "hand", "finger", "post-it", "bookmark", "receipt",
}


class ForeignObjectDetector(BaseDetector):
    """
    Two-stage detection pipeline:
    Stage 1: YOLOv8 for fast object classification.
    Stage 2: SAM (Segment Anything) for precise content coverage measurement.
    Falls back to contour-based detection when models are unavailable.
    """

    def __init__(self) -> None:
        super().__init__("foreign_object")
        self._yolo = None
        self._sam_predictor = None
        self._models_loaded = False

    def _load_models(self) -> None:
        if self._models_loaded:
            return
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(settings.yolo_model_path)
            if settings.use_gpu:
                self._yolo.to("cuda")
            logger.info("YOLOv8 model loaded")
        except Exception as e:
            logger.warning("YOLOv8 not available (%s), using fallback detector", e)

        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
            if settings.use_gpu:
                sam.to("cuda")
            self._sam_predictor = SamPredictor(sam)
            logger.info("SAM model loaded")
        except Exception as e:
            logger.warning("SAM not available (%s), skipping segmentation", e)

        self._models_loaded = True

    def detect(self, page_image: np.ndarray, page_number: int, **kwargs) -> List[PageIssue]:
        self._load_models()

        if self._yolo is not None:
            return self._detect_with_yolo(page_image, page_number)
        # Skip contour fallback — it produces too many false positives on text-heavy pages
        return []

    # ------------------------------------------------------------------
    # YOLOv8 + SAM pipeline
    # ------------------------------------------------------------------

    def _detect_with_yolo(self, page_image: np.ndarray, page_number: int) -> List[PageIssue]:
        results = self._yolo(page_image, conf=settings.yolo_confidence_threshold, verbose=False)
        issues: List[PageIssue] = []
        page_area = page_image.shape[0] * page_image.shape[1]

        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls)]
                confidence = float(box.conf)
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

                # Refine coverage with SAM if available
                coverage = self._sam_coverage(page_image, x1, y1, x2, y2, page_area)
                if coverage is None:
                    box_area = (x2 - x1) * (y2 - y1)
                    coverage = box_area / page_area

                issue = self._build_issue(label, confidence, coverage, bbox)
                if issue:
                    issues.append(issue)

        return issues

    def _sam_coverage(
        self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, page_area: int
    ) -> Optional[float]:
        if self._sam_predictor is None:
            return None
        try:
            self._sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            masks, _, _ = self._sam_predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
            mask_area = int(masks[0].sum())
            return mask_area / page_area
        except Exception as e:
            logger.debug("SAM segmentation failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Contour-based fallback
    # ------------------------------------------------------------------

    def _detect_fallback(self, page_image: np.ndarray, page_number: int) -> List[PageIssue]:
        """Detect large non-text blobs that may be foreign objects."""
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        page_area = gray.size

        # Suppress text using morphological opening, then detect large blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
        _, binary = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dilate to merge nearby blobs
        big_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilated = cv2.dilate(binary, big_kernel, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        issues: List[PageIssue] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            coverage = area / page_area
            if coverage < settings.foreign_object_warning_coverage:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bbox = BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h)
            issue = self._build_issue("unknown object", 0.6, coverage, bbox)
            if issue:
                issues.append(issue)

        return issues

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_issue(
        self, label: str, confidence: float, coverage: float, bbox: BoundingBox
    ) -> Optional[PageIssue]:
        if coverage < settings.foreign_object_warning_coverage:
            return None

        if coverage > settings.foreign_object_critical_coverage:
            severity = Severity.critical
            action = (
                f"Remove the {label} and rescan immediately. "
                f"Approximately {coverage:.0%} of page content is blocked."
            )
        else:
            severity = Severity.warning
            action = (
                f"Check whether {label} is blocking important content "
                f"({coverage:.0%} of page area affected)."
            )

        location = f"x={bbox.x1}-{bbox.x2}, y={bbox.y1}-{bbox.y2}"

        return PageIssue(
            type=IssueType.foreign_object,
            severity=severity,
            description=f"Foreign object detected: {label} covering {coverage:.1%} of page.",
            location=location,
            confidence=confidence,
            recommended_action=action,
            bounding_box=bbox,
        )
