"""Foreign-object detector — flags scanned pages with an object lying on them.

Wraps the fine-tuned EfficientNetV2-S classifier trained in ``pipeline/`` (see
``pipeline/fo_classifier_effv2s.pt``). A page is flagged when P(bad) >= the
recall-tuned threshold stored in the checkpoint. The model loads lazily and is
cached process-wide, and runs on CUDA when available (the user's RTX 5080) so the
GPU is actually exercised; it falls back to CPU otherwise.

If torch or the checkpoint is unavailable, the detector degrades gracefully to a
no-op so the rest of the (CPU-only) pipeline keeps working.
"""
from __future__ import annotations
import logging
import threading
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .base import BaseDetector
from ..models.schemas import PageIssue, IssueType, Severity
from config.settings import settings

logger = logging.getLogger(__name__)

# pipeline/fo_classifier_effv2s.pt, resolved relative to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CKPT = _REPO_ROOT / "pipeline" / "fo_classifier_effv2s.pt"

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class _Model:
    """Lazily-loaded, process-wide singleton holding the torch model + transform."""

    _instance: Optional["_Model"] = None
    _lock = threading.Lock()
    _failed = False  # once loading fails we stop retrying

    def __init__(self) -> None:
        import torch
        from torchvision import models, transforms

        ckpt_path = Path(getattr(settings, "fo_classifier_path", "") or _DEFAULT_CKPT)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"foreign-object checkpoint not found: {ckpt_path}")

        self.device = "cuda" if (settings.use_gpu and torch.cuda.is_available()) else "cpu"
        ck = torch.load(ckpt_path, map_location=self.device)

        model = models.efficientnet_v2_s()
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        model.load_state_dict(ck["state_dict"])
        model.to(self.device).eval()

        self.model = model
        self.threshold = float(ck["threshold"])
        self.bad_idx = ck["classes"].index("bad")
        self.transform = transforms.Compose([
            transforms.Resize((ck["img"], ck["img"])),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
        logger.info(
            "Foreign-object classifier loaded (device=%s, threshold=%.2f) from %s",
            self.device, self.threshold, ckpt_path,
        )

    @classmethod
    def get(cls) -> Optional["_Model"]:
        if cls._failed:
            return None
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None and not cls._failed:
                    try:
                        cls._instance = cls()
                    except Exception as exc:  # torch missing, no checkpoint, etc.
                        cls._failed = True
                        logger.warning("Foreign-object classifier unavailable: %s", exc)
                        return None
        return cls._instance

    def predict_bad_prob(self, page_image: np.ndarray) -> float:
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB) if page_image.ndim == 3 else page_image
        pil = Image.fromarray(rgb)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with torch.autocast(self.device, dtype=torch.float16, enabled=(self.device == "cuda")):
                logits = self.model(tensor)
            prob = torch.softmax(logits.float(), 1)[0, self.bad_idx].item()
        return float(prob)


class ForeignObjectClassifierDetector(BaseDetector):
    """Flags pages whose P(foreign object) meets the model's tuned threshold."""

    def __init__(self) -> None:
        super().__init__("foreign_object")

    def detect(self, page_image: np.ndarray, page_number: int, **kwargs) -> List[PageIssue]:
        if not settings.enable_foreign_object_classifier:
            return []
        model = _Model.get()
        if model is None:
            return []

        prob = model.predict_bad_prob(page_image)
        if prob < model.threshold:
            return []

        # Tuned to minimise false negatives — a flagged page should be rescanned.
        # Severity scales above the decision threshold: a confident detection is
        # critical, a borderline one (just over threshold) is a warning.
        severity = Severity.critical if prob >= model.threshold + 0.15 else Severity.warning
        return [PageIssue(
            type=IssueType.foreign_object,
            severity=severity,
            description=(
                f"Foreign object detected on the page (confidence {prob:.0%}). "
                "An item (bookmark, card, slip, etc.) appears to be lying on the page "
                "and may be obscuring content."
            ),
            location="page",
            confidence=prob,
            recommended_action="Remove the object and rescan the page.",
        )]
