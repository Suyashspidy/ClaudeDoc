"""
Build a REAL-photo holdout test set: render original phone-photo bad PDFs (the true
production-adjacent captures) as 'bad' test images, and REMOVE those same objects
from the training cutout pool so there is zero object-instance leakage into training.

After running this, regenerate the synthetic dataset (step2_composite.py) so train/
val/test composites never use holdout objects. The trained model's recall on
Data/_holdout_real/bad is the honest 'does it catch real objects' metric.

Selection: per object category, hold out ~15% of the source PDFs (>=1), deterministic.
"""
import re
import shutil
from collections import defaultdict
from pathlib import Path

import fitz
from PIL import Image

ROOT = Path(r"D:\E\Data Science Projects\ClaudeDoc\Data")
BAD_DIR = ROOT / "Bad"
CUT_DIR = ROOT / "_objects" / "cutouts"
EXCL = ROOT / "_objects" / "_excluded" / "holdout"
HOLD = ROOT / "_holdout_real"
HOLD_FRAC = 0.15
TARGET_LONG = 1100


def category_of(pdf: Path) -> str:
    s = re.sub(r"\d+", " ", pdf.stem.lower())
    return re.sub(r"\s+", " ", s).strip() or "misc"


def safe(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main():
    (HOLD / "bad").mkdir(parents=True, exist_ok=True)
    EXCL.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for pdf in sorted(BAD_DIR.glob("*.pdf")):
        groups[category_of(pdf)].append(pdf)

    holdout_pdfs = []
    for cat, pdfs in sorted(groups.items()):
        k = max(1, round(len(pdfs) * HOLD_FRAC))
        holdout_pdfs.extend(sorted(pdfs)[-k:])  # deterministic: last k by name

    print(f"Holdout: {len(holdout_pdfs)} PDFs across {len(groups)} categories")
    pages_written, cutouts_moved = 0, 0
    for pdf in holdout_pdfs:
        cat = category_of(pdf)
        stem = safe(pdf.stem)
        # render full phone-photo pages as real 'bad'
        doc = fitz.open(pdf)
        for pno in range(len(doc)):
            page = doc[pno]
            zoom = TARGET_LONG / max(page.rect.width, page.rect.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img.save(HOLD / "bad" / f"{stem}_p{pno}_{safe(cat)}.jpg", quality=90)
            pages_written += 1
        doc.close()
        # pull this PDF's cutouts out of the training pool
        cat_dir = CUT_DIR / safe(cat)
        if cat_dir.exists():
            for cut in cat_dir.glob(f"{stem}_p*.png"):
                shutil.move(str(cut), str(EXCL / cut.name))
                cutouts_moved += 1

    print(f"  real-bad pages -> {HOLD/'bad'}  ({pages_written} imgs)")
    print(f"  holdout cutouts pulled from training pool -> {EXCL}  ({cutouts_moved})")
    remaining = sum(1 for _ in CUT_DIR.rglob("*.png"))
    print(f"  cutouts remaining for training: {remaining}")
    print("\nNEXT: delete Data/_dataset/{train,val,test} and rerun step2_composite.py")


if __name__ == "__main__":
    main()
