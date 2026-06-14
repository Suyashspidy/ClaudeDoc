"""
Step 1 of the foreign-object-classifier data pipeline: segment objects out of the
phone-photo "bad" PDFs and render a contact sheet so we can eyeball cutout quality
BEFORE committing to mass compositing.

What it does:
  1. Groups the bad PDFs by object category (from filename, e.g. "bookmark", "card").
  2. Takes a small sample per category, rasterizes page 0 of each.
  3. Runs rembg (u2net) to cut the salient object out -> transparent PNG.
  4. Writes per-sample originals + cutouts, and one contact_sheet.png
     (original | cutout-on-checkerboard) for review.

This is a PREVIEW. It does not generate training data. If the cutouts look good
we scale up; if u2net grabs the whole page instead of the object, we switch models.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from rembg import new_session, remove

# --- config ---
BAD_DIR = Path(r"D:\E\Data Science Projects\ClaudeDoc\Data\Bad")
STEP1_DIR = Path(r"D:\E\Data Science Projects\ClaudeDoc\Data\_step1")
SAMPLES_PER_CAT = 1          # how many files to sample per category for this preview
RENDER_DPI = 150             # rasterization DPI for the PDF page
THUMB_W = 420                # contact-sheet thumbnail width (px)

# Model can be overridden from the CLI:  python step1_segment_preview.py birefnet-general
MODEL = sys.argv[1] if len(sys.argv) > 1 else "birefnet-general"
OUT_DIR = STEP1_DIR / MODEL  # keep each model's results side-by-side


def category_of(pdf_path: Path) -> str:
    """Normalize a filename to a category key: drop digits/extension, collapse spaces."""
    stem = pdf_path.stem.lower()
    stem = re.sub(r"\d+", " ", stem)        # remove run numbers
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or "misc"


def pick_samples() -> list[Path]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for pdf in sorted(BAD_DIR.glob("*.pdf")):
        groups[category_of(pdf)].append(pdf)
    samples = []
    for cat in sorted(groups):
        samples.extend(sorted(groups[cat])[:SAMPLES_PER_CAT])
    return samples


def render_page0(pdf_path: Path, dpi: int) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def checkerboard(size, square=16):
    w, h = size
    bg = Image.new("RGB", (w, h), (235, 235, 235))
    d = ImageDraw.Draw(bg)
    for y in range(0, h, square):
        for x in range(0, w, square):
            if (x // square + y // square) % 2 == 0:
                d.rectangle([x, y, x + square, y + square], fill=(200, 200, 200))
    return bg


def on_checker(rgba: Image.Image) -> Image.Image:
    bg = checkerboard(rgba.size).convert("RGBA")
    return Image.alpha_composite(bg, rgba).convert("RGB")


def thumb(img: Image.Image, w: int) -> Image.Image:
    if img.width == 0:
        return img
    h = max(1, round(img.height * w / img.width))
    return img.resize((w, h), Image.LANCZOS)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "originals").mkdir(exist_ok=True)
    (OUT_DIR / "cutouts").mkdir(exist_ok=True)

    samples = pick_samples()
    print(f"Sampling {len(samples)} files across categories. Loading rembg '{MODEL}'...")
    session = new_session(MODEL)

    rows = []  # (label, original_thumb, cutout_thumb)
    for i, pdf in enumerate(samples, 1):
        cat = category_of(pdf)
        print(f"[{i}/{len(samples)}] {cat:14s} <- {pdf.name}")
        try:
            orig = render_page0(pdf, RENDER_DPI)
        except Exception as e:
            print(f"    !! render failed: {e}")
            continue
        cut = remove(orig, session=session)  # RGBA
        if cut.mode != "RGBA":
            cut = cut.convert("RGBA")

        stem = re.sub(r"[^a-z0-9]+", "_", pdf.stem.lower()).strip("_")
        orig.save(OUT_DIR / "originals" / f"{stem}.png")
        cut.save(OUT_DIR / "cutouts" / f"{stem}.png")

        rows.append((f"{cat}  ({pdf.name})", thumb(orig, THUMB_W), thumb(on_checker(cut), THUMB_W)))

    if not rows:
        print("No rows produced; aborting contact sheet.")
        return

    # --- build contact sheet: label band + [original | cutout] per row ---
    pad, label_h = 12, 22
    row_h = max(max(o.height, c.height) for _, o, c in rows) + label_h + pad
    sheet_w = THUMB_W * 2 + pad * 3
    sheet_h = row_h * len(rows) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    y = pad
    for label, o, c in rows:
        draw.text((pad, y), label, fill=(0, 0, 0), font=font)
        draw.text((THUMB_W + pad * 2, y), "cutout (rembg)", fill=(90, 90, 90), font=font)
        sheet.paste(o, (pad, y + label_h))
        sheet.paste(c, (THUMB_W + pad * 2, y + label_h))
        y += row_h

    sheet_path = OUT_DIR / "contact_sheet.png"
    sheet.save(sheet_path)
    print(f"\nDone. Contact sheet -> {sheet_path}")
    print(f"Per-sample cutouts -> {OUT_DIR / 'cutouts'}")


if __name__ == "__main__":
    sys.exit(main())
