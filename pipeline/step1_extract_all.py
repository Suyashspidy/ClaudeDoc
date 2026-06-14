"""
Step 1 (full scale): extract every foreign object from the phone-photo "bad" PDFs
into a reusable cutout library, using birefnet-general (chosen after the preview
comparison beat u2net on white paper + sticky notes).

For every page of every PDF in Data/Bad:
  1. Rasterize the page.
  2. birefnet-general -> RGBA cutout (object on transparent background).
  3. Auto-QC the alpha mask:
       - coverage > WHOLEPAGE_COV (mask fills the frame) -> FAIL_WHOLEPAGE
       - coverage < EMPTY_COV  (almost nothing)          -> FAIL_EMPTY
       - else                                            -> OK
  4. Tight-crop OK cutouts to the object bbox; save under cutouts/<category>/.
  5. Write manifest.csv (every page + status) and a contact sheet of the FAILS
     so the small bad minority can be reviewed at a glance.

Output: Data/_objects/   (the object library that step 2 compositing will draw from)
"""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import gpu_init  # noqa: F401  -- must precede onnxruntime/rembg; registers CUDA/cuDNN DLL dirs
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rembg import new_session, remove

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# --- config ---
BAD_DIR = Path(r"D:\E\Data Science Projects\ClaudeDoc\Data\Bad")
OUT_DIR = Path(r"D:\E\Data Science Projects\ClaudeDoc\Data\_objects")
MODEL = "birefnet-general"
RENDER_DPI = 200
ALPHA_THRESH = 16          # alpha above this counts as "object" pixel
WHOLEPAGE_COV = 0.85       # mask covers >85% of frame -> grabbed the page
EMPTY_COV = 0.01           # mask covers <1% -> found nothing
MARGIN = 8                 # px padding kept around the cropped object


def category_of(pdf_path: Path) -> str:
    stem = pdf_path.stem.lower()
    stem = re.sub(r"\d+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or "misc"


def safe(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def render_page(page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def qc_and_crop(rgba: Image.Image):
    """Return (status, cropped_rgba_or_None, coverage)."""
    alpha = np.asarray(rgba.split()[-1])
    mask = alpha > ALPHA_THRESH
    coverage = float(mask.mean())
    if coverage < EMPTY_COV:
        return "FAIL_EMPTY", None, coverage
    if coverage > WHOLEPAGE_COV:
        return "FAIL_WHOLEPAGE", None, coverage
    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - MARGIN); y0 = max(0, y0 - MARGIN)
    x1 = min(rgba.width - 1, x1 + MARGIN); y1 = min(rgba.height - 1, y1 + MARGIN)
    return "OK", rgba.crop((x0, y0, x1 + 1, y1 + 1)), coverage


def thumb(img, w=320):
    if img.width == 0:
        return img
    h = max(1, round(img.height * w / img.width))
    return img.resize((w, h), Image.LANCZOS)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cut_root = OUT_DIR / "cutouts"
    cut_root.mkdir(exist_ok=True)

    print(f"Loading rembg '{MODEL}' on {PROVIDERS[0]}...")
    session = new_session(MODEL, providers=PROVIDERS)
    active = session.inner_session.get_providers()
    print("Active providers:", active)
    if "CUDAExecutionProvider" not in active:
        print("!! WARNING: CUDA not active — running on CPU.")

    pdfs = sorted(BAD_DIR.glob("*.pdf"))
    manifest = []          # rows for CSV
    fails = []             # (label, thumb) for contact sheet
    status_counts = Counter()
    per_cat_ok = defaultdict(int)

    page_total = 0
    for pi, pdf in enumerate(pdfs, 1):
        cat = category_of(pdf)
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            print(f"  !! open failed {pdf.name}: {e}")
            continue
        for pno in range(len(doc)):
            page_total += 1
            try:
                orig = render_page(doc[pno], RENDER_DPI)
            except Exception as e:
                print(f"  !! render {pdf.name} p{pno}: {e}")
                continue
            rgba = remove(orig, session=session)
            if rgba.mode != "RGBA":
                rgba = rgba.convert("RGBA")
            status, crop, cov = qc_and_crop(rgba)
            status_counts[status] += 1

            base = f"{safe(pdf.stem)}_p{pno}"
            if status == "OK":
                cat_dir = cut_root / safe(cat)
                cat_dir.mkdir(exist_ok=True)
                crop.save(cat_dir / f"{base}.png")
                per_cat_ok[cat] += 1
            else:
                if len(fails) < 60:  # cap contact sheet size
                    fails.append((f"{cat} | {pdf.name} p{pno} | {status} cov={cov:.2f}",
                                  thumb(orig)))
            manifest.append([pdf.name, cat, pno, f"{cov:.4f}", status])
        doc.close()
        if pi % 20 == 0:
            print(f"  ...{pi}/{len(pdfs)} pdfs, {page_total} pages, "
                  f"OK={status_counts['OK']} "
                  f"WHOLEPAGE={status_counts['FAIL_WHOLEPAGE']} "
                  f"EMPTY={status_counts['FAIL_EMPTY']}")

    # manifest
    with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdf", "category", "page", "coverage", "status"])
        w.writerows(manifest)

    # fails contact sheet
    if fails:
        cols = 4
        tw = 320
        rows = (len(fails) + cols - 1) // cols
        cell_h = max(t.height for _, t in fails) + 26
        sheet = Image.new("RGB", (cols * (tw + 10) + 10, rows * cell_h + 10), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
        for i, (label, t) in enumerate(fails):
            r, c = divmod(i, cols)
            x = 10 + c * (tw + 10); y = 10 + r * cell_h
            d.text((x, y), label[:60], fill=(150, 0, 0), font=font)
            sheet.paste(t, (x, y + 16))
        sheet.save(OUT_DIR / "fails_contact_sheet.png")

    # summary
    print("\n=== SUMMARY ===")
    print(f"PDFs: {len(pdfs)}   pages: {page_total}")
    for s, n in status_counts.most_common():
        print(f"  {s:16s} {n:4d}  ({n/page_total*100:.1f}%)")
    print("\nOK cutouts per category:")
    for cat in sorted(per_cat_ok):
        print(f"  {cat:14s} {per_cat_ok[cat]}")
    print(f"\nObject library -> {cut_root}")
    print(f"Manifest       -> {OUT_DIR / 'manifest.csv'}")
    if fails:
        print(f"Fails review   -> {OUT_DIR / 'fails_contact_sheet.png'}")


if __name__ == "__main__":
    main()
