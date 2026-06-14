"""Render sample pages from the downloaded base scans into one contact sheet."""
from pathlib import Path
import fitz
from PIL import Image, ImageDraw, ImageFont

SRC = Path(r"D:\E\Data Science Projects\ClaudeDoc\Data\_base_src")
OUT = SRC / "_preview.png"
THUMB_W = 360

rows = []
for pdf in sorted(SRC.glob("*.pdf")):
    doc = fitz.open(pdf)
    n = len(doc)
    # two pages from the middle (skip front matter)
    for frac in (0.45, 0.6):
        pno = int(n * frac)
        pix = doc[pno].get_pixmap(dpi=130)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        h = round(img.height * THUMB_W / img.width)
        rows.append((f"{pdf.stem}  p{pno}/{n}", img.resize((THUMB_W, h), Image.LANCZOS)))
    doc.close()

cols = 4
label_h = 18
cell_h = max(t.height for _, t in rows) + label_h + 8
cell_w = THUMB_W + 10
R = (len(rows) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cell_w + 10, R * cell_h + 10), (255, 255, 255))
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 12)
except Exception:
    font = ImageFont.load_default()
for i, (label, t) in enumerate(rows):
    r, c = divmod(i, cols)
    x, y = 10 + c * cell_w, 10 + r * cell_h
    d.text((x, y), label, fill=(0, 0, 0), font=font)
    sheet.paste(t, (x, y + label_h))
sheet.save(OUT)
print("saved", OUT, "pages:", len(rows))
