from pathlib import Path
import fitz
from PIL import Image, ImageDraw, ImageFont

SRC = Path(r"D:\E\Data Science Projects\ClaudeDoc\Data\Good")
OUT = SRC.parent / "_good_preview.png"
TW = 360
rows = []
for pdf in sorted(SRC.glob("*.pdf")):
    doc = fitz.open(pdf)
    n = len(doc)
    pno = int(n * 0.5)
    pix = doc[pno].get_pixmap(dpi=130)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    h = round(img.height * TW / img.width)
    rows.append((f"{pdf.stem[:28]} p{pno}/{n}", img.resize((TW, h), Image.LANCZOS)))
    doc.close()

cell_h = max(t.height for _, t in rows) + 20
sheet = Image.new("RGB", (len(rows) * (TW + 10) + 10, cell_h + 10), (255, 255, 255))
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 12)
except Exception:
    font = ImageFont.load_default()
for i, (label, t) in enumerate(rows):
    x = 10 + i * (TW + 10)
    d.text((x, 4), label, fill=(0, 0, 0), font=font)
    sheet.paste(t, (x, 20))
sheet.save(OUT)
print("saved", OUT)
