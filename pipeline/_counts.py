from pathlib import Path
import fitz

base = Path(r"D:\E\Data Science Projects\ClaudeDoc\Data\_base_src")
tot = 0
for pdf in sorted(base.glob("*.pdf")):
    n = len(fitz.open(pdf))
    tot += n
    print(f"  {pdf.stem:36s} {n:4d} pages")
print(f"TOTAL base pages: {tot}")

obj = Path(r"D:\E\Data Science Projects\ClaudeDoc\Data\_objects\cutouts")
oc = sum(1 for _ in obj.rglob("*.png"))
cats = sum(1 for d in obj.iterdir() if d.is_dir())
print(f"object cutouts: {oc} across {cats} categories")
