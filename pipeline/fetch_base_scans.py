"""
Download real scanned (library-scanner) public-domain book PDFs from the Internet
Archive to serve as clean BASE pages for compositing. These are genuine scans
(flat pages + real scan texture/grayscale), much closer to Iron Mountain production
modality than the born-digital ebook pages.

Picks prose titles (single-column text), verifies each item is an actual scan, and
downloads the PDF derivative to Data/_base_src/.
"""
import json
import urllib.request
import urllib.parse
from pathlib import Path

OUT = Path(r"D:\E\Data Science Projects\ClaudeDoc\Data\_base_src")
OUT.mkdir(parents=True, exist_ok=True)

# Prose, single-column scanned books (public domain). Avoid dictionaries/encyclopedias.
IDS = [
    "TheNoteBooksOfSamuelButler",
    "AHistoryOfPersia",
    "AkbarTheEmperorOfIndia",
    "JourneysInPersiaAndKurdistanVolII",
]

UA = {"User-Agent": "Mozilla/5.0 (base-scan-fetch)"}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def download(url, dest):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    return total


for ident in IDS:
    print(f"\n=== {ident} ===")
    try:
        meta = get_json(f"https://archive.org/metadata/{ident}")
    except Exception as e:
        print("  metadata failed:", e)
        continue

    files = meta.get("files", [])
    fmts = {f.get("format") for f in files}
    is_scan = any(k in fmts for k in
                  ("Single Page Processed JP2 ZIP", "Abbyy GZ", "DjVuTXT", "Scandata"))
    pdfs = [f for f in files if f.get("name", "").lower().endswith(".pdf")]
    if not pdfs:
        print("  no PDF derivative; skipping")
        continue
    # prefer the 'Text PDF' derivative, else the largest pdf
    pdfs.sort(key=lambda f: (f.get("format") == "Text PDF", int(f.get("size", 0))),
              reverse=True)
    pdf = pdfs[0]
    name = pdf["name"]
    size_mb = int(pdf.get("size", 0)) / 1e6
    print(f"  scan={is_scan}  pdf='{name}'  format='{pdf.get('format')}'  ~{size_mb:.1f} MB")

    dest = OUT / f"{ident}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        print("  already downloaded; skipping")
        continue
    url = f"https://archive.org/download/{ident}/{urllib.parse.quote(name)}"
    try:
        got = download(url, dest)
        print(f"  downloaded {got/1e6:.1f} MB -> {dest}")
    except Exception as e:
        print("  download failed:", e)

print("\nDone. Base scans in", OUT)
